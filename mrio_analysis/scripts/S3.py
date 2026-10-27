# -*- coding: utf-8 -*-
from pathlib import Path
import os
import numpy as np
import pandas as pd


# =====================================================
# S3_H1_I1I2I3: 生成 OD 反推边输入
# =====================================================
# 输入：
#   第五阶段情景分析new\I_H0\Intensity_step2_daily_node_flows\H1\I1\201903\event_window_daily_node_flows.csv
#   ...
#   H1\I2\...
#   H1\I3\...
#
# 输出：
#   第五阶段情景分析new\I_H0\Intensity_step3_edge_inputs\H1\I1\201903\
#       edge_input_real_2019-03-23.csv
#       edge_input_cf_2019-03-23.csv
#       event_window_daily_edge_predicted_counts.csv
#       edge_input_generation_summary.csv
#
# 核心逻辑：
#   1. daily node flow 接 node_id -> SITEREF -> EdgeID；
#   2. 根据 2019 历史 DIR_1 / DIR_2 计算方向比例；
#   3. 将节点总流量拆成 AB / BA 两个方向；
#   4. 多个监测点落在同一 EdgeID 时，按 date + EdgeID 聚合；
#   5. 和 edge template 合并，生成 OD 反推需要的 edge input。
# =====================================================


# =====================================================
# 1. 路径配置
# =====================================================

BASE_DIR = Path(os.environ.get("SCENARIO_DATA_ROOT", "data"))

SCENARIO_ROOT = (
    BASE_DIR /
    '第五阶段情景分析new/I_H0'
)

DAILY_ROOT = (
    SCENARIO_ROOT /
    r"Intensity_step2_daily_node_flows"
)

OUT_ROOT = (
    SCENARIO_ROOT /
    r"Intensity_step3_edge_inputs"
)

OUT_ROOT.mkdir(parents=True, exist_ok=True)

# 节点-边映射文件
NODE_EDGE_MAP_CSV = (
    BASE_DIR /
    '第五阶段_情景分析/I_Eobs_H7/scenario_analysis/step2_node_edge_mapping_check_201912/node_edge_map_130_checked.csv'
)

# 历史边观测，用于计算 DIR_1 / DIR_2 方向比例
EDGE_OBS_CSV = (
    BASE_DIR /
    '第二阶段/final_out/edge_observations_2019.csv'
)

# OD 反推边模板
EDGE_TEMPLATE_CSV = (
    BASE_DIR /
    '第二阶段New/final_out/edge_input_daily_with_ferry.csv'
)


# =====================================================
# 2. 参数
# =====================================================

H_NAME = 'H0'

INTENSITY_SCENARIOS = [
    "I1",
    "I2",
    "I3",
]

MONTHS = [
    "201903",
    "201908",
    "201911",
    "201912",
]

EPS = 1e-6

STRICT_NODE_EDGE_MAP = True

# 多个监测点落在同一条边时：
# mean = 平均，保持之前逻辑；
# sum  = 求和，一般不建议，会放大边流量。
EDGE_AGG_METHOD = "mean"


# =====================================================
# 3. 工具函数
# =====================================================

def clean_siteref_series(s):
    return (
        s.astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
        .str.zfill(8)
    )


def ensure_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]

    if missing:
        raise ValueError(f"{name} 缺少字段: {missing}")


def get_daily_node_path(intensity, month):
    return (
        DAILY_ROOT /
        H_NAME /
        intensity /
        month /
        "event_window_daily_node_flows.csv"
    )


def prepare_node_edge_map(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"找不到 NODE_EDGE_MAP_CSV: {path}")

    node_edge_map = pd.read_csv(path, dtype={"SITEREF": str})

    ensure_cols(
        node_edge_map,
        ["node_id", "SITEREF", "EdgeID"],
        "node_edge_map"
    )

    node_edge_map["node_id"] = pd.to_numeric(
        node_edge_map["node_id"],
        errors="coerce"
    )

    node_edge_map["SITEREF"] = clean_siteref_series(
        node_edge_map["SITEREF"]
    )

    node_edge_map["EdgeID"] = pd.to_numeric(
        node_edge_map["EdgeID"],
        errors="coerce"
    )

    if node_edge_map["node_id"].isna().any():
        bad = node_edge_map[node_edge_map["node_id"].isna()]
        raise ValueError(
            "node_edge_map 存在非法 node_id:\n"
            + str(bad.head(20))
        )

    if node_edge_map["EdgeID"].isna().any():
        bad = node_edge_map[node_edge_map["EdgeID"].isna()]
        raise ValueError(
            "node_edge_map 存在缺失 EdgeID:\n"
            + str(bad[["node_id", "SITEREF", "EdgeID"]].head(20))
        )

    node_edge_map["node_id"] = node_edge_map["node_id"].astype(int)
    node_edge_map["EdgeID"] = node_edge_map["EdgeID"].astype(int)

    node_edge_map = (
        node_edge_map[["node_id", "SITEREF", "EdgeID"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return node_edge_map


def prepare_direction_share(edge_obs_csv):
    edge_obs_csv = Path(edge_obs_csv)

    if not edge_obs_csv.exists():
        raise FileNotFoundError(f"找不到 EDGE_OBS_CSV: {edge_obs_csv}")

    edge_obs = pd.read_csv(
        edge_obs_csv,
        parse_dates=["DATETIME"],
        dtype={"SITEREF": str},
        low_memory=False
    )

    ensure_cols(
        edge_obs,
        ["SITEREF", "DIR_1", "DIR_2"],
        "edge_observations"
    )

    edge_obs["SITEREF"] = clean_siteref_series(edge_obs["SITEREF"])

    edge_obs["DIR_1"] = pd.to_numeric(edge_obs["DIR_1"], errors="coerce")
    edge_obs["DIR_2"] = pd.to_numeric(edge_obs["DIR_2"], errors="coerce")

    edge_obs["dir_sum"] = edge_obs["DIR_1"] + edge_obs["DIR_2"]

    valid = edge_obs[
        edge_obs["DIR_1"].notna() &
        edge_obs["DIR_2"].notna() &
        (edge_obs["dir_sum"] > 0)
    ].copy()

    if valid.empty:
        raise RuntimeError(
            "edge_observations 中没有可用 DIR_1 / DIR_2 方向比例记录。"
        )

    valid["dir1_share_raw"] = valid["DIR_1"] / (valid["dir_sum"] + EPS)
    valid["dir2_share_raw"] = valid["DIR_2"] / (valid["dir_sum"] + EPS)

    dir_share = (
        valid
        .groupby("SITEREF", as_index=False)
        .agg(
            dir1_share=("dir1_share_raw", "mean"),
            dir2_share=("dir2_share_raw", "mean"),
            dir_obs_n=("DATETIME", "count"),
        )
    )

    share_sum = dir_share["dir1_share"] + dir_share["dir2_share"]

    dir_share["dir1_share"] = dir_share["dir1_share"] / (share_sum + EPS)
    dir_share["dir2_share"] = dir_share["dir2_share"] / (share_sum + EPS)

    return dir_share


def prepare_edge_template(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"找不到 EDGE_TEMPLATE_CSV: {path}")

    edge_template = pd.read_csv(path)

    required_cols = [
        "EdgeID",
        "FromNode",
        "ToNode",
        "AB_FF_Time",
        "BA_FF_Time",
        "AB_Capacity",
        "BA_Capacity",
        "Length_m",
    ]

    ensure_cols(edge_template, required_cols, "edge_template")

    edge_template["EdgeID"] = pd.to_numeric(
        edge_template["EdgeID"],
        errors="coerce"
    )

    edge_template = edge_template.dropna(subset=["EdgeID"]).copy()
    edge_template["EdgeID"] = edge_template["EdgeID"].astype(int)

    # 防止 EdgeID 重复导致 merge 膨胀
    edge_template = edge_template.drop_duplicates(subset=["EdgeID"]).copy()

    return edge_template


def load_daily_node_flow(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"找不到 daily node flow: {path}")

    daily = pd.read_csv(path)

    required_cols = [
        "date",
        "node_id",
        "pred_real_daily",
        "pred_cf_daily",
    ]

    ensure_cols(daily, required_cols, "daily node flow")

    daily["date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")

    daily["node_id"] = pd.to_numeric(
        daily["node_id"],
        errors="coerce"
    )

    if daily["node_id"].isna().any():
        bad = daily[daily["node_id"].isna()]
        raise ValueError(
            "daily node flow 存在非法 node_id:\n"
            + str(bad.head(20))
        )

    daily["node_id"] = daily["node_id"].astype(int)

    daily["pred_real_daily"] = pd.to_numeric(
        daily["pred_real_daily"],
        errors="coerce"
    ).fillna(0.0)

    daily["pred_cf_daily"] = pd.to_numeric(
        daily["pred_cf_daily"],
        errors="coerce"
    ).fillna(0.0)

    if "period_type" not in daily.columns:
        daily["period_type"] = "h1_official"

    if "H_name" not in daily.columns:
        daily["H_name"] = H_NAME

    return daily


def aggregate_edge_daily(daily_site):
    if EDGE_AGG_METHOD == "mean":
        edge_daily = (
            daily_site
            .groupby(["date", "EdgeID"], as_index=False)
            .agg(
                real_Count_1=("real_Count_1", "mean"),
                real_Count_2=("real_Count_2", "mean"),
                cf_Count_1=("cf_Count_1", "mean"),
                cf_Count_2=("cf_Count_2", "mean"),

                site_n=("SITEREF", "nunique"),
                node_n=("node_id", "nunique"),

                mean_dir1_share=("dir1_share", "mean"),
                mean_dir2_share=("dir2_share", "mean"),

                period_type=("period_type", "first"),
                H_name=("H_name", "first"),
                intensity=("intensity", "first"),
                month=("month", "first"),
            )
        )

    elif EDGE_AGG_METHOD == "sum":
        edge_daily = (
            daily_site
            .groupby(["date", "EdgeID"], as_index=False)
            .agg(
                real_Count_1=("real_Count_1", "sum"),
                real_Count_2=("real_Count_2", "sum"),
                cf_Count_1=("cf_Count_1", "sum"),
                cf_Count_2=("cf_Count_2", "sum"),

                site_n=("SITEREF", "nunique"),
                node_n=("node_id", "nunique"),

                mean_dir1_share=("dir1_share", "mean"),
                mean_dir2_share=("dir2_share", "mean"),

                period_type=("period_type", "first"),
                H_name=("H_name", "first"),
                intensity=("intensity", "first"),
                month=("month", "first"),
            )
        )

    else:
        raise ValueError(f"未知 EDGE_AGG_METHOD: {EDGE_AGG_METHOD}")

    return edge_daily


# =====================================================
# 4. 单个 H1-intensity-month 处理
# =====================================================

def process_one(intensity, month, node_edge_map, dir_share, edge_template):
    print("\n" + "=" * 80)
    print(f"S3 processing: {H_NAME} {intensity} {month}")
    print("=" * 80)

    daily_path = get_daily_node_path(
        intensity=intensity,
        month=month
    )

    out_dir = OUT_ROOT / H_NAME / intensity / month
    out_dir.mkdir(parents=True, exist_ok=True)

    print("daily node:", daily_path)
    print("out_dir   :", out_dir)

    daily = load_daily_node_flow(daily_path)

    print("daily rows:", len(daily))
    print("日期数:", daily["date"].nunique())
    print("node 数:", daily["node_id"].nunique())
    print("日期列表:", sorted(daily["date"].unique()))

    # =====================================================
    # 1. node_id -> SITEREF -> EdgeID
    # =====================================================

    daily_site = daily.merge(
        node_edge_map,
        on="node_id",
        how="left"
    )

    missing_siteref = int(daily_site["SITEREF"].isna().sum())
    missing_edgeid = int(daily_site["EdgeID"].isna().sum())

    print("合并 node_edge_map 后 rows:", len(daily_site))
    print("缺失 SITEREF:", missing_siteref)
    print("缺失 EdgeID:", missing_edgeid)

    if missing_siteref > 0 or missing_edgeid > 0:
        missing_df = daily_site.loc[
            daily_site["SITEREF"].isna() |
            daily_site["EdgeID"].isna(),
            ["date", "node_id", "SITEREF", "EdgeID"]
        ].drop_duplicates()

        missing_path = out_dir / "missing_node_edge_mapping.csv"

        missing_df.to_csv(
            missing_path,
            index=False,
            encoding="utf-8-sig"
        )

        print("缺失映射已保存:", missing_path)

        if STRICT_NODE_EDGE_MAP:
            raise RuntimeError("node flow 到 EdgeID 映射存在缺失，请先修复。")
        else:
            daily_site = daily_site.dropna(subset=["SITEREF", "EdgeID"]).copy()

    daily_site["EdgeID"] = daily_site["EdgeID"].astype(int)

    # =====================================================
    # 2. 接方向比例
    # =====================================================

    daily_site = daily_site.merge(
        dir_share[["SITEREF", "dir1_share", "dir2_share", "dir_obs_n"]],
        on="SITEREF",
        how="left"
    )

    daily_site["dir1_share"] = daily_site["dir1_share"].fillna(0.5)
    daily_site["dir2_share"] = daily_site["dir2_share"].fillna(0.5)
    daily_site["dir_obs_n"] = daily_site["dir_obs_n"].fillna(0)

    share_sum = daily_site["dir1_share"] + daily_site["dir2_share"]

    daily_site["dir1_share"] = daily_site["dir1_share"] / (share_sum + EPS)
    daily_site["dir2_share"] = daily_site["dir2_share"] / (share_sum + EPS)

    no_dir_n = int((daily_site["dir_obs_n"] == 0).sum())
    print("无方向比例记录数，使用 0.5/0.5:", no_dir_n)

    # =====================================================
    # 3. 节点总流量拆成两个方向
    # =====================================================

    daily_site["real_Count_1"] = (
        daily_site["pred_real_daily"] *
        daily_site["dir1_share"]
    )

    daily_site["real_Count_2"] = (
        daily_site["pred_real_daily"] *
        daily_site["dir2_share"]
    )

    daily_site["cf_Count_1"] = (
        daily_site["pred_cf_daily"] *
        daily_site["dir1_share"]
    )

    daily_site["cf_Count_2"] = (
        daily_site["pred_cf_daily"] *
        daily_site["dir2_share"]
    )

    for c in ["real_Count_1", "real_Count_2", "cf_Count_1", "cf_Count_2"]:
        daily_site[c] = pd.to_numeric(
            daily_site[c],
            errors="coerce"
        ).fillna(0.0).clip(lower=0.0)

    daily_site_csv = out_dir / "daily_site_mapped_flows.csv"

    daily_site.to_csv(
        daily_site_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # 4. 聚合到 EdgeID
    # =====================================================

    edge_daily = aggregate_edge_daily(daily_site)

    edge_daily["real_total_count"] = (
        edge_daily["real_Count_1"] +
        edge_daily["real_Count_2"]
    )

    edge_daily["cf_total_count"] = (
        edge_daily["cf_Count_1"] +
        edge_daily["cf_Count_2"]
    )

    edge_daily["count_loss_cf_minus_real"] = (
        edge_daily["cf_total_count"] -
        edge_daily["real_total_count"]
    )

    edge_daily["count_loss_ratio"] = (
        edge_daily["count_loss_cf_minus_real"] /
        (edge_daily["cf_total_count"] + EPS)
    )

    print("\n聚合到 EdgeID 后:")
    print("edge_daily rows:", len(edge_daily))
    print("日期数:", edge_daily["date"].nunique())
    print("唯一 EdgeID 数:", edge_daily["EdgeID"].nunique())
    print("多站点 edge-date 数:", int((edge_daily["site_n"] > 1).sum()))

    edge_daily_csv = out_dir / "event_window_daily_edge_predicted_counts.csv"

    edge_daily.to_csv(
        edge_daily_csv,
        index=False,
        encoding="utf-8-sig"
    )

    edge_daily_named_csv = (
        out_dir /
        f"{month}_{H_NAME}_{intensity}_daily_edge_predicted_counts.csv"
    )

    edge_daily.to_csv(
        edge_daily_named_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("[SAVED intermediate]", edge_daily_csv)

    # =====================================================
    # 5. 每天生成 real / cf 两套 OD 反推边输入
    # =====================================================

    final_cols = [
        "EdgeID",
        "FromNode",
        "ToNode",
        "AB_FF_Time",
        "BA_FF_Time",
        "AB_Capacity",
        "BA_Capacity",
        "AB_Count",
        "BA_Count",
        "Length_m",
    ]

    dates = sorted(edge_daily["date"].unique())

    summary_rows = []

    for d_str in dates:
        day_edge = edge_daily[edge_daily["date"] == d_str].copy()

        # -----------------------------
        # real / scenario weather
        # -----------------------------

        real_counts = (
            day_edge[[
                "EdgeID",
                "real_Count_1",
                "real_Count_2",
            ]]
            .rename(
                columns={
                    "real_Count_1": "AB_Count",
                    "real_Count_2": "BA_Count",
                }
            )
        )

        real_input = (
            edge_template
            .drop(columns=["AB_Count", "BA_Count"], errors="ignore")
            .merge(real_counts, on="EdgeID", how="left")
        )

        real_input = (
            real_input[final_cols]
            .sort_values("EdgeID")
            .reset_index(drop=True)
        )

        real_path = out_dir / f"edge_input_real_{d_str}.csv"

        real_input.to_csv(
            real_path,
            index=False,
            encoding="utf-8-sig"
        )

        # -----------------------------
        # cf / normal weather
        # -----------------------------

        cf_counts = (
            day_edge[[
                "EdgeID",
                "cf_Count_1",
                "cf_Count_2",
            ]]
            .rename(
                columns={
                    "cf_Count_1": "AB_Count",
                    "cf_Count_2": "BA_Count",
                }
            )
        )

        cf_input = (
            edge_template
            .drop(columns=["AB_Count", "BA_Count"], errors="ignore")
            .merge(cf_counts, on="EdgeID", how="left")
        )

        cf_input = (
            cf_input[final_cols]
            .sort_values("EdgeID")
            .reset_index(drop=True)
        )

        cf_path = out_dir / f"edge_input_cf_{d_str}.csv"

        cf_input.to_csv(
            cf_path,
            index=False,
            encoding="utf-8-sig"
        )

        real_total = float(
            real_input[["AB_Count", "BA_Count"]]
            .sum(skipna=True)
            .sum()
        )

        cf_total = float(
            cf_input[["AB_Count", "BA_Count"]]
            .sum(skipna=True)
            .sum()
        )

        summary_rows.append({
            "H_name": H_NAME,
            "intensity": intensity,
            "month": month,
            "date": d_str,

            "real_file": str(real_path),
            "cf_file": str(cf_path),

            "observed_edges": int(len(day_edge)),

            "real_AB_notna": int(real_input["AB_Count"].notna().sum()),
            "real_BA_notna": int(real_input["BA_Count"].notna().sum()),
            "cf_AB_notna": int(cf_input["AB_Count"].notna().sum()),
            "cf_BA_notna": int(cf_input["BA_Count"].notna().sum()),

            "real_AB_mean": float(real_input["AB_Count"].mean(skipna=True)),
            "real_BA_mean": float(real_input["BA_Count"].mean(skipna=True)),
            "cf_AB_mean": float(cf_input["AB_Count"].mean(skipna=True)),
            "cf_BA_mean": float(cf_input["BA_Count"].mean(skipna=True)),

            "real_total_count": real_total,
            "cf_total_count": cf_total,

            "count_loss_cf_minus_real": cf_total - real_total,
            "count_loss_ratio": (
                (cf_total - real_total) /
                (cf_total + EPS)
            ),
        })

        print(f"[SAVE] {H_NAME} {intensity} {month} {d_str}")
        print("  real:", real_path)
        print("  cf  :", cf_path)

    # =====================================================
    # 6. 保存汇总
    # =====================================================

    summary_df = pd.DataFrame(summary_rows)

    summary_csv = out_dir / "edge_input_generation_summary.csv"

    summary_df.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n[SAVED summary]", summary_csv)
    print(summary_df)

    return summary_df


# =====================================================
# 5. 主程序
# =====================================================

def main():
    print("=" * 80)
    print('S3_H0_I1I2I3: 生成 OD 反推边输入')
    print("=" * 80)

    print("SCENARIO_ROOT:", SCENARIO_ROOT)
    print("DAILY_ROOT:", DAILY_ROOT)
    print("OUT_ROOT:", OUT_ROOT)
    print("NODE_EDGE_MAP_CSV:", NODE_EDGE_MAP_CSV)
    print("EDGE_OBS_CSV:", EDGE_OBS_CSV)
    print("EDGE_TEMPLATE_CSV:", EDGE_TEMPLATE_CSV)
    print("H_NAME:", H_NAME)
    print("INTENSITY_SCENARIOS:", INTENSITY_SCENARIOS)
    print("MONTHS:", MONTHS)
    print("EDGE_AGG_METHOD:", EDGE_AGG_METHOD)

    node_edge_map = prepare_node_edge_map(NODE_EDGE_MAP_CSV)

    print("\nnode_edge_map:")
    print("rows:", len(node_edge_map))
    print("node 数:", node_edge_map["node_id"].nunique())
    print("SITEREF 数:", node_edge_map["SITEREF"].nunique())
    print("EdgeID 数:", node_edge_map["EdgeID"].nunique())

    dir_share = prepare_direction_share(EDGE_OBS_CSV)

    print("\ndir_share:")
    print("SITEREF 数:", dir_share["SITEREF"].nunique())
    print(
        "dir_obs_n min/max:",
        dir_share["dir_obs_n"].min(),
        dir_share["dir_obs_n"].max()
    )

    edge_template = prepare_edge_template(EDGE_TEMPLATE_CSV)

    print("\nedge_template:")
    print("rows:", len(edge_template))
    print("EdgeID 数:", edge_template["EdgeID"].nunique())

    all_summary = []

    for intensity in INTENSITY_SCENARIOS:
        for month in MONTHS:
            summary_df = process_one(
                intensity=intensity,
                month=month,
                node_edge_map=node_edge_map,
                dir_share=dir_share,
                edge_template=edge_template,
            )

            all_summary.append(summary_df)

    all_summary_df = pd.concat(
        all_summary,
        ignore_index=True
    )

    all_summary_csv = (
        OUT_ROOT /
        H_NAME /
        "all_edge_input_generation_summary.csv"
    )

    all_summary_csv.parent.mkdir(parents=True, exist_ok=True)

    all_summary_df.to_csv(
        all_summary_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # 强度汇总
    # =====================================================

    intensity_rows = []

    for intensity, g in all_summary_df.groupby("intensity"):
        real_total = float(g["real_total_count"].sum())
        cf_total = float(g["cf_total_count"].sum())

        loss = cf_total - real_total

        intensity_rows.append({
            "H_name": H_NAME,
            "intensity": intensity,
            "month_count": int(g["month"].nunique()),
            "day_count": int(len(g)),

            "real_total_count": real_total,
            "cf_total_count": cf_total,

            "count_loss_cf_minus_real": loss,
            "count_loss_ratio": loss / (cf_total + EPS),

            "count_loss_percent": loss / (cf_total + EPS) * 100.0,

            "mean_daily_count_loss_percent": float(
                g["count_loss_ratio"].mean() * 100.0
            ),
            "max_daily_count_loss_percent": float(
                g["count_loss_ratio"].max() * 100.0
            ),
            "min_daily_count_loss_percent": float(
                g["count_loss_ratio"].min() * 100.0
            ),
        })

    intensity_summary_df = pd.DataFrame(intensity_rows)

    intensity_summary_csv = (
        OUT_ROOT /
        H_NAME /
        'H0_I1I2I3_edge_input_intensity_summary.csv'
    )

    intensity_summary_df.to_csv(
        intensity_summary_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 80)
    print('✅ S3_H0_I1I2I3 完成')
    print("=" * 80)

    print("总汇总文件:", all_summary_csv)
    print("强度汇总文件:", intensity_summary_csv)

    print("\n强度汇总:")
    print(intensity_summary_df)

    print("\n重点检查：")
    print("1. day_count 应为 21：201903=5, 201908=5, 201911=4, 201912=7。")
    print("2. count_loss_percent 应整体 I1 < I2 < I3。")
    print("3. real_AB_notna / cf_AB_notna 数量应一致。")
    print("4. 输出 edge_input_real_日期.csv 和 edge_input_cf_日期.csv 将用于 S4 OD 反推。")


if __name__ == "__main__":
    main()