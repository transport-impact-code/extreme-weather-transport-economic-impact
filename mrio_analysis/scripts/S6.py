# -*- coding: utf-8 -*-
from pathlib import Path
import os
import numpy as np
import pandas as pd


# =========================================================
# S6_H1_I1I2I3: 由补对角线 OD 生成 capacity factor / transport shock
# =========================================================
# 输入：
#   Intensity_step5_od_filled_diag\H1\I1\201903\2019-03-23\real\region_od_filled_diag.csv
#   Intensity_step5_od_filled_diag\H1\I1\201903\2019-03-23\cf\region_od_filled_diag.csv
#   ...
#
# 输出：
#   Intensity_step6_capacity_factor\H1\I1\201903\2019-03-23\
#       od_capacity_factor_matrix.csv
#       od_loss_matrix.csv
#       od_real_matrix.csv
#       od_cf_matrix.csv
#       regional_transport_factor.csv
#       capacity_factor_summary.csv
#
# 核心逻辑：
#   1. 读取 real/cf OD；
#   2. 计算 OD 层面的有效损失：
#        effective_OD_loss = max(OD_cf - OD_real, 0)
#   3. 计算 OD capacity factor：
#        factor_ij = min(OD_real / OD_cf, 1)
#   4. 计算区域 transport factor：
#        region_factor_r = related_real / related_cf
#      其中 related = outgoing + incoming - diagonal，避免对角线重复。
#   5. factor 限制在 [MIN_FACTOR, 1]。
# =========================================================


# =========================================================
# 1. 路径配置
# =========================================================

BASE_DIR = Path(os.environ.get("SCENARIO_DATA_ROOT", "data"))

SCENARIO_ROOT = (
    BASE_DIR /
    '第五阶段情景分析new/I_H0'
)

S5_ROOT = (
    SCENARIO_ROOT /
    r"Intensity_step5_od_filled_diag"
)

OUT_ROOT = (
    SCENARIO_ROOT /
    r"Intensity_step6_capacity_factor"
)

OUT_ROOT.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. 参数配置
# =========================================================

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

# capacity factor 下限。
# 不建议设成 0，否则后续 MRIO 可能过强断崖。
MIN_FACTOR = 0.05

# 是否把 real > cf 的 OD 单元视为没有损失
CLIP_GAIN_TO_ZERO = True

# 区域冲击是否同时考虑出发和到达 OD
# related = outgoing + incoming - diagonal
REGIONAL_FACTOR_MODE = "out_in"

SKIP_IF_DONE = False


# =========================================================
# 3. 区域名称标准化
# =========================================================

name_map = {
    "Auckland": "Auckland",
    "Bay of Plenty Region": "Bay of Plenty",
    "Canterbury Region": "Canterbury",
    "Gisborne Region": "Gisborne",
    "Hawke's Bay Region": "Hawke's Bay",
    "Manawatū-Whanganui Region": "Manawatū-Whanganui",
    "Manawatu-Whanganui Region": "Manawatū-Whanganui",
    "Marlborough Region": "Marlborough",
    "Nelson-Tasman Region": "Nelson, Tasman",
    "Nelson, Tasman Region": "Nelson, Tasman",
    "Northland Region": "Northland",
    "Otago Region": "Otago",
    "Southland Region": "Southland",
    "Taranaki Region": "Taranaki",
    "Waikato Region": "Waikato",
    "Wellington Region": "Wellington",
    "West Coast Region": "West Coast",

    "Bay of Plenty": "Bay of Plenty",
    "Canterbury": "Canterbury",
    "Gisborne": "Gisborne",
    "Hawke's Bay": "Hawke's Bay",
    "Manawatū-Whanganui": "Manawatū-Whanganui",
    "Manawatu-Whanganui": "Manawatū-Whanganui",
    "Marlborough": "Marlborough",
    "Nelson, Tasman": "Nelson, Tasman",
    "Nelson-Tasman": "Nelson, Tasman",
    "Northland": "Northland",
    "Otago": "Otago",
    "Southland": "Southland",
    "Taranaki": "Taranaki",
    "Waikato": "Waikato",
    "Wellington": "Wellington",
    "West Coast": "West Coast",
}


def normalize_region_name(x):
    if pd.isna(x):
        return x

    x = str(x).strip()

    return name_map.get(x, x)


# =========================================================
# 4. 工具函数
# =========================================================

def ensure_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]

    if missing:
        raise ValueError(f"{name} 缺少字段: {missing}")


def read_od_long(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"找不到 OD 文件: {path}")

    df = pd.read_csv(path)

    ensure_cols(
        df,
        ["O_region", "D_region", "OD_value"],
        str(path)
    )

    df = df[["O_region", "D_region", "OD_value"]].copy()

    df["O_region"] = df["O_region"].apply(normalize_region_name)
    df["D_region"] = df["D_region"].apply(normalize_region_name)

    df["OD_value"] = pd.to_numeric(
        df["OD_value"],
        errors="coerce"
    ).fillna(0.0)

    df = (
        df
        .groupby(["O_region", "D_region"], as_index=False)
        ["OD_value"]
        .sum()
    )

    return df


def long_to_matrix(df, region_order):
    mat = (
        df
        .pivot(
            index="O_region",
            columns="D_region",
            values="OD_value"
        )
        .fillna(0.0)
    )

    mat = mat.reindex(
        index=region_order,
        columns=region_order,
        fill_value=0.0
    )

    return mat.astype(float)


def matrix_to_long(mat, value_name):
    return (
        mat
        .rename_axis(index="O_region", columns="D_region")
        .stack()
        .reset_index(name=value_name)
    )


def get_region_order_from_one_file():
    for intensity in INTENSITY_SCENARIOS:
        for month in MONTHS:
            month_dir = S5_ROOT / H_NAME / intensity / month

            if not month_dir.exists():
                continue

            date_dirs = sorted([p for p in month_dir.iterdir() if p.is_dir()])

            for date_dir in date_dirs:
                f = date_dir / "cf" / "region_od_filled_diag.csv"

                if f.exists():
                    df = read_od_long(f)

                    regions = []

                    for c in ["O_region", "D_region"]:
                        for x in df[c].dropna().tolist():
                            if x not in regions:
                                regions.append(x)

                    return regions

    raise RuntimeError("无法从 S5 输出中识别区域顺序。")


def get_s5_file(intensity, month, date_str, flow_case):
    return (
        S5_ROOT /
        H_NAME /
        intensity /
        month /
        date_str /
        flow_case /
        "region_od_filled_diag.csv"
    )


def get_out_dir(intensity, month, date_str):
    return (
        OUT_ROOT /
        H_NAME /
        intensity /
        month /
        date_str
    )


def discover_tasks():
    rows = []

    for intensity in INTENSITY_SCENARIOS:
        for month in MONTHS:
            month_dir = S5_ROOT / H_NAME / intensity / month

            if not month_dir.exists():
                raise FileNotFoundError(f"找不到 S5 month_dir: {month_dir}")

            date_dirs = sorted([p for p in month_dir.iterdir() if p.is_dir()])

            if len(date_dirs) == 0:
                raise RuntimeError(f"{month_dir} 下没有日期目录。")

            for date_dir in date_dirs:
                date_str = date_dir.name

                real_file = get_s5_file(
                    intensity=intensity,
                    month=month,
                    date_str=date_str,
                    flow_case="real"
                )

                cf_file = get_s5_file(
                    intensity=intensity,
                    month=month,
                    date_str=date_str,
                    flow_case="cf"
                )

                if not real_file.exists():
                    raise FileNotFoundError(f"缺少 real OD: {real_file}")

                if not cf_file.exists():
                    raise FileNotFoundError(f"缺少 cf OD: {cf_file}")

                rows.append({
                    "H_name": H_NAME,
                    "intensity": intensity,
                    "month": month,
                    "date": date_str,
                    "real_file": str(real_file),
                    "cf_file": str(cf_file),
                    "out_dir": str(get_out_dir(intensity, month, date_str)),
                })

    task_df = pd.DataFrame(rows)

    task_df = (
        task_df
        .sort_values(["intensity", "month", "date"])
        .reset_index(drop=True)
    )

    return task_df


def compute_regional_factor(real_mat, cf_mat):
    regions = list(cf_mat.index)

    rows = []

    for region in regions:
        if REGIONAL_FACTOR_MODE == "out":
            cf_related = float(cf_mat.loc[region, :].sum())
            real_related = float(real_mat.loc[region, :].sum())

        elif REGIONAL_FACTOR_MODE == "in":
            cf_related = float(cf_mat.loc[:, region].sum())
            real_related = float(real_mat.loc[:, region].sum())

        elif REGIONAL_FACTOR_MODE == "out_in":
            cf_related = float(
                cf_mat.loc[region, :].sum()
                + cf_mat.loc[:, region].sum()
                - cf_mat.loc[region, region]
            )

            real_related = float(
                real_mat.loc[region, :].sum()
                + real_mat.loc[:, region].sum()
                - real_mat.loc[region, region]
            )

        else:
            raise ValueError(f"未知 REGIONAL_FACTOR_MODE: {REGIONAL_FACTOR_MODE}")

        raw_factor = real_related / (cf_related + EPS)

        if CLIP_GAIN_TO_ZERO:
            factor = min(raw_factor, 1.0)
        else:
            factor = raw_factor

        factor = max(MIN_FACTOR, factor)

        loss = max(cf_related - real_related, 0.0)

        rows.append({
            "region": region,
            "cf_related_OD": cf_related,
            "real_related_OD": real_related,
            "raw_transport_factor": raw_factor,
            "transport_factor": factor,
            "transport_shock": 1.0 - factor,
            "effective_related_OD_loss": loss,
            "effective_related_OD_loss_percent": (
                loss / (cf_related + EPS) * 100.0
            ),
        })

    return pd.DataFrame(rows)


# =========================================================
# 5. 单日期处理
# =========================================================

def process_one(row, region_order):
    intensity = row["intensity"]
    month = row["month"]
    date_str = row["date"]

    out_dir = Path(row["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "capacity_factor_summary.csv"
    regional_csv = out_dir / "regional_transport_factor.csv"
    od_factor_csv = out_dir / "od_capacity_factor_matrix.csv"
    od_loss_csv = out_dir / "od_loss_matrix.csv"

    if (
        SKIP_IF_DONE
        and summary_csv.exists()
        and regional_csv.exists()
        and od_factor_csv.exists()
        and od_loss_csv.exists()
    ):
        print("[SKIP existing]", out_dir)
        return pd.read_csv(summary_csv).iloc[0].to_dict()

    real_long = read_od_long(row["real_file"])
    cf_long = read_od_long(row["cf_file"])

    real_mat = long_to_matrix(real_long, region_order)
    cf_mat = long_to_matrix(cf_long, region_order)

    # 防止负数
    real_mat = real_mat.clip(lower=0.0)
    cf_mat = cf_mat.clip(lower=0.0)

    # =====================================================
    # OD 层面 factor / loss
    # =====================================================

    with np.errstate(divide="ignore", invalid="ignore"):
        raw_factor_values = real_mat.values / (cf_mat.values + EPS)

    raw_factor = pd.DataFrame(
        raw_factor_values,
        index=region_order,
        columns=region_order
    )

    if CLIP_GAIN_TO_ZERO:
        od_factor = raw_factor.clip(lower=MIN_FACTOR, upper=1.0)
    else:
        od_factor = raw_factor.clip(lower=MIN_FACTOR)

    od_loss = cf_mat - real_mat

    if CLIP_GAIN_TO_ZERO:
        od_loss = od_loss.clip(lower=0.0)

    od_loss_percent = od_loss / (cf_mat + EPS) * 100.0

    od_shock = 1.0 - od_factor

    # =====================================================
    # 区域层面 factor
    # =====================================================

    regional_df = compute_regional_factor(
        real_mat=real_mat,
        cf_mat=cf_mat
    )

    regional_df["H_name"] = H_NAME
    regional_df["intensity"] = intensity
    regional_df["month"] = month
    regional_df["date"] = date_str

    # =====================================================
    # 汇总
    # =====================================================

    total_real_OD = float(real_mat.values.sum())
    total_cf_OD = float(cf_mat.values.sum())

    total_raw_loss = total_cf_OD - total_real_OD

    total_effective_OD_loss = float(od_loss.values.sum())

    total_effective_OD_loss_percent = (
        total_effective_OD_loss /
        (total_cf_OD + EPS) *
        100.0
    )

    weighted_capacity_factor = 1.0 - (
        total_effective_OD_loss /
        (total_cf_OD + EPS)
    )

    weighted_capacity_factor = max(
        MIN_FACTOR,
        min(1.0, weighted_capacity_factor)
    )

    min_od_factor = float(od_factor.values.min())
    mean_od_factor = float(od_factor.values.mean())
    max_od_factor = float(od_factor.values.max())

    min_regional_factor = float(regional_df["transport_factor"].min())
    mean_regional_factor = float(regional_df["transport_factor"].mean())
    max_regional_factor = float(regional_df["transport_factor"].max())

    max_regional_shock = float(regional_df["transport_shock"].max())
    mean_regional_shock = float(regional_df["transport_shock"].mean())

    summary = {
        "H_name": H_NAME,
        "intensity": intensity,
        "month": month,
        "date": date_str,

        "real_file": row["real_file"],
        "cf_file": row["cf_file"],
        "out_dir": str(out_dir),

        "total_OD_real": total_real_OD,
        "total_OD_cf": total_cf_OD,

        "raw_OD_loss_cf_minus_real": total_raw_loss,
        "raw_OD_loss_percent": (
            total_raw_loss / (total_cf_OD + EPS) * 100.0
        ),

        "total_effective_OD_loss": total_effective_OD_loss,
        "effective_OD_loss_percent": total_effective_OD_loss_percent,

        "weighted_capacity_factor": weighted_capacity_factor,
        "weighted_transport_shock": 1.0 - weighted_capacity_factor,

        "min_OD_capacity_factor": min_od_factor,
        "mean_OD_capacity_factor": mean_od_factor,
        "max_OD_capacity_factor": max_od_factor,

        "min_regional_transport_factor": min_regional_factor,
        "mean_regional_transport_factor": mean_regional_factor,
        "max_regional_transport_factor": max_regional_factor,

        "mean_regional_transport_shock": mean_regional_shock,
        "max_regional_transport_shock": max_regional_shock,

        "MIN_FACTOR": MIN_FACTOR,
        "CLIP_GAIN_TO_ZERO": CLIP_GAIN_TO_ZERO,
        "REGIONAL_FACTOR_MODE": REGIONAL_FACTOR_MODE,

        "region_count": len(region_order),
    }

    # =====================================================
    # 保存
    # =====================================================

    real_mat.to_csv(
        out_dir / "od_real_matrix.csv",
        encoding="utf-8-sig"
    )

    cf_mat.to_csv(
        out_dir / "od_cf_matrix.csv",
        encoding="utf-8-sig"
    )

    raw_factor.to_csv(
        out_dir / "od_raw_real_to_cf_ratio_matrix.csv",
        encoding="utf-8-sig"
    )

    od_factor.to_csv(
        od_factor_csv,
        encoding="utf-8-sig"
    )

    od_shock.to_csv(
        out_dir / "od_transport_shock_matrix.csv",
        encoding="utf-8-sig"
    )

    od_loss.to_csv(
        od_loss_csv,
        encoding="utf-8-sig"
    )

    od_loss_percent.to_csv(
        out_dir / "od_loss_percent_matrix.csv",
        encoding="utf-8-sig"
    )

    matrix_to_long(
        od_factor,
        "od_capacity_factor"
    ).to_csv(
        out_dir / "od_capacity_factor_long.csv",
        index=False,
        encoding="utf-8-sig"
    )

    matrix_to_long(
        od_loss,
        "effective_OD_loss"
    ).to_csv(
        out_dir / "od_loss_long.csv",
        index=False,
        encoding="utf-8-sig"
    )

    regional_df.to_csv(
        regional_csv,
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame([summary]).to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig"
    )

    return summary


# =========================================================
# 6. 主程序
# =========================================================

def main():
    print("=" * 80)
    print('S6_H0_I1I2I3: 生成 OD capacity factor / regional transport factor')
    print("=" * 80)

    print("SCENARIO_ROOT:", SCENARIO_ROOT)
    print("S5_ROOT:", S5_ROOT)
    print("OUT_ROOT:", OUT_ROOT)
    print("H_NAME:", H_NAME)
    print("INTENSITY_SCENARIOS:", INTENSITY_SCENARIOS)
    print("MONTHS:", MONTHS)
    print("MIN_FACTOR:", MIN_FACTOR)
    print("CLIP_GAIN_TO_ZERO:", CLIP_GAIN_TO_ZERO)
    print("REGIONAL_FACTOR_MODE:", REGIONAL_FACTOR_MODE)
    print("SKIP_IF_DONE:", SKIP_IF_DONE)

    region_order = get_region_order_from_one_file()

    print("\n区域顺序:")
    for i, r in enumerate(region_order, start=1):
        print(f"{i:02d}. {r}")

    if len(region_order) != 15:
        print("⚠️ 警告：区域数量不是 15，请检查区域名称。")

    task_df = discover_tasks()

    task_csv = OUT_ROOT / H_NAME / "s6_capacity_factor_task_list.csv"
    task_csv.parent.mkdir(parents=True, exist_ok=True)

    task_df.to_csv(
        task_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n任务数:", len(task_df))
    print("任务表:", task_csv)

    # 理论任务数：
    # 3 intensities × 21 days = 63
    if len(task_df) != 63:
        print(
            f"⚠️ 当前任务数为 {len(task_df)}，理论 H1 应为 63。"
            "请检查 S5 是否完整。"
        )

    summary_rows = []

    for idx, row in task_df.iterrows():
        print("\n" + "=" * 80)
        print(
            f"[{idx + 1}/{len(task_df)}] "
            f"{row['H_name']} {row['intensity']} {row['month']} {row['date']}"
        )
        print("=" * 80)

        print("real:", row["real_file"])
        print("cf  :", row["cf_file"])
        print("out :", row["out_dir"])

        summary = process_one(
            row=row,
            region_order=region_order
        )

        summary_rows.append(summary)

        print("[DONE]")
        print(f"  total_OD_cf                : {summary['total_OD_cf']:.6f}")
        print(f"  total_OD_real              : {summary['total_OD_real']:.6f}")
        print(f"  effective_OD_loss_percent  : {summary['effective_OD_loss_percent']:.4f}%")
        print(f"  weighted_capacity_factor   : {summary['weighted_capacity_factor']:.6f}")
        print(f"  min regional factor        : {summary['min_regional_transport_factor']:.6f}")
        print(f"  max regional shock         : {summary['max_regional_transport_shock']:.6f}")

    summary_df = pd.DataFrame(summary_rows)

    summary_csv = OUT_ROOT / H_NAME / "all_capacity_factor_summary.csv"

    summary_df.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # 强度汇总
    # =====================================================

    intensity_rows = []

    for intensity, g in summary_df.groupby("intensity"):
        total_cf = float(g["total_OD_cf"].sum())
        total_real = float(g["total_OD_real"].sum())
        total_effective_loss = float(g["total_effective_OD_loss"].sum())

        weighted_capacity_factor = 1.0 - (
            total_effective_loss /
            (total_cf + EPS)
        )

        weighted_capacity_factor = max(
            MIN_FACTOR,
            min(1.0, weighted_capacity_factor)
        )

        intensity_rows.append({
            "H_name": H_NAME,
            "intensity": intensity,
            "month_count": int(g["month"].nunique()),
            "impact_days_count": int(len(g)),

            "total_OD_real": total_real,
            "total_OD_cf": total_cf,

            "total_effective_OD_loss": total_effective_loss,
            "effective_OD_loss_percent": (
                total_effective_loss /
                (total_cf + EPS) *
                100.0
            ),

            "weighted_capacity_factor": weighted_capacity_factor,
            "weighted_transport_shock": 1.0 - weighted_capacity_factor,

            "impact_transport_factor_min": float(
                g["weighted_capacity_factor"].min()
            ),
            "impact_transport_factor_mean": float(
                g["weighted_capacity_factor"].mean()
            ),
            "impact_transport_factor_max": float(
                g["weighted_capacity_factor"].max()
            ),

            "min_event_capacity_factor_matrix": float(
                g["min_OD_capacity_factor"].min()
            ),
            "mean_event_capacity_factor_matrix": float(
                g["mean_OD_capacity_factor"].mean()
            ),

            "min_regional_transport_factor": float(
                g["min_regional_transport_factor"].min()
            ),
            "mean_regional_transport_factor": float(
                g["mean_regional_transport_factor"].mean()
            ),
            "max_regional_transport_shock": float(
                g["max_regional_transport_shock"].max()
            ),
        })

    intensity_summary_df = pd.DataFrame(intensity_rows)

    intensity_summary_csv = (
        OUT_ROOT /
        H_NAME /
        'H0_I1I2I3_capacity_factor_intensity_summary.csv'
    )

    intensity_summary_df.to_csv(
        intensity_summary_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # 月份 × 强度汇总
    # =====================================================

    month_rows = []

    for (intensity, month), g in summary_df.groupby(["intensity", "month"]):
        total_cf = float(g["total_OD_cf"].sum())
        total_real = float(g["total_OD_real"].sum())
        total_effective_loss = float(g["total_effective_OD_loss"].sum())

        month_rows.append({
            "H_name": H_NAME,
            "intensity": intensity,
            "month": month,
            "impact_days_count": int(len(g)),

            "total_OD_real": total_real,
            "total_OD_cf": total_cf,

            "total_effective_OD_loss": total_effective_loss,
            "effective_OD_loss_percent": (
                total_effective_loss /
                (total_cf + EPS) *
                100.0
            ),

            "weighted_capacity_factor": 1.0 - (
                total_effective_loss /
                (total_cf + EPS)
            ),

            "min_regional_transport_factor": float(
                g["min_regional_transport_factor"].min()
            ),
            "mean_regional_transport_factor": float(
                g["mean_regional_transport_factor"].mean()
            ),
            "max_regional_transport_shock": float(
                g["max_regional_transport_shock"].max()
            ),
        })

    month_summary_df = pd.DataFrame(month_rows)

    month_summary_csv = (
        OUT_ROOT /
        H_NAME /
        'H0_I1I2I3_capacity_factor_month_summary.csv'
    )

    month_summary_df.to_csv(
        month_summary_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 80)
    print('✅ S6_H0_I1I2I3 capacity factor 生成完成')
    print("=" * 80)

    print("任务表:", task_csv)
    print("全部日期汇总:", summary_csv)
    print("强度汇总:", intensity_summary_csv)
    print("月份汇总:", month_summary_csv)

    print("\n强度汇总:")
    print(intensity_summary_df)

    print("\n重点检查：")
    print("1. 任务数应为 63：3 强度 × 21 天。")
    print("2. effective_OD_loss_percent 应整体 I1 < I2 < I3。")
    print("3. weighted_capacity_factor 应整体 I1 > I2 > I3。")
    print("4. min_regional_transport_factor 不应大面积等于 MIN_FACTOR。")
    print("5. 下一步是 S7：把 daily regional transport factor 整理成每期 OD / 每期区域冲击序列。")


if __name__ == "__main__":
    main()