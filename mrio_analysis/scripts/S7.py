# -*- coding: utf-8 -*-
from pathlib import Path
import os
import numpy as np
import pandas as pd


# =========================================================
# S7_H1_I1I2I3: 整理每日 OD / 区域冲击为 TT 序列
# =========================================================
# 输入：
#   Intensity_step6_capacity_factor\H1\I1\201903\2019-03-23\
#       regional_transport_factor.csv
#       od_capacity_factor_matrix.csv
#       od_transport_shock_matrix.csv
#       od_real_matrix.csv
#       od_cf_matrix.csv
#       od_loss_matrix.csv
#
# 输出：
#   Intensity_step7_TT_series\H1\I1\
#       time_index_impact_days.csv
#       regional_transport_factor_TT.npy
#       regional_transport_shock_TT.npy
#       od_capacity_factor_TT.npy
#       od_transport_shock_TT.npy
#       OD_real_TT.npy
#       OD_cf_TT.npy
#       OD_effective_loss_TT.npy
#       regional_transport_factor_long.csv
#       od_factor_loss_long.csv
#       daily_OD_summary.csv
#
# 核心逻辑：
#   1. 按日期读取 S6 每日 regional_transport_factor；
#   2. 组装为 T × R 的区域交通能力因子序列；
#   3. 按日期读取 OD factor / shock / real / cf / loss；
#   4. 组装为 T × R × R 的 OD 序列；
#   5. 这里 T = H1 官方窗口内可用冲击天数，共 21 天；
#   6. 下一步 S8 会把这 21 天嵌入 365 天年度序列。
# =========================================================


# =========================================================
# 1. 路径配置
# =========================================================

BASE_DIR = Path(os.environ.get("SCENARIO_DATA_ROOT", "data"))

SCENARIO_ROOT = (
    BASE_DIR /
    '第五阶段情景分析new/I_H0'
)

S6_ROOT = (
    SCENARIO_ROOT /
    r"Intensity_step6_capacity_factor"
)

OUT_ROOT = (
    SCENARIO_ROOT /
    r"Intensity_step7_TT_series"
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


def discover_tasks_one_intensity(intensity):
    rows = []

    for month in MONTHS:
        month_dir = S6_ROOT / H_NAME / intensity / month

        if not month_dir.exists():
            raise FileNotFoundError(f"找不到 S6 month_dir: {month_dir}")

        date_dirs = sorted([p for p in month_dir.iterdir() if p.is_dir()])

        if len(date_dirs) == 0:
            raise RuntimeError(f"{month_dir} 下没有日期目录。")

        for date_dir in date_dirs:
            date_str = date_dir.name

            regional_csv = date_dir / "regional_transport_factor.csv"
            od_factor_csv = date_dir / "od_capacity_factor_matrix.csv"
            od_shock_csv = date_dir / "od_transport_shock_matrix.csv"
            od_real_csv = date_dir / "od_real_matrix.csv"
            od_cf_csv = date_dir / "od_cf_matrix.csv"
            od_loss_csv = date_dir / "od_loss_matrix.csv"
            summary_csv = date_dir / "capacity_factor_summary.csv"

            need_files = [
                regional_csv,
                od_factor_csv,
                od_shock_csv,
                od_real_csv,
                od_cf_csv,
                od_loss_csv,
                summary_csv,
            ]

            for f in need_files:
                if not f.exists():
                    raise FileNotFoundError(f"缺少 S6 输出文件: {f}")

            rows.append({
                "H_name": H_NAME,
                "intensity": intensity,
                "month": month,
                "date": date_str,
                "regional_csv": str(regional_csv),
                "od_factor_csv": str(od_factor_csv),
                "od_shock_csv": str(od_shock_csv),
                "od_real_csv": str(od_real_csv),
                "od_cf_csv": str(od_cf_csv),
                "od_loss_csv": str(od_loss_csv),
                "summary_csv": str(summary_csv),
            })

    task_df = pd.DataFrame(rows)
    task_df["date_dt"] = pd.to_datetime(task_df["date"])

    task_df = (
        task_df
        .sort_values(["date_dt", "month"])
        .reset_index(drop=True)
    )

    task_df["t_index"] = np.arange(1, len(task_df) + 1)

    return task_df


def read_regional_factor(path):
    df = pd.read_csv(path)

    ensure_cols(
        df,
        [
            "region",
            "transport_factor",
            "transport_shock",
            "cf_related_OD",
            "real_related_OD",
            "effective_related_OD_loss",
        ],
        str(path)
    )

    df["region"] = df["region"].apply(normalize_region_name)

    numeric_cols = [
        "transport_factor",
        "transport_shock",
        "cf_related_OD",
        "real_related_OD",
        "effective_related_OD_loss",
        "effective_related_OD_loss_percent",
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df


def read_matrix(path, region_order):
    mat = pd.read_csv(path, index_col=0)

    mat.index = [normalize_region_name(x) for x in mat.index]
    mat.columns = [normalize_region_name(x) for x in mat.columns]

    mat = mat.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    mat = mat.reindex(
        index=region_order,
        columns=region_order,
        fill_value=0.0
    )

    return mat.astype(float)


def get_region_order_from_first_task(task_df):
    first_csv = task_df.iloc[0]["regional_csv"]

    df = read_regional_factor(first_csv)

    regions = df["region"].dropna().tolist()

    # 保持 S6 输出顺序，同时去重
    region_order = []
    for r in regions:
        if r not in region_order:
            region_order.append(r)

    if len(region_order) != 15:
        print("⚠️ 警告：区域数量不是 15，而是:", len(region_order))
        print(region_order)

    return region_order


def matrix_to_long(mat, value_name):
    return (
        mat
        .rename_axis(index="O_region", columns="D_region")
        .stack()
        .reset_index(name=value_name)
    )


def save_region_wide_csv(arr, time_df, region_order, value_name, out_csv):
    df = pd.DataFrame(arr, columns=region_order)
    df.insert(0, "date", time_df["date"].values)
    df.insert(0, "month", time_df["month"].values)
    df.insert(0, "t_index", time_df["t_index"].values)
    df.insert(0, "intensity", time_df["intensity"].values)
    df.insert(0, "H_name", time_df["H_name"].values)

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")


# =========================================================
# 5. 单个强度处理
# =========================================================

def process_one_intensity(intensity):
    print("\n" + "=" * 80)
    print(f"S7 processing: {H_NAME} {intensity}")
    print("=" * 80)

    out_dir = OUT_ROOT / H_NAME / intensity
    out_dir.mkdir(parents=True, exist_ok=True)

    time_index_csv = out_dir / "time_index_impact_days.csv"
    regional_factor_npy = out_dir / "regional_transport_factor_TT.npy"
    od_factor_npy = out_dir / "od_capacity_factor_TT.npy"
    daily_summary_csv = out_dir / "daily_OD_summary.csv"

    if (
        SKIP_IF_DONE
        and time_index_csv.exists()
        and regional_factor_npy.exists()
        and od_factor_npy.exists()
        and daily_summary_csv.exists()
    ):
        print("[SKIP existing]", out_dir)
        old_summary = pd.read_csv(daily_summary_csv)
        return old_summary

    task_df = discover_tasks_one_intensity(intensity)

    if len(task_df) != 21:
        print(
            f"⚠️ 当前 {intensity} 任务数为 {len(task_df)}，理论 H1 应为 21。"
            "请检查 S6 是否完整。"
        )

    region_order = get_region_order_from_first_task(task_df)

    R = len(region_order)
    T = len(task_df)

    print("impact days T:", T)
    print("regions R:", R)
    print("date range:", task_df["date"].min(), "to", task_df["date"].max())
    print("dates:", task_df["date"].tolist())

    # =====================================================
    # 初始化数组
    # =====================================================

    regional_factor_TT = np.ones((T, R), dtype=np.float32)
    regional_shock_TT = np.zeros((T, R), dtype=np.float32)

    regional_cf_related_TT = np.zeros((T, R), dtype=np.float32)
    regional_real_related_TT = np.zeros((T, R), dtype=np.float32)
    regional_loss_related_TT = np.zeros((T, R), dtype=np.float32)

    od_factor_TT = np.ones((T, R, R), dtype=np.float32)
    od_shock_TT = np.zeros((T, R, R), dtype=np.float32)

    OD_real_TT = np.zeros((T, R, R), dtype=np.float32)
    OD_cf_TT = np.zeros((T, R, R), dtype=np.float32)
    OD_effective_loss_TT = np.zeros((T, R, R), dtype=np.float32)

    regional_long_rows = []
    od_long_rows = []
    daily_rows = []

    # =====================================================
    # 逐日读取 S6 输出
    # =====================================================

    for idx, row in task_df.iterrows():
        t0 = int(row["t_index"]) - 1
        date_str = row["date"]
        month = row["month"]

        regional_df = read_regional_factor(row["regional_csv"])

        regional_df = (
            regional_df
            .set_index("region")
            .reindex(region_order)
            .reset_index()
        )

        if regional_df["transport_factor"].isna().any():
            raise RuntimeError(f"{date_str} regional_transport_factor 区域不完整。")

        regional_factor_TT[t0, :] = regional_df["transport_factor"].to_numpy(dtype=np.float32)
        regional_shock_TT[t0, :] = regional_df["transport_shock"].to_numpy(dtype=np.float32)

        regional_cf_related_TT[t0, :] = regional_df["cf_related_OD"].to_numpy(dtype=np.float32)
        regional_real_related_TT[t0, :] = regional_df["real_related_OD"].to_numpy(dtype=np.float32)
        regional_loss_related_TT[t0, :] = regional_df["effective_related_OD_loss"].to_numpy(dtype=np.float32)

        od_factor_mat = read_matrix(row["od_factor_csv"], region_order)
        od_shock_mat = read_matrix(row["od_shock_csv"], region_order)
        od_real_mat = read_matrix(row["od_real_csv"], region_order)
        od_cf_mat = read_matrix(row["od_cf_csv"], region_order)
        od_loss_mat = read_matrix(row["od_loss_csv"], region_order)

        od_factor_TT[t0, :, :] = od_factor_mat.values.astype(np.float32)
        od_shock_TT[t0, :, :] = od_shock_mat.values.astype(np.float32)

        OD_real_TT[t0, :, :] = od_real_mat.values.astype(np.float32)
        OD_cf_TT[t0, :, :] = od_cf_mat.values.astype(np.float32)
        OD_effective_loss_TT[t0, :, :] = od_loss_mat.values.astype(np.float32)

        # regional long
        tmp_reg = regional_df.copy()
        tmp_reg["H_name"] = H_NAME
        tmp_reg["intensity"] = intensity
        tmp_reg["month"] = month
        tmp_reg["date"] = date_str
        tmp_reg["t_index"] = int(row["t_index"])
        regional_long_rows.append(tmp_reg)

        # OD long
        od_long = matrix_to_long(od_factor_mat, "od_capacity_factor")
        od_long["od_transport_shock"] = matrix_to_long(
            od_shock_mat,
            "od_transport_shock"
        )["od_transport_shock"].values

        od_long["OD_real"] = matrix_to_long(
            od_real_mat,
            "OD_real"
        )["OD_real"].values

        od_long["OD_cf"] = matrix_to_long(
            od_cf_mat,
            "OD_cf"
        )["OD_cf"].values

        od_long["effective_OD_loss"] = matrix_to_long(
            od_loss_mat,
            "effective_OD_loss"
        )["effective_OD_loss"].values

        od_long["H_name"] = H_NAME
        od_long["intensity"] = intensity
        od_long["month"] = month
        od_long["date"] = date_str
        od_long["t_index"] = int(row["t_index"])

        od_long_rows.append(od_long)

        total_OD_cf = float(od_cf_mat.values.sum())
        total_OD_real = float(od_real_mat.values.sum())
        total_effective_loss = float(od_loss_mat.values.sum())

        weighted_factor = 1.0 - total_effective_loss / (total_OD_cf + EPS)

        daily_rows.append({
            "H_name": H_NAME,
            "intensity": intensity,
            "month": month,
            "date": date_str,
            "t_index": int(row["t_index"]),

            "total_OD_real": total_OD_real,
            "total_OD_cf": total_OD_cf,
            "total_effective_OD_loss": total_effective_loss,
            "effective_OD_loss_percent": (
                total_effective_loss / (total_OD_cf + EPS) * 100.0
            ),

            "weighted_capacity_factor": weighted_factor,
            "weighted_transport_shock": 1.0 - weighted_factor,

            "min_regional_transport_factor": float(regional_df["transport_factor"].min()),
            "mean_regional_transport_factor": float(regional_df["transport_factor"].mean()),
            "max_regional_transport_factor": float(regional_df["transport_factor"].max()),

            "max_regional_transport_shock": float(regional_df["transport_shock"].max()),
            "mean_regional_transport_shock": float(regional_df["transport_shock"].mean()),

            "min_OD_capacity_factor": float(od_factor_mat.values.min()),
            "mean_OD_capacity_factor": float(od_factor_mat.values.mean()),
            "max_OD_capacity_factor": float(od_factor_mat.values.max()),
        })

    # =====================================================
    # 保存 time index
    # =====================================================

    time_df = task_df[
        [
            "H_name",
            "intensity",
            "month",
            "date",
            "t_index",
            "regional_csv",
            "od_factor_csv",
            "od_real_csv",
            "od_cf_csv",
            "od_loss_csv",
        ]
    ].copy()

    time_df.to_csv(
        time_index_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # 保存 numpy 数组
    # =====================================================

    np.save(out_dir / "regional_transport_factor_TT.npy", regional_factor_TT)
    np.save(out_dir / "regional_transport_shock_TT.npy", regional_shock_TT)

    np.save(out_dir / "regional_cf_related_OD_TT.npy", regional_cf_related_TT)
    np.save(out_dir / "regional_real_related_OD_TT.npy", regional_real_related_TT)
    np.save(out_dir / "regional_effective_loss_related_OD_TT.npy", regional_loss_related_TT)

    np.save(out_dir / "od_capacity_factor_TT.npy", od_factor_TT)
    np.save(out_dir / "od_transport_shock_TT.npy", od_shock_TT)

    np.save(out_dir / "OD_real_TT.npy", OD_real_TT)
    np.save(out_dir / "OD_cf_TT.npy", OD_cf_TT)
    np.save(out_dir / "OD_effective_loss_TT.npy", OD_effective_loss_TT)

    # =====================================================
    # 保存 CSV 宽表
    # =====================================================

    save_region_wide_csv(
        arr=regional_factor_TT,
        time_df=time_df,
        region_order=region_order,
        value_name="transport_factor",
        out_csv=out_dir / "regional_transport_factor_TT.csv"
    )

    save_region_wide_csv(
        arr=regional_shock_TT,
        time_df=time_df,
        region_order=region_order,
        value_name="transport_shock",
        out_csv=out_dir / "regional_transport_shock_TT.csv"
    )

    # =====================================================
    # 保存长表
    # =====================================================

    regional_long_df = pd.concat(regional_long_rows, ignore_index=True)

    regional_long_df.to_csv(
        out_dir / "regional_transport_factor_long.csv",
        index=False,
        encoding="utf-8-sig"
    )

    od_long_df = pd.concat(od_long_rows, ignore_index=True)

    od_long_df.to_csv(
        out_dir / "od_factor_loss_long.csv",
        index=False,
        encoding="utf-8-sig"
    )

    daily_summary_df = pd.DataFrame(daily_rows)

    daily_summary_df.to_csv(
        daily_summary_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # 保存 region order / config
    # =====================================================

    region_df = pd.DataFrame({
        "region_index_0based": np.arange(R),
        "region_index_1based": np.arange(1, R + 1),
        "region": region_order,
    })

    region_df.to_csv(
        out_dir / "region_order.csv",
        index=False,
        encoding="utf-8-sig"
    )

    config_df = pd.DataFrame([{
        "H_name": H_NAME,
        "intensity": intensity,
        "out_dir": str(out_dir),
        "T_impact_days": T,
        "R_regions": R,
        "first_date": time_df["date"].iloc[0],
        "last_date": time_df["date"].iloc[-1],
        "regional_transport_factor_TT_shape": str(regional_factor_TT.shape),
        "od_capacity_factor_TT_shape": str(od_factor_TT.shape),
        "OD_real_TT_shape": str(OD_real_TT.shape),
        "definition": (
            "TT here means observed impact-day sequence. "
            "S8 will embed these impact-day factors into a 365-day annual sequence."
        ),
    }])

    config_df.to_csv(
        out_dir / "s7_config.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # 强度汇总
    # =====================================================

    total_cf = float(daily_summary_df["total_OD_cf"].sum())
    total_real = float(daily_summary_df["total_OD_real"].sum())
    total_loss = float(daily_summary_df["total_effective_OD_loss"].sum())

    intensity_summary = pd.DataFrame([{
        "H_name": H_NAME,
        "intensity": intensity,
        "impact_days_count": T,
        "region_count": R,

        "total_OD_real": total_real,
        "total_OD_cf": total_cf,
        "total_effective_OD_loss": total_loss,
        "effective_OD_loss_percent": total_loss / (total_cf + EPS) * 100.0,

        "weighted_capacity_factor": 1.0 - total_loss / (total_cf + EPS),
        "weighted_transport_shock": total_loss / (total_cf + EPS),

        "impact_transport_factor_min": float(
            daily_summary_df["weighted_capacity_factor"].min()
        ),
        "impact_transport_factor_mean": float(
            daily_summary_df["weighted_capacity_factor"].mean()
        ),
        "impact_transport_factor_max": float(
            daily_summary_df["weighted_capacity_factor"].max()
        ),

        "min_regional_transport_factor": float(
            daily_summary_df["min_regional_transport_factor"].min()
        ),
        "mean_regional_transport_factor": float(
            regional_factor_TT.mean()
        ),
        "max_regional_transport_shock": float(
            regional_shock_TT.max()
        ),

        "min_OD_capacity_factor": float(od_factor_TT.min()),
        "mean_OD_capacity_factor": float(od_factor_TT.mean()),
        "max_OD_transport_shock": float(od_shock_TT.max()),
    }])

    intensity_summary.to_csv(
        out_dir / "S7_intensity_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n[SAVED]", out_dir)
    print("regional_transport_factor_TT:", regional_factor_TT.shape)
    print("od_capacity_factor_TT       :", od_factor_TT.shape)
    print("OD_real_TT                  :", OD_real_TT.shape)
    print("\n强度汇总:")
    print(intensity_summary)

    return intensity_summary


# =========================================================
# 6. 主程序
# =========================================================

def main():
    print("=" * 80)
    print('S7_H0_I1I2I3: 整理每日 OD / 区域冲击为 TT 序列')
    print("=" * 80)

    print("SCENARIO_ROOT:", SCENARIO_ROOT)
    print("S6_ROOT:", S6_ROOT)
    print("OUT_ROOT:", OUT_ROOT)
    print("H_NAME:", H_NAME)
    print("INTENSITY_SCENARIOS:", INTENSITY_SCENARIOS)
    print("MONTHS:", MONTHS)
    print("SKIP_IF_DONE:", SKIP_IF_DONE)

    all_summary = []

    for intensity in INTENSITY_SCENARIOS:
        intensity_summary = process_one_intensity(intensity)
        all_summary.append(intensity_summary)

    all_summary_df = pd.concat(all_summary, ignore_index=True)

    all_summary_csv = (
        OUT_ROOT /
        H_NAME /
        'H0_I1I2I3_S7_TT_series_intensity_summary.csv'
    )

    all_summary_csv.parent.mkdir(parents=True, exist_ok=True)

    all_summary_df.to_csv(
        all_summary_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 80)
    print('✅ S7_H0_I1I2I3 TT 序列整理完成')
    print("=" * 80)

    print("强度总汇总:", all_summary_csv)

    print("\n强度总汇总:")
    print(all_summary_df)

    print("\n重点检查：")
    print("1. 每个强度 impact_days_count 应为 21。")
    print("2. regional_transport_factor_TT.npy shape 应为 (21, 15)。")
    print("3. od_capacity_factor_TT.npy shape 应为 (21, 15, 15)。")
    print("4. effective_OD_loss_percent 应整体 I1 < I2 < I3。")
    print("5. 下一步是 S8：把 21 天冲击嵌入 365 天 VA_TT。")


if __name__ == "__main__":
    main()