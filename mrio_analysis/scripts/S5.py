# -*- coding: utf-8 -*-
from pathlib import Path
import os
import numpy as np
import pandas as pd


# =========================================================
# S5_H1_I1I2I3: OD 反推结果补对角线
# =========================================================
# 输入：
#   Intensity_step4_od_inversion\H1\I1\201903\2019-03-23\real\region_od_final.csv
#   Intensity_step4_od_inversion\H1\I1\201903\2019-03-23\cf\region_od_final.csv
#   ...
#
# 输出：
#   Intensity_step5_od_filled_diag\H1\I1\201903\2019-03-23\real\
#       region_od_filled_diag.csv
#       region_od_filled_diag_matrix.csv
#       diag_before_after.csv
#
# 对角线补值逻辑：
#   OD_ii = ALPHA * (row_sum_i + col_sum_i) / 2
#
# 其中 row_sum_i / col_sum_i 使用补值前的非对角线 OD。
# =========================================================


# =========================================================
# 1. 路径配置
# =========================================================

BASE_DIR = Path(os.environ.get("SCENARIO_DATA_ROOT", "data"))

SCENARIO_ROOT = (
    BASE_DIR /
    '第五阶段情景分析new/I_H0'
)

OD_ROOT = (
    SCENARIO_ROOT /
    r"Intensity_step4_od_inversion"
)

OUT_ROOT = (
    SCENARIO_ROOT /
    r"Intensity_step5_od_filled_diag"
)

OUT_ROOT.mkdir(parents=True, exist_ok=True)

# 用这个文件确定 15 个 MRIO 区域顺序
REGION_ORDER_CSV = (
    BASE_DIR /
    '第二阶段New/final_out/region15_od_seed_from_gdp.csv'
)


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

FLOW_CASES = [
    "real",
    "cf",
]

# 对角线补值系数
ALPHA = 0.5

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


def load_region_order():
    if not REGION_ORDER_CSV.exists():
        raise FileNotFoundError(f"找不到 REGION_ORDER_CSV: {REGION_ORDER_CSV}")

    df = pd.read_csv(REGION_ORDER_CSV)

    ensure_cols(
        df,
        ["O_region", "D_region"],
        "REGION_ORDER_CSV"
    )

    regions = []

    for c in ["O_region", "D_region"]:
        for x in df[c].dropna().tolist():
            name = normalize_region_name(x)
            if name not in regions:
                regions.append(name)

    if len(regions) != 15:
        print("⚠️ 警告：识别到的区域数不是 15，而是:", len(regions))
        print(regions)

    return regions


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

    # 防止重复 OD 对
    df = (
        df
        .groupby(["O_region", "D_region"], as_index=False)
        ["OD_value"]
        .sum()
    )

    return df


def long_to_matrix(od_long, region_order):
    mat = (
        od_long
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


def matrix_to_long(mat):
    return (
        mat
        .rename_axis(index="O_region", columns="D_region")
        .stack()
        .reset_index(name="OD_value")
    )


def fill_diag_transport(mat, alpha=0.5):
    """
    对角线补值：

        OD_ii = alpha * (row_sum_i + col_sum_i) / 2

    row_sum_i / col_sum_i 使用补值前的非对角线规模。
    """

    mat2 = mat.copy().astype(float)

    old_diag = pd.Series(
        np.diag(mat2.values),
        index=mat2.index
    )

    row_sum = mat2.sum(axis=1) - old_diag
    col_sum = mat2.sum(axis=0) - old_diag

    for region in mat2.index:
        mat2.loc[region, region] = (
            alpha *
            (row_sum.loc[region] + col_sum.loc[region]) /
            2.0
        )

    return mat2


def get_input_path(intensity, month, date_str, flow_case):
    return (
        OD_ROOT /
        H_NAME /
        intensity /
        month /
        date_str /
        flow_case /
        "region_od_final.csv"
    )


def get_output_dir(intensity, month, date_str, flow_case):
    return (
        OUT_ROOT /
        H_NAME /
        intensity /
        month /
        date_str /
        flow_case
    )


def discover_tasks():
    rows = []

    for intensity in INTENSITY_SCENARIOS:
        for month in MONTHS:
            month_dir = OD_ROOT / H_NAME / intensity / month

            if not month_dir.exists():
                raise FileNotFoundError(f"找不到 S4 month_dir: {month_dir}")

            date_dirs = sorted([p for p in month_dir.iterdir() if p.is_dir()])

            if len(date_dirs) == 0:
                raise RuntimeError(f"{month_dir} 下没有日期目录。")

            for date_dir in date_dirs:
                date_str = date_dir.name

                for flow_case in FLOW_CASES:
                    in_path = get_input_path(
                        intensity=intensity,
                        month=month,
                        date_str=date_str,
                        flow_case=flow_case
                    )

                    out_dir = get_output_dir(
                        intensity=intensity,
                        month=month,
                        date_str=date_str,
                        flow_case=flow_case
                    )

                    rows.append({
                        "H_name": H_NAME,
                        "intensity": intensity,
                        "month": month,
                        "date": date_str,
                        "flow_case": flow_case,
                        "input_file": str(in_path),
                        "out_dir": str(out_dir),
                    })

    task_df = pd.DataFrame(rows)

    task_df = (
        task_df
        .sort_values(["intensity", "month", "date", "flow_case"])
        .reset_index(drop=True)
    )

    return task_df


# =========================================================
# 5. 单个 OD 文件补对角线
# =========================================================

def process_one(row, region_order):
    intensity = row["intensity"]
    month = row["month"]
    date_str = row["date"]
    flow_case = row["flow_case"]

    in_path = Path(row["input_file"])
    out_dir = Path(row["out_dir"])

    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_csv = out_dir / "region_od_filled_diag_matrix.csv"
    long_csv = out_dir / "region_od_filled_diag.csv"
    diag_csv = out_dir / "diag_before_after.csv"

    if SKIP_IF_DONE and matrix_csv.exists() and long_csv.exists() and diag_csv.exists():
        print("[SKIP existing]", out_dir)

        od_filled_long = pd.read_csv(long_csv)

        total_after = float(od_filled_long["OD_value"].sum())

        return {
            "H_name": H_NAME,
            "intensity": intensity,
            "month": month,
            "date": date_str,
            "flow_case": flow_case,
            "input_file": str(in_path),
            "output_long_file": str(long_csv),
            "output_matrix_file": str(matrix_csv),
            "diag_file": str(diag_csv),
            "alpha": ALPHA,
            "total_before": np.nan,
            "total_after": total_after,
            "diag_sum_before": np.nan,
            "diag_sum_after": np.nan,
            "diag_added_sum": np.nan,
            "non_diag_sum": np.nan,
            "diag_to_non_diag_ratio": np.nan,
            "min_diag_after": np.nan,
            "max_diag_after": np.nan,
            "mean_diag_after": np.nan,
            "region_count": len(region_order),
            "matrix_rows": 15,
            "matrix_cols": 15,
            "skipped_existing": True,
        }

    od_long = read_od_long(in_path)

    od_mat = long_to_matrix(
        od_long=od_long,
        region_order=region_order
    )

    diag_before = pd.Series(
        np.diag(od_mat.values),
        index=od_mat.index,
        name="diag_before"
    )

    total_before = float(od_mat.values.sum())
    non_diag_sum = float(
        total_before - diag_before.sum()
    )

    od_filled = fill_diag_transport(
        od_mat,
        alpha=ALPHA
    )

    diag_after = pd.Series(
        np.diag(od_filled.values),
        index=od_filled.index,
        name="diag_after"
    )

    total_after = float(od_filled.values.sum())

    od_filled_long = matrix_to_long(od_filled)

    od_filled.to_csv(
        matrix_csv,
        encoding="utf-8-sig"
    )

    od_filled_long.to_csv(
        long_csv,
        index=False,
        encoding="utf-8-sig"
    )

    diag_df = pd.concat(
        [diag_before, diag_after],
        axis=1
    )

    diag_df["diag_added"] = (
        diag_df["diag_after"] -
        diag_df["diag_before"]
    )

    diag_df["H_name"] = H_NAME
    diag_df["intensity"] = intensity
    diag_df["month"] = month
    diag_df["date"] = date_str
    diag_df["flow_case"] = flow_case

    diag_df.to_csv(
        diag_csv,
        encoding="utf-8-sig"
    )

    return {
        "H_name": H_NAME,
        "intensity": intensity,
        "month": month,
        "date": date_str,
        "flow_case": flow_case,

        "input_file": str(in_path),
        "output_long_file": str(long_csv),
        "output_matrix_file": str(matrix_csv),
        "diag_file": str(diag_csv),

        "alpha": ALPHA,

        "total_before": total_before,
        "total_after": total_after,

        "diag_sum_before": float(diag_before.sum()),
        "diag_sum_after": float(diag_after.sum()),
        "diag_added_sum": float(diag_after.sum() - diag_before.sum()),

        "non_diag_sum": non_diag_sum,
        "diag_to_non_diag_ratio": float(
            diag_after.sum() / (non_diag_sum + EPS)
        ),

        "min_diag_after": float(diag_after.min()),
        "max_diag_after": float(diag_after.max()),
        "mean_diag_after": float(diag_after.mean()),

        "region_count": len(region_order),
        "matrix_rows": int(od_filled.shape[0]),
        "matrix_cols": int(od_filled.shape[1]),

        "skipped_existing": False,
    }


# =========================================================
# 6. 主程序
# =========================================================

def main():
    print("=" * 80)
    print('S5_H0_I1I2I3: OD 补对角线')
    print("=" * 80)

    print("SCENARIO_ROOT:", SCENARIO_ROOT)
    print("OD_ROOT:", OD_ROOT)
    print("OUT_ROOT:", OUT_ROOT)
    print("REGION_ORDER_CSV:", REGION_ORDER_CSV)
    print("H_NAME:", H_NAME)
    print("INTENSITY_SCENARIOS:", INTENSITY_SCENARIOS)
    print("MONTHS:", MONTHS)
    print("FLOW_CASES:", FLOW_CASES)
    print("ALPHA:", ALPHA)
    print("SKIP_IF_DONE:", SKIP_IF_DONE)

    region_order = load_region_order()

    print("\n区域顺序:")
    for i, r in enumerate(region_order, start=1):
        print(f"{i:02d}. {r}")

    task_df = discover_tasks()

    task_csv = OUT_ROOT / H_NAME / "s5_fill_diag_task_list.csv"
    task_csv.parent.mkdir(parents=True, exist_ok=True)

    task_df.to_csv(
        task_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n任务数:", len(task_df))
    print("任务表:", task_csv)

    # 理论任务数：
    # 3 intensities × 21 days × 2 real/cf = 126
    if len(task_df) != 126:
        print(
            f"⚠️ 当前任务数为 {len(task_df)}，理论 H1 应为 126。"
            "请检查 S4 是否完整。"
        )

    summary_rows = []

    for idx, row in task_df.iterrows():
        print("\n" + "=" * 80)
        print(
            f"[{idx + 1}/{len(task_df)}] "
            f"{row['H_name']} {row['intensity']} {row['month']} "
            f"{row['date']} {row['flow_case']}"
        )
        print("=" * 80)

        print("input:", row["input_file"])
        print("out  :", row["out_dir"])

        summary = process_one(
            row=row,
            region_order=region_order
        )

        summary_rows.append(summary)

        print("[DONE]")
        print(f"  before total : {summary['total_before']:.6f}")
        print(f"  after total  : {summary['total_after']:.6f}")
        print(f"  non-diag sum : {summary['non_diag_sum']:.6f}")
        print(f"  diag after   : {summary['diag_sum_after']:.6f}")

    summary_df = pd.DataFrame(summary_rows)

    summary_csv = OUT_ROOT / H_NAME / "fill_diag_summary.csv"

    summary_df.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # real / cf 补对角线后 OD 总量检查
    # =====================================================

    pivot = (
        summary_df
        .pivot_table(
            index=["H_name", "intensity", "month", "date"],
            columns="flow_case",
            values="total_after",
            aggfunc="first"
        )
        .reset_index()
    )

    check_csv = OUT_ROOT / H_NAME / "real_cf_total_od_after_diag_check.csv"

    if "real" in pivot.columns and "cf" in pivot.columns:
        pivot["real_to_cf_ratio_after_diag"] = (
            pivot["real"] /
            (pivot["cf"] + EPS)
        )

        pivot["od_loss_cf_minus_real_after_diag"] = (
            pivot["cf"] -
            pivot["real"]
        )

        pivot["od_loss_ratio_after_diag"] = (
            pivot["od_loss_cf_minus_real_after_diag"] /
            (pivot["cf"] + EPS)
        )

        pivot["od_loss_percent_after_diag"] = (
            pivot["od_loss_ratio_after_diag"] *
            100.0
        )

    pivot.to_csv(
        check_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # 强度汇总
    # =====================================================

    intensity_rows = []

    if "real" in pivot.columns and "cf" in pivot.columns:
        for intensity, g in pivot.groupby("intensity"):
            total_real = float(g["real"].sum())
            total_cf = float(g["cf"].sum())
            loss = total_cf - total_real

            intensity_rows.append({
                "H_name": H_NAME,
                "intensity": intensity,
                "month_count": int(g["month"].nunique()),
                "day_count": int(len(g)),

                "total_OD_real_after_diag": total_real,
                "total_OD_cf_after_diag": total_cf,

                "od_loss_cf_minus_real_after_diag": loss,
                "od_loss_percent_after_diag": (
                    loss / (total_cf + EPS) * 100.0
                ),

                "mean_daily_od_loss_percent_after_diag": float(
                    g["od_loss_percent_after_diag"].mean()
                ),
                "max_daily_od_loss_percent_after_diag": float(
                    g["od_loss_percent_after_diag"].max()
                ),
                "min_daily_od_loss_percent_after_diag": float(
                    g["od_loss_percent_after_diag"].min()
                ),

                "weighted_real_to_cf_ratio_after_diag": float(
                    total_real / (total_cf + EPS)
                ),
            })

    intensity_summary_df = pd.DataFrame(intensity_rows)

    intensity_summary_csv = (
        OUT_ROOT /
        H_NAME /
        'H0_I1I2I3_fill_diag_intensity_summary.csv'
    )

    intensity_summary_df.to_csv(
        intensity_summary_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 80)
    print('✅ S5_H0_I1I2I3 OD 补对角线完成')
    print("=" * 80)

    print("任务表:", task_csv)
    print("补对角线总汇总:", summary_csv)
    print("real/cf 检查:", check_csv)
    print("强度汇总:", intensity_summary_csv)

    print("\n强度汇总:")
    print(intensity_summary_df)

    print("\n重点检查：")
    print("1. 任务数应为 126。")
    print("2. 每个输出文件 region_od_filled_diag.csv 应为 15×15 = 225 行。")
    print("3. weighted_real_to_cf_ratio_after_diag 应整体 I1 > I2 > I3。")
    print("4. od_loss_percent_after_diag 应整体 I1 < I2 < I3。")
    print("5. 下一步是 S6：生成 capacity factor。")


if __name__ == "__main__":
    main()