# -*- coding: utf-8 -*-
from pathlib import Path
import os
import numpy as np
import pandas as pd


# =====================================================
# 生成 low / middle / high 库存情景表
# =====================================================
# 输入：
#   sector_stock_days_AES2019\sector_stock_days_for_model.csv
#
# 输出：
#   sector_stock_days_AES2019_low\sector_stock_days_for_model.csv
#   sector_stock_days_AES2019_middle\sector_stock_days_for_model.csv
#   sector_stock_days_AES2019_high\sector_stock_days_for_model.csv
#   sector_stock_days_low_middle_high_comparison.csv
#
# 设定：
#   low    = baseline × 0.5
#   middle = baseline × 1.0
#   high   = baseline × 2.0
# =====================================================


BASE_DIR = Path(os.environ.get("SCENARIO_DATA_ROOT", "data"))

SCENARIO_ROOT = (
    BASE_DIR /
    '第五阶段情景分析new/I_H0'
)

BASE_STOCK_CSV = (
    SCENARIO_ROOT /
    'sector_stock_days_AES2019_30/sector_stock_days_for_model.csv'
)

OUT_LOW_DIR = (
    SCENARIO_ROOT /
    r"sector_stock_days_AES2019_low"
)

OUT_MIDDLE_DIR = (
    SCENARIO_ROOT /
    r"sector_stock_days_AES2019_middle"
)

OUT_HIGH_DIR = (
    SCENARIO_ROOT /
    r"sector_stock_days_AES2019_high"
)

OUT_LOW_DIR.mkdir(parents=True, exist_ok=True)
OUT_MIDDLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_HIGH_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 库存情景参数
# =====================================================

# low：基准库存 × 0.5，但最低不能低于 1 天
LOW_MULTIPLIER = 0.5
LOW_MIN_DAYS = 1.0
LOW_MAX_DAYS = 90.0

# middle：基准库存 × 1.0
MIDDLE_MULTIPLIER = 1.0
MIDDLE_MIN_DAYS = 1.0
MIDDLE_MAX_DAYS = 180.0

# high：基准库存 × 2.0，最高不超过 180 天
HIGH_MULTIPLIER = 2.0
HIGH_MIN_DAYS = 1.0
HIGH_MAX_DAYS = 180.0


def make_inventory_scenario(df, multiplier, min_days, max_days, scenario_name):
    out = df.copy()

    out["stock_days_baseline"] = pd.to_numeric(
        out["stock_days_for_model"],
        errors="coerce"
    )

    out["stock_multiplier"] = multiplier
    out["inventory_scenario"] = scenario_name

    out["stock_days_for_model"] = np.clip(
        out["stock_days_baseline"] * multiplier,
        min_days,
        max_days
    )

    out["stock_days_min_bound"] = min_days
    out["stock_days_max_bound"] = max_days

    return out


def safe_select_columns(df, cols):
    """
    comparison 表里有些字段可能不存在，用这个避免 KeyError。
    """
    existing_cols = [c for c in cols if c in df.columns]
    return df[existing_cols].copy()


def main():
    if not BASE_STOCK_CSV.exists():
        raise FileNotFoundError(f"找不到基础库存表: {BASE_STOCK_CSV}")

    base_df = pd.read_csv(BASE_STOCK_CSV)

    required = ["sector", "sector_index", "stock_days_for_model"]

    missing = [c for c in required if c not in base_df.columns]

    if missing:
        raise ValueError(f"基础库存表缺少字段: {missing}")

    base_df["stock_days_for_model"] = pd.to_numeric(
        base_df["stock_days_for_model"],
        errors="coerce"
    )

    if base_df["stock_days_for_model"].isna().any():
        raise ValueError("基础库存表存在无法识别的 stock_days_for_model。")

    # =====================================================
    # 生成三套库存表
    # =====================================================

    low_df = make_inventory_scenario(
        df=base_df,
        multiplier=LOW_MULTIPLIER,
        min_days=LOW_MIN_DAYS,
        max_days=LOW_MAX_DAYS,
        scenario_name="low"
    )

    middle_df = make_inventory_scenario(
        df=base_df,
        multiplier=MIDDLE_MULTIPLIER,
        min_days=MIDDLE_MIN_DAYS,
        max_days=MIDDLE_MAX_DAYS,
        scenario_name="middle"
    )

    high_df = make_inventory_scenario(
        df=base_df,
        multiplier=HIGH_MULTIPLIER,
        min_days=HIGH_MIN_DAYS,
        max_days=HIGH_MAX_DAYS,
        scenario_name="high"
    )

    low_csv = OUT_LOW_DIR / "sector_stock_days_for_model.csv"
    middle_csv = OUT_MIDDLE_DIR / "sector_stock_days_for_model.csv"
    high_csv = OUT_HIGH_DIR / "sector_stock_days_for_model.csv"

    low_df.to_csv(
        low_csv,
        index=False,
        encoding="utf-8-sig"
    )

    middle_df.to_csv(
        middle_csv,
        index=False,
        encoding="utf-8-sig"
    )

    high_df.to_csv(
        high_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # comparison 表
    # =====================================================

    comparison_base_cols = [
        "sector",
        "sector_index",
        "stock_days_for_model",
        "stock_days_raw",
        "closing_stocks_NZD_million",
        "annual_intermediate_input_NZD_million",
        "aes_codes",
        "source_note",
    ]

    comparison = safe_select_columns(
        base_df,
        comparison_base_cols
    )

    comparison = comparison.rename(
        columns={
            "stock_days_for_model": "stock_days_baseline"
        }
    )

    comparison["stock_days_low"] = low_df["stock_days_for_model"]
    comparison["stock_days_middle"] = middle_df["stock_days_for_model"]
    comparison["stock_days_high"] = high_df["stock_days_for_model"]

    comparison_csv = (
        SCENARIO_ROOT /
        "sector_stock_days_low_middle_high_comparison.csv"
    )

    comparison.to_csv(
        comparison_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # 打印检查
    # =====================================================

    print("=" * 90)
    print("low / middle / high 库存表生成完成")
    print("=" * 90)
    print("baseline:", BASE_STOCK_CSV)
    print("low     :", low_csv)
    print("middle  :", middle_csv)
    print("high    :", high_csv)
    print("compare :", comparison_csv)

    print("\n库存天数对比：")
    show_cols = [
        "sector_index",
        "sector",
        "stock_days_baseline",
        "stock_days_low",
        "stock_days_middle",
        "stock_days_high",
    ]

    print(
        comparison[show_cols].to_string(index=False)
    )

    print("\n均值检查：")
    print(f"baseline mean: {comparison['stock_days_baseline'].mean():.3f}")
    print(f"low mean     : {comparison['stock_days_low'].mean():.3f}")
    print(f"middle mean  : {comparison['stock_days_middle'].mean():.3f}")
    print(f"high mean    : {comparison['stock_days_high'].mean():.3f}")


if __name__ == "__main__":
    main()