# -*- coding: utf-8 -*-
from pathlib import Path
import os
import numpy as np
import pandas as pd
import re
from difflib import get_close_matches


# =====================================================
# S9-0: 从 AES 2019 构造 MRIO 19 行业库存天数表
# =====================================================
# 核心逻辑：
#   1. 读取 AES 2019 财务长表；
#   2. 抽取 H17/H18/H19/H21/H22；
#   3. 选择 Level 1 行业层级；
#   4. 计算库存天数：
#        average_stock = (H21 + H22) / 2
#        denominator = H19；如果 H19 缺失，用 H17 + H18
#        stock_days_raw = average_stock / denominator * 365
#   5. 将 AES Level 1 行业映射到 MRIO 19 行业；
#   6. 输出 sector_stock_days_for_model.csv
# =====================================================


# =====================================================
# 1. 路径
# =====================================================

BASE_DIR = Path(os.environ.get("SCENARIO_DATA_ROOT", "data"))

SCENARIO_ROOT = (
    BASE_DIR /
    '第五阶段情景分析new/I_H0'
)

AES_PATH = (
    SCENARIO_ROOT /
    r"annual-enterprise-survey-2019-financial-year-provisional-csv.csv"
)

# 优先从你 S8 输出里找 sector_names.csv
# 如果你的 H 文件夹名字不是 H1，请改这里
SECTOR_NAMES_CANDIDATES = [
    SCENARIO_ROOT / 'Intensity_step8_annual_365_VA_TT_H0_eta3.5/I1/MRIO模型/sector_names.csv',
    SCENARIO_ROOT / 'Intensity_step8_annual_365_VA_TT_H0_eta3.5/I1/MRIO模型/sector_names.csv',
    BASE_DIR / '第三阶段/MRIO模型/sector_names.csv',
]

OUT_DIR = (
    SCENARIO_ROOT /
    '库存/sector_stock_days_AES2019_middle'
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. 参数
# =====================================================

YEAR = 2019

# 先使用 Level 1，避免 Level 3/4 和 Level 1 混用造成重复
AES_LEVEL = "Level 1"

# 模型中库存天数裁剪范围
# 原始农业会到 194 天甚至更高，直接放进去会让缓冲过强
MIN_STOCK_DAYS = 1.0
MAX_STOCK_DAYS = 90.0

# 如果某个 MRIO 行业映射不到，使用全行业中位数兜底
USE_MEDIAN_FOR_MISSING = True


# =====================================================
# 3. 工具函数
# =====================================================

def clean_value(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    # 常见缺失 / 保密符号
    if x in ["", "C", "S", "..", "NA", "nan", "None"]:
        return np.nan

    x = x.replace(",", "")

    try:
        return float(x)
    except Exception:
        return np.nan


def read_sector_names():
    for p in SECTOR_NAMES_CANDIDATES:
        if p.exists():
            df = pd.read_csv(p)

            if df.shape[1] == 1:
                sectors = df.iloc[:, 0].astype(str).tolist()
            elif "sector" in df.columns:
                sectors = df["sector"].astype(str).tolist()
            elif "sector_name" in df.columns:
                sectors = df["sector_name"].astype(str).tolist()
            else:
                sectors = df.iloc[:, -1].astype(str).tolist()

            print("使用 sector_names:", p)
            print("sector count:", len(sectors))
            print(sectors)

            return sectors, p

    raise FileNotFoundError(
        "找不到 sector_names.csv。请检查 SECTOR_NAMES_CANDIDATES。"
    )


def norm_text(x):
    x = str(x).lower().strip()
    x = x.replace("&", "and")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def first_match_sector_to_aes(sector, aes_names):
    """
    尝试把 MRIO sector 自动匹配到 AES Level 1 industry。
    这只是初始匹配，最后仍然输出 mapping_review 文件给你检查。
    """

    s = norm_text(sector)

    # 手动关键词规则，优先于 fuzzy match
    rules = [
        ("agriculture", "Agriculture, Forestry and Fishing"),
        ("forestry", "Agriculture, Forestry and Fishing"),
        ("fishing", "Agriculture, Forestry and Fishing"),

        ("mining", "Mining"),

        ("manufacturing", "Manufacturing"),

        ("electricity", "Electricity, Gas, Water and Waste Services"),
        ("gas", "Electricity, Gas, Water and Waste Services"),
        ("water", "Electricity, Gas, Water and Waste Services"),
        ("waste", "Electricity, Gas, Water and Waste Services"),

        ("construction", "Construction"),

        ("wholesale", "Wholesale Trade"),

        ("retail", "Retail Trade and Accommodation"),
        ("accommodation", "Retail Trade and Accommodation"),
        ("food", "Retail Trade and Accommodation"),

        ("transport", "Transport, Postal and Warehousing"),
        ("postal", "Transport, Postal and Warehousing"),
        ("warehousing", "Transport, Postal and Warehousing"),

        ("information", "Information Media and Telecommunications"),
        ("media", "Information Media and Telecommunications"),
        ("telecommunications", "Information Media and Telecommunications"),

        ("financial", "Financial and Insurance Services"),
        ("insurance", "Financial and Insurance Services"),

        ("rental", "Rental, Hiring and Real Estate Services"),
        ("hiring", "Rental, Hiring and Real Estate Services"),
        ("real estate", "Rental, Hiring and Real Estate Services"),

        ("professional", "Professional, Scientific, Technical, Administrative and Support Services"),
        ("scientific", "Professional, Scientific, Technical, Administrative and Support Services"),
        ("technical", "Professional, Scientific, Technical, Administrative and Support Services"),
        ("administrative", "Professional, Scientific, Technical, Administrative and Support Services"),
        ("support", "Professional, Scientific, Technical, Administrative and Support Services"),

        ("public", "Public Order, Safety and Regulatory Services"),
        ("safety", "Public Order, Safety and Regulatory Services"),
        ("regulatory", "Public Order, Safety and Regulatory Services"),

        ("education", "Education and Training"),

        ("health", "Health Care and Social Assistance"),
        ("social assistance", "Health Care and Social Assistance"),

        ("arts", "Arts, Recreation and Other Services"),
        ("recreation", "Arts, Recreation and Other Services"),
        ("other services", "Arts, Recreation and Other Services"),
    ]

    aes_set = set(aes_names)

    for keyword, aes_name in rules:
        if keyword in s and aes_name in aes_set:
            return aes_name, "keyword"

    # fuzzy match 兜底
    norm_to_original = {norm_text(x): x for x in aes_names}
    candidates = list(norm_to_original.keys())
    matches = get_close_matches(s, candidates, n=1, cutoff=0.45)

    if matches:
        return norm_to_original[matches[0]], "fuzzy"

    return None, "missing"


# =====================================================
# 4. 读取 AES 并计算 Level 1 库存天数
# =====================================================

def build_aes_level1_stock_days():
    if not AES_PATH.exists():
        raise FileNotFoundError(f"找不到 AES 文件: {AES_PATH}")

    df = pd.read_csv(AES_PATH, dtype=str)

    df["Value_num"] = df["Value"].apply(clean_value)

    df = df[df["Year"].astype(str) == str(YEAR)].copy()

    # 需要的变量
    need_codes = ["H17", "H18", "H19", "H21", "H22"]

    sub = df[df["Variable_code"].isin(need_codes)].copy()

    # 只选 Level 1
    sub_l1 = sub[
        sub["Industry_aggregation_NZSIOC"].astype(str) == AES_LEVEL
    ].copy()

    id_cols = [
        "Industry_aggregation_NZSIOC",
        "Industry_code_NZSIOC",
        "Industry_name_NZSIOC",
        "Industry_code_ANZSIC06",
    ]

    wide = (
        sub_l1
        .pivot_table(
            index=id_cols,
            columns="Variable_code",
            values="Value_num",
            aggfunc="first"
        )
        .reset_index()
    )

    wide.columns.name = None

    for c in need_codes:
        if c not in wide.columns:
            wide[c] = np.nan

    wide = wide.rename(columns={
        "H17": "purchases_goods_for_resale",
        "H18": "other_purchases_operating_expenses",
        "H19": "purchases_operating_expenses",
        "H21": "opening_stocks",
        "H22": "closing_stocks",
    })

    # 去掉 All industries
    wide = wide[
        wide["Industry_name_NZSIOC"].astype(str).str.lower().ne("all industries")
    ].copy()

    # 分母：优先 H19；如果 H19 缺失，则 H17 + H18
    fallback_denominator = (
        wide["purchases_goods_for_resale"].fillna(0.0)
        + wide["other_purchases_operating_expenses"].fillna(0.0)
    )

    wide["denominator_source"] = np.where(
        wide["purchases_operating_expenses"].notna()
        & (wide["purchases_operating_expenses"] > 0),
        "H19",
        "H17_plus_H18"
    )

    wide["denominator_for_stock_days"] = np.where(
        wide["denominator_source"] == "H19",
        wide["purchases_operating_expenses"],
        fallback_denominator
    )

    # 如果 fallback 还是 0，设为 NaN
    wide.loc[
        wide["denominator_for_stock_days"] <= 0,
        "denominator_for_stock_days"
    ] = np.nan

    wide["average_stocks"] = (
        wide["opening_stocks"] + wide["closing_stocks"]
    ) / 2.0

    wide["stock_days_raw"] = (
        wide["average_stocks"] /
        wide["denominator_for_stock_days"]
    ) * 365.0

    wide["stock_days_raw"] = wide["stock_days_raw"].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # 模型使用版本：裁剪
    wide["stock_days_clipped"] = (
        wide["stock_days_raw"]
        .clip(lower=MIN_STOCK_DAYS, upper=MAX_STOCK_DAYS)
    )

    median_days = wide["stock_days_clipped"].median(skipna=True)

    wide["stock_days_for_mapping"] = wide["stock_days_clipped"].copy()

    if USE_MEDIAN_FOR_MISSING:
        wide["stock_days_for_mapping"] = (
            wide["stock_days_for_mapping"].fillna(median_days)
        )

    wide["is_stock_days_imputed"] = wide["stock_days_raw"].isna().astype(int)

    wide.to_csv(
        OUT_DIR / "AES2019_Level1_stock_days_H19_fallback_H17H18.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nAES Level 1 库存天数:")
    show_cols = [
        "Industry_code_NZSIOC",
        "Industry_name_NZSIOC",
        "purchases_goods_for_resale",
        "other_purchases_operating_expenses",
        "purchases_operating_expenses",
        "denominator_source",
        "denominator_for_stock_days",
        "opening_stocks",
        "closing_stocks",
        "average_stocks",
        "stock_days_raw",
        "stock_days_clipped",
        "stock_days_for_mapping",
        "is_stock_days_imputed",
    ]

    print(wide[show_cols].to_string(index=False))

    print("\nstock_days_raw 描述统计:")
    print(wide["stock_days_raw"].describe())

    print("\nstock_days_for_mapping 描述统计:")
    print(wide["stock_days_for_mapping"].describe())

    return wide


# =====================================================
# 5. 映射到 MRIO 19 行业
# =====================================================

def build_mrio_sector_stock_days(aes_l1):
    sector_names, sector_path = read_sector_names()

    aes_names = aes_l1["Industry_name_NZSIOC"].astype(str).tolist()

    rows = []

    for sector_index, sector in enumerate(sector_names):
        aes_name, match_method = first_match_sector_to_aes(sector, aes_names)

        if aes_name is None:
            rows.append({
                "sector_index": sector_index,
                "sector": sector,
                "aes_industry_name": None,
                "match_method": match_method,
                "stock_days_raw": np.nan,
                "stock_days_for_model": np.nan,
                "needs_manual_review": 1,
            })
            continue

        g = aes_l1[
            aes_l1["Industry_name_NZSIOC"].astype(str) == aes_name
        ].iloc[0]

        rows.append({
            "sector_index": sector_index,
            "sector": sector,
            "aes_industry_name": aes_name,
            "match_method": match_method,

            "aes_code": g["Industry_code_NZSIOC"],
            "denominator_source": g["denominator_source"],

            "purchases_goods_for_resale": g["purchases_goods_for_resale"],
            "other_purchases_operating_expenses": g["other_purchases_operating_expenses"],
            "purchases_operating_expenses": g["purchases_operating_expenses"],
            "denominator_for_stock_days": g["denominator_for_stock_days"],

            "opening_stocks": g["opening_stocks"],
            "closing_stocks": g["closing_stocks"],
            "average_stocks": g["average_stocks"],

            "stock_days_raw": g["stock_days_raw"],
            "stock_days_clipped": g["stock_days_clipped"],
            "stock_days_for_model": g["stock_days_for_mapping"],

            "is_stock_days_imputed_from_Aes": g["is_stock_days_imputed"],

            # fuzzy 或 missing 都建议人工看一眼
            "needs_manual_review": 1 if match_method in ["fuzzy", "missing"] else 0,
        })

    out = pd.DataFrame(rows)

    # 如果还有缺失，用 MRIO 映射后的中位数补
    median_model = out["stock_days_for_model"].median(skipna=True)

    out["stock_days_for_model"] = (
        pd.to_numeric(out["stock_days_for_model"], errors="coerce")
        .fillna(median_model)
        .clip(lower=MIN_STOCK_DAYS, upper=MAX_STOCK_DAYS)
    )

    # 输出完整审查表
    out.to_csv(
        OUT_DIR / "MRIO19_sector_stock_days_mapping_review.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 输出 S9 需要的最简表
    final = out[[
        "sector_index",
        "sector",
        "stock_days_for_model",
    ]].copy()

    final.to_csv(
        OUT_DIR / "sector_stock_days_for_model.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 也输出 low / high 敏感性版本
    low = final.copy()
    low["stock_days_for_model"] = (
        low["stock_days_for_model"] * 0.25
    ).clip(lower=MIN_STOCK_DAYS, upper=MAX_STOCK_DAYS)

    low.to_csv(
        OUT_DIR / "sector_stock_days_for_model_low.csv",
        index=False,
        encoding="utf-8-sig"
    )

    high = final.copy()
    high["stock_days_for_model"] = (
        high["stock_days_for_model"] * 1.5
    ).clip(lower=MIN_STOCK_DAYS, upper=MAX_STOCK_DAYS)

    high.to_csv(
        OUT_DIR / "sector_stock_days_for_model_high.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nMRIO 19行业库存天数映射结果:")
    print(out[[
        "sector_index",
        "sector",
        "aes_industry_name",
        "match_method",
        "denominator_source",
        "stock_days_raw",
        "stock_days_for_model",
        "needs_manual_review",
    ]].to_string(index=False))

    print("\n最终 S9 输入文件:")
    print(OUT_DIR / "sector_stock_days_for_model.csv")

    print("\n请重点检查:")
    print("1. MRIO19_sector_stock_days_mapping_review.csv 里面 needs_manual_review=1 的行。")
    print("2. Manufacturing / Wholesale / Retail 是否已经不再是 NaN。")
    print("3. Transport, Postal and Warehousing 的 stock_days 是否约为 6.7 天。")
    print("4. 是否有行业被错误匹配到 AES 行业。")

    return out


# =====================================================
# 6. main
# =====================================================

def main():
    print("=" * 80)
    print("S9-0: 从 AES 2019 构造 MRIO 19 行业库存天数")
    print("=" * 80)

    print("AES_PATH:", AES_PATH)
    print("OUT_DIR:", OUT_DIR)
    print("AES_LEVEL:", AES_LEVEL)
    print("MIN_STOCK_DAYS:", MIN_STOCK_DAYS)
    print("MAX_STOCK_DAYS:", MAX_STOCK_DAYS)

    aes_l1 = build_aes_level1_stock_days()

    mrio_stock = build_mrio_sector_stock_days(aes_l1)

    print("\n✅ 完成")
    print("输出目录:", OUT_DIR)
    print("核心文件:")
    print(" - AES2019_Level1_stock_days_H19_fallback_H17H18.csv")
    print(" - MRIO19_sector_stock_days_mapping_review.csv")
    print(" - sector_stock_days_for_model.csv")
    print(" - sector_stock_days_for_model_low.csv")
    print(" - sector_stock_days_for_model_high.csv")


if __name__ == "__main__":
    main()