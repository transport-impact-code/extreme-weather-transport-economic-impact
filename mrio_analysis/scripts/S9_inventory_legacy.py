# -*- coding: utf-8 -*-
from pathlib import Path
import os
import numpy as np
import pandas as pd


# =====================================================
# S8.5: 用 AES 2019 Closing stocks + MRIO 中间投入
#       计算 19 行业库存天数
# =====================================================


# =====================================================
# 1. 路径
# =====================================================

BASE_DIR = Path(os.environ.get("SCENARIO_DATA_ROOT", "data"))

# 改成你的 AES 2019 CSV 路径
AES_PATH = Path(
    BASE_DIR / '第五阶段情景分析new/I_H0/annual-enterprise-survey-2019-financial-year-provisional-csv.csv'
)

MRIO_PATH = (
    BASE_DIR /
    '第三阶段/MRIO_15regions_19sectors_FULL.csv'
)

OUT_DIR = (
    BASE_DIR /
    '第五阶段情景分析new/I_H0/sector_stock_days_AES2019_30'
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. 参数
# =====================================================

RR = 15
NN = 19
RRNN = RR * NN

DAYS_PER_YEAR = 365.0

# 是否对库存天数做上下限裁剪
# raw_stock_days 会完整保存；
# stock_days_for_model 用于 S9。
USE_CLIPPED_STOCK_DAYS_FOR_MODEL = True

MIN_STOCK_DAYS = 1.0
MAX_STOCK_DAYS = 30.0

# 缺失行业使用默认值
DEFAULT_SERVICE_STOCK_DAYS = 7.0


# =====================================================
# 3. MRIO 19 行业 -> AES NZSIOC 代码映射
# =====================================================
# 注意：
#   这里用的是 Closing stocks 的行业代码。
#   不要同时使用父级和子级，否则会重复计算。
# =====================================================

MRIO_TO_AES_CODES = {
    # GH 拆分
    "Accommodation": ["GH211"],
    "Food and beverage services": ["GH212"],
    "Retail Trade": ["GH11", "GH12", "GH13"],

    # AA / BB 拆分
    "Agriculture": ["AA11", "AA12", "AA13", "AA14", "AA32"],
    "Forestry, Fishing, and Mining": ["AA21", "AA31", "BB"],

    # Level 1 直接对应
    "Manufacturing": ["CC"],
    "Electricity, Gas, Water, and Waste services": ["DD"],
    "Construction": ["EE"],
    "Wholesale Trade": ["FF"],
    "Transport, Postal and Warehousing": ["II"],
    "Financial and Insurance Services": ["KK"],
    "Rental, Hiring and Real Estate Services": ["LL"],

    # 后半部分通常在 AES 前100行后面
    # 如果代码不存在，程序会报 warning 并输出 missing。
    "Professional, Scientific, and Technical Services": ["MN11"],
    "Administrative and Support Services": ["MN21"],
    "Public Administration and Safety": ["OO"],
    "Education and Training": ["PP"],
    "Health Care and Social Assistance": ["QQ"],

    # 你的 MRIO 这个行业名把 Information Media / Telecom 和 Other Services 合并了
    # AES 中 JJ 是信息通信，RS 通常是 Arts, Recreation and Other Services
    # 如果你的 AES 2019 里没有 RS，程序会自动只用 JJ，并记录 warning。
    "Information Media, Telecommunications and Other Services": ["JJ", "RS"],

    # Owner-occupied property operation 没有直接企业库存项
    # 这里先用 Rental, Hiring and Real Estate Services 作为代理。
    "Owner-Occupied Property Operation": ["LL"],
}


# =====================================================
# 4. 工具函数
# =====================================================

def clean_value_to_float(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    if s in ["", "..", "C", "S", "nan"]:
        return np.nan

    s = s.replace(",", "")

    return pd.to_numeric(s, errors="coerce")


def read_aes_closing_stocks():
    if not AES_PATH.exists():
        raise FileNotFoundError(f"找不到 AES_PATH: {AES_PATH}")

    df = pd.read_csv(AES_PATH)

    required = [
        "Year",
        "Industry_aggregation_NZSIOC",
        "Industry_code_NZSIOC",
        "Industry_name_NZSIOC",
        "Units",
        "Variable_name",
        "Value",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"AES 文件缺少字段: {missing}")

    stocks = df[
        (df["Year"] == 2019) &
        (
            df["Variable_name"]
            .astype(str)
            .str.strip()
            .str.lower()
            == "closing stocks"
        )
    ].copy()

    stocks["Value_num"] = stocks["Value"].apply(clean_value_to_float)

    stocks = stocks.dropna(subset=["Value_num"]).copy()

    stocks["Industry_code_NZSIOC"] = (
        stocks["Industry_code_NZSIOC"]
        .astype(str)
        .str.strip()
    )

    return stocks


def read_mrio_inputs():
    if not MRIO_PATH.exists():
        raise FileNotFoundError(f"找不到 MRIO_PATH: {MRIO_PATH}")

    df = pd.read_csv(MRIO_PATH, header=None)

    Z_cols_start = 2
    Z_cols_end = 2 + RRNN

    F_cols_start = Z_cols_end
    F_cols_end = Z_cols_end + RR

    X_col = F_cols_end

    header_row_1 = df.iloc[0, :]
    header_row_2 = df.iloc[1, :]

    data = df.iloc[2:-1, :].copy()

    region_names = []

    for r in range(RR):
        region_names.append(str(header_row_1.iloc[2 + r * NN]).strip())

    sector_names = []

    for n in range(NN):
        sector_names.append(str(header_row_2.iloc[2 + n]).strip())

    zz_s = (
        data
        .iloc[:, Z_cols_start:Z_cols_end]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy()
    )

    xx_s = (
        data
        .iloc[:, X_col]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy()
        .reshape(1, -1)
    )

    return zz_s, xx_s, region_names, sector_names


def get_stock_by_codes(stocks, codes):
    rows = []

    missing_codes = []

    for code in codes:
        g = stocks[stocks["Industry_code_NZSIOC"] == code].copy()

        if g.empty:
            missing_codes.append(code)
            continue

        rows.append(g)

    if len(rows) == 0:
        return 0.0, pd.DataFrame(), missing_codes

    use_df = pd.concat(rows, ignore_index=True)

    total_stock = float(use_df["Value_num"].sum())

    return total_stock, use_df, missing_codes


def compute_sector_intermediate_inputs(zz_s, sector_names):
    rows = []

    for n, sector in enumerate(sector_names):
        cols = [r * NN + n for r in range(RR)]

        annual_intermediate_input = float(zz_s[:, cols].sum())

        rows.append({
            "sector": sector,
            "sector_index": n,
            "annual_intermediate_input_NZD_million": annual_intermediate_input,
            "daily_intermediate_input_NZD_million": (
                annual_intermediate_input / DAYS_PER_YEAR
            ),
        })

    return pd.DataFrame(rows)


# =====================================================
# 5. main
# =====================================================

def main():
    print("=" * 80)
    print("S8.5: Build sector stock days from AES 2019")
    print("=" * 80)

    print("AES_PATH:", AES_PATH)
    print("MRIO_PATH:", MRIO_PATH)
    print("OUT_DIR:", OUT_DIR)
    print("USE_CLIPPED_STOCK_DAYS_FOR_MODEL:", USE_CLIPPED_STOCK_DAYS_FOR_MODEL)
    print("MIN_STOCK_DAYS:", MIN_STOCK_DAYS)
    print("MAX_STOCK_DAYS:", MAX_STOCK_DAYS)

    stocks = read_aes_closing_stocks()

    print("\nAES 2019 Closing stocks rows:", len(stocks))
    print("Units:")
    print(stocks["Units"].value_counts(dropna=False))

    zz_s, xx_s, region_names, sector_names = read_mrio_inputs()

    print("\nMRIO sectors:")
    for i, s in enumerate(sector_names):
        print(f"{i:02d}. {s}")

    input_df = compute_sector_intermediate_inputs(
        zz_s=zz_s,
        sector_names=sector_names
    )

    stock_rows = []
    mapping_rows = []

    for sector in sector_names:
        codes = MRIO_TO_AES_CODES.get(sector, [])

        if len(codes) == 0:
            closing_stock = np.nan
            used_df = pd.DataFrame()
            missing_codes = []
            note = "no_mapping_use_default"
        else:
            closing_stock, used_df, missing_codes = get_stock_by_codes(
                stocks=stocks,
                codes=codes
            )

            if len(used_df) == 0:
                closing_stock = np.nan
                note = "mapping_codes_not_found_use_default"
            else:
                note = "AES2019_closing_stocks"

        input_row = input_df[input_df["sector"] == sector].iloc[0]

        annual_input = float(
            input_row["annual_intermediate_input_NZD_million"]
        )

        daily_input = float(
            input_row["daily_intermediate_input_NZD_million"]
        )

        if pd.isna(closing_stock) or daily_input <= 0:
            raw_stock_days = DEFAULT_SERVICE_STOCK_DAYS
            source_note = note + "_default"
        else:
            raw_stock_days = closing_stock / daily_input
            source_note = note

        if USE_CLIPPED_STOCK_DAYS_FOR_MODEL:
            stock_days_for_model = float(
                np.clip(
                    raw_stock_days,
                    MIN_STOCK_DAYS,
                    MAX_STOCK_DAYS
                )
            )
        else:
            stock_days_for_model = float(raw_stock_days)

        stock_rows.append({
            "sector": sector,
            "sector_index": int(input_row["sector_index"]),
            "aes_codes": ";".join(codes),
            "missing_aes_codes": ";".join(missing_codes),

            "closing_stocks_NZD_million": closing_stock,
            "annual_intermediate_input_NZD_million": annual_input,
            "daily_intermediate_input_NZD_million": daily_input,

            "stock_days_raw": float(raw_stock_days),
            "stock_days_for_model": stock_days_for_model,

            "source_note": source_note,
            "USE_CLIPPED_STOCK_DAYS_FOR_MODEL": USE_CLIPPED_STOCK_DAYS_FOR_MODEL,
            "MIN_STOCK_DAYS": MIN_STOCK_DAYS,
            "MAX_STOCK_DAYS": MAX_STOCK_DAYS,
        })

        if len(used_df) > 0:
            for _, r in used_df.iterrows():
                mapping_rows.append({
                    "mrio_sector": sector,
                    "aes_code_used": r["Industry_code_NZSIOC"],
                    "aes_level": r["Industry_aggregation_NZSIOC"],
                    "aes_name": r["Industry_name_NZSIOC"],
                    "units": r["Units"],
                    "closing_stock_NZD_million": r["Value_num"],
                    "Industry_code_ANZSIC06": r.get("Industry_code_ANZSIC06", ""),
                })
        else:
            mapping_rows.append({
                "mrio_sector": sector,
                "aes_code_used": "",
                "aes_level": "",
                "aes_name": "",
                "units": "",
                "closing_stock_NZD_million": np.nan,
                "Industry_code_ANZSIC06": "",
            })

    stock_df = pd.DataFrame(stock_rows)

    mapping_df = pd.DataFrame(mapping_rows)

    # 排序为 MRIO 19 行业顺序
    stock_df = stock_df.sort_values("sector_index").reset_index(drop=True)

    out_raw = OUT_DIR / "sector_stock_days_raw.csv"
    out_model = OUT_DIR / "sector_stock_days_for_model.csv"
    out_mapping = OUT_DIR / "aes_to_mrio_stock_mapping_check.csv"
    out_input = OUT_DIR / "mrio_sector_intermediate_inputs.csv"

    stock_df.to_csv(
        out_raw,
        index=False,
        encoding="utf-8-sig"
    )

    stock_df[
        [
            "sector",
            "sector_index",
            "stock_days_for_model",
            "stock_days_raw",
            "closing_stocks_NZD_million",
            "annual_intermediate_input_NZD_million",
            "aes_codes",
            "missing_aes_codes",
            "source_note",
        ]
    ].to_csv(
        out_model,
        index=False,
        encoding="utf-8-sig"
    )

    mapping_df.to_csv(
        out_mapping,
        index=False,
        encoding="utf-8-sig"
    )

    input_df.to_csv(
        out_input,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 80)
    print("✅ sector stock days 已生成")
    print("=" * 80)

    print("raw:", out_raw)
    print("model:", out_model)
    print("mapping:", out_mapping)
    print("MRIO inputs:", out_input)

    print("\n19行业库存天数：")
    print(
        stock_df[
            [
                "sector_index",
                "sector",
                "aes_codes",
                "closing_stocks_NZD_million",
                "annual_intermediate_input_NZD_million",
                "stock_days_raw",
                "stock_days_for_model",
                "missing_aes_codes",
                "source_note",
            ]
        ].to_string(index=False)
    )

    missing = stock_df[
        stock_df["missing_aes_codes"].astype(str).str.len() > 0
    ]

    if not missing.empty:
        print("\n⚠️ 有 AES code 没找到：")
        print(
            missing[
                ["sector", "aes_codes", "missing_aes_codes"]
            ].to_string(index=False)
        )

    print("\n下一步：")
    print("1. 先检查 stock_days_raw 是否有特别离谱的行业。")
    print("2. S9 动态 MRIO 用 sector_stock_days_for_model.csv。")
    print("3. 如果你觉得 90 天上限太高/太低，可以改 MAX_STOCK_DAYS 再跑。")


if __name__ == "__main__":
    main()