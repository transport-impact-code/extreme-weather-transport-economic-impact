# -*- coding: utf-8 -*-
from pathlib import Path
import os
import numpy as np
import pandas as pd


# =====================================================
# S8_H1_I1I2I3:
# 把 S7 的 21天区域交通冲击嵌入 2019年365期 VA_TT
# =====================================================
# 输入：
#   Intensity_step7_TT_series\H1\I1\
#       time_index_impact_days.csv
#       regional_transport_factor_TT.npy     # shape = (21, 15)
#       region_order.csv
#
# 输出：
#   Intensity_step8_annual_365_VA_TT_H1_eta3.5\
#       I1\MRIO模型\
#           VA_TT_transport_only.npy          # shape = (19, 15, 365)
#           zz_s.npy
#           ff_s.npy
#           va_s.npy
#           xx_s.npy
#           region_names.csv
#           sector_names.csv
#           time_index.csv
#           annual_g_rt_transport_factor.csv
#           annual_g_rt_effective_eta3.5.csv
#           annual_365_event_embedding_summary.csv
#           annual_365_event_membership_long.csv
#           annual_365_VA_TT_config.csv
#
# 核心逻辑：
#   1. S7 已经得到 H1 官方窗口内 21天 regional_transport_factor；
#   2. S8 把这 21天嵌入 2019 年 365 天；
#   3. 非灾害日 VA_TT = 1；
#   4. 灾害日只冲击 Transport, Postal and Warehousing 行业；
#   5. 使用 g_rt_effective = g_rt ** ETA 放大交通冲击；
#   6. I1/I2/I3 分别输出一套 MRIO 输入。
# =====================================================


# =====================================================
# 1. 参数配置
# =====================================================

YEAR = 2019
TT_ANNUAL = 365

RR = 15
NN = 19
RRNN = RR * NN

H_NAME = 'H0'

INTENSITY_SCENARIOS = [
    "I1",
    "I2",
    "I3",
]

ETA = 3.5
EPS = 1e-6

# 当前和前面保持一致：不设交通因子下限，不额外加全国共同冲击
USE_TRANSPORT_FACTOR_FLOOR = False
TRANSPORT_FACTOR_FLOOR = 0.05

USE_NATIONAL_COMMON_SHOCK = False


# =====================================================
# 2. 路径配置
# =====================================================

BASE_DIR = Path(os.environ.get("SCENARIO_DATA_ROOT", "data"))

SCENARIO_ROOT = (
    BASE_DIR /
    '第五阶段情景分析new/I_H0'
)

S7_ROOT = (
    SCENARIO_ROOT /
    r"Intensity_step7_TT_series"
)

OUT_ROOT = (
    SCENARIO_ROOT /
    rf"Intensity_step8_annual_365_VA_TT_{H_NAME}_eta{ETA:g}"
)

OUT_ROOT.mkdir(parents=True, exist_ok=True)


# MRIO 原始表候选路径
MRIO_PATH_CANDIDATES = [
    BASE_DIR / '第三阶段/MRIO_15regions_19sectors_FULL.csv',
    BASE_DIR / r"MRIO_15regions_19sectors_FULL.csv",
    BASE_DIR / '第三阶段迁移/MRIO_15regions_19sectors_FULL.csv',
]


# =====================================================
# 3. 区域名称映射
# =====================================================

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
    x = str(x).strip()
    return name_map.get(x, x)


# =====================================================
# 4. 读取 MRIO
# =====================================================

def resolve_mrio_path():
    for p in MRIO_PATH_CANDIDATES:
        if p.exists():
            return p

    msg = "没有找到 MRIO_15regions_19sectors_FULL.csv。请检查以下路径：\n"
    msg += "\n".join(str(p) for p in MRIO_PATH_CANDIDATES)
    raise FileNotFoundError(msg)


def read_mrio_inputs(mrio_path):
    df = pd.read_csv(mrio_path, header=None)

    Z_cols_start = 2
    Z_cols_end = 2 + RRNN

    F_cols_start = Z_cols_end
    F_cols_end = Z_cols_end + RR

    X_col = F_cols_end

    header_row_1 = df.iloc[0, :]
    header_row_2 = df.iloc[1, :]

    data = df.iloc[2:-1, :].copy()

    region_names = []
    sector_names = []

    for r in range(RR):
        region_names.append(
            normalize_region_name(header_row_1.iloc[2 + r * NN])
        )

    for n in range(NN):
        sector_names.append(
            str(header_row_2.iloc[2 + n]).strip()
        )

    zz_s = (
        data
        .iloc[:, Z_cols_start:Z_cols_end]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy()
    )

    ff_s = (
        data
        .iloc[:, F_cols_start:F_cols_end]
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

    va_row = df.iloc[-1, :]

    va_s = (
        pd.to_numeric(
            va_row.iloc[Z_cols_start:Z_cols_end],
            errors="coerce"
        )
        .fillna(0.0)
        .to_numpy()
        .reshape(1, -1)
    )

    print("=" * 80)
    print("MRIO 输入检查")
    print("=" * 80)
    print("MRIO_PATH:", mrio_path)
    print("zz_s:", zz_s.shape)
    print("ff_s:", ff_s.shape)
    print("va_s:", va_s.shape)
    print("xx_s:", xx_s.shape)
    print("region_names:", len(region_names), region_names)
    print("sector_names:", len(sector_names), sector_names)

    if zz_s.shape != (RRNN, RRNN):
        raise ValueError(f"zz_s shape 不对: {zz_s.shape}")

    if ff_s.shape != (RRNN, RR):
        raise ValueError(f"ff_s shape 不对: {ff_s.shape}")

    if va_s.shape != (1, RRNN):
        raise ValueError(f"va_s shape 不对: {va_s.shape}")

    if xx_s.shape != (1, RRNN):
        raise ValueError(f"xx_s shape 不对: {xx_s.shape}")

    col_err = np.max(
        np.abs(zz_s.sum(axis=0) + va_s.flatten() - xx_s.flatten())
    )

    row_err = np.max(
        np.abs(zz_s.sum(axis=1) + ff_s.sum(axis=1) - xx_s.flatten())
    )

    print("最大列平衡误差:", col_err)
    print("最大行平衡误差:", row_err)

    return zz_s, ff_s, va_s, xx_s, region_names, sector_names


def find_transport_sector(sector_names):
    candidates = []

    for i, s in enumerate(sector_names):
        s_lower = str(s).lower()

        if (
            "transport" in s_lower
            or "postal" in s_lower
            or "warehousing" in s_lower
        ):
            candidates.append((i, s))

    print("\ntransport 行业候选:")
    print(candidates)

    if len(candidates) == 0:
        raise ValueError("没有找到 transport 行业，请手动指定 transport_idx。")

    transport_idx = candidates[0][0]
    transport_name = candidates[0][1]

    print("使用 transport_idx:", transport_idx)
    print("transport_name:", transport_name)

    return transport_idx, transport_name


# =====================================================
# 5. 年度 time index
# =====================================================

def build_annual_time_index():
    annual_dates = pd.date_range(
        start=f"{YEAR}-01-01",
        end=f"{YEAR}-12-31",
        freq="D"
    )

    if len(annual_dates) != TT_ANNUAL:
        raise ValueError(f"年度日期长度不是365: {len(annual_dates)}")

    time_index = pd.DataFrame({
        "t": np.arange(1, TT_ANNUAL + 1),
        "date": annual_dates,
    })

    time_index["date_str"] = time_index["date"].dt.strftime("%Y-%m-%d")
    time_index["period_type"] = "normal"
    time_index["event"] = "normal"
    time_index["event_local_t"] = np.nan
    time_index["H_name"] = H_NAME

    date_to_index = {
        pd.Timestamp(d).normalize(): i
        for i, d in enumerate(time_index["date"])
    }

    return time_index, date_to_index


def append_label(old_value, new_value):
    old_value = str(old_value)

    if old_value in ["normal", "", "nan", "None"]:
        return new_value

    parts = old_value.split("+")

    if new_value in parts:
        return old_value

    return old_value + "+" + new_value


def infer_event_name(month):
    month = str(month)

    if month in ["201903", "201908", "201911", "201912"]:
        return month

    return month


# =====================================================
# 6. 读取 S7 输出
# =====================================================

def load_s7_intensity(intensity):
    s7_dir = S7_ROOT / H_NAME / intensity

    factor_path = s7_dir / "regional_transport_factor_TT.npy"
    time_index_path = s7_dir / "time_index_impact_days.csv"
    region_order_path = s7_dir / "region_order.csv"
    od_factor_path = s7_dir / "od_capacity_factor_TT.npy"
    od_shock_path = s7_dir / "od_transport_shock_TT.npy"
    daily_summary_path = s7_dir / "daily_OD_summary.csv"

    required = [
        factor_path,
        time_index_path,
        region_order_path,
        od_factor_path,
        od_shock_path,
        daily_summary_path,
    ]

    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"缺少 S7 输出文件: {p}")

    g_rt_T_R = np.load(factor_path)

    time_event = pd.read_csv(time_index_path)
    time_event["date"] = pd.to_datetime(time_event["date"])
    time_event["date_str"] = time_event["date"].dt.strftime("%Y-%m-%d")

    region_order_df = pd.read_csv(region_order_path)

    if "region" not in region_order_df.columns:
        raise ValueError(f"{region_order_path} 缺少 region 字段。")

    s7_regions = [
        normalize_region_name(x)
        for x in region_order_df["region"].astype(str).tolist()
    ]

    od_factor_TT = np.load(od_factor_path)
    od_shock_TT = np.load(od_shock_path)

    daily_summary = pd.read_csv(daily_summary_path)

    print("\n" + "=" * 80)
    print(f"S7 输入检查: {intensity}")
    print("=" * 80)
    print("s7_dir:", s7_dir)
    print("regional_transport_factor_TT:", g_rt_T_R.shape)
    print("time_event:", time_event.shape)
    print("s7_regions:", len(s7_regions), s7_regions)
    print("od_capacity_factor_TT:", od_factor_TT.shape)
    print("od_transport_shock_TT:", od_shock_TT.shape)

    if g_rt_T_R.shape[0] != len(time_event):
        raise ValueError(
            f"{intensity} regional_transport_factor_TT 时间维不等于 time_event 行数: "
            f"{g_rt_T_R.shape[0]} vs {len(time_event)}"
        )

    if g_rt_T_R.shape[1] != len(s7_regions):
        raise ValueError(
            f"{intensity} regional_transport_factor_TT 区域维不等于 region_order 行数: "
            f"{g_rt_T_R.shape[1]} vs {len(s7_regions)}"
        )

    return {
        "s7_dir": s7_dir,
        "g_rt_T_R": g_rt_T_R,
        "time_event": time_event,
        "s7_regions": s7_regions,
        "od_factor_TT": od_factor_TT,
        "od_shock_TT": od_shock_TT,
        "daily_summary": daily_summary,
    }


def align_factor_to_mrio_regions(g_rt_T_R, s7_regions, mrio_regions):
    """
    S7: T × R_s7
    输出：T × R_mrio，按 MRIO 区域顺序排列。
    """

    df = pd.DataFrame(
        g_rt_T_R,
        columns=s7_regions
    )

    missing = [r for r in mrio_regions if r not in df.columns]

    if missing:
        raise ValueError(
            "S7 区域中缺少 MRIO 区域：\n"
            + str(missing)
        )

    df = df.reindex(columns=mrio_regions)

    return df.to_numpy(dtype=float)


def compute_effective_factor(g_rt):
    g = np.clip(g_rt, 0.0, 1.0)

    if USE_TRANSPORT_FACTOR_FLOOR:
        g = np.clip(g, TRANSPORT_FACTOR_FLOOR, 1.0)

    g_eff = g ** ETA

    if USE_TRANSPORT_FACTOR_FLOOR:
        g_eff = np.clip(g_eff, TRANSPORT_FACTOR_FLOOR, 1.0)

    return g_eff


# =====================================================
# 7. 单个强度嵌入 365 天
# =====================================================

def build_one_intensity_annual(
    intensity,
    zz_s,
    ff_s,
    va_s,
    xx_s,
    region_names,
    sector_names,
    transport_idx,
    transport_name,
):
    print("\n" + "#" * 80)
    print(f"构造年度 VA_TT: {H_NAME} {intensity} eta{ETA:g}")
    print("#" * 80)

    save_folder = OUT_ROOT / intensity / "MRIO模型"
    save_folder.mkdir(parents=True, exist_ok=True)

    s7 = load_s7_intensity(intensity)

    g_rt_T_R_raw = s7["g_rt_T_R"]
    time_event = s7["time_event"]
    s7_regions = s7["s7_regions"]

    g_rt_T_R = align_factor_to_mrio_regions(
        g_rt_T_R=g_rt_T_R_raw,
        s7_regions=s7_regions,
        mrio_regions=region_names
    )

    g_rt_T_R = np.clip(g_rt_T_R, 0.0, 1.0)

    if USE_NATIONAL_COMMON_SHOCK:
        national_profile = np.nanmean(g_rt_T_R, axis=1)
        g_rt_T_R = np.minimum(
            g_rt_T_R,
            national_profile.reshape(-1, 1)
        )
        print("[INFO] USE_NATIONAL_COMMON_SHOCK applied.")

    g_rt_eff_T_R = compute_effective_factor(g_rt_T_R)

    T_impact = g_rt_T_R.shape[0]

    time_index, date_to_index = build_annual_time_index()

    # VA_TT: sector × region × day
    VA_TT_annual = np.ones((NN, RR, TT_ANNUAL), dtype=float)

    annual_g_rt = np.ones((RR, TT_ANNUAL), dtype=float)
    annual_g_rt_effective = np.ones((RR, TT_ANNUAL), dtype=float)

    membership_rows = []
    event_summary_rows = []

    inserted_days = 0
    skipped_days = 0

    for local_idx, row in time_event.iterrows():
        date_norm = pd.Timestamp(row["date"]).normalize()
        date_str = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
        month = str(row["month"])
        event_name = infer_event_name(month)

        if date_norm not in date_to_index:
            skipped_days += 1
            continue

        annual_idx = date_to_index[date_norm]

        g_day = g_rt_T_R[local_idx, :]
        g_eff_day = g_rt_eff_T_R[local_idx, :]

        # 如果同一天被多个事件覆盖，取更严重冲击，即更小 factor
        annual_g_rt[:, annual_idx] = np.minimum(
            annual_g_rt[:, annual_idx],
            g_day
        )

        annual_g_rt_effective[:, annual_idx] = np.minimum(
            annual_g_rt_effective[:, annual_idx],
            g_eff_day
        )

        # 只冲击交通行业
        VA_TT_annual[transport_idx, :, annual_idx] = np.minimum(
            VA_TT_annual[transport_idx, :, annual_idx],
            g_eff_day
        )

        event_label = f"{event_name}_{H_NAME}"
        period_label = "h1_official"

        time_index.loc[annual_idx, "event"] = append_label(
            time_index.loc[annual_idx, "event"],
            event_label
        )

        time_index.loc[annual_idx, "period_type"] = append_label(
            time_index.loc[annual_idx, "period_type"],
            period_label
        )

        time_index.loc[annual_idx, "event_local_t"] = int(row["t_index"])

        membership_rows.append({
            "H_name": H_NAME,
            "intensity": intensity,
            "event": event_name,
            "month": month,
            "date": date_str,
            "annual_t": int(time_index.loc[annual_idx, "t"]),
            "event_local_t": int(row["t_index"]),
            "period_type": period_label,
            "source_s7_dir": str(s7["s7_dir"]),
            "regional_factor_min": float(np.min(g_day)),
            "regional_factor_mean": float(np.mean(g_day)),
            "regional_factor_max": float(np.max(g_day)),
            f"regional_factor_eta{ETA:g}_min": float(np.min(g_eff_day)),
            f"regional_factor_eta{ETA:g}_mean": float(np.mean(g_eff_day)),
            f"regional_factor_eta{ETA:g}_max": float(np.max(g_eff_day)),
        })

        inserted_days += 1

    # 每个事件汇总
    membership_df = pd.DataFrame(membership_rows)

    for event_name, g in membership_df.groupby("event"):
        event_summary_rows.append({
            "H_name": H_NAME,
            "intensity": intensity,
            "event": event_name,
            "start_date": g["date"].min(),
            "end_date": g["date"].max(),
            "inserted_days": int(len(g)),
            "regional_factor_min": float(g["regional_factor_min"].min()),
            "regional_factor_mean": float(g["regional_factor_mean"].mean()),
            "regional_factor_max": float(g["regional_factor_max"].max()),
            f"regional_factor_eta{ETA:g}_min": float(g[f"regional_factor_eta{ETA:g}_min"].min()),
            f"regional_factor_eta{ETA:g}_mean": float(g[f"regional_factor_eta{ETA:g}_mean"].mean()),
            f"regional_factor_eta{ETA:g}_max": float(g[f"regional_factor_eta{ETA:g}_max"].max()),
        })

    event_summary_df = pd.DataFrame(event_summary_rows)

    # =====================================================
    # 检查
    # =====================================================

    transport_slice = VA_TT_annual[transport_idx, :, :]

    impact_mask = time_index["period_type"].astype(str) != "normal"
    impact_indices = np.where(impact_mask.to_numpy())[0]

    impact_days_count = int(impact_mask.sum())

    if impact_days_count > 0:
        impact_transport = transport_slice[:, impact_indices]
        impact_transport_min = float(np.nanmin(impact_transport))
        impact_transport_mean = float(np.nanmean(impact_transport))
        impact_transport_max = float(np.nanmax(impact_transport))
    else:
        impact_transport_min = np.nan
        impact_transport_mean = np.nan
        impact_transport_max = np.nan

    print("\n" + "=" * 80)
    print(f"[{intensity}] 年度 VA_TT 检查")
    print("=" * 80)
    print("VA_TT_annual shape:", VA_TT_annual.shape)
    print("transport_idx:", transport_idx)
    print("transport_name:", transport_name)
    print("S7 impact days:", T_impact)
    print("inserted_days:", inserted_days)
    print("skipped_days:", skipped_days)
    print("annual impact_days_count:", impact_days_count)
    print("impact transport factor min/mean/max:",
          impact_transport_min,
          impact_transport_mean,
          impact_transport_max)
    print("annual transport factor min/mean/max:",
          float(np.nanmin(transport_slice)),
          float(np.nanmean(transport_slice)),
          float(np.nanmax(transport_slice)))

    # =====================================================
    # 保存
    # =====================================================

    np.save(save_folder / "VA_TT_transport_only.npy", VA_TT_annual)

    np.save(save_folder / "zz_s.npy", zz_s)
    np.save(save_folder / "ff_s.npy", ff_s)
    np.save(save_folder / "va_s.npy", va_s)
    np.save(save_folder / "xx_s.npy", xx_s)

    pd.DataFrame({"region": region_names}).to_csv(
        save_folder / "region_names.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame({"sector": sector_names}).to_csv(
        save_folder / "sector_names.csv",
        index=False,
        encoding="utf-8-sig"
    )

    time_index.to_csv(
        save_folder / "time_index.csv",
        index=False,
        encoding="utf-8-sig"
    )

    annual_cols = [
        f"t{int(time_index.loc[i, 't'])}_{time_index.loc[i, 'date_str']}"
        for i in range(TT_ANNUAL)
    ]

    pd.DataFrame(
        annual_g_rt,
        index=region_names,
        columns=annual_cols
    ).to_csv(
        save_folder / "annual_g_rt_transport_factor.csv",
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        annual_g_rt_effective,
        index=region_names,
        columns=annual_cols
    ).to_csv(
        save_folder / f"annual_g_rt_effective_eta{ETA:g}.csv",
        encoding="utf-8-sig"
    )

    event_summary_df.to_csv(
        save_folder / "annual_365_event_embedding_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    membership_df.to_csv(
        save_folder / "annual_365_event_membership_long.csv",
        index=False,
        encoding="utf-8-sig"
    )

    config_df = pd.DataFrame([{
        "H_name": H_NAME,
        "intensity": intensity,
        "year": YEAR,
        "TT_ANNUAL": TT_ANNUAL,
        "eta": ETA,
        "transport_idx": transport_idx,
        "transport_name": transport_name,
        "save_folder": str(save_folder),

        "source_s7_dir": str(s7["s7_dir"]),
        "S7_impact_days": int(T_impact),
        "inserted_days": int(inserted_days),
        "skipped_days": int(skipped_days),
        "impact_days_count": int(impact_days_count),

        "impact_transport_factor_min": impact_transport_min,
        "impact_transport_factor_mean": impact_transport_mean,
        "impact_transport_factor_max": impact_transport_max,

        "annual_transport_factor_min": float(np.nanmin(transport_slice)),
        "annual_transport_factor_mean": float(np.nanmean(transport_slice)),
        "annual_transport_factor_max": float(np.nanmax(transport_slice)),

        "regional_transport_factor_TT_input_shape": str(g_rt_T_R.shape),
        "VA_TT_shape": str(VA_TT_annual.shape),

        "USE_TRANSPORT_FACTOR_FLOOR": USE_TRANSPORT_FACTOR_FLOOR,
        "TRANSPORT_FACTOR_FLOOR": TRANSPORT_FACTOR_FLOOR,
        "USE_NATIONAL_COMMON_SHOCK": USE_NATIONAL_COMMON_SHOCK,

        "definition": (
            "VA_TT is an annual 365-day productivity factor tensor. "
            "Only Transport, Postal and Warehousing sector is shocked. "
            "For impact dates, transport factor equals regional_transport_factor ** ETA. "
            "All non-impact dates and non-transport sectors remain 1."
        ),
    }])

    config_df.to_csv(
        save_folder / "annual_365_VA_TT_config.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n[SAVED]", save_folder)
    print("VA_TT:", save_folder / "VA_TT_transport_only.npy")
    print("config:", save_folder / "annual_365_VA_TT_config.csv")
    print("time_index:", save_folder / "time_index.csv")

    return {
        "H_name": H_NAME,
        "intensity": intensity,
        "save_folder": str(save_folder),
        "S7_impact_days": int(T_impact),
        "inserted_days": int(inserted_days),
        "skipped_days": int(skipped_days),
        "impact_days_count": int(impact_days_count),
        "impact_transport_factor_min": impact_transport_min,
        "impact_transport_factor_mean": impact_transport_mean,
        "impact_transport_factor_max": impact_transport_max,
        "annual_transport_factor_min": float(np.nanmin(transport_slice)),
        "annual_transport_factor_mean": float(np.nanmean(transport_slice)),
        "annual_transport_factor_max": float(np.nanmax(transport_slice)),
    }


# =====================================================
# 8. main
# =====================================================

def main():
    print("=" * 80)
    print('S8_H0_I1I2I3: 构造年度 365期 VA_TT')
    print("=" * 80)

    print("SCENARIO_ROOT:", SCENARIO_ROOT)
    print("S7_ROOT:", S7_ROOT)
    print("OUT_ROOT:", OUT_ROOT)
    print("H_NAME:", H_NAME)
    print("INTENSITY_SCENARIOS:", INTENSITY_SCENARIOS)
    print("ETA:", ETA)
    print("USE_TRANSPORT_FACTOR_FLOOR:", USE_TRANSPORT_FACTOR_FLOOR)
    print("USE_NATIONAL_COMMON_SHOCK:", USE_NATIONAL_COMMON_SHOCK)

    mrio_path = resolve_mrio_path()

    zz_s, ff_s, va_s, xx_s, region_names, sector_names = read_mrio_inputs(
        mrio_path
    )

    transport_idx, transport_name = find_transport_sector(sector_names)

    all_rows = []

    for intensity in INTENSITY_SCENARIOS:
        row = build_one_intensity_annual(
            intensity=intensity,
            zz_s=zz_s,
            ff_s=ff_s,
            va_s=va_s,
            xx_s=xx_s,
            region_names=region_names,
            sector_names=sector_names,
            transport_idx=transport_idx,
            transport_name=transport_name,
        )

        all_rows.append(row)

    all_summary_df = pd.DataFrame(all_rows)

    all_summary_csv = (
        OUT_ROOT /
        f"all_intensity_annual_365_VA_TT_summary_{H_NAME}_eta{ETA:g}.csv"
    )

    all_summary_df.to_csv(
        all_summary_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 80)
    print('✅ S8_H0_I1I2I3 年度 365期 VA_TT 生成完成')
    print("=" * 80)

    print("汇总文件:", all_summary_csv)
    print(all_summary_df)

    print("\n重点检查：")
    print("1. 每个强度 impact_days_count 应为 21。")
    print("2. VA_TT_transport_only.npy shape 应为 (19, 15, 365)。")
    print("3. impact_transport_factor_min 应该 I1 > I2 > I3。")
    print("4. annual_transport_factor_mean 应该 I1 > I2 > I3，但非常接近 1，因为只有21天有冲击。")
    print("5. 下一步是 S9：动态 MRIO 模拟。")

    print("\nS9 base_path 对应：")
    for row in all_rows:
        print(row["intensity"], "=>", row["save_folder"])


if __name__ == "__main__":
    main()