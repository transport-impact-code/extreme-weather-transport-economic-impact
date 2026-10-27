import os
import pandas as pd
import numpy as np

# =========================
# 路径
# =========================

INPUT_DIR = 'data/weather_flow_month'

OUTPUT_DIR = 'artifacts/labeled'


# =========================
# 分位数阈值参数
# =========================

PRCP_Q = 0.99
WDSP_Q = 0.99
MXSPD_Q = 0.99
TEMP_LOW_Q = 0.01

USE_MXSPD_IF_EXISTS = True

# 雪/冰代理
SNOW_TEMP_TH = 2.0
SNOW_RH_TH = 85.0

# =========================
# 已知灾害事件窗口
# 当前版本：只按时间窗口，不按区域
# =========================

EVENT_WINDOWS = {
    "201903": [
        ("2019-03-24 00:00:00", "2019-03-29 23:00:00"),
    ],
    "201908": [
        ("2019-08-09 00:00:00", "2019-08-12 23:00:00"),
    ],
    "201911": [
        ("2019-11-20 00:00:00", "2019-11-20 23:00:00"),
    ],
    "201912": [
        ("2019-12-02 00:00:00", "2019-12-08 23:00:00"),
    ],
}


# =========================
# 工具函数
# =========================

def safe_numeric(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        df[col] = np.nan
    return df


def compute_global_thresholds(files):
    all_parts = []

    print("\n[INFO] 计算全局阈值...")

    for f in files:
        path = os.path.join(INPUT_DIR, f)
        df = pd.read_csv(path, low_memory=False)

        for col in ["PRCP", "WDSP", "MXSPD", "TEMP", "RH"]:
            df = safe_numeric(df, col)

        all_parts.append(df[["PRCP", "WDSP", "MXSPD", "TEMP", "RH"]])

    all_weather = pd.concat(all_parts, axis=0, ignore_index=True)

    thresholds = {
        "prcp_thr": all_weather["PRCP"].quantile(PRCP_Q),
        "wdsp_thr": all_weather["WDSP"].quantile(WDSP_Q),
        "mxspd_thr": all_weather["MXSPD"].quantile(MXSPD_Q),
        "temp_low_thr": all_weather["TEMP"].quantile(TEMP_LOW_Q),
    }

    print("\n========== Global thresholds ==========")
    print(f"PRCP  p{int(PRCP_Q * 100)} = {thresholds['prcp_thr']:.6f}")
    print(f"WDSP  p{int(WDSP_Q * 100)} = {thresholds['wdsp_thr']:.6f}")
    print(f"MXSPD p{int(MXSPD_Q * 100)} = {thresholds['mxspd_thr']:.6f}")
    print(f"TEMP  p{int(TEMP_LOW_Q * 100)} = {thresholds['temp_low_thr']:.6f}")
    print(f"SNOW proxy: TEMP <= {SNOW_TEMP_TH}, RH >= {SNOW_RH_TH}")

    return thresholds


def build_weather_flags(df, thresholds):
    for col in ["PRCP", "WDSP", "MXSPD", "TEMP", "RH"]:
        df = safe_numeric(df, col)

    prcp_thr = thresholds["prcp_thr"]
    wdsp_thr = thresholds["wdsp_thr"]
    mxspd_thr = thresholds["mxspd_thr"]
    temp_low_thr = thresholds["temp_low_thr"]

    # 1. 极端降雨
    df["RAIN_FLAG"] = (
        df["PRCP"].fillna(-999) >= prcp_thr
    ).astype(int)

    # 2. 极端风
    if USE_MXSPD_IF_EXISTS and "MXSPD" in df.columns and df["MXSPD"].notna().sum() > 0:
        df["WIND_FLAG"] = (
            (df["WDSP"].fillna(-999) >= wdsp_thr) |
            (df["MXSPD"].fillna(-999) >= mxspd_thr)
        ).astype(int)
    else:
        df["WIND_FLAG"] = (
            df["WDSP"].fillna(-999) >= wdsp_thr
        ).astype(int)

    # 3. 极端低温
    df["COLD_FLAG"] = (
        df["TEMP"].fillna(999) <= temp_low_thr
    ).astype(int)

    # 4. 雪/冰代理
    df["SNOW_ICE_FLAG"] = (
        (df["TEMP"].fillna(999) <= SNOW_TEMP_TH) &
        (df["RH"].fillna(-999) >= SNOW_RH_TH)
    ).astype(int)

    # 气象极端综合标签
    df["WEATHER_EXTREME_FLAG"] = (
        (df["RAIN_FLAG"] == 1) |
        (df["WIND_FLAG"] == 1) |
        (df["COLD_FLAG"] == 1) |
        (df["SNOW_ICE_FLAG"] == 1)
    ).astype(int)

    return df


def build_event_flag(df, month):
    df["EVENT_FLAG"] = 0

    if month not in EVENT_WINDOWS:
        return df

    for start, end in EVENT_WINDOWS[month]:
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        mask = (
            (df["DATETIME"] >= start) &
            (df["DATETIME"] <= end)
        )

        df.loc[mask, "EVENT_FLAG"] = 1

    return df


def print_flag_summary(df, month):
    print("\n========== Flag ratio ==========")

    flag_cols = [
        "RAIN_FLAG",
        "WIND_FLAG",
        "COLD_FLAG",
        "SNOW_ICE_FLAG",
        "WEATHER_EXTREME_FLAG",
        "EVENT_FLAG",
        "EXTREME_FLAG",
    ]

    for col in flag_cols:
        print(f"{col:22s}: {df[col].mean():.4f}")

    if df["EVENT_FLAG"].sum() > 0:
        print("\n事件窗口时间范围:")
        print(df.loc[df["EVENT_FLAG"] == 1, "DATETIME"].min())
        print(df.loc[df["EVENT_FLAG"] == 1, "DATETIME"].max())

    print("\n========== Top PRCP ==========")
    print(
        df.sort_values("PRCP", ascending=False)
          .head(8)[["DATETIME", "PRCP", "RAIN_FLAG", "WEATHER_EXTREME_FLAG", "EVENT_FLAG", "EXTREME_FLAG"]]
    )

    print("\n========== Top WDSP ==========")
    print(
        df.sort_values("WDSP", ascending=False)
          .head(8)[["DATETIME", "WDSP", "WIND_FLAG", "WEATHER_EXTREME_FLAG", "EVENT_FLAG", "EXTREME_FLAG"]]
    )

    if "MXSPD" in df.columns:
        print("\n========== Top MXSPD ==========")
        print(
            df.sort_values("MXSPD", ascending=False)
              .head(8)[["DATETIME", "MXSPD", "WIND_FLAG", "WEATHER_EXTREME_FLAG", "EVENT_FLAG", "EXTREME_FLAG"]]
        )

    print("\n========== Lowest TEMP ==========")
    print(
        df.sort_values("TEMP", ascending=True)
          .head(8)[["DATETIME", "TEMP", "RH", "COLD_FLAG", "SNOW_ICE_FLAG", "WEATHER_EXTREME_FLAG", "EVENT_FLAG", "EXTREME_FLAG"]]
    )


# =========================
# 主程序
# =========================

def main():

    files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.endswith(".csv") and f[:6].isdigit()
    ])

    print("总文件数:", len(files))

    if len(files) == 0:
        raise RuntimeError("INPUT_DIR 里没有月份 csv 文件")

    thresholds = compute_global_thresholds(files)

    summary_rows = []

    for f in files:
        print("\n==============================")
        print("处理:", f)

        month = f[:6]

        path = os.path.join(INPUT_DIR, f)
        df = pd.read_csv(path, low_memory=False)

        df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
        df = df.dropna(subset=["DATETIME"]).copy()

        # 1. 气象变量极端标签
        df = build_weather_flags(df, thresholds)

        # 2. 事件窗口标签
        df = build_event_flag(df, month)

        # 3. 最终标签
        df["EXTREME_FLAG"] = (
            (df["WEATHER_EXTREME_FLAG"] == 1) |
            (df["EVENT_FLAG"] == 1)
        ).astype(int)

        print_flag_summary(df, month)

        summary_rows.append({
            "file": f,
            "month": month,
            "records": len(df),
            "RAIN_FLAG_ratio": df["RAIN_FLAG"].mean(),
            "WIND_FLAG_ratio": df["WIND_FLAG"].mean(),
            "COLD_FLAG_ratio": df["COLD_FLAG"].mean(),
            "SNOW_ICE_FLAG_ratio": df["SNOW_ICE_FLAG"].mean(),
            "WEATHER_EXTREME_FLAG_ratio": df["WEATHER_EXTREME_FLAG"].mean(),
            "EVENT_FLAG_ratio": df["EVENT_FLAG"].mean(),
            "EXTREME_FLAG_ratio": df["EXTREME_FLAG"].mean(),
            "PRCP_max": df["PRCP"].max(),
            "WDSP_max": df["WDSP"].max(),
            "MXSPD_max": df["MXSPD"].max() if "MXSPD" in df.columns else np.nan,
            "TEMP_min": df["TEMP"].min(),
        })

        out_path = os.path.join(
            OUTPUT_DIR,
            f.replace(".csv", "_extreme_multihazard_event_v1.csv")
        )

        df.to_csv(out_path, index=False, encoding="utf-8-sig")

        print("\n✅ 已保存:", out_path)

    summary_df = pd.DataFrame(summary_rows)

    summary_path = os.path.join(
        OUTPUT_DIR,
        "multihazard_event_flag_summary.csv"
    )

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n🎉 Step 1 多灾种 + 事件标签完成！")
    print("Summary saved:", summary_path)


