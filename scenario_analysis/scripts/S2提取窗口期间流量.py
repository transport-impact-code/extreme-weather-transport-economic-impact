# -*- coding: utf-8 -*-
from pathlib import Path
import os
import numpy as np
import pandas as pd


# =====================================================
# S2_H1_I1I2I3: S1 npz -> hourly / daily node flows
# =====================================================
# 输入：
#   第五阶段情景分析new\I_H0\Intensity_step1_prediction\H1\I1
#   第五阶段情景分析new\I_H0\Intensity_step1_prediction\H1\I2
#   第五阶段情景分析new\I_H0\Intensity_step1_prediction\H1\I3
#
# 输出：
#   第五阶段情景分析new\I_H0\Intensity_step2_daily_node_flows\H1\I1\201903\
#       event_window_hourly_node_flows.csv
#       event_window_daily_node_flows.csv
#       event_window_daily_summary.csv
#
# 核心逻辑：
#   1. 读取 S1 的 pred_real / pred_cf / true；
#   2. 展开 S × Tout × node；
#   3. 同一个 time-node 可能被多个 horizon 预测到，先取均值；
#   4. 截取 H1 官方灾害窗口；
#   5. 按 date-node 聚合为日均流量。
# =====================================================


# =====================================================
# 1. 路径配置
# =====================================================

BASE_DIR = Path(os.environ.get("SCENARIO_DATA_ROOT", "data"))

SCENARIO_ROOT = (
    BASE_DIR /
    '第五阶段情景分析new/I_H0'
)

S1_ROOT = (
    SCENARIO_ROOT /
    r"Intensity_step1_prediction"
)

OUT_ROOT = (
    SCENARIO_ROOT /
    r"Intensity_step2_daily_node_flows"
)

OUT_ROOT.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. 参数配置
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

EVENT_WINDOWS_H1 = {
    "201903": ("2019-03-23 00:00:00", "2019-03-27 23:00:00"),
    "201908": ("2019-08-10 00:00:00", "2019-08-14 23:00:00"),
    "201911": ("2019-11-19 00:00:00", "2019-11-22 23:00:00"),
    "201912": ("2019-12-01 00:00:00", "2019-12-08 23:00:00"),
}

EPS = 1e-6

SKIP_IF_DONE = False


# =====================================================
# 3. 工具函数
# =====================================================

def get_s1_npz_path(intensity, month):
    return (
        S1_ROOT /
        H_NAME /
        intensity /
        f"{month}_national_multihazard_event_results.npz"
    )


def safe_ratio(a, b, eps=EPS):
    return a / (b + eps)


def load_time_matrix(data, S, Tout):
    """
    优先使用 S1 保存的 future_time_axis。
    如果不存在，再用 time_axis + horizon 重构。
    """

    if "future_time_axis" in data.files:
        future_time_axis = data["future_time_axis"]

        future_time_mat = pd.to_datetime(
            future_time_axis.reshape(-1)
        ).to_numpy().reshape(S, Tout)

        return future_time_mat

    if "time_axis" not in data.files:
        raise ValueError("npz 中既没有 future_time_axis，也没有 time_axis。")

    time_axis = pd.to_datetime(data["time_axis"])

    if len(time_axis) != S:
        raise ValueError(
            f"time_axis 长度 {len(time_axis)} 与 S={S} 不一致。"
        )

    future_time_mat = np.empty((S, Tout), dtype="datetime64[ns]")

    for s in range(S):
        for h in range(Tout):
            future_time_mat[s, h] = time_axis[s] + pd.Timedelta(hours=h)

    return future_time_mat


def flatten_npz_to_long(data, month, intensity):
    """
    输出长表：
      time, node_id, pred_real, pred_cf, true_flow, horizon, sample_index
    """

    required = ["pred_real", "pred_cf", "true"]

    missing = [k for k in required if k not in data.files]

    if missing:
        raise ValueError(f"npz 缺少字段: {missing}")

    pred_real = data["pred_real"]   # S, Tout, N, 1
    pred_cf = data["pred_cf"]
    true = data["true"]

    if pred_real.shape != pred_cf.shape or pred_real.shape != true.shape:
        raise ValueError(
            f"pred_real/pred_cf/true shape 不一致: "
            f"{pred_real.shape}, {pred_cf.shape}, {true.shape}"
        )

    S, Tout, N, C = pred_real.shape

    if C != 1:
        raise ValueError(f"最后一维 C 应为 1，现在是 {C}")

    future_time_mat = load_time_matrix(
        data=data,
        S=S,
        Tout=Tout
    )

    # event_flag 可选
    if "event_flag" in data.files:
        event_flag = data["event_flag"]
        if event_flag.shape != pred_real.shape:
            print(
                "[WARN] event_flag shape 与 pred_real 不一致，忽略 event_flag:",
                event_flag.shape,
                pred_real.shape
            )
            event_flag = None
    else:
        event_flag = None

    # raw_extreme_flag 可选
    if "raw_extreme_flag" in data.files:
        raw_extreme_flag = data["raw_extreme_flag"]
        if raw_extreme_flag.shape != pred_real.shape:
            print(
                "[WARN] raw_extreme_flag shape 与 pred_real 不一致，忽略 raw_extreme_flag:",
                raw_extreme_flag.shape,
                pred_real.shape
            )
            raw_extreme_flag = None
    else:
        raw_extreme_flag = None

    records = []

    for s in range(S):
        for h in range(Tout):
            t = pd.Timestamp(future_time_mat[s, h])

            for node_id in range(N):
                row = {
                    "H_name": H_NAME,
                    "intensity": intensity,
                    "month": month,

                    "time": t,
                    "node_id": node_id,

                    "pred_real": float(pred_real[s, h, node_id, 0]),
                    "pred_cf": float(pred_cf[s, h, node_id, 0]),
                    "true_flow": float(true[s, h, node_id, 0]),

                    "horizon": h + 1,
                    "sample_index": s,
                }

                if event_flag is not None:
                    row["event_flag"] = float(event_flag[s, h, node_id, 0])

                if raw_extreme_flag is not None:
                    row["raw_extreme_flag"] = float(
                        raw_extreme_flag[s, h, node_id, 0]
                    )

                records.append(row)

    df = pd.DataFrame(records)

    return df


def aggregate_to_hourly(df):
    """
    同一个 time-node 由多个 horizon 预测到，取均值。
    """

    agg_dict = {
        "pred_real": ("pred_real", "mean"),
        "pred_cf": ("pred_cf", "mean"),
        "true_flow": ("true_flow", "mean"),
        "n_predictions": ("horizon", "count"),
        "mean_horizon": ("horizon", "mean"),
        "min_horizon": ("horizon", "min"),
        "max_horizon": ("horizon", "max"),
    }

    if "event_flag" in df.columns:
        agg_dict["event_flag"] = ("event_flag", "max")
        agg_dict["event_flag_mean"] = ("event_flag", "mean")

    if "raw_extreme_flag" in df.columns:
        agg_dict["raw_extreme_flag"] = ("raw_extreme_flag", "max")
        agg_dict["raw_extreme_flag_mean"] = ("raw_extreme_flag", "mean")

    df_hourly = (
        df
        .groupby(
            ["H_name", "intensity", "month", "time", "node_id"],
            as_index=False
        )
        .agg(**agg_dict)
    )

    df_hourly["delta_real_minus_cf"] = (
        df_hourly["pred_real"] - df_hourly["pred_cf"]
    )

    df_hourly["loss_cf_minus_real"] = (
        df_hourly["pred_cf"] - df_hourly["pred_real"]
    )

    df_hourly["ratio_real_to_cf"] = safe_ratio(
        df_hourly["pred_real"],
        df_hourly["pred_cf"]
    )

    df_hourly["loss_percent_cf_minus_real"] = (
        df_hourly["loss_cf_minus_real"] /
        (df_hourly["pred_cf"] + EPS) *
        100.0
    )

    return df_hourly


def filter_event_window(df_hourly, month):
    start = pd.Timestamp(EVENT_WINDOWS_H1[month][0])
    end = pd.Timestamp(EVENT_WINDOWS_H1[month][1])

    out = df_hourly[
        (df_hourly["time"] >= start) &
        (df_hourly["time"] <= end)
    ].copy()

    out["date"] = out["time"].dt.strftime("%Y-%m-%d")
    out["period_type"] = "h1_official"
    out["event_start"] = str(start)
    out["event_end"] = str(end)

    return out


def aggregate_to_daily(df_window):
    agg_dict = {
        "pred_real_daily": ("pred_real", "mean"),
        "pred_cf_daily": ("pred_cf", "mean"),
        "true_flow_daily": ("true_flow", "mean"),

        "n_hours": ("time", "nunique"),
        "n_predictions_total": ("n_predictions", "sum"),
        "mean_n_predictions_per_hour": ("n_predictions", "mean"),
    }

    if "event_flag" in df_window.columns:
        agg_dict["event_flag_daily"] = ("event_flag", "max")
        agg_dict["event_flag_daily_mean"] = ("event_flag", "mean")

    if "raw_extreme_flag" in df_window.columns:
        agg_dict["raw_extreme_flag_daily"] = ("raw_extreme_flag", "max")
        agg_dict["raw_extreme_flag_daily_mean"] = ("raw_extreme_flag", "mean")

    df_daily = (
        df_window
        .groupby(
            ["H_name", "intensity", "month", "date", "node_id"],
            as_index=False
        )
        .agg(**agg_dict)
    )

    df_daily["delta_real_minus_cf"] = (
        df_daily["pred_real_daily"] -
        df_daily["pred_cf_daily"]
    )

    df_daily["loss_cf_minus_real"] = (
        df_daily["pred_cf_daily"] -
        df_daily["pred_real_daily"]
    )

    df_daily["ratio_real_to_cf"] = safe_ratio(
        df_daily["pred_real_daily"],
        df_daily["pred_cf_daily"]
    )

    df_daily["loss_percent_cf_minus_real"] = (
        df_daily["loss_cf_minus_real"] /
        (df_daily["pred_cf_daily"] + EPS) *
        100.0
    )

    df_daily["positive_loss_cf_minus_real"] = (
        df_daily["loss_cf_minus_real"].clip(lower=0.0)
    )

    df_daily["period_type"] = "h1_official"

    return df_daily


def summarize_daily(df_daily):
    rows = []

    for date, g in df_daily.groupby("date"):
        total_real = float(g["pred_real_daily"].sum())
        total_cf = float(g["pred_cf_daily"].sum())
        total_true = float(g["true_flow_daily"].sum())

        loss = total_cf - total_real
        positive_loss = float(g["positive_loss_cf_minus_real"].sum())

        rows.append({
            "H_name": g["H_name"].iloc[0],
            "intensity": g["intensity"].iloc[0],
            "month": g["month"].iloc[0],
            "date": date,
            "node_count": int(g["node_id"].nunique()),
            "mean_n_hours": float(g["n_hours"].mean()),
            "min_n_hours": int(g["n_hours"].min()),
            "max_n_hours": int(g["n_hours"].max()),

            "total_true_flow_daily": total_true,
            "total_real_flow_daily": total_real,
            "total_cf_flow_daily": total_cf,

            "loss_cf_minus_real": loss,
            "loss_percent_cf_minus_real": loss / (total_cf + EPS) * 100.0,

            "positive_loss_cf_minus_real": positive_loss,
            "positive_loss_percent_cf_minus_real": (
                positive_loss / (total_cf + EPS) * 100.0
            ),

            "mean_ratio_real_to_cf": float(g["ratio_real_to_cf"].mean()),
            "weighted_ratio_real_to_cf": float(
                total_real / (total_cf + EPS)
            ),
        })

    return pd.DataFrame(rows)


# =====================================================
# 4. 单个情景处理
# =====================================================

def process_one(intensity, month):
    print("\n" + "=" * 80)
    print(f"S2 processing: {H_NAME} {intensity} {month}")
    print("=" * 80)

    npz_path = get_s1_npz_path(
        intensity=intensity,
        month=month
    )

    if not npz_path.exists():
        raise FileNotFoundError(f"找不到 S1 npz: {npz_path}")

    out_dir = OUT_ROOT / H_NAME / intensity / month
    out_dir.mkdir(parents=True, exist_ok=True)

    hourly_csv = out_dir / "event_window_hourly_node_flows.csv"
    daily_csv = out_dir / "event_window_daily_node_flows.csv"
    daily_summary_csv = out_dir / "event_window_daily_summary.csv"
    config_csv = out_dir / "s2_config.csv"

    if SKIP_IF_DONE and hourly_csv.exists() and daily_csv.exists():
        print("[SKIP existing]", out_dir)
        df_daily = pd.read_csv(daily_csv)
        summary_df = pd.read_csv(daily_summary_csv)
        return summary_df

    data = np.load(npz_path, allow_pickle=True)

    print("npz:", npz_path)
    print("npz fields:", data.files)

    pred_real = data["pred_real"]
    pred_cf = data["pred_cf"]
    true = data["true"]

    print("pred_real:", pred_real.shape)
    print("pred_cf  :", pred_cf.shape)
    print("true     :", true.shape)

    # =====================================================
    # 1. 展开 S × horizon × node
    # =====================================================

    df_long = flatten_npz_to_long(
        data=data,
        month=month,
        intensity=intensity
    )

    print("flatten rows:", len(df_long))
    print("time range:", df_long["time"].min(), "to", df_long["time"].max())

    # =====================================================
    # 2. 同一 time-node 多 horizon 平均
    # =====================================================

    df_hourly_all = aggregate_to_hourly(df_long)

    print("hourly all rows:", len(df_hourly_all))
    print("hourly all time range:", df_hourly_all["time"].min(), "to", df_hourly_all["time"].max())

    # =====================================================
    # 3. 截取 H1 官方窗口
    # =====================================================

    df_window = filter_event_window(
        df_hourly=df_hourly_all,
        month=month
    )

    if df_window.empty:
        raise RuntimeError(f"{H_NAME} {intensity} {month} H1窗口截取后为空。")

    print("window rows:", len(df_window))
    print("window hours:", df_window["time"].nunique())
    print("window dates:", sorted(df_window["date"].unique()))
    print("window nodes:", df_window["node_id"].nunique())

    # =====================================================
    # 4. 按日聚合
    # =====================================================

    df_daily = aggregate_to_daily(df_window)

    summary_df = summarize_daily(df_daily)

    # =====================================================
    # 5. 保存
    # =====================================================

    df_window.to_csv(
        hourly_csv,
        index=False,
        encoding="utf-8-sig"
    )

    df_daily.to_csv(
        daily_csv,
        index=False,
        encoding="utf-8-sig"
    )

    summary_df.to_csv(
        daily_summary_csv,
        index=False,
        encoding="utf-8-sig"
    )

    config_df = pd.DataFrame([{
        "H_name": H_NAME,
        "intensity": intensity,
        "month": month,
        "s1_npz": str(npz_path),
        "out_dir": str(out_dir),
        "event_start": EVENT_WINDOWS_H1[month][0],
        "event_end": EVENT_WINDOWS_H1[month][1],
        "hourly_csv": str(hourly_csv),
        "daily_csv": str(daily_csv),
        "daily_summary_csv": str(daily_summary_csv),
        "pred_real_shape": str(pred_real.shape),
        "pred_cf_shape": str(pred_cf.shape),
        "true_shape": str(true.shape),
        "hour_count": int(df_window["time"].nunique()),
        "date_count": int(df_window["date"].nunique()),
        "node_count": int(df_window["node_id"].nunique()),
    }])

    config_df.to_csv(
        config_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("[SAVED hourly]", hourly_csv)
    print("[SAVED daily ]", daily_csv)
    print("[SAVED summary]", daily_summary_csv)

    print("\n每日汇总：")
    print(summary_df)

    return summary_df


# =====================================================
# 5. 主程序
# =====================================================

def main():
    print("=" * 80)
    print('S2_H0_I1I2I3: S1 npz -> daily node flows')
    print("=" * 80)

    print("SCENARIO_ROOT:", SCENARIO_ROOT)
    print("S1_ROOT:", S1_ROOT)
    print("OUT_ROOT:", OUT_ROOT)
    print("H_NAME:", H_NAME)
    print("INTENSITY_SCENARIOS:", INTENSITY_SCENARIOS)
    print("MONTHS:", MONTHS)
    print("SKIP_IF_DONE:", SKIP_IF_DONE)

    all_summary_rows = []

    for intensity in INTENSITY_SCENARIOS:
        for month in MONTHS:
            summary_df = process_one(
                intensity=intensity,
                month=month
            )

            all_summary_rows.append(summary_df)

    all_summary_df = pd.concat(
        all_summary_rows,
        ignore_index=True
    )

    all_summary_csv = (
        OUT_ROOT /
        H_NAME /
        'all_H0_I1I2I3_daily_node_flow_summary.csv'
    )

    all_summary_df.to_csv(
        all_summary_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # 情景总汇总
    # =====================================================

    scenario_rows = []

    for intensity, g in all_summary_df.groupby("intensity"):
        total_real = float(g["total_real_flow_daily"].sum())
        total_cf = float(g["total_cf_flow_daily"].sum())

        loss = total_cf - total_real

        scenario_rows.append({
            "H_name": H_NAME,
            "intensity": intensity,
            "month_count": int(g["month"].nunique()),
            "day_count": int(len(g)),
            "total_real_flow_daily_sum": total_real,
            "total_cf_flow_daily_sum": total_cf,
            "loss_cf_minus_real_sum": loss,
            "loss_percent_cf_minus_real": (
                loss / (total_cf + EPS) * 100.0
            ),
            "positive_loss_sum": float(
                g["positive_loss_cf_minus_real"].sum()
            ),
            "positive_loss_percent": float(
                g["positive_loss_cf_minus_real"].sum() /
                (total_cf + EPS) *
                100.0
            ),
            "mean_daily_loss_percent": float(
                g["loss_percent_cf_minus_real"].mean()
            ),
            "max_daily_loss_percent": float(
                g["loss_percent_cf_minus_real"].max()
            ),
            "min_daily_loss_percent": float(
                g["loss_percent_cf_minus_real"].min()
            ),
            "weighted_ratio_real_to_cf_all": float(
                total_real / (total_cf + EPS)
            ),
        })

    scenario_summary_df = pd.DataFrame(scenario_rows)

    scenario_summary_csv = (
        OUT_ROOT /
        H_NAME /
        'H0_I1I2I3_daily_node_flow_intensity_summary.csv'
    )

    scenario_summary_df.to_csv(
        scenario_summary_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 80)
    print('✅ S2 H0 I1/I2/I3 完成')
    print("=" * 80)

    print("每日总汇总:", all_summary_csv)
    print("强度汇总:", scenario_summary_csv)

    print("\n强度汇总：")
    print(scenario_summary_df)

    print("\n重点检查：")
    print("1. day_count 应为 21 左右：201903=5, 201908=5, 201911=4, 201912=7。")
    print("2. I1/I2/I3 的 loss_percent_cf_minus_real 应逐步增大。")
    print("3. 输出 daily 文件将用于 S3 生成 OD 反推边。")
    print('4. S3 需要读取路径：Intensity_step2_daily_node_flows/H0/I1/I2/I3/month/event_window_daily_node_flows.csv')


if __name__ == "__main__":
    main()