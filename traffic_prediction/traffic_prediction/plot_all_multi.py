import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from .model import STGNNWeatherSeq2Seq


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 配置
# =========================

MONTH = "201903"

MONTH_NPZ = 'artifacts/samples/201903_samples.npz'

ADJ_PATH = 'data/adjacency_distance_130x130_exp.csv'

CKPT_PATH = 'artifacts/checkpoints/stgnn_weather_multihazard_event_v1_24to12.pth'

NORM_PATH = 'artifacts/checkpoints/norm_stats_multihazard_event_v1.json'

OUT_DIR = 'artifacts/counterfactual'

NUM_NODES = 130
Tin = 24
Tout = 12
BATCH_SIZE = 64
ADJ_ALREADY_NORMALIZED = True

# 必须和训练时一致
WEATHER_FORCE = 5.0

# 只放大事件窗口内 h=0 的反事实结果
CF_SCALE = 1

# =========================
# 不同月份事件窗口
# =========================

EVENT_WINDOWS = {
    "201903": ("2019-03-23 00:00:00", "2019-03-27 23:00:00"),
    "201908": ("2019-08-10 00:00:00", "2019-08-14 23:00:00"),
    "201911": ("2019-11-19 00:00:00", "2019-11-22 23:00:00"),
    "201912": ("2019-12-01 00:00:00", "2019-12-08 23:00:00"),
}

EVENT_START = None
EVENT_END = None


# =========================
# Dataset
# =========================

class TrafficMonthDataset(Dataset):

    def __init__(self, npz_path, mean, std):

        data = np.load(npz_path)

        self.X_hist_raw = np.nan_to_num(data["X_hist"]).astype(np.float32)
        self.X_future_raw = np.nan_to_num(data["X_future"]).astype(np.float32)
        self.Y_raw = np.nan_to_num(data["Y"]).astype(np.float32)

        self.EXTREME_FLAG = np.nan_to_num(data["EXTREME_FLAG"]).astype(np.float32)

        self.RAIN_FLAG = np.nan_to_num(data["RAIN_FLAG"]).astype(np.float32) if "RAIN_FLAG" in data else None
        self.WIND_FLAG = np.nan_to_num(data["WIND_FLAG"]).astype(np.float32) if "WIND_FLAG" in data else None
        self.COLD_FLAG = np.nan_to_num(data["COLD_FLAG"]).astype(np.float32) if "COLD_FLAG" in data else None
        self.SNOW_ICE_FLAG = np.nan_to_num(data["SNOW_ICE_FLAG"]).astype(np.float32) if "SNOW_ICE_FLAG" in data else None
        self.WEATHER_EXTREME_FLAG = np.nan_to_num(data["WEATHER_EXTREME_FLAG"]).astype(np.float32) if "WEATHER_EXTREME_FLAG" in data else None
        self.EVENT_FLAG = np.nan_to_num(data["EVENT_FLAG"]).astype(np.float32) if "EVENT_FLAG" in data else None

        self.X_hist = (self.X_hist_raw - mean) / std

        weather_mean = mean[1:5]
        weather_std = std[1:5]

        self.X_future = (self.X_future_raw - weather_mean) / weather_std
        self.Y = (self.Y_raw - mean[0]) / std[0]

    def __len__(self):
        return self.X_hist.shape[0]

    def __getitem__(self, idx):

        item = {
            "x_hist": torch.tensor(self.X_hist[idx], dtype=torch.float32),
            "x_future": torch.tensor(self.X_future[idx], dtype=torch.float32),
            "x_future_raw": torch.tensor(self.X_future_raw[idx], dtype=torch.float32),
            "y": torch.tensor(self.Y[idx], dtype=torch.float32),
            "extreme_flag": torch.tensor(self.EXTREME_FLAG[idx], dtype=torch.float32),
        }

        if self.EVENT_FLAG is not None:
            item["event_flag_npz"] = torch.tensor(self.EVENT_FLAG[idx], dtype=torch.float32)

        return item


# =========================
# 工具函数
# =========================

def load_norm(path):

    with open(path, "r") as f:
        norm = json.load(f)

    mean = np.array(norm["mean"], dtype=np.float32)
    std = np.array(norm["std"], dtype=np.float32)

    print("mean:", mean)
    print("std :", std)

    return mean, std


def load_adj(path, num_nodes, device, already_normalized=True):

    adj_df = pd.read_csv(path, index_col=0)
    adj_df = adj_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    A = torch.tensor(adj_df.values, dtype=torch.float32)

    if already_normalized:
        return A.to(device)

    I = torch.eye(num_nodes)
    A_hat = A + I
    D = torch.diag(torch.sum(A_hat, dim=1).clamp(min=1e-12) ** -0.5)

    return (D @ A_hat @ D).to(device)


def denorm_flow(x, mean, std):

    return x * std[0] + mean[0]


def build_base_times(month, S):

    times = pd.date_range(
        start=f"{month[:4]}-{month[4:6]}-01",
        periods=Tin + Tout + S,
        freq="h"
    )

    return times[Tin: Tin + S]


def build_event_time_mask(base_times, batch_start, B, Tout, device):

    batch_base_times = base_times[batch_start: batch_start + B]

    mask_np = np.zeros((B, Tout), dtype=bool)

    for k in range(Tout):
        horizon_times = batch_base_times + pd.to_timedelta(k, unit="h")
        mask_np[:, k] = (
            (horizon_times >= EVENT_START) &
            (horizon_times <= EVENT_END)
        )

    return torch.tensor(
        mask_np,
        dtype=torch.bool,
        device=device
    ).view(B, Tout, 1, 1)


def build_counterfactual_raw(x_future_raw, event_flag, mean):

    cf_future_raw = x_future_raw.clone()

    normal_raw = torch.zeros_like(cf_future_raw)

    normal_raw[..., 0] = float(mean[1])   # TEMP
    normal_raw[..., 1] = 0.0              # PRCP
    normal_raw[..., 2] = float(mean[3])   # WDSP
    normal_raw[..., 3] = 70.0             # RH

    mask = event_flag.expand_as(cf_future_raw)

    cf_future_raw = torch.where(
        mask == 1,
        normal_raw,
        cf_future_raw
    )

    return cf_future_raw


# =========================
# 主程序
# =========================

def main():

    print("DEVICE:", DEVICE)
    print("MONTH:", MONTH)
    print("EVENT_START:", EVENT_START)
    print("EVENT_END  :", EVENT_END)
    print("WEATHER_FORCE:", WEATHER_FORCE)
    print("CF_SCALE:", CF_SCALE)

    mean, std = load_norm(NORM_PATH)

    dataset = TrafficMonthDataset(MONTH_NPZ, mean, std)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    S = len(dataset)
    base_times = build_base_times(MONTH, S)

    print("[INFO] Samples:", S)
    print("[INFO] first h=0 time:", base_times[0])
    print("[INFO] last  h=0 time:", base_times[-1])

    A_norm = load_adj(
        ADJ_PATH,
        NUM_NODES,
        DEVICE,
        ADJ_ALREADY_NORMALIZED
    )

    model = STGNNWeatherSeq2Seq(
        num_nodes=NUM_NODES,
        weather_force=WEATHER_FORCE
    ).to(DEVICE)

    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()

    preds_real = []
    preds_cf = []
    trues = []
    event_flags = []
    raw_flags = []

    sample_offset = 0

    with torch.no_grad():

        for batch_i, batch in enumerate(loader):

            x_hist = batch["x_hist"].to(DEVICE)
            x_future = batch["x_future"].to(DEVICE)
            x_future_raw = batch["x_future_raw"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            extreme_flag = batch["extreme_flag"].to(DEVICE)

            B = x_future.shape[0]

            event_time_mask = build_event_time_mask(
                base_times=base_times,
                batch_start=sample_offset,
                B=B,
                Tout=Tout,
                device=DEVICE
            )

            event_flag = ((extreme_flag == 1) & event_time_mask).float()

            pred_real = model(
                x_hist=x_hist,
                x_future=x_future,
                A_norm=A_norm,
                x_future_raw=x_future_raw
            )

            cf_weather_norm = x_future.clone()

            cf_weather_raw = build_counterfactual_raw(
                x_future_raw=x_future_raw,
                event_flag=event_flag,
                mean=mean
            )

            if batch_i == 0:
                print("[DEBUG] batch0 event records:", int(event_flag.sum().item()))

            pred_cf = model(
                x_hist=x_hist,
                x_future=cf_weather_norm,
                A_norm=A_norm,
                x_future_raw=cf_weather_raw
            )

            preds_real.append(pred_real.cpu())
            preds_cf.append(pred_cf.cpu())
            trues.append(y.cpu())
            event_flags.append(event_flag.cpu())
            raw_flags.append(extreme_flag.cpu())

            sample_offset += B

    pred_real = torch.cat(preds_real).numpy()
    pred_cf = torch.cat(preds_cf).numpy()
    true = torch.cat(trues).numpy()
    event_flag = torch.cat(event_flags).numpy()
    raw_extreme_flag = torch.cat(raw_flags).numpy()

    print("[INFO] raw EXTREME_FLAG records:", int(raw_extreme_flag.sum()))
    print("[INFO] event_flag records:", int(event_flag.sum()))

    pred_real_denorm = denorm_flow(pred_real, mean, std)
    pred_cf_denorm = denorm_flow(pred_cf, mean, std)
    true_denorm = denorm_flow(true, mean, std)

    pred_real_denorm = np.maximum(pred_real_denorm, 0)
    pred_cf_denorm = np.maximum(pred_cf_denorm, 0)
    true_denorm = np.maximum(true_denorm, 0)

    h = 0

    event_window_mask = (
        (base_times >= EVENT_START) &
        (base_times <= EVENT_END)
    )

    pred_cf_denorm[event_window_mask, h, :, :] *= CF_SCALE

    time_axis = base_times

    real_total_series = pred_real_denorm[:, h, :, 0].sum(axis=1)
    cf_total_series = pred_cf_denorm[:, h, :, 0].sum(axis=1)
    true_total_series = true_denorm[:, h, :, 0].sum(axis=1)

    real_avg_series = pred_real_denorm[:, h, :, 0].mean(axis=1)
    cf_avg_series = pred_cf_denorm[:, h, :, 0].mean(axis=1)
    true_avg_series = true_denorm[:, h, :, 0].mean(axis=1)

    event_flag_series = event_flag[:, h, :, 0].max(axis=1)

    real_total = real_total_series[event_window_mask].sum()
    cf_total = cf_total_series[event_window_mask].sum()

    impact = cf_total - real_total
    impact_percent = impact / (cf_total + 1e-6) * 100

    positive_ratio = (
        (cf_total_series[event_window_mask] - real_total_series[event_window_mask]) > 0
    ).mean() * 100

    print("\n========== National Event Window Impact, h=0 ==========")
    print(f"CF scale in event only = {CF_SCALE:.2f}")
    print(f"Hours                  = {event_window_mask.sum()}")
    print(f"Real total flow        = {real_total:.2f}")
    print(f"Counterfactual flow    = {cf_total:.2f}")
    print(f"Impact CF - Real       = {impact:.2f}")
    print(f"Impact %               = {impact_percent:.2f}%")
    print(f"Positive ratio CF>Real = {positive_ratio:.2f}%")
    print(f"Real mean total flow   = {real_total_series[event_window_mask].mean():.2f}")
    print(f"CF mean total flow     = {cf_total_series[event_window_mask].mean():.2f}")

    summary = {
        "month": MONTH,
        "event_start": str(EVENT_START),
        "event_end": str(EVENT_END),
        "weather_force": float(WEATHER_FORCE),
        "cf_scale_event_only": float(CF_SCALE),
        "scaled_horizon": int(h),
        "hours": int(event_window_mask.sum()),
        "real_total": float(real_total),
        "cf_total": float(cf_total),
        "impact": float(impact),
        "impact_percent": float(impact_percent),
        "positive_ratio": float(positive_ratio),
        "real_mean_total_flow": float(real_total_series[event_window_mask].mean()),
        "cf_mean_total_flow": float(cf_total_series[event_window_mask].mean()),
    }

    summary_path = os.path.join(
        OUT_DIR,
        f"{MONTH}_national_multihazard_event_impact_summary.csv"
    )

    pd.DataFrame([summary]).to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("Saved summary:", summary_path)

    result_path = os.path.join(
        OUT_DIR,
        f"{MONTH}_national_multihazard_event_results.npz"
    )

    np.savez_compressed(
        result_path,
        pred_real=pred_real_denorm,
        pred_cf=pred_cf_denorm,
        true=true_denorm,
        event_flag=event_flag,
        raw_extreme_flag=raw_extreme_flag,
        time_axis=np.array(time_axis.astype(str)),
        event_start=str(EVENT_START),
        event_end=str(EVENT_END),
        weather_force=float(WEATHER_FORCE),
        cf_scale_event_only=float(CF_SCALE),
        scaled_horizon=int(h),
    )

    print("Saved:", result_path)

    event_times = np.array(time_axis)[event_flag_series == 1]

    # =========================
    # 图1：全国总流量
    # =========================

    plt.figure(figsize=(14, 5))

    plt.plot(time_axis, true_total_series, label="Observed Total Flow", linewidth=1.8)
    plt.plot(time_axis, real_total_series, label="Predicted Total under Real Weather", linewidth=1.8)
    plt.plot(time_axis, cf_total_series, label="Counterfactual Total under Normal Weather", linewidth=1.8)

    if len(event_times) > 0:
        plt.scatter(
            event_times,
            true_total_series[event_flag_series == 1],
            s=22,
            label="Event Extreme Weather",
            zorder=5
        )

    plt.axvspan(EVENT_START, EVENT_END, alpha=0.12, label="Event Window")

    plt.title(f"{MONTH} National Total Flow: Multi-hazard Event Counterfactual")
    plt.xlabel("Time")
    plt.ylabel("Total Flow")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    fig_total_path = os.path.join(
        OUT_DIR,
        f"{MONTH}_national_total_multihazard_event.png"
    )

    plt.savefig(fig_total_path, dpi=200)
    plt.close()

    print("Saved:", fig_total_path)

    # =========================
    # 图2：全国平均流量
    # =========================

    plt.figure(figsize=(14, 5))

    plt.plot(time_axis, true_avg_series, label="Observed Average Flow", linewidth=1.8)
    plt.plot(time_axis, real_avg_series, label="Predicted Average under Real Weather", linewidth=1.8)
    plt.plot(time_axis, cf_avg_series, label="Counterfactual Average under Normal Weather", linewidth=1.8)

    if len(event_times) > 0:
        plt.scatter(
            event_times,
            true_avg_series[event_flag_series == 1],
            s=22,
            label="Event Extreme Weather",
            zorder=5
        )

    plt.axvspan(EVENT_START, EVENT_END, alpha=0.12, label="Event Window")

    plt.title(f"{MONTH} National Average Flow: Multi-hazard Event Counterfactual")
    plt.xlabel("Time")
    plt.ylabel("Average Flow")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    fig_avg_path = os.path.join(
        OUT_DIR,
        f"{MONTH}_national_average_multihazard_event.png"
    )

    plt.savefig(fig_avg_path, dpi=200)
    plt.close()

    print("Saved:", fig_avg_path)

    # =========================
    # 图3：全国总流量 Delta
    # =========================

    delta_total = cf_total_series - real_total_series

    plt.figure(figsize=(14, 4))

    plt.plot(time_axis, delta_total, label="Counterfactual - Real", linewidth=2)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.axvspan(EVENT_START, EVENT_END, alpha=0.12, label="Event Window")

    plt.title(f"{MONTH} National Total Counterfactual Impact")
    plt.xlabel("Time")
    plt.ylabel("Total Flow Difference")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    delta_path = os.path.join(
        OUT_DIR,
        f"{MONTH}_national_total_delta_multihazard_event.png"
    )

    plt.savefig(delta_path, dpi=200)
    plt.close()

    print("Saved:", delta_path)

    # =========================
    # 图4：事件窗口附近 zoom
    # =========================

    zoom_start = EVENT_START - pd.Timedelta(days=2)
    zoom_end = EVENT_END + pd.Timedelta(days=2)

    zoom_mask = (
        (time_axis >= zoom_start) &
        (time_axis <= zoom_end)
    )

    plt.figure(figsize=(14, 4))

    plt.plot(
        np.array(time_axis)[zoom_mask],
        delta_total[zoom_mask],
        label="Counterfactual - Real",
        linewidth=2
    )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.axvspan(EVENT_START, EVENT_END, alpha=0.12, label="Event Window")

    plt.title(f"{MONTH} National Total Counterfactual Impact: Event Zoom")
    plt.xlabel("Time")
    plt.ylabel("Total Flow Difference")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    zoom_path = os.path.join(
        OUT_DIR,
        f"{MONTH}_national_total_delta_zoom_multihazard_event.png"
    )

    plt.savefig(zoom_path, dpi=200)
    plt.close()

    print("Saved:", zoom_path)

    print("\nDone.")


