# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from model import STGNNWeatherSeq2Seq


# =====================================================
# S1_H1_rawonly_delta_I1I2I3
# =====================================================
# 核心口径：
#
# H1 = H0 官方灾害窗口
#
# 旧口径 raw-only 干预：
#   x_future_norm 始终保持观测天气的标准化输入不变；
#   只改变 x_future_raw。
#
# 固定正常天气 raw：
#   TEMP = norm mean TEMP
#   PRCP = 0
#   WDSP = norm mean WDSP
#   RH   = 70
#
# CF:
#   norm = observed norm
#   raw  = normal raw
#
# I1:
#   norm = observed norm
#   raw  = observed raw
#
# I2/I3:
#   norm = observed norm
#   raw  = normal raw + k * (observed raw - normal raw)
#
# 其中：
#   I1: k = 1
#   I2: k = 2
#   I3: k = 3
#
# 这样 I1 应该复现旧 I100 量级。
# =====================================================


# =====================================================
# 0. 全局配置
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(os.environ.get("SCENARIO_DATA_ROOT", "data"))

SAMPLE_ROOT = (
    BASE_DIR /
    r"有天气new\samples_extreme_multihazard_event_v1"
)

ADJ_PATH = (
    BASE_DIR /
    'data/adjacency_distance_130x130_exp.csv'
)

CKPT_PATH = (
    BASE_DIR /
    r"有天气new\ckpt_extreme_multihazard_event_v1"
    '/stgnn_weather_multihazard_event_v1_24to12.pth'
)

NORM_PATH = (
    BASE_DIR /
    r"有天气new\ckpt_extreme_multihazard_event_v1"
    '/norm_stats_multihazard_event_v1.json'
)

# 新输出目录：避免覆盖刚才 good-weather 版本
OUT_ROOT = (
    BASE_DIR /
    '第五阶段情景分析new/I_H0'
    '/Intensity_step1_prediction'
)

OUT_ROOT.mkdir(parents=True, exist_ok=True)

H_NAME = 'H0'

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

INTENSITY_SCENARIOS = {
    "I1": 1.0,
    "I2": 2.0,
    "I3": 3.0,
}

NUM_NODES = 130
Tin = 24
Tout = 12
BATCH_SIZE = 64

ADJ_ALREADY_NORMALIZED = True
WEATHER_FORCE = 5.0

CLIP_FLOW_TO_ZERO = True

# raw weather 顺序：
# 0 TEMP
# 1 PRCP
# 2 WDSP
# 3 RH
WEATHER_VAR_NAMES = ["TEMP", "PRCP", "WDSP", "RH"]

# 固定正常天气设置
NORMAL_PRCP = 0.0
NORMAL_RH = 70.0

# 物理约束
CLIP_PRCP_NONNEGATIVE = True
CLIP_WDSP_NONNEGATIVE = True
CLIP_RH_0_100 = True

PLOT_FIGURES = True


# =====================================================
# 1. Dataset
# =====================================================

class TrafficMonthDataset(Dataset):
    def __init__(self, npz_path, mean, std):
        data = np.load(npz_path, allow_pickle=True)

        self.X_hist_raw = np.nan_to_num(data["X_hist"]).astype(np.float32)
        self.X_future_raw = np.nan_to_num(data["X_future"]).astype(np.float32)
        self.Y_raw = np.nan_to_num(data["Y"]).astype(np.float32)

        self.EXTREME_FLAG = np.nan_to_num(
            data["EXTREME_FLAG"]
        ).astype(np.float32)

        self.X_hist = (self.X_hist_raw - mean) / std

        weather_mean = mean[1:5]
        weather_std = std[1:5]

        self.X_future_norm = (self.X_future_raw - weather_mean) / weather_std
        self.Y = (self.Y_raw - mean[0]) / std[0]

    def __len__(self):
        return self.X_hist.shape[0]

    def __getitem__(self, idx):
        return {
            "x_hist": torch.tensor(self.X_hist[idx], dtype=torch.float32),
            "x_future_norm_obs": torch.tensor(self.X_future_norm[idx], dtype=torch.float32),
            "x_future_raw_obs": torch.tensor(self.X_future_raw[idx], dtype=torch.float32),
            "y": torch.tensor(self.Y[idx], dtype=torch.float32),
            "extreme_flag": torch.tensor(self.EXTREME_FLAG[idx], dtype=torch.float32),
        }


# =====================================================
# 2. 基础工具
# =====================================================

def load_norm(path):
    with open(path, "r", encoding="utf-8") as f:
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
    out = x * std[0] + mean[0]

    if CLIP_FLOW_TO_ZERO:
        out = np.maximum(out, 0.0)

    return out


def build_base_times(month, S):
    times = pd.date_range(
        start=f"{month[:4]}-{month[4:6]}-01",
        periods=Tin + Tout + S,
        freq="h"
    )

    return times[Tin: Tin + S]


def build_future_time_matrix(base_times, S, Tout):
    out = np.empty((S, Tout), dtype="datetime64[ns]")

    for s in range(S):
        for h in range(Tout):
            out[s, h] = base_times[s] + pd.Timedelta(hours=h)

    return out


def build_future_time_axis(base_times, S, Tout):
    out = np.empty((S, Tout), dtype=object)

    for s in range(S):
        for h in range(Tout):
            out[s, h] = str(base_times[s] + pd.Timedelta(hours=h))

    return out


def build_time_mask_np(future_time_mat, start, end):
    return (
        (future_time_mat >= np.datetime64(start))
        & (future_time_mat <= np.datetime64(end))
    )


def make_event_flag_torch(
    base_times,
    batch_start,
    B,
    Tout,
    event_start,
    event_end,
    extreme_flag,
    device
):
    batch_base_times = base_times[batch_start: batch_start + B]

    time_mask = np.zeros((B, Tout), dtype=bool)

    for h in range(Tout):
        horizon_times = batch_base_times + pd.to_timedelta(h, unit="h")

        time_mask[:, h] = (
            (horizon_times >= event_start)
            & (horizon_times <= event_end)
        )

    time_mask_t = torch.tensor(
        time_mask,
        dtype=torch.bool,
        device=device
    ).view(B, Tout, 1, 1)

    event_flag = ((extreme_flag == 1) & time_mask_t).float()

    return event_flag, time_mask_t.float()


def apply_weather_physical_constraints_torch(raw_weather):
    out = raw_weather.clone()

    if CLIP_PRCP_NONNEGATIVE:
        out[..., 1] = torch.clamp(out[..., 1], min=0.0)

    if CLIP_WDSP_NONNEGATIVE:
        out[..., 2] = torch.clamp(out[..., 2], min=0.0)

    if CLIP_RH_0_100:
        out[..., 3] = torch.clamp(out[..., 3], min=0.0, max=100.0)

    return out


def build_normal_raw_tensor(x_future_raw_obs, mean):
    """
    固定正常天气 raw：
      TEMP = mean[1]
      PRCP = 0
      WDSP = mean[3]
      RH   = 70
    shape 与 x_future_raw_obs 一致。
    """

    normal = torch.zeros_like(x_future_raw_obs)

    normal[..., 0] = float(mean[1])       # TEMP
    normal[..., 1] = float(NORMAL_PRCP)   # PRCP
    normal[..., 2] = float(mean[3])       # WDSP
    normal[..., 3] = float(NORMAL_RH)     # RH

    return normal


# =====================================================
# 3. 绘图
# =====================================================

def save_plot(
    out_dir,
    month,
    intensity_name,
    time_axis,
    true_total_series,
    scenario_total_series,
    cf_total_series,
    event_flag_series,
    event_start,
    event_end,
):
    out_dir = Path(out_dir)

    event_times = np.array(time_axis)[event_flag_series == 1]

    plt.figure(figsize=(14, 5))

    plt.plot(
        time_axis,
        true_total_series,
        label="Observed Total Flow",
        linewidth=1.8
    )

    plt.plot(
        time_axis,
        scenario_total_series,
        label=f"Predicted Total under {intensity_name}",
        linewidth=1.8
    )

    plt.plot(
        time_axis,
        cf_total_series,
        label="Counterfactual Total under Fixed Normal Weather",
        linewidth=1.8
    )

    if len(event_times) > 0:
        plt.scatter(
            event_times,
            true_total_series[event_flag_series == 1],
            s=20,
            label='H0 Event Points',
            zorder=5
        )

    plt.axvspan(
        event_start,
        event_end,
        alpha=0.12,
        label='H0 Official Window'
    )

    plt.title(f"{month} National Total Flow: {H_NAME}_{intensity_name}")
    plt.xlabel("Time")
    plt.ylabel("Total Flow")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    fig_path = out_dir / f"{month}_{H_NAME}_{intensity_name}_national_total_flow.png"

    plt.savefig(fig_path, dpi=220)
    plt.close()

    delta_total = cf_total_series - scenario_total_series

    plt.figure(figsize=(14, 4))

    plt.plot(
        time_axis,
        delta_total,
        label=f"Counterfactual - {intensity_name}",
        linewidth=2
    )

    plt.axhline(0, linestyle="--", linewidth=1)

    plt.axvspan(
        event_start,
        event_end,
        alpha=0.12,
        label='H0 Official Window'
    )

    plt.title(f"{month} National Total Impact: {H_NAME}_{intensity_name}")
    plt.xlabel("Time")
    plt.ylabel("Total Flow Difference")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    delta_path = out_dir / f"{month}_{H_NAME}_{intensity_name}_national_total_delta.png"

    plt.savefig(delta_path, dpi=220)
    plt.close()


# =====================================================
# 4. 单月运行
# =====================================================

def run_one_month(month, model, A_norm, mean, std):
    print("\n" + "#" * 80)
    print("MONTH:", month)
    print("#" * 80)

    month_npz = SAMPLE_ROOT / f"{month}_samples.npz"

    if not month_npz.exists():
        raise FileNotFoundError(f"找不到样本文件: {month_npz}")

    event_start = pd.Timestamp(EVENT_WINDOWS_H1[month][0])
    event_end = pd.Timestamp(EVENT_WINDOWS_H1[month][1])

    print("H_NAME:", H_NAME)
    print("EVENT_START:", event_start)
    print("EVENT_END  :", event_end)

    dataset = TrafficMonthDataset(month_npz, mean, std)

    S = len(dataset)
    base_times = build_base_times(month, S)
    future_time_mat = build_future_time_matrix(base_times, S, Tout)
    future_time_axis = build_future_time_axis(base_times, S, Tout)

    print("[INFO] Samples:", S)
    print("[INFO] first h=0 time:", base_times[0])
    print("[INFO] last  h=0 time:", base_times[-1])
    print("[INFO] latest future time:", pd.Timestamp(future_time_mat[-1, -1]))

    h1_time_mask_np = build_time_mask_np(
        future_time_mat,
        event_start,
        event_end
    )

    selected_times = future_time_mat[h1_time_mask_np]

    if len(selected_times) == 0:
        raise RuntimeError(f"{month}: H1 窗口内没有可用 sample-horizon。")

    print('[INFO] H0 selected unique hours:', len(pd.unique(selected_times)))
    print(
        '[INFO] H0 selected time range:',
        pd.Timestamp(selected_times.min()),
        "to",
        pd.Timestamp(selected_times.max())
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    preds_scenario = {
        intensity_name: []
        for intensity_name in INTENSITY_SCENARIOS.keys()
    }

    preds_cf = []
    trues = []
    event_flags = []
    raw_extreme_flags = []

    weather_diag_rows = []

    sample_offset = 0

    with torch.no_grad():
        for batch_i, batch in enumerate(loader):
            x_hist = batch["x_hist"].to(DEVICE)
            x_future_norm_obs = batch["x_future_norm_obs"].to(DEVICE)
            x_future_raw_obs = batch["x_future_raw_obs"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            extreme_flag = batch["extreme_flag"].to(DEVICE)

            B = x_future_raw_obs.shape[0]

            event_flag, h1_time_mask_t = make_event_flag_torch(
                base_times=base_times,
                batch_start=sample_offset,
                B=B,
                Tout=Tout,
                event_start=event_start,
                event_end=event_end,
                extreme_flag=extreme_flag,
                device=DEVICE
            )

            event_mask_4d = event_flag.expand_as(x_future_raw_obs) == 1

            normal_raw = build_normal_raw_tensor(
                x_future_raw_obs=x_future_raw_obs,
                mean=mean
            )

            # =================================================
            # CF: raw-only 正常天气
            # norm 不变，raw 在 event_flag 位置换成 normal
            # =================================================

            cf_raw = torch.where(
                event_mask_4d,
                normal_raw,
                x_future_raw_obs
            )

            cf_raw = apply_weather_physical_constraints_torch(cf_raw)

            pred_cf = model(
                x_hist=x_hist,
                x_future=x_future_norm_obs,
                A_norm=A_norm,
                x_future_raw=cf_raw
            )

            preds_cf.append(pred_cf.cpu())

            # =================================================
            # I1/I2/I3: raw-only delta 天气强度
            # scenario_raw = normal + k * (obs - normal)
            # norm 不变
            # =================================================

            delta_raw = x_future_raw_obs - normal_raw

            for intensity_name, k in INTENSITY_SCENARIOS.items():
                k = float(k)

                scenario_event_raw = normal_raw + k * delta_raw

                scenario_event_raw = apply_weather_physical_constraints_torch(
                    scenario_event_raw
                )

                scenario_raw = torch.where(
                    event_mask_4d,
                    scenario_event_raw,
                    x_future_raw_obs
                )

                pred_s = model(
                    x_hist=x_hist,
                    x_future=x_future_norm_obs,
                    A_norm=A_norm,
                    x_future_raw=scenario_raw
                )

                preds_scenario[intensity_name].append(pred_s.cpu())

                if batch_i == 0:
                    ef = event_mask_4d
                    if ef.sum().item() > 0:
                        obs_vals = x_future_raw_obs[ef].detach().cpu().numpy()
                        normal_vals = normal_raw[ef].detach().cpu().numpy()
                        scenario_vals = scenario_raw[ef].detach().cpu().numpy()

                        weather_diag_rows.append({
                            "month": month,
                            "intensity": intensity_name,
                            "k_delta": k,
                            "batch": batch_i,
                            "event_weather_records": int(ef.sum().item()),
                            "obs_weather_mean_all_vars": float(np.mean(obs_vals)),
                            "normal_weather_mean_all_vars": float(np.mean(normal_vals)),
                            "scenario_weather_mean_all_vars": float(np.mean(scenario_vals)),
                            "delta_weather_mean_all_vars": float(np.mean(obs_vals - normal_vals)),
                        })

            trues.append(y.cpu())
            event_flags.append(event_flag.cpu())
            raw_extreme_flags.append(extreme_flag.cpu())

            if batch_i == 0:
                print("[DEBUG] batch0 event_flag records:", int(event_flag.sum().item()))

            sample_offset += B

    # =====================================================
    # 合并预测并反归一化
    # =====================================================

    pred_cf_norm = torch.cat(preds_cf).numpy()
    true_norm = torch.cat(trues).numpy()
    event_flag_np = torch.cat(event_flags).numpy()
    raw_extreme_flag_np = torch.cat(raw_extreme_flags).numpy()

    pred_cf_denorm = denorm_flow(pred_cf_norm, mean, std)
    true_denorm = denorm_flow(true_norm, mean, std)

    pred_scenario_denorm = {}

    for intensity_name in INTENSITY_SCENARIOS.keys():
        arr_norm = torch.cat(preds_scenario[intensity_name]).numpy()
        pred_scenario_denorm[intensity_name] = denorm_flow(arr_norm, mean, std)

    print("[INFO] raw EXTREME_FLAG records:", int(raw_extreme_flag_np.sum()))
    print('[INFO] H0 event_flag records:', int(event_flag_np.sum()))

    # =====================================================
    # 保存结果
    # =====================================================

    h = 0
    time_axis = base_times

    event_window_mask_h0 = (
        (base_times >= event_start)
        & (base_times <= event_end)
    )

    true_total_series = true_denorm[:, h, :, 0].sum(axis=1)
    cf_total_series = pred_cf_denorm[:, h, :, 0].sum(axis=1)
    event_flag_series = event_flag_np[:, h, :, 0].max(axis=1)

    summary_rows = []

    for intensity_name, k in INTENSITY_SCENARIOS.items():
        out_dir = OUT_ROOT / H_NAME / intensity_name
        out_dir.mkdir(parents=True, exist_ok=True)

        pred_real_denorm = pred_scenario_denorm[intensity_name]

        scenario_total_series = pred_real_denorm[:, h, :, 0].sum(axis=1)

        scenario_event = pred_real_denorm[event_window_mask_h0, h, :, 0]
        cf_event = pred_cf_denorm[event_window_mask_h0, h, :, 0]

        scenario_total = float(scenario_event.sum())
        cf_total = float(cf_event.sum())

        impact = cf_total - scenario_total
        impact_percent = impact / (cf_total + 1e-6) * 100.0

        positive_impact = float(
            np.maximum(cf_event - scenario_event, 0.0).sum()
        )

        positive_impact_percent = positive_impact / (cf_total + 1e-6) * 100.0

        positive_ratio = (
            (
                cf_total_series[event_window_mask_h0]
                - scenario_total_series[event_window_mask_h0]
            ) > 0
        ).mean() * 100.0

        h1_hours_available_h0 = int(event_window_mask_h0.sum())

        print('\n========== H0 Raw-only Delta Impact, h=0 ==========')
        print("MONTH:", month)
        print("H_NAME:", H_NAME)
        print("INTENSITY:", intensity_name)
        print(f"k_delta                       = {float(k):.2f}")
        print(f"H1 hours available h=0        = {h1_hours_available_h0}")
        print(f"Scenario total flow           = {scenario_total:.2f}")
        print(f"Counterfactual flow           = {cf_total:.2f}")
        print(f"Impact CF - Scenario          = {impact:.2f}")
        print(f"Impact %                      = {impact_percent:.2f}%")
        print(f"Positive impact               = {positive_impact:.2f}")
        print(f"Positive impact %             = {positive_impact_percent:.2f}%")
        print(f"Positive ratio CF>S           = {positive_ratio:.2f}%")

        summary = {
            "month": month,
            "H_name": H_NAME,
            "intensity": intensity_name,
            "weather_delta_multiplier": float(k),

            "event_start": str(event_start),
            "event_end": str(event_end),

            "scaled_horizon": int(h),
            "h1_hours_available_h0": h1_hours_available_h0,
            "selected_unique_hours_all_horizons": int(len(pd.unique(selected_times))),
            "selected_days_all_horizons": int(
                len(pd.unique(pd.to_datetime(selected_times).date))
            ),

            'event_flag_records_H0': int(event_flag_np.sum()),
            "raw_extreme_flag_records": int(raw_extreme_flag_np.sum()),

            "scenario_total": scenario_total,
            "cf_total": cf_total,
            "impact": impact,
            "impact_percent": impact_percent,
            "positive_impact": positive_impact,
            "positive_impact_percent": positive_impact_percent,
            "positive_ratio": positive_ratio,

            "scenario_mean_total_flow": float(
                scenario_total_series[event_window_mask_h0].mean()
            ),
            "cf_mean_total_flow": float(
                cf_total_series[event_window_mask_h0].mean()
            ),

            "intervention_mode": "raw_only",
            "normal_weather_definition": (
                f"TEMP=norm_mean_TEMP({float(mean[1])}), "
                f"PRCP={NORMAL_PRCP}, "
                f"WDSP=norm_mean_WDSP({float(mean[3])}), "
                f"RH={NORMAL_RH}."
            ),
            "intensity_definition": (
                "I1 = normal + 1*(observed-normal) = observed; "
                "I2 = normal + 2*(observed-normal); "
                "I3 = normal + 3*(observed-normal). "
                "Only x_future_raw is modified; x_future_norm remains observed."
            ),
            "counterfactual_definition": (
                "At event_flag positions, x_future_raw is replaced by fixed normal weather; "
                "x_future_norm remains observed."
            ),
            "event_flag_definition": (
                'event_flag = EXTREME_FLAG × H0 official time window.'
            ),
            "weather_physical_constraints": (
                "PRCP>=0, WDSP>=0, RH clipped to [0,100]."
            ),
        }

        summary_rows.append(summary)

        summary_path = out_dir / f"{month}_{H_NAME}_{intensity_name}_event_impact_summary.csv"

        pd.DataFrame([summary]).to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig"
        )

        result_path = out_dir / f"{month}_national_multihazard_event_results.npz"

        np.savez_compressed(
            result_path,

            pred_real=pred_real_denorm.astype(np.float32),
            pred_cf=pred_cf_denorm.astype(np.float32),
            true=true_denorm.astype(np.float32),

            event_flag=event_flag_np.astype(np.float32),
            raw_extreme_flag=raw_extreme_flag_np.astype(np.float32),

            time_axis=np.array(time_axis.astype(str)),
            future_time_axis=future_time_axis,

            event_start=str(event_start),
            event_end=str(event_end),

            h0_event_start=str(event_start),
            h0_event_end=str(event_end),

            H_name=H_NAME,
            intensity_scenario=intensity_name,
            weather_delta_multiplier=float(k),

            weather_var_names=np.array(WEATHER_VAR_NAMES),

            normal_TEMP=float(mean[1]),
            normal_PRCP=float(NORMAL_PRCP),
            normal_WDSP=float(mean[3]),
            normal_RH=float(NORMAL_RH),

            weather_force=float(WEATHER_FORCE),
            scaled_horizon=int(h),
            month=month,
            Tin=Tin,
            Tout=Tout,

            intervention_mode="raw_only",
            normal_weather_definition=summary["normal_weather_definition"],
            intensity_definition=summary["intensity_definition"],
            counterfactual_definition=summary["counterfactual_definition"],
            event_flag_definition=summary["event_flag_definition"],
            weather_physical_constraints=summary["weather_physical_constraints"],
        )

        print("Saved summary:", summary_path)
        print("Saved result :", result_path)

        if PLOT_FIGURES:
            save_plot(
                out_dir=out_dir,
                month=month,
                intensity_name=intensity_name,
                time_axis=time_axis,
                true_total_series=true_total_series,
                scenario_total_series=scenario_total_series,
                cf_total_series=cf_total_series,
                event_flag_series=event_flag_series,
                event_start=event_start,
                event_end=event_end,
            )

    if len(weather_diag_rows) > 0:
        diag_df = pd.DataFrame(weather_diag_rows)
        diag_path = OUT_ROOT / H_NAME / f"{month}_rawonly_delta_batch0_diagnostics.csv"
        diag_df.to_csv(diag_path, index=False, encoding="utf-8-sig")
        print("[SAVED weather diagnostics]", diag_path)

    return summary_rows


# =====================================================
# 5. 主程序
# =====================================================

def main():
    print("=" * 80)
    print('S1_H0_rawonly_delta_I1I2I3')
    print("=" * 80)

    print("DEVICE:", DEVICE)
    print("BASE_DIR:", BASE_DIR)
    print("SAMPLE_ROOT:", SAMPLE_ROOT)
    print("OUT_ROOT:", OUT_ROOT)
    print("H_NAME:", H_NAME)
    print("MONTHS:", MONTHS)
    print("INTENSITY_SCENARIOS:", INTENSITY_SCENARIOS)
    print("WEATHER_FORCE:", WEATHER_FORCE)
    print("NORMAL_PRCP:", NORMAL_PRCP)
    print("NORMAL_RH:", NORMAL_RH)

    mean, std = load_norm(NORM_PATH)

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

    state = torch.load(
        CKPT_PATH,
        map_location=DEVICE
    )

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    model.eval()

    print("[INFO] Model loaded.")

    all_summary_rows = []

    for month in MONTHS:
        rows = run_one_month(
            month=month,
            model=model,
            A_norm=A_norm,
            mean=mean,
            std=std
        )

        all_summary_rows.extend(rows)

    all_summary_df = pd.DataFrame(all_summary_rows)

    all_summary_path = (
        OUT_ROOT /
        H_NAME /
        f"all_months_{H_NAME}_I1I2I3_rawonly_delta_event_impact_summary.csv"
    )

    all_summary_df.to_csv(
        all_summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 80)
    print('✅ S1 H0 I1/I2/I3 raw-only delta 完成')
    print("=" * 80)
    print("[SAVED]", all_summary_path)
    print(all_summary_df)

    print("\n重点检查：")
    print("1. I1 是否基本复现旧 I100 的 impact_percent。")
    print("2. 201903 I1 应该接近旧版约 12% 左右。")
    print("3. 201908 I1 应该接近旧版约 20% 左右。")
    print("4. I2/I3 是否随 k 增强而整体冲击增强；若不单调，再单独处理强度变量。")
    print('5. 输出 npz 是否在 H0/I1、H0/I2、H0/I3 下。')


if __name__ == "__main__":
    main()