import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from .dataset_train import (
    TrafficRawDataset,
    build_split_indices_by_month,
    compute_norm_stats_from_indices,
    NormalizedSubset,
    save_norm_stats_json,
)

from .model import STGNNWeatherSeq2Seq


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 路径配置
# =========================

SAMPLES_DIR = 'artifacts/samples'

ADJ_PATH = 'data/adjacency_distance_130x130_exp.csv'

SAVE_DIR = 'artifacts/checkpoints'

CKPT_PATH = os.path.join(SAVE_DIR, "stgnn_weather_multihazard_event_v1_24to12.pth")
NORM_PATH = os.path.join(SAVE_DIR, "norm_stats_multihazard_event_v1.json")
LOSS_PATH = os.path.join(SAVE_DIR, "loss_curve_multihazard_event_v1.npz")
LOSS_PNG  = os.path.join(SAVE_DIR, "loss_curve_multihazard_event_v1.png")

# =========================
# 数据维度
# =========================

NUM_NODES = 130
Tin = 24
Tout = 12

# =========================
# 训练超参数
# =========================

BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
SEED = 42

NUM_WORKERS = 2
PIN_MEMORY = (DEVICE == "cuda")

EXTREME_ALPHA = 5.0

# 天气惩罚强度：建议先用2.0，不要一开始就5.0
WEATHER_FORCE = 5.0

# 方向约束：保证反事实流量 >= 极端天气流量
DIRECTION_BETA = 1.0
DIRECTION_MARGIN = 0.0

USE_AMP = False
GRAD_CLIP_NORM = 5.0

# =========================
# 按月份划分
# =========================

TEST_MONTHS = ["201903", "201908", "201911", "201912"]
VAL_MONTHS  = ["201304", "201305", "201306", "201309"]

# =========================
# 邻接矩阵
# =========================

ADJ_ALREADY_NORMALIZED = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_adj(path: str, num_nodes: int, device: str, already_normalized: bool):
    adj_df = pd.read_csv(path, index_col=0)
    adj_df = adj_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    A = torch.tensor(adj_df.values, dtype=torch.float32)

    if already_normalized:
        return A.to(device)

    I = torch.eye(num_nodes)
    A_hat = A + I
    D = torch.diag(torch.sum(A_hat, dim=1).clamp(min=1e-12) ** -0.5)

    return (D @ A_hat @ D).to(device)


def weighted_extreme_mae(pred, y, extreme_flag, alpha=3.0):
    base_loss = torch.abs(pred - y)
    weight = 1.0 + alpha * extreme_flag
    return (base_loss * weight).mean()


def directional_counterfactual_loss(pred_real, pred_cf, extreme_flag, margin=0.0):
    """
    目标：
    在 extreme_flag == 1 的位置，要求 pred_cf >= pred_real。

    如果 pred_cf < pred_real，则产生惩罚。
    """
    violation = torch.relu(pred_real - pred_cf + margin)
    loss = (violation * extreme_flag).sum() / (extreme_flag.sum() + 1e-6)
    return loss


def build_counterfactual_raw(x_future_raw, extreme_flag, norm_stats):
    """
    构造反事实 raw weather。

    x_future_raw: (B, 12, N, 4)
    顺序: TEMP, PRCP, WDSP, RH

    只在 extreme_flag == 1 的位置替换成普通天气。
    x_future_norm 不改，只改 raw weather，让反事实主要作用于 penalty。
    """

    cf_future_raw = x_future_raw.clone()

    normal_raw = torch.zeros_like(cf_future_raw)

    # norm_stats["mean"] 顺序:
    # [FLOW, TEMP, PRCP, WDSP, RH]
    temp_mean = float(norm_stats["mean"][1])
    wdsp_mean = float(norm_stats["mean"][3])

    normal_raw[..., 0] = temp_mean   # TEMP
    normal_raw[..., 1] = 0.0         # PRCP
    normal_raw[..., 2] = wdsp_mean   # WDSP
    normal_raw[..., 3] = 70.0        # RH

    mask = extreme_flag.expand_as(cf_future_raw)

    cf_future_raw = torch.where(
        mask == 1,
        normal_raw,
        cf_future_raw
    )

    return cf_future_raw


def plot_losses(train_losses, val_losses, train_main_losses, train_dir_losses, save_path):
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_losses, label="Train Total")
    plt.plot(epochs, val_losses, label="Val Total")
    plt.plot(epochs, train_main_losses, label="Train Pred Loss", linestyle="--")
    plt.plot(epochs, train_dir_losses, label="Train Direction Loss", linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("STGNN 24→12 Training Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    set_seed(SEED)

    print("[INFO] DEVICE:", DEVICE)
    print("[INFO] TASK: multihazard event model + directional counterfactual loss")
    print("[INFO] SAMPLES_DIR:", SAMPLES_DIR)
    print("[INFO] EXTREME_ALPHA:", EXTREME_ALPHA)
    print("[INFO] WEATHER_FORCE:", WEATHER_FORCE)
    print("[INFO] DIRECTION_BETA:", DIRECTION_BETA)

    # -------------------------------------------------
    # 1. Raw dataset
    # -------------------------------------------------

    raw_ds = TrafficRawDataset(SAMPLES_DIR)

    train_idx, val_idx, test_idx = build_split_indices_by_month(
        raw_ds,
        test_months=TEST_MONTHS,
        val_months=VAL_MONTHS
    )

    print(f"[INFO] Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError("train/val/test empty, check month split.")

    # -------------------------------------------------
    # 2. Norm stats
    # -------------------------------------------------

    norm_stats = compute_norm_stats_from_indices(raw_ds, train_idx)
    save_norm_stats_json(norm_stats, NORM_PATH)

    # -------------------------------------------------
    # 3. Normalized datasets
    # -------------------------------------------------

    train_set = NormalizedSubset(raw_ds, train_idx, norm_stats)
    val_set = NormalizedSubset(raw_ds, val_idx, norm_stats)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    print("[INFO] DataLoader ready")

    # -------------------------------------------------
    # 4. Adjacency
    # -------------------------------------------------

    A_norm = load_adj(
        ADJ_PATH,
        NUM_NODES,
        DEVICE,
        ADJ_ALREADY_NORMALIZED
    )

    # -------------------------------------------------
    # 5. Model
    # -------------------------------------------------

    model = STGNNWeatherSeq2Seq(
        num_nodes=NUM_NODES,
        weather_force=WEATHER_FORCE,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_val = float("inf")

    train_losses = []
    val_losses = []
    train_main_losses = []
    train_dir_losses = []

    # -------------------------------------------------
    # 6. Training loop
    # -------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        model.train()

        tr_total = 0.0
        tr_main = 0.0
        tr_dir = 0.0

        for batch in train_loader:

            x_hist = batch["x_hist"].to(DEVICE, non_blocking=True)
            x_future = batch["x_future"].to(DEVICE, non_blocking=True)
            x_future_raw = batch["x_future_raw"].to(DEVICE, non_blocking=True)
            y = batch["y"].to(DEVICE, non_blocking=True)
            extreme_flag = batch["extreme_flag"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):

                # 真实天气预测
                pred_real = model(
                    x_hist=x_hist,
                    x_future=x_future,
                    A_norm=A_norm,
                    x_future_raw=x_future_raw,
                    y_true=y
                )

                # 构造反事实 raw weather
                cf_future_raw = build_counterfactual_raw(
                    x_future_raw=x_future_raw,
                    extreme_flag=extreme_flag,
                    norm_stats=norm_stats
                )

                # 反事实预测
                # 注意：x_future 不变，只改 raw weather
                pred_cf = model(
                    x_hist=x_hist,
                    x_future=x_future,
                    A_norm=A_norm,
                    x_future_raw=cf_future_raw,
                    y_true=y
                )

                # 主预测 loss
                loss_pred = weighted_extreme_mae(
                    pred_real,
                    y,
                    extreme_flag,
                    alpha=EXTREME_ALPHA
                )

                # 方向约束 loss
                loss_dir = directional_counterfactual_loss(
                    pred_real,
                    pred_cf,
                    extreme_flag,
                    margin=DIRECTION_MARGIN
                )

                loss = loss_pred + DIRECTION_BETA * loss_dir

            scaler.scale(loss).backward()

            if GRAD_CLIP_NORM is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            scaler.step(optimizer)
            scaler.update()

            tr_total += loss.item()
            tr_main += loss_pred.item()
            tr_dir += loss_dir.item()

        tr_total /= max(len(train_loader), 1)
        tr_main /= max(len(train_loader), 1)
        tr_dir /= max(len(train_loader), 1)

        train_losses.append(tr_total)
        train_main_losses.append(tr_main)
        train_dir_losses.append(tr_dir)

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        model.eval()

        va_total = 0.0

        with torch.no_grad():

            for batch in val_loader:

                x_hist = batch["x_hist"].to(DEVICE, non_blocking=True)
                x_future = batch["x_future"].to(DEVICE, non_blocking=True)
                x_future_raw = batch["x_future_raw"].to(DEVICE, non_blocking=True)
                y = batch["y"].to(DEVICE, non_blocking=True)
                extreme_flag = batch["extreme_flag"].to(DEVICE, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=USE_AMP):

                    pred_real = model(
                        x_hist=x_hist,
                        x_future=x_future,
                        A_norm=A_norm,
                        x_future_raw=x_future_raw
                    )

                    cf_future_raw = build_counterfactual_raw(
                        x_future_raw=x_future_raw,
                        extreme_flag=extreme_flag,
                        norm_stats=norm_stats
                    )

                    pred_cf = model(
                        x_hist=x_hist,
                        x_future=x_future,
                        A_norm=A_norm,
                        x_future_raw=cf_future_raw
                    )

                    loss_pred = weighted_extreme_mae(
                        pred_real,
                        y,
                        extreme_flag,
                        alpha=EXTREME_ALPHA
                    )

                    loss_dir = directional_counterfactual_loss(
                        pred_real,
                        pred_cf,
                        extreme_flag,
                        margin=DIRECTION_MARGIN
                    )

                    loss = loss_pred + DIRECTION_BETA * loss_dir

                va_total += loss.item()

        va_total /= max(len(val_loader), 1)
        val_losses.append(va_total)

        print(
            f"Epoch {epoch:03d} | "
            f"Train total {tr_total:.6f} | "
            f"Pred {tr_main:.6f} | "
            f"Dir {tr_dir:.6f} | "
            f"Val total {va_total:.6f}"
        )

        if va_total < best_val:

            best_val = va_total
            torch.save(model.state_dict(), CKPT_PATH)

            print("[SAVE] Best model ->", CKPT_PATH)

    # -------------------------------------------------
    # 7. Save curves
    # -------------------------------------------------

    np.savez(
        LOSS_PATH,
        train_loss=np.array(train_losses, dtype=np.float32),
        val_loss=np.array(val_losses, dtype=np.float32),
        train_pred_loss=np.array(train_main_losses, dtype=np.float32),
        train_direction_loss=np.array(train_dir_losses, dtype=np.float32),
    )

    plot_losses(
        train_losses=train_losses,
        val_losses=val_losses,
        train_main_losses=train_main_losses,
        train_dir_losses=train_dir_losses,
        save_path=LOSS_PNG
    )

    print("[DONE] Training finished.")
    print(f"[INFO] Best val loss = {best_val:.6f}")
    print("[SAVE] Model ->", CKPT_PATH)
    print("[SAVE] Norm  ->", NORM_PATH)
    print("[SAVE] Loss  ->", LOSS_PATH)
    print("[SAVE] Plot  ->", LOSS_PNG)


