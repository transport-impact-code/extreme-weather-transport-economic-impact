import os
import re
import json
import numpy as np
import torch
from torch.utils.data import Dataset


# =========================
# 工具：从文件名提取月份
# =========================

def _infer_month_from_filename(fname: str):
    m = re.match(r"^(\d{6})_samples\.npz$", os.path.basename(fname))
    return m.group(1) if m else ""


# =========================
# 原始 Dataset
# =========================

class TrafficRawDataset(Dataset):

    def __init__(self, samples_dir):

        self.X_hist = []
        self.X_future = []
        self.Y = []
        self.EXTREME_FLAG = []
        self.sample_month = []

        files = sorted([
            f for f in os.listdir(samples_dir)
            if f.endswith("_samples.npz")
        ])

        print("[INFO] loading files:", len(files))

        for fname in files:

            month = _infer_month_from_filename(fname)
            path = os.path.join(samples_dir, fname)

            data = np.load(path)

            Xh = data["X_hist"]
            Xf = data["X_future"]
            Y = data["Y"]
            extreme_flag = data["EXTREME_FLAG"]

            Xh = np.nan_to_num(Xh)
            Xf = np.nan_to_num(Xf)
            Y = np.nan_to_num(Y)
            extreme_flag = np.nan_to_num(extreme_flag)

            self.X_hist.append(Xh.astype(np.float32))
            self.X_future.append(Xf.astype(np.float32))
            self.Y.append(Y.astype(np.float32))
            self.EXTREME_FLAG.append(extreme_flag.astype(np.float32))

            self.sample_month.extend([month] * Xh.shape[0])

        self.X_hist = np.concatenate(self.X_hist, axis=0)
        self.X_future = np.concatenate(self.X_future, axis=0)
        self.Y = np.concatenate(self.Y, axis=0)
        self.EXTREME_FLAG = np.concatenate(self.EXTREME_FLAG, axis=0)

        self.sample_month = np.array(self.sample_month)

        print("[INFO] total samples:", self.X_hist.shape[0])
        print("[INFO] X_hist:", self.X_hist.shape)
        print("[INFO] X_future:", self.X_future.shape)
        print("[INFO] Y:", self.Y.shape)

    def __len__(self):
        return self.X_hist.shape[0]

    def __getitem__(self, idx):

        return {
            "x_hist": torch.tensor(self.X_hist[idx]),
            "x_future": torch.tensor(self.X_future[idx]),
            "x_future_raw": torch.tensor(self.X_future[idx]),  # ⭐关键
            "y": torch.tensor(self.Y[idx]),
            "extreme_flag": torch.tensor(self.EXTREME_FLAG[idx]),
        }


# =========================
# 划分 train / val / test
# =========================

def build_split_indices_by_month(raw_ds, test_months, val_months):

    train_idx = []
    val_idx = []
    test_idx = []

    for i, m in enumerate(raw_ds.sample_month):

        if m in test_months:
            test_idx.append(i)
        elif m in val_months:
            val_idx.append(i)
        else:
            train_idx.append(i)

    return train_idx, val_idx, test_idx


# =========================
# 计算归一化参数
# =========================

def compute_norm_stats_from_indices(raw_ds, indices):

    Xh = raw_ds.X_hist[indices]  # (B,24,N,5)

    # reshape成 (所有样本点, feature)
    Xh = Xh.reshape(-1, Xh.shape[-1])  # (B*24*N, 5)

    mean = Xh.mean(axis=0)
    std = Xh.std(axis=0)

    std[std < 1e-6] = 1.0  # 防止除0

    print("[INFO] Norm mean:", mean)
    print("[INFO] Norm std :", std)

    return {
        "mean": mean.tolist(),
        "std": std.tolist()
    }


# =========================
# 保存归一化参数
# =========================

def save_norm_stats_json(norm_stats, path):

    with open(path, "w") as f:
        json.dump(norm_stats, f, indent=4)

    print("[INFO] norm stats saved:", path)


# =========================
# Normalized Dataset
# =========================

class NormalizedSubset(Dataset):

    def __init__(self, raw_ds, indices, norm_stats):

        self.raw_ds = raw_ds
        self.indices = indices

        self.mean = np.array(norm_stats["mean"], dtype=np.float32)
        self.std = np.array(norm_stats["std"], dtype=np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):

        idx = self.indices[i]

        Xh = self.raw_ds.X_hist[idx].copy()
        Xf_raw = self.raw_ds.X_future[idx].copy()
        Y = self.raw_ds.Y[idx].copy()
        EXTREME_FLAG = self.raw_ds.EXTREME_FLAG[idx].copy()

        # =========================
        # 历史归一化
        # =========================

        Xh = (Xh - self.mean) / self.std

        # =========================
        # 未来天气归一化
        # =========================

        weather_mean = self.mean[1:5]
        weather_std = self.std[1:5]

        Xf = (Xf_raw - weather_mean) / weather_std

        # =========================
        # flow归一化
        # =========================

        flow_mean = self.mean[0]
        flow_std = self.std[0]

        Y = (Y - flow_mean) / flow_std

        return {
            "x_hist": torch.tensor(Xh),
            "x_future": torch.tensor(Xf),
            "x_future_raw": torch.tensor(Xf_raw),  # ⭐核心
            "y": torch.tensor(Y),
            "extreme_flag": torch.tensor(EXTREME_FLAG),
        }