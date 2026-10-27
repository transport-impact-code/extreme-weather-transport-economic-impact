import os
import pandas as pd
import numpy as np

# =========================
# 路径
# =========================

DATA_DIR = 'artifacts/labeled'

NODE_FILE = 'data/site_locations_130.csv'

SAVE_DIR = 'artifacts/samples'


# =========================
# 参数
# =========================

Tin = 24
Tout = 12

# 模型真正使用的核心特征
FEATURE_COLS = ["FLOW", "TEMP", "PRCP", "WDSP", "RH", "EXTREME_FLAG"]

# 额外保存的标签，方便后续分析
EXTRA_FLAG_COLS = [
    "RAIN_FLAG",
    "WIND_FLAG",
    "COLD_FLAG",
    "SNOW_ICE_FLAG",
    "WEATHER_EXTREME_FLAG",
    "EVENT_FLAG",
]

ALL_COLS = FEATURE_COLS + EXTRA_FLAG_COLS

# =========================
# 读取固定节点顺序
# =========================

def main():
    nodes = (
        pd.read_csv(NODE_FILE, dtype={"SITEREF": str})["SITEREF"]
        .astype(str)
        .str.strip()
        .str.zfill(8)
        .tolist()
    )

    N = len(nodes)
    print(f"[INFO] 固定节点数 N = {N}")

    # =========================
    # 文件列表
    # =========================

    csv_files = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.endswith("_extreme_multihazard_event_v1.csv")
    ])

    print(f"[INFO] 共发现 {len(csv_files)} 个月份文件")

    if len(csv_files) == 0:
        raise RuntimeError("没有找到 _extreme_multihazard_event_v1.csv 文件，请检查 DATA_DIR")

    # =========================
    # 主循环
    # =========================

    for fname in csv_files:

        month = fname[:6]

        print("\n==============================")
        print(f"[PROCESS] {month} | {fname}")

        csv_path = os.path.join(DATA_DIR, fname)

        df = pd.read_csv(
            csv_path,
            dtype={"SITEREF": str},
            low_memory=False
        )

        # =========================
        # 基础检查
        # =========================

        if "SITEREF" not in df.columns:
            print(f"[SKIP] {fname} 没有 SITEREF 列")
            continue

        if "DATETIME" not in df.columns:
            print(f"[SKIP] {fname} 没有 DATETIME 列")
            continue

        missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
        if len(missing_cols) > 0:
            raise ValueError(f"[ERROR] {fname} 缺少核心列: {missing_cols}")

        # 如果某些额外标签不存在，就补0
        for col in EXTRA_FLAG_COLS:
            if col not in df.columns:
                print(f"[WARN] {fname} 缺少 {col}，自动补0")
                df[col] = 0

        # =========================
        # 基础处理
        # =========================

        df["SITEREF"] = df["SITEREF"].astype(str).str.strip().str.zfill(8)
        df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")

        df = df.dropna(subset=["DATETIME", "SITEREF"]).copy()

        for col in ALL_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 只保留130个节点
        df = df[df["SITEREF"].isin(nodes)].copy()

        if len(df) == 0:
            print(f"[SKIP] {month} 过滤130节点后为空")
            continue

        # =========================
        # 同一时间同一节点聚合
        # =========================

        agg_dict = {
            "FLOW": "sum",
            "TEMP": "mean",
            "PRCP": "mean",
            "WDSP": "mean",
            "RH": "mean",

            # 所有标签用 max
            "EXTREME_FLAG": "max",
            "RAIN_FLAG": "max",
            "WIND_FLAG": "max",
            "COLD_FLAG": "max",
            "SNOW_ICE_FLAG": "max",
            "WEATHER_EXTREME_FLAG": "max",
            "EVENT_FLAG": "max",
        }

        df = (
            df.groupby(["DATETIME", "SITEREF"], as_index=False)
            .agg(agg_dict)
        )

        df = df.sort_values(["DATETIME", "SITEREF"]).reset_index(drop=True)

        # =========================
        # 补齐完整小时序列
        # =========================

        min_time = df["DATETIME"].min()
        max_time = df["DATETIME"].max()

        full_times = pd.date_range(min_time, max_time, freq="h")
        T = len(full_times)

        print(f"[INFO] 时间范围: {min_time} 到 {max_time}")
        print(f"[INFO] 完整小时数 T = {T}")

        if T < Tin + Tout:
            print(f"[SKIP] {month} 小时数不足")
            continue

        full_index = pd.MultiIndex.from_product(
            [full_times, nodes],
            names=["DATETIME", "SITEREF"]
        )

        df_full = (
            df.set_index(["DATETIME", "SITEREF"])
              .reindex(full_index)
              .reset_index()
        )

        # =========================
        # 缺失检查
        # =========================

        missing_ratio = df_full[ALL_COLS].isna().mean()

        print("[INFO] 缺失比例:")
        print(missing_ratio)

        # =========================
        # 缺失填补
        # =========================

        # FLOW 缺失：认为该小时该节点无记录，填0
        df_full["FLOW"] = df_full["FLOW"].fillna(0)

        # 天气缺失：按节点前后填充，再整体均值兜底
        for col in ["TEMP", "PRCP", "WDSP", "RH"]:

            df_full[col] = (
                df_full.groupby("SITEREF")[col]
                .transform(lambda s: s.ffill().bfill())
            )

            df_full[col] = df_full[col].fillna(df_full[col].mean())

        # 标签缺失：填0，并确保0/1
        flag_cols = [
            "EXTREME_FLAG",
            "RAIN_FLAG",
            "WIND_FLAG",
            "COLD_FLAG",
            "SNOW_ICE_FLAG",
            "WEATHER_EXTREME_FLAG",
            "EVENT_FLAG",
        ]

        for col in flag_cols:
            df_full[col] = df_full[col].fillna(0)
            df_full[col] = (df_full[col] > 0).astype(np.float32)

        # =========================
        # 排序，保证 tensor 节点顺序正确
        # =========================

        df_full["SITEREF"] = pd.Categorical(
            df_full["SITEREF"],
            categories=nodes,
            ordered=True
        )

        df_full = df_full.sort_values(["DATETIME", "SITEREF"]).reset_index(drop=True)

        # =========================
        # 构建核心 tensor: (T, N, 6)
        # 顺序: FLOW, TEMP, PRCP, WDSP, RH, EXTREME_FLAG
        # =========================

        values = df_full[FEATURE_COLS].values.astype(np.float32)

        if values.shape[0] != T * N:
            raise ValueError(
                f"[ERROR] {month}: 数据长度不匹配, values={values.shape[0]}, T*N={T*N}"
            )

        tensor = values.reshape(T, N, len(FEATURE_COLS))

        print(f"[INFO] tensor shape = {tensor.shape}")

        # =========================
        # 额外 flag tensor: (T, N, 6)
        # =========================

        extra_values = df_full[EXTRA_FLAG_COLS].values.astype(np.float32)
        extra_tensor = extra_values.reshape(T, N, len(EXTRA_FLAG_COLS))

        # =========================
        # 构建样本
        # =========================

        X_hist_list = []
        X_future_list = []
        Y_list = []
        EXTREME_FLAG_list = []

        RAIN_FLAG_list = []
        WIND_FLAG_list = []
        COLD_FLAG_list = []
        SNOW_ICE_FLAG_list = []
        WEATHER_EXTREME_FLAG_list = []
        EVENT_FLAG_list = []

        for t in range(Tin - 1, T - Tout):

            # 历史输入：过去24小时 flow + weather
            X_hist_list.append(
                tensor[t - Tin + 1: t + 1, :, 0:5]
            )

            # 未来输入：未来12小时 weather
            X_future_list.append(
                tensor[t + 1: t + 1 + Tout, :, 1:5]
            )

            # 未来12小时 flow
            Y_list.append(
                tensor[t + 1: t + 1 + Tout, :, 0:1]
            )

            # 未来12小时总 extreme flag
            EXTREME_FLAG_list.append(
                tensor[t + 1: t + 1 + Tout, :, 5:6]
            )

            # 未来12小时子标签
            flags_future = extra_tensor[t + 1: t + 1 + Tout, :, :]

            RAIN_FLAG_list.append(flags_future[:, :, 0:1])
            WIND_FLAG_list.append(flags_future[:, :, 1:2])
            COLD_FLAG_list.append(flags_future[:, :, 2:3])
            SNOW_ICE_FLAG_list.append(flags_future[:, :, 3:4])
            WEATHER_EXTREME_FLAG_list.append(flags_future[:, :, 4:5])
            EVENT_FLAG_list.append(flags_future[:, :, 5:6])

        X_hist = np.stack(X_hist_list).astype(np.float32)
        X_future = np.stack(X_future_list).astype(np.float32)
        Y = np.stack(Y_list).astype(np.float32)
        EXTREME_FLAG = np.stack(EXTREME_FLAG_list).astype(np.float32)

        RAIN_FLAG = np.stack(RAIN_FLAG_list).astype(np.float32)
        WIND_FLAG = np.stack(WIND_FLAG_list).astype(np.float32)
        COLD_FLAG = np.stack(COLD_FLAG_list).astype(np.float32)
        SNOW_ICE_FLAG = np.stack(SNOW_ICE_FLAG_list).astype(np.float32)
        WEATHER_EXTREME_FLAG = np.stack(WEATHER_EXTREME_FLAG_list).astype(np.float32)
        EVENT_FLAG = np.stack(EVENT_FLAG_list).astype(np.float32)

        print(
            f"[OK] 样本数 = {X_hist.shape[0]} | "
            f"X_hist {X_hist.shape} | "
            f"X_future {X_future.shape} | "
            f"Y {Y.shape} | "
            f"EXTREME_FLAG {EXTREME_FLAG.shape}"
        )

        print(f"[INFO] EXTREME_FLAG 占比          : {EXTREME_FLAG.mean():.4f}")
        print(f"[INFO] WEATHER_EXTREME_FLAG 占比  : {WEATHER_EXTREME_FLAG.mean():.4f}")
        print(f"[INFO] EVENT_FLAG 占比            : {EVENT_FLAG.mean():.4f}")
        print(f"[INFO] RAIN_FLAG 占比             : {RAIN_FLAG.mean():.4f}")
        print(f"[INFO] WIND_FLAG 占比             : {WIND_FLAG.mean():.4f}")
        print(f"[INFO] COLD_FLAG 占比             : {COLD_FLAG.mean():.4f}")
        print(f"[INFO] SNOW_ICE_FLAG 占比         : {SNOW_ICE_FLAG.mean():.4f}")

        # =========================
        # 保存
        # =========================

        save_path = os.path.join(SAVE_DIR, f"{month}_samples.npz")

        np.savez_compressed(
            save_path,
            X_hist=X_hist,
            X_future=X_future,
            Y=Y,

            # 总标签，训练主要用这个
            EXTREME_FLAG=EXTREME_FLAG,

            # 子标签，后续分析用
            RAIN_FLAG=RAIN_FLAG,
            WIND_FLAG=WIND_FLAG,
            COLD_FLAG=COLD_FLAG,
            SNOW_ICE_FLAG=SNOW_ICE_FLAG,
            WEATHER_EXTREME_FLAG=WEATHER_EXTREME_FLAG,
            EVENT_FLAG=EVENT_FLAG,

            nodes=np.array(nodes),
            times=full_times.astype(str).values
        )

        print(f"[SAVED] {save_path}")

    print("\n🎉 [ALL DONE] 多灾种 + 事件标签样本构建完成")