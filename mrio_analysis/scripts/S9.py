# -*- coding: utf-8 -*-
from pathlib import Path
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# =====================================================
# S9-2_clean:
# 动态 MRIO + AES 行业库存天数 + sector-major checked 输入
# =====================================================
# 核心逻辑与基准 MRIO 模拟一致：
#   每期每个部门实际产出 =
#       min(库存约束, 生产能力约束, 订单约束)
#
# 与基准代码的对应：
#   zz_s             -> 中间投入矩阵
#   ff_s             -> 最终需求矩阵
#   VA_TT            -> 生产能力冲击
#   mat_order        -> 订单矩阵
#   mat_stock        -> 当前库存
#   mat_stock_obj    -> 目标库存
#   mat_ezc          -> 单位产出投入系数
#   mat_ezz          -> 库存缺口分配回原始供应商的比例
#   xx_TT            -> 动态产出路径
#
# 本版修正：
#   1. 不再读取旧 MRIO模型；
#   2. 读取 MRIO模型_sector_major_checked；
#   3. 使用 sector-major 顺序：
#          index = sector * RR + region
#   4. 使用 AES 2019 推算的行业库存天数；
#   5. 库存类别按供应行业聚合，即 19 × 285；
#   6. 输出 I1/I2/I3 动态 MRIO 结果。
# =====================================================


# =====================================================
# 1. 路径配置
# =====================================================

BASE_DIR = Path(os.environ.get("SCENARIO_DATA_ROOT", "data"))

SCENARIO_ROOT = (
    BASE_DIR /
    '第五阶段情景分析new/I_H0'
)

# Step8 + S9-1 后的输入根目录
INPUT_ROOT_CANDIDATES = [
    SCENARIO_ROOT / r"Intensity_step8_annual_365_VA_TT_H0_eta3.5",
    SCENARIO_ROOT / 'Intensity_step8_annual_365_VA_TT_H0_eta3.5',
    SCENARIO_ROOT / r"Intensity_step8_annual_365_VA_TT_eta3.5",
]

INPUT_ROOT = None
for p in INPUT_ROOT_CANDIDATES:
    if p.exists():
        INPUT_ROOT = p
        break

if INPUT_ROOT is None:
    raise FileNotFoundError(
        "找不到 Step8 输入目录，请检查：\n"
        + "\n".join(str(p) for p in INPUT_ROOT_CANDIDATES)
    )


# 库存天数文件：使用 S9-0 生成的 middle 版本
STOCK_DAYS_CSV = (
    SCENARIO_ROOT /
    '库存/sector_stock_days_AES2019_middle/sector_stock_days_for_model.csv'
)

# 如要做敏感性，可以改成：
# sector_stock_days_for_model_low.csv
# sector_stock_days_for_model_high.csv

RUN_TAG = "dynamic_mrio_clean_AES2019_middle_sector_major"

H_NAME = "H0"

INTENSITY_SCENARIOS = [
    "I1",
    "I2",
    "I3",
]


# =====================================================
# 2. 模型参数
# =====================================================

RR = 15
NN = 19
RRNN = RR * NN

TT = 365
EPS = 1e-9
DAYS_PER_YEAR = 365.0

# 关键投入识别
# 与基准代码中的 mat_key 对应：
# 基准代码用外部 mat_key；
# 这里用投入份额自动识别关键投入类别。
KEY_SHARE_CUTOFF = 0.05
MIN_KEY_CATEGORIES = 3
MAX_KEY_CATEGORIES = 8

# 如果你想更接近“所有投入都可约束”，可以改成 "all_positive"
# 当前推荐 "share"
KEY_MODE = "share"  # "share" or "all_positive"

PLOT_FIGURES = True


# =====================================================
# 3. 工具函数
# =====================================================

def read_name_csv(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"找不到名称文件: {path}")

    df = pd.read_csv(path)

    if df.shape[1] == 1:
        return df.iloc[:, 0].astype(str).tolist()

    for c in ["region", "region_name", "sector", "sector_name", "name"]:
        if c in df.columns:
            return df[c].astype(str).tolist()

    return df.iloc[:, -1].astype(str).tolist()


def get_impact_mask(time_index):
    """
    兼容不同版本 time_index：
      1. 如果有 is_impact_day，用它；
      2. 如果有 period_type，非 normal 视为冲击期；
      3. 否则全部 False。
    """

    if "is_impact_day" in time_index.columns:
        return (
            pd.to_numeric(time_index["is_impact_day"], errors="coerce")
            .fillna(0)
            .astype(int)
            .to_numpy() == 1
        )

    if "period_type" in time_index.columns:
        return (
            time_index["period_type"]
            .astype(str)
            .str.lower()
            .ne("normal")
            .to_numpy()
        )

    return np.zeros(len(time_index), dtype=bool)


def load_mrio_inputs(base_path):
    """
    读取已经经过 S9-1 检查和重排的 sector-major 输入。
    """

    base_path = Path(base_path)

    required = [
        "zz_s.npy",
        "ff_s.npy",
        "va_s.npy",
        "xx_s.npy",
        "VA_TT_transport_only.npy",
        "region_names.csv",
        "sector_names.csv",
        "time_index.csv",
    ]

    for fname in required:
        p = base_path / fname
        if not p.exists():
            raise FileNotFoundError(f"缺少输入文件: {p}")

    zz_s = np.load(base_path / "zz_s.npy")
    ff_s = np.load(base_path / "ff_s.npy")
    va_s = np.load(base_path / "va_s.npy")
    xx_s = np.load(base_path / "xx_s.npy")
    VA_TT = np.load(base_path / "VA_TT_transport_only.npy")

    region_names = read_name_csv(base_path / "region_names.csv")
    sector_names = read_name_csv(base_path / "sector_names.csv")

    time_index = pd.read_csv(base_path / "time_index.csv")

    if "date" in time_index.columns:
        time_index["date"] = pd.to_datetime(time_index["date"])

    return (
        zz_s,
        ff_s,
        va_s,
        xx_s,
        VA_TT,
        region_names,
        sector_names,
        time_index,
    )


def load_sector_stock_days(stock_csv, sector_names):
    """
    读取 S9-0 生成的 sector_stock_days_for_model.csv。
    """

    stock_csv = Path(stock_csv)

    if not stock_csv.exists():
        raise FileNotFoundError(f"找不到库存天数文件: {stock_csv}")

    df = pd.read_csv(stock_csv)

    if "stock_days_for_model" not in df.columns:
        raise ValueError(f"{stock_csv} 缺少 stock_days_for_model 字段。")

    df["stock_days_for_model"] = pd.to_numeric(
        df["stock_days_for_model"],
        errors="coerce"
    )

    if df["stock_days_for_model"].isna().any():
        raise ValueError("库存天数表中存在 NaN。")

    stock_days = np.zeros(len(sector_names), dtype=float)

    # 优先按 sector_index
    if "sector_index" in df.columns:
        tmp = df.copy()
        tmp["sector_index"] = pd.to_numeric(
            tmp["sector_index"],
            errors="coerce"
        )

        for _, row in tmp.dropna(subset=["sector_index"]).iterrows():
            idx = int(row["sector_index"])

            if 0 <= idx < len(sector_names):
                stock_days[idx] = float(row["stock_days_for_model"])

    # 如果 sector_index 没匹配全，再按 sector 名称匹配
    if (stock_days <= 0).any() and "sector" in df.columns:
        stock_map = {
            str(row["sector"]).strip(): float(row["stock_days_for_model"])
            for _, row in df.iterrows()
        }

        for i, s in enumerate(sector_names):
            if stock_days[i] <= 0 and str(s).strip() in stock_map:
                stock_days[i] = stock_map[str(s).strip()]

    if (stock_days <= 0).any():
        bad = [sector_names[i] for i in np.where(stock_days <= 0)[0]]
        raise ValueError(f"以下行业没有匹配到库存天数: {bad}")

    stock_df = pd.DataFrame({
        "sector_index": np.arange(len(sector_names)),
        "sector": sector_names,
        "stock_days_for_model": stock_days,
    })

    return stock_days, stock_df


def get_sector_index_for_activity(activity_index):
    """
    sector-major:
        index = sector * RR + region
    """
    return activity_index // RR


def get_region_index_for_activity(activity_index):
    """
    sector-major:
        index = sector * RR + region
    """
    return activity_index % RR


def aggregate_inputs_by_supplier_sector(mat):
    """
    把 285×285 的供应矩阵按供应方行业聚合为 19×285。

    输入 mat:
        行 = 供应活动 i
        列 = 使用活动 j

    输出 out:
        行 = 供应行业类别 s
        列 = 使用活动 j
    """

    out = np.zeros((NN, RRNN), dtype=float)

    for i in range(RRNN):
        s_idx = get_sector_index_for_activity(i)
        out[s_idx, :] += mat[i, :]

    return out


def build_mat_ezz(zz_daily, zz_cat):
    """
    mat_ezz[i, j] =
        原始供应活动 i 在其供应行业类别中的份额。

    用于把库存缺口从 19×285 分配回 285×285 原始供应商。

    对应基准代码：
        mat_ezz[i, j] = zz_s[i, j] / zz_s_c[mat_index[i, j], j]
    """

    mat_ezz = np.zeros((RRNN, RRNN), dtype=float)

    for i in range(RRNN):
        s_idx = get_sector_index_for_activity(i)
        denom = zz_cat[s_idx, :]

        mat_ezz[i, :] = np.where(
            denom > EPS,
            zz_daily[i, :] / (denom + EPS),
            0.0
        )

    return mat_ezz


def disaggregate_stock_gap_to_original_suppliers(stock_gap_cat, mat_ezz):
    """
    把 19×285 的库存缺口按照历史供应区域比例分配回 285×285。

    stock_gap_cat:
        行 = 供应行业类别
        列 = 使用活动

    out:
        行 = 原始供应活动
        列 = 使用活动
    """

    out = np.zeros((RRNN, RRNN), dtype=float)

    for i in range(RRNN):
        s_idx = get_sector_index_for_activity(i)
        out[i, :] = stock_gap_cat[s_idx, :] * mat_ezz[i, :]

    return out


def build_key_input_mask(zz_cat):
    """
    构造关键投入类别矩阵 key_mask。

    key_mask[s, j] = True 表示：
        对使用活动 j 来说，供应行业类别 s 是关键投入。

    与基准代码 mat_key 对应。
    """

    key_mask = np.zeros((NN, RRNN), dtype=bool)

    if KEY_MODE == "all_positive":
        key_mask = zz_cat > EPS
        return key_mask

    if KEY_MODE != "share":
        raise ValueError(f"未知 KEY_MODE: {KEY_MODE}")

    input_sum = zz_cat.sum(axis=0)

    for j in range(RRNN):
        if input_sum[j] <= EPS:
            continue

        shares = zz_cat[:, j] / (input_sum[j] + EPS)

        selected = np.where(shares >= KEY_SHARE_CUTOFF)[0].tolist()

        # 至少选 MIN_KEY_CATEGORIES 个
        if len(selected) < MIN_KEY_CATEGORIES:
            top_idx = np.argsort(shares)[::-1]
            selected = top_idx[:MIN_KEY_CATEGORIES].tolist()

        # 最多选 MAX_KEY_CATEGORIES 个
        selected = sorted(
            selected,
            key=lambda x: shares[x],
            reverse=True
        )[:MAX_KEY_CATEGORIES]

        for s_idx in selected:
            if shares[s_idx] > 0:
                key_mask[s_idx, j] = True

    return key_mask


# =====================================================
# 4. 动态 MRIO 主模拟
# =====================================================

def dynamic_mrio_simulation(
    zz_s,
    ff_s,
    va_s,
    xx_s,
    VA_TT,
    stock_days_by_sector,
    region_names,
    sector_names,
    time_index,
):
    """
    动态 MRIO 主循环。

    每天：
      1. 读取当天 VA_TT；
      2. 计算每个部门订单；
      3. 计算库存约束、能力约束、订单约束；
      4. 取最小值作为实际产出；
      5. 按订单比例分配产出；
      6. 更新库存；
      7. 库存缺口形成下一期订单。
    """

    if zz_s.shape != (RRNN, RRNN):
        raise ValueError(f"zz_s shape 错误: {zz_s.shape}, 应为 {(RRNN, RRNN)}")

    if ff_s.shape != (RRNN, RR):
        raise ValueError(f"ff_s shape 错误: {ff_s.shape}, 应为 {(RRNN, RR)}")

    if va_s.shape != (1, RRNN):
        raise ValueError(f"va_s shape 错误: {va_s.shape}, 应为 {(1, RRNN)}")

    if xx_s.shape != (1, RRNN):
        raise ValueError(f"xx_s shape 错误: {xx_s.shape}, 应为 {(1, RRNN)}")

    if VA_TT.shape != (NN, RR, TT):
        raise ValueError(f"VA_TT shape 错误: {VA_TT.shape}, 应为 {(NN, RR, TT)}")

    # =================================================
    # 4.1 年度 MRIO 转日尺度
    # =================================================

    zz_daily = np.nan_to_num(
        zz_s.astype(float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ) / DAYS_PER_YEAR

    ff_daily = np.nan_to_num(
        ff_s.astype(float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ) / DAYS_PER_YEAR

    va_daily = np.nan_to_num(
        va_s.astype(float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ) / DAYS_PER_YEAR

    xx_daily = np.nan_to_num(
        xx_s.astype(float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ) / DAYS_PER_YEAR

    # =================================================
    # 4.2 库存类别系统
    # =================================================
    # zz_cat:
    #   19 × 285
    #   行 = 供应行业类别
    #   列 = 使用活动
    # =================================================

    zz_cat = aggregate_inputs_by_supplier_sector(zz_daily)

    # 单位产出投入系数：
    # 每生产 1 单位 j，需要多少供应行业类别 s 的投入
    mat_ezc = zz_cat / (xx_daily + EPS)

    # 原始供应活动在其供应行业类别中的份额
    mat_ezz = build_mat_ezz(
        zz_daily=zz_daily,
        zz_cat=zz_cat
    )

    # 关键投入类别
    key_mask = build_key_input_mask(zz_cat)
    key_count = key_mask.sum(axis=0)

    print("\n" + "=" * 80)
    print("库存系统检查")
    print("=" * 80)
    print("zz_cat:", zz_cat.shape)
    print("mat_ezc:", mat_ezc.shape)
    print("mat_ezz:", mat_ezz.shape)
    print("key_mask:", key_mask.shape)
    print("KEY_MODE:", KEY_MODE)
    print("KEY_SHARE_CUTOFF:", KEY_SHARE_CUTOFF)
    print("MIN_KEY_CATEGORIES:", MIN_KEY_CATEGORIES)
    print("MAX_KEY_CATEGORIES:", MAX_KEY_CATEGORIES)
    print(
        "每个活动关键投入类别数量 min/mean/max:",
        int(key_count.min()),
        float(key_count.mean()),
        int(key_count.max())
    )

    # 每个活动 j 的库存天数由其所属行业决定
    stock_days_activity = np.zeros(RRNN, dtype=float)

    for j in range(RRNN):
        s_idx = get_sector_index_for_activity(j)
        stock_days_activity[j] = stock_days_by_sector[s_idx]

    # 初始库存：
    #   库存 = 正常每日投入需求 × 库存天数
    mat_stock = zz_cat * stock_days_activity.reshape(1, -1)

    # 目标库存：
    #   当前初始库存 + 1 天正常投入需求
    mat_stock_obj = mat_stock + zz_cat

    # 初始订单：
    #   中间投入订单 + 最终需求订单
    #   shape = 285 × (285+15)
    mat_order = np.hstack([zz_daily, ff_daily])

    # 输出时间序列
    xx_TT = np.zeros((TT, RRNN), dtype=float)

    # 诊断指标
    binding_inventory_count = np.zeros(TT, dtype=int)
    binding_capacity_count = np.zeros(TT, dtype=int)
    binding_order_count = np.zeros(TT, dtype=int)

    min_inventory_constraint_TT = np.zeros(TT, dtype=float)
    mean_inventory_constraint_TT = np.zeros(TT, dtype=float)

    total_order_TT = np.zeros(TT, dtype=float)
    total_stock_TT = np.zeros(TT, dtype=float)

    # =================================================
    # 4.3 动态循环
    # =================================================

    for t in range(TT):
        if t == 0 or (t + 1) % 25 == 0 or (t + 1) == TT:
            print(f"Running period {t + 1}/{TT}")

        # VA_TT: sector × region × time
        # reshape 后天然是 sector-major:
        #   sector0-region0..14, sector1-region0..14, ...
        va_t = VA_TT[:, :, t].reshape(1, -1)

        # 每个供应活动收到的总订单
        mat_order_sum = mat_order.sum(axis=1)

        xx_t = np.zeros((1, RRNN), dtype=float)

        inventory_constraints = np.full(RRNN, np.inf, dtype=float)

        # ---------------------------------------------
        # 对 285 个部门逐个计算实际产出
        # ---------------------------------------------
        for j in range(RRNN):
            # 库存约束
            key_idx = np.where(key_mask[:, j])[0]

            if len(key_idx) > 0:
                coeff = mat_ezc[key_idx, j]
                stock = mat_stock[key_idx, j]

                valid = coeff > EPS

                if valid.any():
                    inventory_constraints[j] = np.min(
                        stock[valid] / (coeff[valid] + EPS)
                    )

            inventory_constraint = inventory_constraints[j]

            # 生产能力约束
            capacity_constraint = va_t[0, j] * xx_daily[0, j]

            # 订单约束
            order_constraint = mat_order_sum[j]

            constraints = np.array([
                inventory_constraint,
                capacity_constraint,
                order_constraint,
            ])

            q = float(np.min(constraints))

            if not np.isfinite(q):
                q = 0.0

            q = max(q, 0.0)

            xx_t[0, j] = q

            binding = int(np.argmin(constraints))

            if binding == 0:
                binding_inventory_count[t] += 1
            elif binding == 1:
                binding_capacity_count[t] += 1
            else:
                binding_order_count[t] += 1

        finite_inventory = inventory_constraints[
            np.isfinite(inventory_constraints)
        ]

        if len(finite_inventory) > 0:
            min_inventory_constraint_TT[t] = float(finite_inventory.min())
            mean_inventory_constraint_TT[t] = float(finite_inventory.mean())
        else:
            min_inventory_constraint_TT[t] = np.nan
            mean_inventory_constraint_TT[t] = np.nan

        total_order_TT[t] = float(mat_order_sum.sum())
        total_stock_TT[t] = float(mat_stock.sum())

        # ---------------------------------------------
        # 按订单比例分配产出
        # ---------------------------------------------

        with np.errstate(divide="ignore", invalid="ignore"):
            order_share = mat_order / (mat_order_sum[:, None] + EPS)

        order_share = np.nan_to_num(
            order_share,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # xx_t_dis:
        #   行 = 供应活动
        #   列 = 需求方，包括 285 个中间需求 + 15 个最终需求
        xx_t_dis = xx_t.T * order_share

        # ---------------------------------------------
        # 库存消耗
        # ---------------------------------------------

        mat_stock_use = mat_ezc * xx_t

        # ---------------------------------------------
        # 中间交付补充库存
        # ---------------------------------------------

        intermediate_delivery = xx_t_dis[:, :RRNN]

        mat_stock_add = aggregate_inputs_by_supplier_sector(
            intermediate_delivery
        )

        # ---------------------------------------------
        # 更新库存
        # ---------------------------------------------

        mat_stock = mat_stock - mat_stock_use + mat_stock_add
        mat_stock = np.maximum(mat_stock, 0.0)

        # ---------------------------------------------
        # 库存缺口形成下一期中间品订单
        # ---------------------------------------------

        mat_stock_gap = mat_stock_obj - mat_stock
        mat_stock_gap = np.maximum(mat_stock_gap, 0.0)

        mat_order[:, :RRNN] = disaggregate_stock_gap_to_original_suppliers(
            stock_gap_cat=mat_stock_gap,
            mat_ezz=mat_ezz
        )

        # 最终需求保持正常日尺度
        mat_order[:, RRNN:RRNN + RR] = ff_daily

        # 保存当期产出
        xx_TT[t, :] = xx_t

    # =================================================
    # 4.4 诊断表
    # =================================================

    diag_df = pd.DataFrame({
        "t": np.arange(1, TT + 1),
        "date": (
            time_index["date"].values
            if "date" in time_index.columns
            else np.arange(1, TT + 1)
        ),
        "binding_inventory_count": binding_inventory_count,
        "binding_capacity_count": binding_capacity_count,
        "binding_order_count": binding_order_count,
        "min_inventory_constraint": min_inventory_constraint_TT,
        "mean_inventory_constraint": mean_inventory_constraint_TT,
        "total_order": total_order_TT,
        "total_stock": total_stock_TT,
    })

    aux = {
        "zz_daily": zz_daily,
        "ff_daily": ff_daily,
        "va_daily": va_daily,
        "xx_daily": xx_daily,
        "zz_cat": zz_cat,
        "mat_ezc": mat_ezc,
        "stock_days_activity": stock_days_activity,
        "key_count": key_count,
    }

    return xx_TT, diag_df, aux


# =====================================================
# 5. 结果汇总与保存
# =====================================================

def summarize_and_save(
    intensity,
    base_path,
    output_path,
    xx_TT,
    diag_df,
    aux,
    region_names,
    sector_names,
    time_index,
    stock_df,
):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    xx_daily_base = aux["xx_daily"]

    base_total_output_daily = float(xx_daily_base.sum())
    base_total_output_annual = float(base_total_output_daily * DAYS_PER_YEAR)

    total_output_daily = xx_TT.sum(axis=1)

    daily_loss = base_total_output_daily - total_output_daily
    daily_loss = np.maximum(daily_loss, 0.0)

    daily_loss_percent = (
        daily_loss /
        (base_total_output_daily + EPS) *
        100.0
    )

    result_df = time_index.copy()

    if len(result_df) != TT:
        raise ValueError(f"time_index 行数不是 {TT}: {len(result_df)}")

    result_df["t"] = np.arange(1, TT + 1)

    impact_mask = get_impact_mask(result_df)

    if "is_impact_day" not in result_df.columns:
        result_df["is_impact_day"] = impact_mask.astype(int)

    if "period_type" not in result_df.columns:
        result_df["period_type"] = np.where(
            impact_mask,
            "impact",
            "normal"
        )

    result_df["total_output_daily_NZD_million"] = total_output_daily
    result_df["base_total_output_daily_NZD_million"] = base_total_output_daily
    result_df["daily_loss_NZD_million"] = daily_loss
    result_df["daily_loss_percent"] = daily_loss_percent

    result_df["total_output_annualized_NZD_million"] = (
        result_df["total_output_daily_NZD_million"] *
        DAYS_PER_YEAR
    )

    result_df["base_total_output_annualized_NZD_million"] = (
        base_total_output_annual
    )

    annual_total_output = float(total_output_daily.sum())

    annual_loss = base_total_output_annual - annual_total_output
    annual_loss = max(annual_loss, 0.0)

    annual_loss_percent = (
        annual_loss /
        (base_total_output_annual + EPS) *
        100.0
    )

    impact_df = result_df[
        result_df["is_impact_day"].astype(int) == 1
    ].copy()

    if len(impact_df) > 0:
        impact_days_count = int(len(impact_df))
        impact_total_loss = float(
            impact_df["daily_loss_NZD_million"].sum()
        )
        impact_average_loss_percent = float(
            impact_df["daily_loss_percent"].mean()
        )
    else:
        impact_days_count = 0
        impact_total_loss = np.nan
        impact_average_loss_percent = np.nan

    peak_idx = int(result_df["daily_loss_percent"].idxmax())
    peak_row = result_df.loc[peak_idx]

    # =================================================
    # reshape:
    #   xx_TT: TT × 285
    #   sector-major -> TT × NN × RR
    # =================================================

    xx_base_daily_3d = xx_daily_base.reshape(NN, RR)
    xx_TT_3d = xx_TT.reshape(TT, NN, RR)

    # =================================================
    # 行业输出与损失
    # =================================================

    sector_base_daily = xx_base_daily_3d.sum(axis=1)
    sector_output_daily = xx_TT_3d.sum(axis=2)

    sector_loss_daily = (
        sector_base_daily.reshape(1, NN) -
        sector_output_daily
    )

    sector_loss_daily = np.maximum(sector_loss_daily, 0.0)

    sector_base_annual = sector_base_daily * DAYS_PER_YEAR
    sector_actual_annual = sector_output_daily.sum(axis=0)
    sector_loss_annual = sector_loss_daily.sum(axis=0)

    sector_loss_df = pd.DataFrame({
        "sector_index": np.arange(NN),
        "sector": sector_names,
        "base_output_annual_NZD_million": sector_base_annual,
        "actual_output_annual_NZD_million": sector_actual_annual,
        "loss_NZD_million": sector_loss_annual,
        "loss_percent": (
            sector_loss_annual /
            (sector_base_annual + EPS) *
            100.0
        ),
        "max_daily_loss_NZD_million": sector_loss_daily.max(axis=0),
        "max_daily_loss_percent": (
            sector_loss_daily /
            (sector_base_daily.reshape(1, NN) + EPS) *
            100.0
        ).max(axis=0),
    })

    sector_loss_df = sector_loss_df.sort_values(
        "loss_NZD_million",
        ascending=False
    )

    # =================================================
    # 区域输出与损失
    # =================================================

    region_base_daily = xx_base_daily_3d.sum(axis=0)
    region_output_daily = xx_TT_3d.sum(axis=1)

    region_loss_daily = (
        region_base_daily.reshape(1, RR) -
        region_output_daily
    )

    region_loss_daily = np.maximum(region_loss_daily, 0.0)

    region_base_annual = region_base_daily * DAYS_PER_YEAR
    region_actual_annual = region_output_daily.sum(axis=0)
    region_loss_annual = region_loss_daily.sum(axis=0)

    region_loss_df = pd.DataFrame({
        "region_index": np.arange(RR),
        "region": region_names,
        "base_output_annual_NZD_million": region_base_annual,
        "actual_output_annual_NZD_million": region_actual_annual,
        "loss_NZD_million": region_loss_annual,
        "loss_percent": (
            region_loss_annual /
            (region_base_annual + EPS) *
            100.0
        ),
        "max_daily_loss_NZD_million": region_loss_daily.max(axis=0),
        "max_daily_loss_percent": (
            region_loss_daily /
            (region_base_daily.reshape(1, RR) + EPS) *
            100.0
        ).max(axis=0),
    })

    region_loss_df = region_loss_df.sort_values(
        "loss_NZD_million",
        ascending=False
    )

    # =================================================
    # 逐日行业/区域输出表
    # =================================================

    sector_output_tt_df = pd.DataFrame(
        sector_output_daily,
        columns=sector_names
    )
    sector_output_tt_df.insert(0, "t", np.arange(1, TT + 1))

    sector_loss_tt_df = pd.DataFrame(
        sector_loss_daily,
        columns=sector_names
    )
    sector_loss_tt_df.insert(0, "t", np.arange(1, TT + 1))

    region_output_tt_df = pd.DataFrame(
        region_output_daily,
        columns=region_names
    )
    region_output_tt_df.insert(0, "t", np.arange(1, TT + 1))

    region_loss_tt_df = pd.DataFrame(
        region_loss_daily,
        columns=region_names
    )
    region_loss_tt_df.insert(0, "t", np.arange(1, TT + 1))

    if "date" in result_df.columns:
        sector_output_tt_df.insert(1, "date", result_df["date"].values)
        sector_loss_tt_df.insert(1, "date", result_df["date"].values)
        region_output_tt_df.insert(1, "date", result_df["date"].values)
        region_loss_tt_df.insert(1, "date", result_df["date"].values)

    # =================================================
    # 事件汇总
    # =================================================

    event_rows = []

    if "event" in result_df.columns:
        for event_name in ["201903", "201908", "201911", "201912"]:
            mask_event = result_df["event"].astype(str).str.contains(
                event_name,
                na=False
            )

            if mask_event.sum() == 0:
                continue

            event_rows.append({
                "H_name": H_NAME,
                "intensity": intensity,
                "event": event_name,
                "days": int(mask_event.sum()),
                "start_date": (
                    str(result_df.loc[mask_event, "date"].min())
                    if "date" in result_df.columns
                    else ""
                ),
                "end_date": (
                    str(result_df.loc[mask_event, "date"].max())
                    if "date" in result_df.columns
                    else ""
                ),
                "event_loss_NZD_million": float(
                    result_df.loc[
                        mask_event,
                        "daily_loss_NZD_million"
                    ].sum()
                ),
                "event_average_loss_percent": float(
                    result_df.loc[
                        mask_event,
                        "daily_loss_percent"
                    ].mean()
                ),
                "event_max_daily_loss_percent": float(
                    result_df.loc[
                        mask_event,
                        "daily_loss_percent"
                    ].max()
                ),
            })

    event_summary_df = pd.DataFrame(event_rows)

    # =================================================
    # 总汇总
    # =================================================

    summary = {
        "H_name": H_NAME,
        "intensity": intensity,
        "RUN_TAG": RUN_TAG,
        "base_path": str(base_path),
        "output_path": str(output_path),
        "STOCK_DAYS_CSV": str(STOCK_DAYS_CSV),

        "base_total_output_NZD_million_daily": base_total_output_daily,
        "base_total_output_NZD_million_annual": base_total_output_annual,
        "annual_total_output_NZD_million": annual_total_output,

        "annual_loss_NZD_million": annual_loss,
        "annual_loss_percent": annual_loss_percent,

        "impact_days_count": impact_days_count,
        "impact_total_loss_NZD_million": impact_total_loss,
        "impact_average_loss_percent": impact_average_loss_percent,

        "peak_loss_date": (
            str(peak_row["date"])
            if "date" in result_df.columns
            else ""
        ),
        "peak_loss_period_type": (
            str(peak_row["period_type"])
            if "period_type" in result_df.columns
            else ""
        ),
        "max_daily_loss_percent": float(peak_row["daily_loss_percent"]),
        "max_daily_loss_NZD_million": float(
            peak_row["daily_loss_NZD_million"]
        ),

        "sector_stock_days_mean": float(
            stock_df["stock_days_for_model"].mean()
        ),
        "sector_stock_days_min": float(
            stock_df["stock_days_for_model"].min()
        ),
        "sector_stock_days_max": float(
            stock_df["stock_days_for_model"].max()
        ),

        "KEY_MODE": KEY_MODE,
        "KEY_SHARE_CUTOFF": KEY_SHARE_CUTOFF,
        "MIN_KEY_CATEGORIES": MIN_KEY_CATEGORIES,
        "MAX_KEY_CATEGORIES": MAX_KEY_CATEGORIES,

        "binding_inventory_count_sum": int(
            diag_df["binding_inventory_count"].sum()
        ),
        "binding_capacity_count_sum": int(
            diag_df["binding_capacity_count"].sum()
        ),
        "binding_order_count_sum": int(
            diag_df["binding_order_count"].sum()
        ),

        "mrio_order": "sector-major, index = sector * RR + region",
    }

    summary_df = pd.DataFrame([summary])

    # =================================================
    # 保存
    # =================================================

    np.save(output_path / "xx_TT.npy", xx_TT)
    np.save(output_path / "xx_TT_daily.npy", xx_TT)
    np.save(output_path / "xx_TT_annualized.npy", xx_TT * DAYS_PER_YEAR)

    result_df.to_csv(
        output_path / "dynamic_mrio_daily_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    result_df.to_csv(
        output_path / "total_output_tt.csv",
        index=False,
        encoding="utf-8-sig"
    )

    diag_df.to_csv(
        output_path / "dynamic_mrio_diagnostics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary_df.to_csv(
        output_path / "dynamic_mrio_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    event_summary_df.to_csv(
        output_path / "event_loss_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    sector_output_tt_df.to_csv(
        output_path / "sector_output_tt.csv",
        index=False,
        encoding="utf-8-sig"
    )

    sector_loss_tt_df.to_csv(
        output_path / "sector_loss_tt.csv",
        index=False,
        encoding="utf-8-sig"
    )

    region_output_tt_df.to_csv(
        output_path / "region_output_tt.csv",
        index=False,
        encoding="utf-8-sig"
    )

    region_loss_tt_df.to_csv(
        output_path / "region_loss_tt.csv",
        index=False,
        encoding="utf-8-sig"
    )

    sector_loss_df.to_csv(
        output_path / "sector_loss_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    region_loss_df.to_csv(
        output_path / "region_loss_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 兼容旧命名
    region_loss_df.to_csv(
        output_path / "regional_loss_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    sector_loss_df.to_csv(
        output_path / "max_sector_loss.csv",
        index=False,
        encoding="utf-8-sig"
    )

    region_loss_df.to_csv(
        output_path / "max_region_loss.csv",
        index=False,
        encoding="utf-8-sig"
    )

    stock_df.to_csv(
        output_path / "sector_stock_days_used.csv",
        index=False,
        encoding="utf-8-sig"
    )

    config_df = pd.DataFrame([{
        "H_name": H_NAME,
        "intensity": intensity,
        "RUN_TAG": RUN_TAG,
        "INPUT_ROOT": str(INPUT_ROOT),
        "base_path": str(base_path),
        "STOCK_DAYS_CSV": str(STOCK_DAYS_CSV),
        "DAYS_PER_YEAR": DAYS_PER_YEAR,
        "TT": TT,
        "RR": RR,
        "NN": NN,
        "RRNN": RRNN,
        "mrio_order": "sector-major, index = sector * RR + region",
        "input_folder_required": "MRIO模型_sector_major_checked",
        "inventory_category": (
            "supplier sectors aggregated to 19 input categories"
        ),
        "inventory_definition": (
            "initial stock = daily intermediate input requirement "
            "* AES-derived sector stock days"
        ),
        "dynamic_rule": (
            "actual output = min(inventory constraint, "
            "capacity constraint, order constraint)"
        ),
        "KEY_MODE": KEY_MODE,
        "KEY_SHARE_CUTOFF": KEY_SHARE_CUTOFF,
        "MIN_KEY_CATEGORIES": MIN_KEY_CATEGORIES,
        "MAX_KEY_CATEGORIES": MAX_KEY_CATEGORIES,
    }])

    config_df.to_csv(
        output_path / "dynamic_mrio_config.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # =================================================
    # 绘图
    # =================================================

    if PLOT_FIGURES:
        plt.figure(figsize=(12, 4))

        plt.plot(
            result_df["t"],
            result_df["total_output_daily_NZD_million"],
            linewidth=1.6,
            label="Dynamic MRIO output"
        )

        plt.axhline(
            base_total_output_daily,
            linestyle="--",
            linewidth=1.0,
            label="Baseline daily output"
        )

        impact_days = result_df[
            result_df["is_impact_day"].astype(int) == 1
        ]

        if len(impact_days) > 0:
            plt.scatter(
                impact_days["t"],
                impact_days["total_output_daily_NZD_million"],
                s=10,
                label="Impact days"
            )

        plt.xlabel("Day of year")
        plt.ylabel("Daily total output (NZD million)")
        plt.title(f"Dynamic MRIO total output - {H_NAME} {intensity}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            output_path / "total_output_timeseries.png",
            dpi=220
        )

        plt.close()

        plt.figure(figsize=(12, 4))

        plt.plot(
            result_df["t"],
            result_df["daily_loss_percent"],
            linewidth=1.6
        )

        plt.xlabel("Day of year")
        plt.ylabel("Daily output loss (%)")
        plt.title(f"Daily output loss rate - {H_NAME} {intensity}")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            output_path / "daily_loss_rate_365.png",
            dpi=220
        )

        plt.close()

    return summary_df, sector_loss_df, region_loss_df


# =====================================================
# 6. 单个强度运行
# =====================================================

def run_one_intensity(intensity):
    print("\n" + "=" * 80)
    print(f"Running clean dynamic MRIO: {H_NAME} {intensity}")
    print("=" * 80)

    base_path = (
        INPUT_ROOT /
        intensity /
        "MRIO模型_sector_major_checked"
    )

    output_path = base_path / RUN_TAG

    if not base_path.exists():
        raise FileNotFoundError(
            f"找不到 sector-major checked 输入目录: {base_path}\n"
            "请先运行 S9-1 统一 MRIO 顺序。"
        )

    print("base_path:", base_path)
    print("output_path:", output_path)
    print("STOCK_DAYS_CSV:", STOCK_DAYS_CSV)
    print("RUN_TAG:", RUN_TAG)

    (
        zz_s,
        ff_s,
        va_s,
        xx_s,
        VA_TT,
        region_names,
        sector_names,
        time_index,
    ) = load_mrio_inputs(base_path)

    stock_days, stock_df = load_sector_stock_days(
        STOCK_DAYS_CSV,
        sector_names
    )

    print("\n" + "=" * 80)
    print("输入检查")
    print("=" * 80)
    print("zz_s:", zz_s.shape)
    print("ff_s:", ff_s.shape)
    print("va_s:", va_s.shape)
    print("xx_s:", xx_s.shape)
    print("VA_TT:", VA_TT.shape)
    print("region_names:", len(region_names), region_names)
    print("sector_names:", len(sector_names), sector_names)
    print("time_index:", time_index.shape)

    print("\n库存天数:")
    print(stock_df.to_string(index=False))

    # 检查交通行业冲击是否存在
    transport_idx = None

    for i, s in enumerate(sector_names):
        s_low = str(s).lower()
        if (
            "transport" in s_low
            or "postal" in s_low
            or "warehousing" in s_low
        ):
            transport_idx = i
            break

    if transport_idx is not None:
        transport_slice = VA_TT[transport_idx, :, :]
        print("\n交通行业 VA_TT 检查:")
        print("transport_idx:", transport_idx)
        print("transport sector:", sector_names[transport_idx])
        print("min/mean/max:",
              float(np.nanmin(transport_slice)),
              float(np.nanmean(transport_slice)),
              float(np.nanmax(transport_slice)))
    else:
        print("[WARN] 没有找到交通行业。")

    xx_TT, diag_df, aux = dynamic_mrio_simulation(
        zz_s=zz_s,
        ff_s=ff_s,
        va_s=va_s,
        xx_s=xx_s,
        VA_TT=VA_TT,
        stock_days_by_sector=stock_days,
        region_names=region_names,
        sector_names=sector_names,
        time_index=time_index,
    )

    summary_df, sector_loss_df, region_loss_df = summarize_and_save(
        intensity=intensity,
        base_path=base_path,
        output_path=output_path,
        xx_TT=xx_TT,
        diag_df=diag_df,
        aux=aux,
        region_names=region_names,
        sector_names=sector_names,
        time_index=time_index,
        stock_df=stock_df,
    )

    print("\n[DONE]", intensity)
    print(summary_df.to_string(index=False))

    print("\n行业损失 Top 10:")
    print(sector_loss_df.head(10).to_string(index=False))

    print("\n区域损失 Top 10:")
    print(region_loss_df.head(10).to_string(index=False))

    print("\n输出目录:")
    print(output_path)

    return summary_df


# =====================================================
# 7. main
# =====================================================

def main():
    print("=" * 80)
    print("S9-2 clean: 动态 MRIO + AES 库存 + sector-major checked")
    print("=" * 80)

    print("SCENARIO_ROOT:", SCENARIO_ROOT)
    print("INPUT_ROOT:", INPUT_ROOT)
    print("STOCK_DAYS_CSV:", STOCK_DAYS_CSV)
    print("RUN_TAG:", RUN_TAG)
    print("H_NAME:", H_NAME)
    print("INTENSITY_SCENARIOS:", INTENSITY_SCENARIOS)
    print("KEY_MODE:", KEY_MODE)
    print("KEY_SHARE_CUTOFF:", KEY_SHARE_CUTOFF)
    print("MIN_KEY_CATEGORIES:", MIN_KEY_CATEGORIES)
    print("MAX_KEY_CATEGORIES:", MAX_KEY_CATEGORIES)

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"找不到 INPUT_ROOT: {INPUT_ROOT}")

    if not STOCK_DAYS_CSV.exists():
        raise FileNotFoundError(f"找不到 STOCK_DAYS_CSV: {STOCK_DAYS_CSV}")

    all_summary = []

    for intensity in INTENSITY_SCENARIOS:
        summary_df = run_one_intensity(intensity)
        all_summary.append(summary_df)

    all_summary_df = pd.concat(
        all_summary,
        ignore_index=True
    )

    all_summary_csv = (
        INPUT_ROOT /
        f"all_intensity_dynamic_mrio_summary_{H_NAME}_{RUN_TAG}.csv"
    )

    all_summary_df.to_csv(
        all_summary_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 80)
    print("✅ S9-2 clean 动态 MRIO 完成")
    print("=" * 80)

    print("总汇总:", all_summary_csv)

    print("\n总汇总:")
    print(all_summary_df.to_string(index=False))

    print("\n重点检查：")
    print("1. annual_loss_percent 应整体 I1 < I2 < I3。")
    print("2. max_daily_loss_percent 应整体 I1 < I2 < I3。")
    print("3. peak_loss_date 应主要在冲击窗口或冲击后恢复期。")
    print("4. sector_loss_summary.csv 中 Transport 应有明显直接损失。")
    print("5. dynamic_mrio_diagnostics.csv 可检查 inventory/capacity/order 哪类约束主导。")
    print("6. 若结果极端大或极端小，下一步检查 KEY_MODE 和库存天数敏感性。")

    print("\n每个强度输出目录：")
    for intensity in INTENSITY_SCENARIOS:
        out_dir = (
            INPUT_ROOT /
            intensity /
            "MRIO模型_sector_major_checked" /
            RUN_TAG
        )
        print(intensity, "=>", out_dir)


if __name__ == "__main__":
    main()