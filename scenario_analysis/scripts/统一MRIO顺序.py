# -*- coding: utf-8 -*-
from pathlib import Path
import os
import shutil
import numpy as np
import pandas as pd


# =====================================================
# S9-1: 检查并统一 MRIO 285 部门排列顺序
# =====================================================
# 目的：
#   1. 判断原始 MRIO 是 region-major 还是 sector-major；
#   2. 统一转换为 S9 使用的 sector-major 顺序：
#        index = sector * RR + region
#   3. 为 I1/I2/I3 生成新的 S9 输入目录：
#        MRIO模型_sector_major_checked
#
# 背景：
#   VA_TT.shape = (NN, RR, TT)
#   VA_TT[:, :, t].reshape(1, -1)
#   默认展开顺序是 sector-major：
#       sector0-region0, sector0-region1, ...
#
#   所以 zz_s / ff_s / va_s / xx_s 也必须是 sector-major。
# =====================================================


# =====================================================
# 1. 参数
# =====================================================

BASE_DIR = Path(os.environ.get("SCENARIO_DATA_ROOT", "data"))

SCENARIO_ROOT = (
    BASE_DIR /
    '第五阶段情景分析new/I_H0'
)

RR = 15
NN = 19
RRNN = RR * NN

INTENSITY_SCENARIOS = ["I1", "I2", "I3"]

# Step8 输出候选目录
STEP8_ROOT_CANDIDATES = [
    SCENARIO_ROOT / r"Intensity_step8_annual_365_VA_TT_H0_eta3.5",
    SCENARIO_ROOT / 'Intensity_step8_annual_365_VA_TT_H0_eta3.5',
    SCENARIO_ROOT / r"Intensity_step8_annual_365_VA_TT_eta3.5",
]

# 原始 MRIO 表候选路径
MRIO_PATH_CANDIDATES = [
    BASE_DIR / '第三阶段/MRIO_15regions_19sectors_FULL.csv',
    BASE_DIR / r"MRIO_15regions_19sectors_FULL.csv",
    BASE_DIR / '第三阶段迁移/MRIO_15regions_19sectors_FULL.csv',
]


# =====================================================
# 2. 工具函数
# =====================================================

def resolve_existing_path(candidates, name):
    for p in candidates:
        if p.exists():
            print(f"[FOUND] {name}:", p)
            return p

    msg = f"找不到 {name}，检查以下候选路径：\n"
    msg += "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(msg)


def read_name_csv(path):
    df = pd.read_csv(path)

    if df.shape[1] == 1:
        return df.iloc[:, 0].astype(str).tolist()

    for c in ["region", "region_name", "sector", "sector_name", "name"]:
        if c in df.columns:
            return df[c].astype(str).tolist()

    return df.iloc[:, -1].astype(str).tolist()


def read_raw_mrio_activity_order(mrio_path):
    """
    从原始 MRIO CSV 的前两行读取 285 个部门的 region / sector 标签。
    根据 S8 读取逻辑：
      header_row_1 = region
      header_row_2 = sector
      Z columns start = 2
    """

    df = pd.read_csv(mrio_path, header=None)

    header_region = df.iloc[0, 2:2 + RRNN].astype(str).str.strip().tolist()
    header_sector = df.iloc[1, 2:2 + RRNN].astype(str).str.strip().tolist()

    activity_order = pd.DataFrame({
        "raw_index": np.arange(RRNN),
        "raw_region": header_region,
        "raw_sector": header_sector,
    })

    return activity_order


def infer_order(activity_order):
    """
    判断原始 MRIO 的排列方式：
    - region-major:
        前 NN=19 个位置区域相同，行业变化
    - sector-major:
        前 RR=15 个位置行业相同，区域变化
    """

    first_NN_regions = activity_order.loc[:NN - 1, "raw_region"].tolist()
    first_NN_sectors = activity_order.loc[:NN - 1, "raw_sector"].tolist()

    first_RR_regions = activity_order.loc[:RR - 1, "raw_region"].tolist()
    first_RR_sectors = activity_order.loc[:RR - 1, "raw_sector"].tolist()

    is_region_major = (
        len(set(first_NN_regions)) == 1
        and len(set(first_NN_sectors)) == NN
    )

    is_sector_major = (
        len(set(first_RR_sectors)) == 1
        and len(set(first_RR_regions)) == RR
    )

    if is_region_major and not is_sector_major:
        return "region-major"

    if is_sector_major and not is_region_major:
        return "sector-major"

    if is_region_major and is_sector_major:
        return "ambiguous"

    return "unknown"


def build_sector_major_raw_indices(order_type):
    """
    返回一个长度 285 的 raw index 列表。
    列表顺序是 sector-major。
    也就是：
        new_index = sector * RR + region
        raw_index = ?
    """

    if order_type == "sector-major":
        return np.arange(RRNN, dtype=int)

    if order_type == "region-major":
        idx = []

        for s in range(NN):
            for r in range(RR):
                raw_i = r * NN + s
                idx.append(raw_i)

        return np.array(idx, dtype=int)

    raise ValueError(f"无法为 order_type={order_type} 构造重排索引。")


def check_transport_positions(sector_names, region_names):
    """
    打印交通行业在 sector-major 和 region-major 下的索引位置。
    """

    transport_candidates = []

    for i, s in enumerate(sector_names):
        s_low = str(s).lower()
        if (
            "transport" in s_low
            or "postal" in s_low
            or "warehousing" in s_low
        ):
            transport_candidates.append((i, s))

    print("\n交通行业候选:")
    print(transport_candidates)

    if len(transport_candidates) == 0:
        print("[WARN] 没找到交通行业。")
        return

    transport_idx = transport_candidates[0][0]
    transport_name = transport_candidates[0][1]

    print("使用 transport_idx:", transport_idx)
    print("transport_name:", transport_name)

    print("\n如果是 sector-major，交通行业 15 个区域的 index 是:")
    sector_major_indices = [
        transport_idx * RR + r
        for r in range(RR)
    ]
    print(sector_major_indices)

    print("\n如果是 region-major，交通行业 15 个区域的 index 是:")
    region_major_indices = [
        r * NN + transport_idx
        for r in range(RR)
    ]
    print(region_major_indices)


def reorder_arrays_to_sector_major(
    zz_s,
    ff_s,
    va_s,
    xx_s,
    sector_major_raw_indices,
):
    """
    把 MRIO 矩阵统一到 sector-major。
    """

    idx = sector_major_raw_indices

    zz_new = zz_s[np.ix_(idx, idx)]
    ff_new = ff_s[idx, :]
    va_new = va_s[:, idx]
    xx_new = xx_s[:, idx]

    return zz_new, ff_new, va_new, xx_new


def make_activity_mapping(activity_order, sector_major_raw_indices):
    mapping = activity_order.iloc[sector_major_raw_indices].copy()
    mapping = mapping.reset_index(drop=True)
    mapping.insert(0, "sector_major_index", np.arange(RRNN))
    return mapping


# =====================================================
# 3. 主流程
# =====================================================

def main():
    print("=" * 80)
    print("S9-1: 检查并统一 MRIO 排列顺序")
    print("=" * 80)

    step8_root = resolve_existing_path(
        STEP8_ROOT_CANDIDATES,
        "STEP8_ROOT"
    )

    mrio_path = resolve_existing_path(
        MRIO_PATH_CANDIDATES,
        "MRIO 原始表"
    )

    # -------------------------------------------------
    # 读取原始 MRIO 表头并判断顺序
    # -------------------------------------------------

    activity_order = read_raw_mrio_activity_order(mrio_path)

    order_type = infer_order(activity_order)

    print("\n" + "=" * 80)
    print("原始 MRIO 285 部门顺序判断")
    print("=" * 80)
    print("order_type:", order_type)

    print("\n原始 MRIO 前 40 个活动:")
    print(activity_order.head(40).to_string(index=False))

    if order_type not in ["region-major", "sector-major"]:
        raise ValueError(
            f"无法可靠判断 MRIO 顺序: {order_type}。"
            "请查看 activity_order_raw.csv。"
        )

    sector_major_raw_indices = build_sector_major_raw_indices(order_type)

    mapping = make_activity_mapping(
        activity_order=activity_order,
        sector_major_raw_indices=sector_major_raw_indices
    )

    # 保存全局检查文件
    check_root = step8_root / "_S9_1_order_check"
    check_root.mkdir(parents=True, exist_ok=True)

    activity_order.to_csv(
        check_root / "activity_order_raw.csv",
        index=False,
        encoding="utf-8-sig"
    )

    mapping.to_csv(
        check_root / "activity_order_sector_major_mapping.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame([{
        "order_type_detected": order_type,
        "RR": RR,
        "NN": NN,
        "RRNN": RRNN,
        "mrio_path": str(mrio_path),
        "step8_root": str(step8_root),
        "definition": (
            "If original MRIO is region-major, matrices are reordered to "
            "sector-major for S9. Sector-major uses index = sector * RR + region."
        ),
    }]).to_csv(
        check_root / "S9_1_order_check_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n检查文件已保存:")
    print(check_root)

    # -------------------------------------------------
    # 对 I1/I2/I3 生成 sector-major checked 输入
    # -------------------------------------------------

    for intensity in INTENSITY_SCENARIOS:
        print("\n" + "#" * 80)
        print(f"处理 {intensity}")
        print("#" * 80)

        in_dir = step8_root / intensity / "MRIO模型"
        out_dir = step8_root / intensity / "MRIO模型_sector_major_checked"

        if not in_dir.exists():
            raise FileNotFoundError(f"找不到输入目录: {in_dir}")

        out_dir.mkdir(parents=True, exist_ok=True)

        # 读取输入
        zz_s = np.load(in_dir / "zz_s.npy")
        ff_s = np.load(in_dir / "ff_s.npy")
        va_s = np.load(in_dir / "va_s.npy")
        xx_s = np.load(in_dir / "xx_s.npy")
        VA_TT = np.load(in_dir / "VA_TT_transport_only.npy")

        region_names = read_name_csv(in_dir / "region_names.csv")
        sector_names = read_name_csv(in_dir / "sector_names.csv")

        print("in_dir:", in_dir)
        print("out_dir:", out_dir)
        print("zz_s:", zz_s.shape)
        print("ff_s:", ff_s.shape)
        print("va_s:", va_s.shape)
        print("xx_s:", xx_s.shape)
        print("VA_TT:", VA_TT.shape)
        print("region count:", len(region_names))
        print("sector count:", len(sector_names))

        if zz_s.shape != (RRNN, RRNN):
            raise ValueError(f"zz_s shape 错误: {zz_s.shape}")

        if ff_s.shape != (RRNN, RR):
            raise ValueError(f"ff_s shape 错误: {ff_s.shape}")

        if va_s.shape != (1, RRNN):
            raise ValueError(f"va_s shape 错误: {va_s.shape}")

        if xx_s.shape != (1, RRNN):
            raise ValueError(f"xx_s shape 错误: {xx_s.shape}")

        if VA_TT.shape[0] != NN or VA_TT.shape[1] != RR:
            raise ValueError(f"VA_TT shape 错误: {VA_TT.shape}")

        check_transport_positions(
            sector_names=sector_names,
            region_names=region_names
        )

        # 重排 MRIO 矩阵
        zz_new, ff_new, va_new, xx_new = reorder_arrays_to_sector_major(
            zz_s=zz_s,
            ff_s=ff_s,
            va_s=va_s,
            xx_s=xx_s,
            sector_major_raw_indices=sector_major_raw_indices
        )

        # VA_TT 不重排！
        # 因为 VA_TT 本身是 (sector, region, time)，
        # S9 flatten 后天然是 sector-major。
        VA_TT_new = VA_TT.copy()

        # 保存新输入
        np.save(out_dir / "zz_s.npy", zz_new)
        np.save(out_dir / "ff_s.npy", ff_new)
        np.save(out_dir / "va_s.npy", va_new)
        np.save(out_dir / "xx_s.npy", xx_new)
        np.save(out_dir / "VA_TT_transport_only.npy", VA_TT_new)

        # 复制 csv
        for fname in [
            "region_names.csv",
            "sector_names.csv",
            "time_index.csv",
            "annual_g_rt_transport_factor.csv",
        ]:
            src = in_dir / fname
            if src.exists():
                shutil.copy2(src, out_dir / fname)

        # effective eta 文件名字可能有 eta3.5
        for f in in_dir.glob("annual_g_rt_effective_eta*.csv"):
            shutil.copy2(f, out_dir / f.name)

        # 保存当前强度的映射和说明
        mapping.to_csv(
            out_dir / "activity_order_sector_major_mapping.csv",
            index=False,
            encoding="utf-8-sig"
        )

        pd.DataFrame([{
            "intensity": intensity,
            "source_dir": str(in_dir),
            "output_dir": str(out_dir),
            "original_order_type": order_type,
            "target_order_type": "sector-major",
            "zz_s_shape": str(zz_new.shape),
            "ff_s_shape": str(ff_new.shape),
            "va_s_shape": str(va_new.shape),
            "xx_s_shape": str(xx_new.shape),
            "VA_TT_shape": str(VA_TT_new.shape),
            "note": (
                "MRIO arrays were reordered to sector-major if needed. "
                "VA_TT was not reordered because it is stored as sector × region × time."
            )
        }]).to_csv(
            out_dir / "S9_1_sector_major_checked_config.csv",
            index=False,
            encoding="utf-8-sig"
        )

        # 平衡检查
        col_err = np.max(
            np.abs(zz_new.sum(axis=0) + va_new.flatten() - xx_new.flatten())
        )

        row_err = np.max(
            np.abs(zz_new.sum(axis=1) + ff_new.sum(axis=1) - xx_new.flatten())
        )

        print("\n平衡检查:")
        print("max column balance error:", col_err)
        print("max row balance error:", row_err)

        print("\n[SAVED]", out_dir)

    print("\n" + "=" * 80)
    print("✅ S9-1 完成：MRIO 输入已统一为 sector-major")
    print("=" * 80)

    print("原始顺序:", order_type)
    print("新 S9 输入目录示例:")
    for intensity in INTENSITY_SCENARIOS:
        print(
            intensity,
            "=>",
            step8_root / intensity / "MRIO模型_sector_major_checked"
        )

    print("\n下一步：")
    print("新的 S9 不要再读 MRIO模型，而是读 MRIO模型_sector_major_checked。")


if __name__ == "__main__":
    main()