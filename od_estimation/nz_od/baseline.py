# Extracted without changing function bodies. See provenance.json.

import os

from pathlib import Path

from collections import defaultdict

import numpy as np

import pandas as pd

import networkx as nx

N_ITER = 5

MAX_UE_ITER = 50

REL_GAP_TOL = 1e-4

BPR_ALPHA = 0.15

BPR_BETA = 4.0

MIN_CAPACITY = 1e-6

ALPHA_UPDATE = 0.20

MIN_FACTOR = 0.50

MAX_FACTOR = 1.50

EPS = 1e-6

DO_GLOBAL_SCALING_EACH_ITER = True

STOP_IF_FACTOR_MEAN_CLOSE_TO_ONE = 0.01

STOP_IF_OD_RATIO_CLOSE_TO_ONE = 0.01

def ensure_required_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少字段: {missing}")

def bpr_time(t0: float, x: float, c: float, alpha: float = 0.15, beta: float = 4.0) -> float:
    if pd.isna(t0):
        return np.nan
    c = max(float(c), MIN_CAPACITY)
    vc = float(x) / c
    return float(t0) * (1.0 + alpha * (vc ** beta))

def build_graph(edge_df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, row in edge_df.iterrows():
        fn = int(row["FromNode"])
        tn = int(row["ToNode"])
        eid = int(row["EdgeID"])

        if pd.notna(row["AB_time"]):
            G.add_edge(fn, tn, weight=float(row["AB_time"]), edge_id=eid, direction="AB")

        if pd.notna(row["BA_time"]):
            G.add_edge(tn, fn, weight=float(row["BA_time"]), edge_id=eid, direction="BA")
    return G

def aon_load(od_table: pd.DataFrame, G: nx.DiGraph):
    aux_AB = defaultdict(float)
    aux_BA = defaultdict(float)

    total_demand = 0.0
    loaded_demand = 0.0
    skipped = 0
    sptt = 0.0

    for _, row in od_table.iterrows():
        o = int(row["O_node"])
        d = int(row["D_node"])
        q = float(row["OD_value"])

        if q <= 0:
            continue

        total_demand += q

        try:
            path = nx.shortest_path(G, o, d, weight="weight")
            path_cost = nx.shortest_path_length(G, o, d, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            skipped += 1
            continue

        loaded_demand += q
        sptt += q * path_cost

        for u, v in zip(path[:-1], path[1:]):
            ed = G[u][v]
            eid = ed["edge_id"]
            direc = ed["direction"]
            if direc == "AB":
                aux_AB[eid] += q
            else:
                aux_BA[eid] += q

    return aux_AB, aux_BA, total_demand, loaded_demand, skipped, sptt

def calc_tstt(edge_df: pd.DataFrame) -> float:
    ab = (edge_df["AB_flow"] * edge_df["AB_time"]).sum(skipna=True)
    ba = (edge_df["BA_flow"] * edge_df["BA_time"]).sum(skipna=True)
    return float(ab + ba)

def run_ue(node_od_df: pd.DataFrame, edges_raw: pd.DataFrame):
    edges = edges_raw.copy()

    for c in ["AB_FF_Time", "BA_FF_Time", "AB_Capacity", "BA_Capacity"]:
        edges[c] = pd.to_numeric(edges[c], errors="coerce")

    edges["AB_Capacity"] = edges["AB_Capacity"].fillna(MIN_CAPACITY).clip(lower=MIN_CAPACITY)
    edges["BA_Capacity"] = edges["BA_Capacity"].fillna(MIN_CAPACITY).clip(lower=MIN_CAPACITY)

    edges["AB_flow"] = 0.0
    edges["BA_flow"] = 0.0
    edges["AB_time"] = edges["AB_FF_Time"].copy()
    edges["BA_time"] = edges["BA_FF_Time"].copy()

    iter_logs = []

    for it in range(1, MAX_UE_ITER + 1):
        G = build_graph(edges)
        aux_AB, aux_BA, total_demand, loaded_demand, skipped, sptt = aon_load(node_od_df, G)

        step = 1.0 / it

        old_AB = edges["AB_flow"].copy()
        old_BA = edges["BA_flow"].copy()

        edges["aux_AB_flow"] = edges["EdgeID"].map(aux_AB).fillna(0.0)
        edges["aux_BA_flow"] = edges["EdgeID"].map(aux_BA).fillna(0.0)

        edges["AB_flow"] = old_AB + step * (edges["aux_AB_flow"] - old_AB)
        edges["BA_flow"] = old_BA + step * (edges["aux_BA_flow"] - old_BA)

        edges["AB_time"] = edges.apply(
            lambda r: bpr_time(r["AB_FF_Time"], r["AB_flow"], r["AB_Capacity"], BPR_ALPHA, BPR_BETA),
            axis=1
        )
        edges["BA_time"] = edges.apply(
            lambda r: bpr_time(r["BA_FF_Time"], r["BA_flow"], r["BA_Capacity"], BPR_ALPHA, BPR_BETA),
            axis=1
        )

        tstt = calc_tstt(edges)
        rel_gap = abs(tstt - sptt) / sptt if sptt > 0 else np.nan

        iter_logs.append({
            "iter": it,
            "step": step,
            "total_demand": total_demand,
            "loaded_demand": loaded_demand,
            "skipped_od": skipped,
            "SPTT": sptt,
            "TSTT": tstt,
            "rel_gap": rel_gap,
            "max_AB_flow_change": (edges["AB_flow"] - old_AB).abs().max(),
            "max_BA_flow_change": (edges["BA_flow"] - old_BA).abs().max()
        })

        if pd.notna(rel_gap) and rel_gap < REL_GAP_TOL:
            break

    edges.drop(columns=["aux_AB_flow", "aux_BA_flow"], inplace=True, errors="ignore")

    def safe_diff(model, obs):
        if pd.isna(obs):
            return np.nan
        return model - obs

    def safe_rel_err(model, obs):
        if pd.isna(obs) or obs == 0:
            return np.nan
        return (model - obs) / obs

    edges["AB_diff"] = edges.apply(lambda r: safe_diff(r["AB_flow"], r.get("AB_Count", np.nan)), axis=1)
    edges["BA_diff"] = edges.apply(lambda r: safe_diff(r["BA_flow"], r.get("BA_Count", np.nan)), axis=1)
    edges["AB_rel_err"] = edges.apply(lambda r: safe_rel_err(r["AB_flow"], r.get("AB_Count", np.nan)), axis=1)
    edges["BA_rel_err"] = edges.apply(lambda r: safe_rel_err(r["BA_flow"], r.get("BA_Count", np.nan)), axis=1)

    return edges, pd.DataFrame(iter_logs)

def expand_region_to_node_od(region_od_df: pd.DataFrame, access_df: pd.DataFrame) -> pd.DataFrame:
    access_dict = {}
    for region_name, grp in access_df.groupby("Region15"):
        access_dict[region_name] = grp[["NodeID", "weight"]].to_dict("records")

    records = []
    for _, row in region_od_df.iterrows():
        o_region = row["O_region"]
        d_region = row["D_region"]
        od_val = float(row["OD_value"])

        if o_region == d_region or od_val <= 0:
            continue
        if o_region not in access_dict or d_region not in access_dict:
            continue

        for o_item in access_dict[o_region]:
            for d_item in access_dict[d_region]:
                pair_weight = float(o_item["weight"]) * float(d_item["weight"])
                node_od = od_val * pair_weight
                records.append({
                    "O_region": o_region,
                    "D_region": d_region,
                    "O_node": int(o_item["NodeID"]),
                    "D_node": int(d_item["NodeID"]),
                    "pair_weight": pair_weight,
                    "OD_value": node_od
                })

    return pd.DataFrame(records)

def global_scale_region_od(region_od_df: pd.DataFrame, edge_flow_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    ab_obs = edge_flow_df.loc[edge_flow_df["AB_Count"].notna(), "AB_Count"].sum()
    ab_mod = edge_flow_df.loc[edge_flow_df["AB_Count"].notna(), "AB_flow"].sum()

    ba_obs = edge_flow_df.loc[edge_flow_df["BA_Count"].notna(), "BA_Count"].sum()
    ba_mod = edge_flow_df.loc[edge_flow_df["BA_Count"].notna(), "BA_flow"].sum()

    all_obs = ab_obs + ba_obs
    all_mod = ab_mod + ba_mod

    lam = (all_obs / all_mod) if all_mod > 0 else 1.0

    out = region_od_df.copy()
    for col in ["OD_value_old", "factor_region_smoothed"]:
        if col in out.columns:
            out = out.drop(columns=[col])

    mask_non_diag = out["O_region"] != out["D_region"]
    out.loc[mask_non_diag, "OD_value"] = out.loc[mask_non_diag, "OD_value"] * lam
    out.loc[~mask_non_diag, "OD_value"] = 0.0
    return out, lam

def update_region_od(region_od_df: pd.DataFrame, node_od_df: pd.DataFrame, edge_flow_df: pd.DataFrame):
    G = nx.DiGraph()

    for _, row in edge_flow_df.iterrows():
        fn = int(row["FromNode"])
        tn = int(row["ToNode"])
        eid = int(row["EdgeID"])

        if pd.notna(row["AB_time"]):
            G.add_edge(fn, tn, weight=float(row["AB_time"]), edge_id=eid, direction="AB")
        if pd.notna(row["BA_time"]):
            G.add_edge(tn, fn, weight=float(row["BA_time"]), edge_id=eid, direction="BA")

    edge_ratio = {}
    for _, row in edge_flow_df.iterrows():
        eid = int(row["EdgeID"])
        if pd.notna(row["AB_Count"]):
            edge_ratio[(eid, "AB")] = (float(row["AB_Count"]) + EPS) / (float(row["AB_flow"]) + EPS)
        if pd.notna(row["BA_Count"]):
            edge_ratio[(eid, "BA")] = (float(row["BA_Count"]) + EPS) / (float(row["BA_flow"]) + EPS)

    node_records = []
    skipped = 0
    used_obs_paths = 0

    for _, row in node_od_df.iterrows():
        o_region = row["O_region"]
        d_region = row["D_region"]
        o_node = int(row["O_node"])
        d_node = int(row["D_node"])
        q = float(row["OD_value"])

        if q <= 0:
            continue

        try:
            path = nx.shortest_path(G, o_node, d_node, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            skipped += 1
            continue

        ratios = []
        weights = []

        for u, v in zip(path[:-1], path[1:]):
            ed = G[u][v]
            eid = ed["edge_id"]
            direc = ed["direction"]
            t = float(ed["weight"])

            if (eid, direc) in edge_ratio:
                ratios.append(edge_ratio[(eid, direc)])
                weights.append(t)

        if len(ratios) == 0:
            factor_node = 1.0
        else:
            used_obs_paths += 1
            factor_node = np.average(ratios, weights=weights)

        factor_node = max(MIN_FACTOR, min(MAX_FACTOR, factor_node))

        node_records.append({
            "O_region": o_region,
            "D_region": d_region,
            "OD_value": q,
            "factor_node": factor_node
        })

    node_update_df = pd.DataFrame(node_records)

    if node_update_df.empty:
        tmp = region_od_df.copy()
        if "factor_region_smoothed" not in tmp.columns:
            tmp["factor_region_smoothed"] = 1.0
        return tmp, pd.DataFrame(), {"skipped": skipped, "used_obs_paths": used_obs_paths}

    region_factor_df = (
        node_update_df.groupby(["O_region", "D_region"], as_index=False)
        .apply(
            lambda g: pd.Series({
                "factor_region_raw": np.average(g["factor_node"], weights=g["OD_value"]),
                "node_od_sum": g["OD_value"].sum(),
                "node_pair_count": len(g)
            }),
            include_groups=False
        )
        .reset_index(drop=True)
    )

    region_factor_df["factor_region_raw"] = region_factor_df["factor_region_raw"].clip(lower=MIN_FACTOR, upper=MAX_FACTOR)
    region_factor_df["factor_region_smoothed"] = region_factor_df["factor_region_raw"] ** ALPHA_UPDATE

    region_od_base = region_od_df.copy()
    for col in ["factor_region_smoothed", "OD_value_old"]:
        if col in region_od_base.columns:
            region_od_base = region_od_base.drop(columns=[col])

    region_od_new = region_od_base.merge(
        region_factor_df[["O_region", "D_region", "factor_region_smoothed"]],
        on=["O_region", "D_region"],
        how="left"
    )

    region_od_new["factor_region_smoothed"] = region_od_new["factor_region_smoothed"].fillna(1.0)

    mask_non_diag = region_od_new["O_region"] != region_od_new["D_region"]
    region_od_new.loc[mask_non_diag, "OD_value_old"] = region_od_new.loc[mask_non_diag, "OD_value"]
    region_od_new.loc[mask_non_diag, "OD_value"] = (
        region_od_new.loc[mask_non_diag, "OD_value_old"] *
        region_od_new.loc[mask_non_diag, "factor_region_smoothed"]
    )
    region_od_new.loc[~mask_non_diag, "OD_value"] = 0.0

    diag = {"skipped": skipped, "used_obs_paths": used_obs_paths}
    return region_od_new, region_factor_df, diag

def main():
    print("=" * 80)
    print("读取固定输入")
    print("=" * 80)

    access_df = pd.read_csv(ACCESS_CSV)
    edges_raw = pd.read_csv(EDGES_CSV)
    region_od_df = pd.read_csv(INITIAL_REGION_OD_CSV)

    ensure_required_columns(access_df, ["Region15", "NodeID", "weight"], "access")
    ensure_required_columns(edges_raw, ["EdgeID", "FromNode", "ToNode", "AB_FF_Time", "BA_FF_Time", "AB_Capacity", "BA_Capacity"], "edges")
    ensure_required_columns(region_od_df, ["O_region", "D_region", "OD_value"], "region_od")

    access_df["NodeID"] = access_df["NodeID"].astype(int)
    access_df["weight"] = pd.to_numeric(access_df["weight"], errors="coerce").fillna(0.0)

    edges_raw["EdgeID"] = edges_raw["EdgeID"].astype(int)
    edges_raw["FromNode"] = edges_raw["FromNode"].astype(int)
    edges_raw["ToNode"] = edges_raw["ToNode"].astype(int)

    region_od_df["OD_value"] = pd.to_numeric(region_od_df["OD_value"], errors="coerce").fillna(0.0)

    summary_rows = []
    current_region_od = region_od_df.copy()

    for i in range(1, N_ITER + 1):
        print("\n" + "=" * 80)
        print(f"开始第 {i} 轮")
        print("=" * 80)

        # ---------- S5: 区域OD -> 节点OD ----------
        node_od_df = expand_region_to_node_od(current_region_od, access_df)
        node_od_csv = ITER_DIR / f"node_od_iter{i}.csv"
        node_od_df.to_csv(node_od_csv, index=False, encoding="utf-8-sig")

        # ---------- S6: UE ----------
        edge_flow_df, ue_log_df = run_ue(node_od_df, edges_raw)
        edge_flow_csv = ITER_DIR / f"edge_flow_iter{i}.csv"
        ue_log_csv = ITER_DIR / f"ue_log_iter{i}.csv"
        edge_flow_df.to_csv(edge_flow_csv, index=False, encoding="utf-8-sig")
        ue_log_df.to_csv(ue_log_csv, index=False, encoding="utf-8-sig")

        # ---------- S8: 结构更新 ----------
        updated_region_od, factor_df, diag = update_region_od(current_region_od, node_od_df, edge_flow_df)

        # ---------- 每轮总量缩放（放在结构更新后） ----------
        lambda_all = 1.0
        if DO_GLOBAL_SCALING_EACH_ITER:
            updated_region_od, lambda_all = global_scale_region_od(updated_region_od, edge_flow_df)

        region_od_csv = ITER_DIR / f"region_od_iter{i}.csv"
        factor_csv = ITER_DIR / f"region_factor_iter{i}.csv"
        updated_region_od.to_csv(region_od_csv, index=False, encoding="utf-8-sig")
        factor_df.to_csv(factor_csv, index=False, encoding="utf-8-sig")

        ab_diff_mean = edge_flow_df["AB_diff"].mean(skipna=True)
        ba_diff_mean = edge_flow_df["BA_diff"].mean(skipna=True)
        ab_diff_std = edge_flow_df["AB_diff"].std(skipna=True)
        ba_diff_std = edge_flow_df["BA_diff"].std(skipna=True)

        total_region_od_old = current_region_od.loc[current_region_od["O_region"] != current_region_od["D_region"], "OD_value"].sum()
        total_region_od_new = updated_region_od.loc[updated_region_od["O_region"] != updated_region_od["D_region"], "OD_value"].sum()

        last_rel_gap = ue_log_df["rel_gap"].iloc[-1] if not ue_log_df.empty else np.nan
        last_skipped = ue_log_df["skipped_od"].iloc[-1] if not ue_log_df.empty else np.nan
        last_loaded = ue_log_df["loaded_demand"].iloc[-1] if not ue_log_df.empty else np.nan

        factor_mean = factor_df["factor_region_smoothed"].mean(skipna=True) if not factor_df.empty else np.nan
        factor_std = factor_df["factor_region_smoothed"].std(skipna=True) if not factor_df.empty else np.nan
        od_total_ratio = (total_region_od_new / total_region_od_old) if total_region_od_old > 0 else np.nan

        summary_rows.append({
            "iter": i,
            "node_od_records": len(node_od_df),
            "loaded_demand": last_loaded,
            "skipped_od": last_skipped,
            "rel_gap": last_rel_gap,
            "AB_diff_mean": ab_diff_mean,
            "BA_diff_mean": ba_diff_mean,
            "AB_diff_std": ab_diff_std,
            "BA_diff_std": ba_diff_std,
            "region_od_total_old": total_region_od_old,
            "region_od_total_new": total_region_od_new,
            "od_total_ratio": od_total_ratio,
            "lambda_all": lambda_all,
            "update_skipped_paths": diag["skipped"],
            "update_used_obs_paths": diag["used_obs_paths"],
            "factor_mean": factor_mean,
            "factor_std": factor_std
        })

        print(f"第 {i} 轮完成：")
        print(f"  node_od_records      = {len(node_od_df)}")
        print(f"  loaded_demand        = {last_loaded}")
        print(f"  skipped_od           = {last_skipped}")
        print(f"  rel_gap              = {last_rel_gap}")
        print(f"  AB_diff_mean/std     = {ab_diff_mean:.6f} / {ab_diff_std:.6f}")
        print(f"  BA_diff_mean/std     = {ba_diff_mean:.6f} / {ba_diff_std:.6f}")
        print(f"  region_od_total(old) = {total_region_od_old:.6f}")
        print(f"  region_od_total(new) = {total_region_od_new:.6f}")
        print(f"  od_total_ratio       = {od_total_ratio:.6f}")
        print(f"  lambda_all           = {lambda_all:.6f}")
        print(f"  factor_mean/std      = {factor_mean:.6f} / {factor_std:.6f}")

        current_region_od = updated_region_od.copy()

        # 提前停止：更新已经很小
        if (
            pd.notna(factor_mean) and
            pd.notna(od_total_ratio) and
            abs(factor_mean - 1.0) < STOP_IF_FACTOR_MEAN_CLOSE_TO_ONE and
            abs(od_total_ratio - 1.0) < STOP_IF_OD_RATIO_CLOSE_TO_ONE
        ):
            print(f"\n提前停止：第 {i} 轮后更新已很小。")
            break

    final_region_od_csv = ITER_DIR / "region_od_final.csv"
    current_region_od.to_csv(final_region_od_csv, index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = ITER_DIR / "iteration_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("全部迭代完成")
    print("=" * 80)
    print("最终区域OD:", final_region_od_csv)
    print("迭代汇总表:", summary_csv)
    print("\n迭代汇总：")
    print(summary_df)
