#!/usr/bin/env python3
"""E6 (rebuttal): put numbers behind the claim that fixed recipes are brittle.

Reviewer #3 D1: "'Fixed recipes such as BM25-only, dense-only, fixed fusion, or fixed
reranking depth are simple but brittle' -- it would help to clarify how the selectivity
and budget change today with the query mix and what impact does it have. Can it be
quantified as drop in throughput or latency?"

The introduction asserted brittleness without measuring it. This script measures it: it
sweeps the *amount* of workload-mix drift between the labeled and the served window and
records, for every fixed recipe and for CWC, how the served window's SLO-violation rate,
mean/p95 latency, achievable throughput (1/mean service time) and nDCG@10 move.

Drift knob d in [0, 0.45]: an item of selectivity s joins the labeled window with
probability p(s) = 0.5 + d for the cheap stratum, 0.5 for the middle one, 0.5 - d for the
expensive one. d = 0 is the i.i.d. (no-drift) control; larger d skews the served window
towards the expensive stratum, which is exactly the mix change an operator sees when a
filtered/hard query class grows.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fast import build_tables  # noqa: E402
from scripts.certified_window_compiler import (  # noqa: E402
    constrained_score, load_items, telemetry_vector,
)

RECIPES = {
    "BM25-only": ["bm25_pre", "bm25_post"],
    "Dense-only": ["dense_e5_base_v2_pre", "dense_e5_base_v2_post", "dense_pre", "dense_post"],
    "Fixed-fusion(RRF)": ["rrf_pre", "rrf_post", "rrf_dense_e5_base_v2_pre",
                          "rrf_dense_e5_base_v2_post"],
    "Fixed-rerank@50": ["rerank_depth_50"],
    "Fixed-rerank@100": ["rerank_depth_100"],
    "SOTA-SPLADE": ["external_splade_cocondenser_ensembledistil_pre",
                    "external_splade_cocondenser_ensembledistil_post"],
    "SOTA-Qwen3": ["external_qwen3_embedding_06b_pre", "external_qwen3_embedding_06b_post"],
}


def run_block(df, budget, d, n_cells, seed):
    rng = np.random.default_rng(seed)
    items, TEL, LAT, ND, SEL, actions = build_tables(df)
    p_train = {0.1: 0.5 + d, 0.5: 0.5, 1.0: 0.5 - d}
    train_items, eval_items = [], []
    for it in items:
        s = SEL[it]
        (train_items if rng.random() < p_train.get(s, 0.5) else eval_items).append(it)
    if len(train_items) < 2 * n_cells or len(eval_items) < n_cells:
        return []

    def metrics(it, a):
        v = LAT.get((it, a))
        if v is None:
            return float("nan"), float("nan"), float("nan")
        ndv = ND[(it, a)]
        return v, ndv, constrained_score(ndv, v, budget)

    # --- CWC (repaired: selection-split binding + selection-split abstain) ---
    perm = rng.permutation(len(train_items))
    half = len(train_items) // 2
    sel_items = [train_items[i] for i in perm[:half]]
    fit = sel_items + eval_items
    scaler = StandardScaler().fit(np.vstack([TEL[it] for it in fit]))
    km = KMeans(n_clusters=n_cells, n_init=10, random_state=seed).fit(
        scaler.transform(np.vstack([TEL[it] for it in fit])))
    cell_of = {it: int(km.predict(scaler.transform(TEL[it].reshape(1, -1)))[0])
               for it in items}
    plan_by_cell = {}
    for c in range(n_cells):
        ci = [it for it in sel_items if cell_of[it] == c]
        if not ci:
            plan_by_cell[c] = "bm25_pre"; continue
        best, bv = "bm25_pre", -1e9
        for a in actions:
            v = [metrics(it, a)[2] for it in ci]
            v = [x for x in v if not math.isnan(x)]
            if v and float(np.mean(v)) > bv:
                bv, best = float(np.mean(v)), a
        plan_by_cell[c] = best

    def sviol(a):
        v = [1.0 if metrics(it, a)[0] > budget else 0.0
             for it in sel_items if not math.isnan(metrics(it, a)[0])]
        return float(np.mean(v)) if v else 1.0

    feas = [a for a in actions if sviol(a) <= 1e-9]
    fallback = (max(feas, key=lambda a: float(np.nanmean([metrics(it, a)[1] for it in sel_items])))
                if feas else "bm25_pre")
    deploy = {}
    for c in range(n_cells):
        ci = [it for it in sel_items if cell_of[it] == c]
        if not ci:
            deploy[c] = fallback; continue
        vp = float(np.mean([1.0 if metrics(it, plan_by_cell[c])[0] > budget else 0.0 for it in ci]))
        vf = float(np.mean([1.0 if metrics(it, fallback)[0] > budget else 0.0 for it in ci]))
        deploy[c] = fallback if vf < vp else plan_by_cell[c]

    def report(name, chooser):
        lats, nds = [], []
        for it in eval_items:
            a = chooser(it)
            if a is None:
                continue
            lt, ndv, _ = metrics(it, a)
            if math.isnan(lt):
                continue
            lats.append(lt); nds.append(ndv)
        if not lats:
            return None
        return {"drift_d": d, "budget_ms": budget, "seed": seed, "policy": name,
                "n_eval": len(lats),
                "violation": float(np.mean([1.0 if x > budget else 0.0 for x in lats])),
                "mean_ms": float(np.mean(lats)), "p95_ms": float(np.percentile(lats, 95)),
                "throughput_qps": 1000.0 / float(np.mean(lats)),
                "ndcg10": float(np.mean(nds))}

    rows = []
    for name, cands in RECIPES.items():
        avail = [a for a in cands if a in actions]
        if not avail:
            continue
        r = report(name, lambda it, av=avail: max(
            av, key=lambda a: (metrics(it, a)[1] if not math.isnan(metrics(it, a)[1]) else -1)))
        if r:
            rows.append(r)
    sb = max(actions, key=lambda a: float(np.nanmean([metrics(it, a)[2] for it in train_items])))
    r = report("StaticBest-cal", lambda it: sb)
    if r:
        rows.append(r)
    r = report("CWC", lambda it: deploy[cell_of[it]])
    if r:
        rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action-rows", type=Path,
                    default=ROOT / "results/larger_workload/all_action_rows_multi.csv")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--output-dir", type=Path, default=Path("rebuttal/results/r6_drift_motivation"))
    ap.add_argument("--budgets-ms", default="80,120")
    ap.add_argument("--drifts", default="0.0,0.15,0.3,0.45")
    ap.add_argument("--n-cells", type=int, default=4)
    ap.add_argument("--seeds", default="13,17,23,29,31")
    args = ap.parse_args()

    df = load_items(args.action_rows, args.dataset or None)
    rows = []
    for b in [float(x) for x in args.budgets_ms.split(",")]:
        for d in [float(x) for x in args.drifts.split(",")]:
            for s in [int(x) for x in args.seeds.split(",")]:
                rows.extend(run_block(df, b, d, args.n_cells, s))
            print(f"  done budget={b:g} drift={d}", flush=True)
    res = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.output_dir / "raw_results.csv", index=False)
    agg = res.groupby(["budget_ms", "policy", "drift_d"]).agg(
        violation=("violation", "mean"), mean_ms=("mean_ms", "mean"),
        p95_ms=("p95_ms", "mean"), throughput_qps=("throughput_qps", "mean"),
        ndcg10=("ndcg10", "mean")).reset_index()
    agg.to_csv(args.output_dir / "aggregate.csv", index=False)
    print(agg.to_string(index=False))

    # drift sensitivity: change from the no-drift control to the strongest drift
    d0, d1 = agg["drift_d"].min(), agg["drift_d"].max()
    sens = []
    for b in agg["budget_ms"].unique():
        for pol in agg["policy"].unique():
            a = agg[(agg.budget_ms == b) & (agg.policy == pol) & (agg.drift_d == d0)]
            z = agg[(agg.budget_ms == b) & (agg.policy == pol) & (agg.drift_d == d1)]
            if len(a) and len(z):
                sens.append({
                    "budget_ms": b, "policy": pol,
                    "violation_no_drift": float(a.violation.iloc[0]),
                    "violation_max_drift": float(z.violation.iloc[0]),
                    "violation_delta": float(z.violation.iloc[0] - a.violation.iloc[0]),
                    "p95_ratio": float(z.p95_ms.iloc[0] / max(a.p95_ms.iloc[0], 1e-9)),
                    "throughput_ratio": float(z.throughput_qps.iloc[0]
                                              / max(a.throughput_qps.iloc[0], 1e-9)),
                    "ndcg_delta": float(z.ndcg10.iloc[0] - a.ndcg10.iloc[0]),
                })
    sdf = pd.DataFrame(sens).sort_values(["budget_ms", "violation_delta"], ascending=[True, False])
    sdf.to_csv(args.output_dir / "drift_sensitivity.csv", index=False)
    print(sdf.to_string(index=False))
    (args.output_dir / "summary.json").write_text(json.dumps(
        {"drift_levels": sorted(agg["drift_d"].unique().tolist()),
         "sensitivity": sdf.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
