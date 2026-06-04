#!/usr/bin/env python3
"""Average-rank statistical gate: certified window-compiler (CWC) vs SOTA operators + baselines.

Per block (dataset x budget x seed) under the bounded workload shift, score every method by
its mean constrained utility on the eval window, write per-block scores to rank_gate_input.csv.
Significance is computed separately by scripts/verify_rank_significance.py.
"""
from __future__ import annotations
import argparse, math, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.certified_window_compiler import load_items, telemetry_vector, constrained_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

CANDIDATE = "CWC-certified"
SOTA_OPS = {
    "Static-Qwen3":    ["external_qwen3_embedding_06b_pre", "external_qwen3_embedding_06b_post"],
    "Static-SPLADE":   ["external_splade_cocondenser_ensembledistil_pre", "external_splade_cocondenser_ensembledistil_post"],
    "Static-ColBERTv2":["external_colbertv2_pre", "external_colbertv2_post"],
    "Static-E5":       ["dense_e5_base_v2_pre", "dense_e5_base_v2_post"],
}


def block_scores(df, dataset, budget, seed, p_train_by_sel, n_cells):
    sub = df[df.dataset == dataset]
    items = sorted(sub["item"].unique())
    by_item = {it: sub[sub["item"] == it] for it in items}
    rng = np.random.default_rng(seed)
    train, ev = [], []
    for it in items:
        s = float(by_item[it].iloc[0]["selectivity"])
        (train if rng.random() < p_train_by_sel.get(s, 0.5) else ev).append(it)
    if len(train) < 2 * n_cells or len(ev) < n_cells:
        return None
    perm = rng.permutation(len(train)); half = len(train) // 2
    sel = [train[i] for i in perm[:half]]; cal = [train[i] for i in perm[half:]]
    actions = sorted(sub["action"].unique())

    def cs(it, a):
        r = by_item[it]; r = r[r.action == a]
        if len(r) == 0: return float("nan")
        r = r.iloc[0]; return constrained_score(float(r.ndcg10), float(r.latency_ms), budget)

    def lat(it, a):
        r = by_item[it]; r = r[r.action == a]
        return float(r.iloc[0].latency_ms) if len(r) else float("nan")

    def ndcg(it, a):
        r = by_item[it]; r = r[r.action == a]
        return float(r.iloc[0].ndcg10) if len(r) else float("nan")

    fit_items = sel + ev
    F = np.vstack([telemetry_vector(by_item[it]) for it in fit_items])
    scaler = StandardScaler().fit(F)
    km = KMeans(n_clusters=n_cells, n_init=10, random_state=seed).fit(scaler.transform(F))
    def cell(it):
        return int(km.predict(scaler.transform(telemetry_vector(by_item[it]).reshape(1, -1)))[0])
    sel_cell = {it: cell(it) for it in sel}; ev_cell = {it: cell(it) for it in ev}; cal_cell = {it: cell(it) for it in cal}

    plan = {}
    for c in range(n_cells):
        ci = [it for it in sel if sel_cell[it] == c]
        if not ci: plan[c] = "bm25_pre"; continue
        best, bv = "bm25_pre", -1e9
        for a in actions:
            v = [cs(it, a) for it in ci]; v = [x for x in v if not math.isnan(x)]
            if v and float(np.mean(v)) > bv: bv, best = float(np.mean(v)), a
        plan[c] = best

    def trv(a):
        v = [1.0 if lat(it, a) > budget else 0.0 for it in train if not math.isnan(lat(it, a))]
        return float(np.mean(v)) if v else 1.0
    feas = [a for a in actions if trv(a) <= 1e-9]
    fb = max(feas, key=lambda a: float(np.nanmean([ndcg(it, a) for it in train]))) if feas else "bm25_pre"

    deploy = dict(plan)
    for c in range(n_cells):
        ci = [it for it in cal if cal_cell[it] == c]
        if ci and float(np.mean([1.0 if lat(it, plan[c]) > budget else 0.0 for it in ci])) > 0.5:
            deploy[c] = fb

    scores = {}
    scores[CANDIDATE] = float(np.nanmean([cs(it, deploy[ev_cell[it]]) for it in ev]))
    sb = max(actions, key=lambda a: float(np.nanmean([cs(it, a) for it in train])))
    scores["StaticBest-cal"] = float(np.nanmean([cs(it, sb) for it in ev]))
    for name, ops in SOTA_OPS.items():
        ops = [o for o in ops if o in actions]
        if not ops: continue
        scores[name] = float(np.nanmean([max(cs(it, o) for o in ops) for it in ev]))
    p95 = {a: float(np.nanpercentile([lat(it, a) for it in train if not math.isnan(lat(it, a))], 95)) for a in actions}
    feasc = [a for a in actions if p95[a] <= budget] or actions
    cg = max(feasc, key=lambda a: float(np.nanmean([ndcg(it, a) for it in train])))
    scores["CostGreedy-cal"] = float(np.nanmean([cs(it, cg) for it in ev]))
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--action-rows", type=Path, default=Path("results/larger_workload/all_action_rows_multi.csv"))
    ap.add_argument("--output-dir", type=Path, default=Path("results/rank_gate_cwc"))
    ap.add_argument("--budgets-ms", default="40,80,120,160")
    ap.add_argument("--seeds", default="13,17,23,29,31")
    ap.add_argument("--n-cells", type=int, default=4)
    ap.add_argument("--p-train-by-sel", default="0.1:0.8,0.5:0.5,1.0:0.2")
    args = ap.parse_args()
    df = load_items(args.action_rows, None)
    budgets = [float(x) for x in args.budgets_ms.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    pts = {float(kv.split(":")[0]): float(kv.split(":")[1]) for kv in args.p_train_by_sel.split(",")}
    rank_rows = []
    for d in sorted(df.dataset.unique()):
        for b in budgets:
            for s in seeds:
                sc = block_scores(df, d, b, s, pts, args.n_cells)
                if not sc: continue
                block = f"{d}|b={b:g}|seed={s}"
                for m, v in sc.items():
                    rank_rows.append({"block": block, "method": m, "mean_score": v})
    outdir = args.output_dir / "rank_gate_cwc"
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rank_rows).to_csv(outdir / "rank_gate_input.csv", index=False)
    print("blocks", len({r["block"] for r in rank_rows}), "methods", len({r["method"] for r in rank_rows}))


if __name__ == "__main__":
    main()
