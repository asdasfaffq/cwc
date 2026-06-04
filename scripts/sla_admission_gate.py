#!/usr/bin/env python3
"""SLA admission-control rank gate.

Reuses the EXACT same 7-method pool and block split as scripts/rank_gate_cwc.py
(the canonical 45-block study), but instead of the soft constrained-utility score
it reports, per (method, block), the realized eval-window SLO-violation rate and
mean nDCG@10 of that method's deployed plan. The admission utility under an SLA
budget tau is then nDCG if the window is admitted (realized violation <= tau)
and a penalty otherwise -- the standard window-level admission-control semantics.

No fabricated numbers: every per-method (ndcg, violation) is computed from the
same logged action rows the main rank gate uses.
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


def block_nv(df, dataset, budget, seed, p_train_by_sel, n_cells):
    """Return {method: (ndcg_ev, violation_ev)} using the exact main-gate construction."""
    sub = df[df.dataset == dataset]
    items = sorted(sub["item"].unique())
    by_item = {it: sub[sub["item"] == it] for it in items}
    # O(1) lookup tables: (item, action) -> latency / ndcg10
    LAT, NDCG = {}, {}
    for it in items:
        r = by_item[it]
        LAT[it] = dict(zip(r["action"], r["latency_ms"].astype(float)))
        NDCG[it] = dict(zip(r["action"], r["ndcg10"].astype(float)))
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

    def lat(it, a):
        return LAT[it].get(a, float("nan"))
    def ndcg(it, a):
        return NDCG[it].get(a, float("nan"))
    def cs(it, a):
        l = LAT[it].get(a); n = NDCG[it].get(a)
        if l is None or n is None or math.isnan(l) or math.isnan(n): return float("nan")
        return constrained_score(float(n), float(l), budget)

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

    # per-item chosen action for each method, then realized violation + mean nDCG on ev
    def nv(chosen):
        lats = [lat(it, chosen(it)) for it in ev]
        nds  = [ndcg(it, chosen(it)) for it in ev]
        lats = [x for x in lats if not math.isnan(x)]
        nds  = [x for x in nds if not math.isnan(x)]
        viol = float(np.mean([1.0 if l > budget else 0.0 for l in lats])) if lats else 1.0
        return (float(np.mean(nds)) if nds else 0.0, viol)

    out = {}
    out[CANDIDATE] = nv(lambda it: deploy[ev_cell[it]])
    sb = max(actions, key=lambda a: float(np.nanmean([cs(it, a) for it in train])))
    out["StaticBest-cal"] = nv(lambda it: sb)
    for name, ops in SOTA_OPS.items():
        ops_a = [o for o in ops if o in actions]
        if not ops_a: continue
        def chooser(it, ops_a=ops_a):
            return max(ops_a, key=lambda o: (cs(it, o) if not math.isnan(cs(it, o)) else -1e9))
        out[name] = nv(chooser)
    p95 = {a: float(np.nanpercentile([lat(it, a) for it in train if not math.isnan(lat(it, a))], 95)) for a in actions}
    feasc = [a for a in actions if p95[a] <= budget] or actions
    cg = max(feasc, key=lambda a: float(np.nanmean([ndcg(it, a) for it in train])))
    out["CostGreedy-cal"] = nv(lambda it: cg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--action-rows", type=Path, default=Path("results/larger_workload/all_action_rows_multi.csv"))
    ap.add_argument("--output", type=Path, default=Path("results/sla_admission/admission_nv.csv"))
    ap.add_argument("--budgets-ms", default="80,120,160")
    ap.add_argument("--seeds", default="13,17,23,29,31")
    ap.add_argument("--n-cells", type=int, default=4)
    ap.add_argument("--p-train-by-sel", default="0.1:0.8,0.5:0.5,1.0:0.2")
    args = ap.parse_args()
    df = load_items(args.action_rows, None)
    budgets = [float(x) for x in args.budgets_ms.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    pts = {float(kv.split(":")[0]): float(kv.split(":")[1]) for kv in args.p_train_by_sel.split(",")}
    rows = []
    for d in sorted(df.dataset.unique()):
        for b in budgets:
            for s in seeds:
                r = block_nv(df, d, b, s, pts, args.n_cells)
                if not r: continue
                block = f"{d}|b={b:g}|seed={s}"
                for m, (nd, v) in r.items():
                    rows.append({"block": block, "budget": b, "method": m, "ndcg": nd, "violation": v})
    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print("blocks", out.block.nunique(), "methods", out.method.nunique(), "rows", len(out))


if __name__ == "__main__":
    main()
