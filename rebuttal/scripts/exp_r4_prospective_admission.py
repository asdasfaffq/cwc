#!/usr/bin/env python3
"""E4 (rebuttal): SLA admission control decided with pre-deployment information only.

Reviewer #2 W2/D2: "the admission utility appears to depend on the realized
evaluation-window violation rate, which is only known after execution. Please evaluate
admission using only information available before deployment. A direct comparison between
certificate-based admission and post-hoc admission using realized violations would
clarify."

The paper's Table VI scored a deployed window by its *realized* violation, so the
admission objective was retrospective. Here the admission DECISION is taken strictly
before execution, and only the payoff is settled afterwards. We reuse the paper's 45-block
protocol (3 datasets x 3 budgets x 5 seeds) and put every candidate deployment through
every admission rule:

  deployments  CWC (repaired) and the fixed / calibration-aware recipes it competes with
  rules        certificate     R_cert of Prop. 2 with the observable Gamma of E3
                               (only CWC can produce one)
               eb-nocorrection L_hat + empirical-Bernstein slack, shift ignored
                               (the strongest prospective signal a baseline has)
               point-estimate  L_hat on the calibration split: no slack, no shift term
               posthoc-oracle  the realized eval-window violation -- NOT deployable,
                               included only as the ceiling any rule could reach

  admit  <=>  signal <= tau
  payoff      withheld -> 0 ; admitted and realized <= tau -> nDCG@10 ;
              admitted and realized  > tau -> -penalty  (an SLA breach)

The operator-facing number is the breach rate *among admitted windows*: a sound
prospective rule keeps it at or below delta no matter what it costs in coverage.
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
from scripts.certified_window_compiler import constrained_score, load_items  # noqa: E402
from exp_r3_gamma_audit import eb_eps, observable_gamma, strata_labels  # noqa: E402

SOTA_OPS = {
    "Static-Qwen3": ["external_qwen3_embedding_06b_pre", "external_qwen3_embedding_06b_post"],
    "Static-SPLADE": ["external_splade_cocondenser_ensembledistil_pre",
                      "external_splade_cocondenser_ensembledistil_post"],
    "Static-ColBERTv2": ["external_colbertv2_pre", "external_colbertv2_post"],
    "Static-E5": ["dense_e5_base_v2_pre", "dense_e5_base_v2_post"],
}


def run_block(df, dataset, budget, p_train_by_sel, n_cells, delta, seed, coarsening):
    rng = np.random.default_rng(seed)
    items, TEL, LAT, ND, SEL, actions = build_tables(df)
    train_items, eval_items = [], []
    for it in items:
        (train_items if rng.random() < p_train_by_sel.get(SEL[it], 0.5) else eval_items).append(it)
    if len(train_items) < 2 * n_cells or len(eval_items) < n_cells:
        return []

    perm = rng.permutation(len(train_items))
    half = len(train_items) // 2
    sel_items = [train_items[i] for i in perm[:half]]
    cal_items = [train_items[i] for i in perm[half:]]
    fit = sel_items + eval_items
    scaler = StandardScaler().fit(np.vstack([TEL[it] for it in fit]))
    km = KMeans(n_clusters=n_cells, n_init=20, random_state=seed).fit(
        scaler.transform(np.vstack([TEL[it] for it in fit])))
    cell_of = {it: int(km.predict(scaler.transform(TEL[it].reshape(1, -1)))[0]) for it in items}

    def metrics(it, a):
        v = LAT.get((it, a))
        if v is None:
            return float("nan"), float("nan"), float("nan")
        ndv = ND[(it, a)]
        return v, ndv, constrained_score(ndv, v, budget)

    # ---- CWC (repaired: selection-split binding + selection-split abstain) ----
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

    strata = strata_labels(TEL, SEL, items, coarsening, ref_items=items)
    gamma, _ = observable_gamma(cell_of, strata, train_items, eval_items, n_cells)
    M = len(eval_items)
    mass = {c: sum(1 for it in eval_items if cell_of[it] == c) for c in range(n_cells)}

    # ---- competing deployments ----
    sb = max(actions, key=lambda a: float(np.nanmean([metrics(it, a)[2] for it in train_items])))
    p95 = {a: float(np.nanpercentile([metrics(it, a)[0] for it in train_items], 95)) for a in actions}
    feasc = [a for a in actions if p95[a] <= budget] or actions
    cg = max(feasc, key=lambda a: float(np.nanmean([metrics(it, a)[1] for it in train_items])))
    policies: dict = {"CWC": None, "StaticBest-cal": sb, "CostGreedy-cal": cg}
    for name, ops in SOTA_OPS.items():
        ops = [o for o in ops if o in actions]
        if ops:
            policies[name] = ops  # per-query best of that operator's pre/post variants

    rows = []
    for name, spec in policies.items():
        if spec is None:
            def chooser(it):
                return deploy[cell_of[it]]
        elif isinstance(spec, list):
            def chooser(it, ops=spec):
                return max(ops, key=lambda a: (metrics(it, a)[2]
                                               if not math.isnan(metrics(it, a)[2]) else -1e9))
        else:
            def chooser(it, a=spec):
                return a

        R_cert, R_eb, R_point = 0.0, 0.0, 0.0
        for c in range(n_cells):
            ci = [it for it in cal_items if cell_of[it] == c]
            ls = [1.0 if metrics(it, chooser(it))[0] > budget else 0.0 for it in ci]
            lh = float(np.mean(ls)) if ls else 1.0
            eps = eb_eps(ls, n_cells, delta) if ls else 1.0
            g = gamma[c]
            w = mass[c] / M
            R_cert += w * (1.0 if not np.isfinite(g) else min(g * (lh + eps), 1.0))
            R_eb += w * min(lh + eps, 1.0)
            R_point += w * lh

        lats = [metrics(it, chooser(it))[0] for it in eval_items]
        nds = [metrics(it, chooser(it))[1] for it in eval_items]
        realized = float(np.mean([1.0 if x > budget else 0.0 for x in lats if not math.isnan(x)]))
        rows.append({
            "dataset": dataset, "budget_ms": budget, "seed": seed, "method": name,
            "certificate": R_cert if name == "CWC" else float("nan"),
            "eb-nocorrection": R_eb, "point-estimate": R_point,
            "posthoc-oracle": realized, "realized": realized,
            "ndcg10": float(np.nanmean(nds))})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action-rows", type=Path,
                    default=ROOT / "results/larger_workload/all_action_rows_multi.csv")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("rebuttal/results/r4_prospective_admission"))
    ap.add_argument("--budgets-ms", default="80,120,160")
    ap.add_argument("--n-cells", type=int, default=4)
    ap.add_argument("--seeds", default="13,17,23,29,31")
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--taus", default="0.05,0.10,0.20")
    ap.add_argument("--penalty", type=float, default=1.0)
    ap.add_argument("--coarsening", default="sel")
    ap.add_argument("--p-train-by-sel", default="0.1:0.8,0.5:0.5,1.0:0.2")
    ap.add_argument("--pool", action="store_true",
                    help="treat all collections as one workload window (the regime where "
                         "per-cell calibration counts are large enough for the certificate "
                         "to be usable); otherwise block per collection as in Table VI")
    args = ap.parse_args()

    df_all = load_items(args.action_rows, None)
    p_train = {float(kv.split(":")[0]): float(kv.split(":")[1])
               for kv in args.p_train_by_sel.split(",") if kv.strip()}
    rows = []
    groups = ([("pooled", df_all)] if args.pool
              else [(ds, df_all[df_all["dataset"] == ds]) for ds in sorted(df_all["dataset"].unique())])
    for ds, df in groups:
        for b in [float(x) for x in args.budgets_ms.split(",")]:
            for s in [int(x) for x in args.seeds.split(",")]:
                rows.extend(run_block(df, ds, b, p_train, args.n_cells, args.delta, s,
                                      args.coarsening))
            print(f"  done {ds} budget={b:g}", flush=True)
    bl = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bl.to_csv(args.output_dir / "blocks.csv", index=False)

    out = []
    for tau in [float(x) for x in args.taus.split(",")]:
        for rule in ["certificate", "eb-nocorrection", "point-estimate", "posthoc-oracle"]:
            for method in bl["method"].unique():
                sub = bl[bl["method"] == method]
                sig = sub[rule]
                if sig.isna().all():
                    continue  # a certificate exists only for CWC
                admit = sig <= tau
                breach = admit & (sub["realized"] > tau)
                util = np.where(admit, np.where(sub["realized"] <= tau, sub["ndcg10"],
                                                -args.penalty), 0.0)
                out.append({
                    "tau": tau, "rule": rule, "prospective": rule != "posthoc-oracle",
                    "method": method, "blocks": int(len(sub)),
                    "admitted_frac": float(admit.mean()), "breaches": int(breach.sum()),
                    "breach_rate_among_admitted": float(breach.sum() / max(1, admit.sum())),
                    "mean_utility": float(util.mean()),
                    "mean_realized_violation": float(sub["realized"].mean()),
                    "mean_ndcg": float(sub["ndcg10"].mean())})
    res = pd.DataFrame(out)
    res.to_csv(args.output_dir / "admission.csv", index=False)
    for tau in sorted(res["tau"].unique()):
        print(f"\n=== tau = {tau} ===")
        print(res[res.tau == tau].sort_values(["rule", "method"]).to_string(index=False))
    (args.output_dir / "summary.json").write_text(json.dumps(
        {"n_blocks_per_method": int(bl.groupby("method").size().max()), "delta": args.delta,
         "penalty": args.penalty, "table": res.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
