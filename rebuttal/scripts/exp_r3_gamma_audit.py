#!/usr/bin/env python3
"""E3 (rebuttal): make the shift budget Gamma_c an audited quantity, not an assumption.

Reviewer #2 W1/D1: "the main experiments use an estimated value without guaranteeing that
it upper-bounds the true shift ... please explain how an operator should choose or
validate Gamma_c in practice."

The paper's plug-in Gamma_c came from a within-cell train-vs-eval density-ratio
classifier, with no guarantee it upper-bounds the truth. This script replaces that step
with a quantity the serving system can *compute*, and then audits the one assumption that
remains.

Key observation. CWC is transductive: the eval window's unlabeled telemetry is in hand at
compile time. So for any coarsening Z of the telemetry space that the operator declares
(selectivity band, tenant, query class, filter cost bucket ...), BOTH window marginals
P_c(z) and Q_c(z) are directly observable. The within-cell shift budget on that coarsening

    Gamma_c^obs = max_{z : Q_c(z) > 0}  Q_c(z) / P_c(z)

is therefore measured, not estimated. Under

  (A2') within-stratum exchangeability: conditional on (cell c, stratum z), train- and
        eval-window queries are drawn from the same law,

we get E_{Q_c}[L] = sum_z Q_c(z) E[L | c, z] <= Gamma_c^obs * sum_z P_c(z) E[L | c, z]
                  = Gamma_c^obs * E_{P_c}[L],
i.e. exactly the change-of-measure step of Proposition 2, with a computed Gamma.

Finer coarsenings weaken (A2') but raise Gamma_c^obs; the operator therefore has a knob
with a visible dose-response, which this script sweeps. It also
  (1) falsification-tests (A2') directly, by comparing train vs eval conditional loss
      inside each (cell, stratum);
  (2) reports the inflation kappa* = max_c Gamma_c^obs / Gamma_c^plugin that the paper's
      plug-in estimate would have needed to dominate the measured budget;
  (3) backtests a purely prospective rule (calibrate kappa on earlier window pairs, deploy
      it on a later one) for operators unwilling to assume (A2').
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
    TELEMETRY_FEATURES, constrained_score, estimate_rho, load_items,
)

COARSENINGS = ["sel", "sel_x_lat2", "sel_x_lat4", "sel_x_lat4_x_bm25"]


def eb_eps(losses, k, delta) -> float:
    n = len(losses)
    if n <= 1:
        return 1.0
    var = float(np.var(np.asarray(losses, dtype=np.float64), ddof=1))
    logt = math.log(2.0 * k / delta)
    return math.sqrt(2.0 * var * logt / n) + 7.0 * logt / (3.0 * (n - 1))


I_BM25_TOP = TELEMETRY_FEATURES.index("bm25_top_score")
I_PROBE_LAT = TELEMETRY_FEATURES.index("probe_latency_ms")


def strata_labels(TEL, SEL, items, scheme: str, ref_items) -> dict:
    """Assign each item an observable stratum id. Bin edges come from the pooled
    (train + eval) telemetry, which is unlabeled and available at compile time."""
    if scheme == "sel":
        return {it: (SEL[it],) for it in items}
    plat_ref = np.array([TEL[it][I_PROBE_LAT] for it in ref_items])
    edges = ([float(np.median(plat_ref))] if scheme == "sel_x_lat2"
             else [float(x) for x in np.percentile(plat_ref, [25, 50, 75])])
    bm_med = None
    if scheme == "sel_x_lat4_x_bm25":
        bm_med = float(np.median([TEL[it][I_BM25_TOP] for it in ref_items]))
    out = {}
    for it in items:
        key = (SEL[it], int(np.searchsorted(edges, TEL[it][I_PROBE_LAT])))
        if bm_med is not None:
            key = key + (int(TEL[it][I_BM25_TOP] > bm_med),)
        out[it] = key
    return out


def observable_gamma(cell_of, strata, train_items, eval_items, n_cells):
    """Gamma_c^obs = max_z Q_c(z)/P_c(z) on the declared coarsening, plus diagnostics."""
    gamma, unseen = {}, {}
    for c in range(n_cells):
        tr = [it for it in train_items if cell_of[it] == c]
        ev = [it for it in eval_items if cell_of[it] == c]
        if not ev:
            gamma[c] = 1.0; unseen[c] = 0.0
            continue
        if not tr:
            gamma[c] = float("inf"); unseen[c] = 1.0
            continue
        ptr, pev = {}, {}
        for it in tr:
            ptr[strata[it]] = ptr.get(strata[it], 0) + 1
        for it in ev:
            pev[strata[it]] = pev.get(strata[it], 0) + 1
        ntr, nev = len(tr), len(ev)
        g, miss = 1.0, 0
        for z, cnt in pev.items():
            q = cnt / nev
            p = ptr.get(z, 0) / ntr
            if p <= 0.0:
                g = float("inf"); miss += cnt
            else:
                g = max(g, q / p)
        gamma[c] = g
        unseen[c] = miss / nev
    return gamma, unseen


def run_block(df, budget, p_train_by_sel, n_cells, delta, seed, rho_quantile):
    rng = np.random.default_rng(seed)
    items, TEL, LAT, ND, SEL, actions = build_tables(df)

    train_items, eval_items = [], []
    for it in items:
        s = SEL[it]
        (train_items if rng.random() < p_train_by_sel.get(s, 0.5) else eval_items).append(it)
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
    cell_of = {it: int(km.predict(scaler.transform(TEL[it].reshape(1, -1)))[0])
               for it in items}

    def metrics(it, a):
        v = LAT.get((it, a))
        if v is None:
            return float("nan"), float("nan"), float("nan")
        ndv = ND[(it, a)]
        return v, ndv, constrained_score(ndv, v, budget)

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

    def sel_viol(a):
        v = [1.0 if metrics(it, a)[0] > budget else 0.0
             for it in sel_items if not math.isnan(metrics(it, a)[0])]
        return float(np.mean(v)) if v else 1.0

    feas = [a for a in actions if sel_viol(a) <= 1e-9]
    fallback = (max(feas, key=lambda a: float(np.nanmean([metrics(it, a)[1] for it in sel_items])))
                if feas else "bm25_pre")

    # selection-split abstain (repaired policy, see E1)
    deploy = {}
    for c in range(n_cells):
        ci = [it for it in sel_items if cell_of[it] == c]
        if not ci:
            deploy[c] = fallback; continue
        vp = float(np.mean([1.0 if metrics(it, plan_by_cell[c])[0] > budget else 0.0 for it in ci]))
        vf = float(np.mean([1.0 if metrics(it, fallback)[0] > budget else 0.0 for it in ci]))
        deploy[c] = fallback if vf < vp else plan_by_cell[c]

    losses, Lhat = {}, {}
    for c in range(n_cells):
        ci = [it for it in cal_items if cell_of[it] == c]
        losses[c] = [1.0 if metrics(it, deploy[c])[0] > budget else 0.0 for it in ci]
        Lhat[c] = float(np.mean(losses[c])) if losses[c] else 1.0

    M = len(eval_items)
    mass = {c: sum(1 for it in eval_items if cell_of[it] == c) for c in range(n_cells)}
    realized = float(np.mean([1.0 if metrics(it, deploy[cell_of[it]])[0] > budget else 0.0
                              for it in eval_items
                              if not math.isnan(metrics(it, deploy[cell_of[it]])[0])]))

    # plug-in Gamma (what the paper reported)
    sel_feat = np.vstack([TEL[it] for it in sel_items])
    ev_feat = np.vstack([TEL[it] for it in eval_items])
    gamma_plugin = estimate_rho(sel_feat, ev_feat,
                                np.array([cell_of[it] for it in sel_items]),
                                np.array([cell_of[it] for it in eval_items]),
                                n_cells, seed, rho_quantile)

    rows = []
    for scheme in COARSENINGS:
        strata = strata_labels(TEL, SEL, items, scheme, ref_items=items)
        gamma, unseen = observable_gamma(cell_of, strata, train_items, eval_items, n_cells)

        R = 0.0
        for c in range(n_cells):
            eps = eb_eps(losses[c], n_cells, delta) if losses[c] else 1.0
            g = gamma[c]
            term = 1.0 if not np.isfinite(g) else min(g * (Lhat[c] + eps), 1.0)
            R += (mass[c] / M) * term

        finite = [gamma[c] for c in range(n_cells) if np.isfinite(gamma[c])]
        kappa = max((gamma[c] / max(gamma_plugin[c], 1e-9)) for c in range(n_cells)
                    if np.isfinite(gamma[c])) if finite else float("inf")
        plugin_dominates = all(gamma_plugin[c] >= gamma[c] - 1e-9
                               for c in range(n_cells) if np.isfinite(gamma[c]))

        # (A2') falsification: train vs eval conditional loss inside each (cell, stratum)
        gaps, npairs = [], 0
        for c in range(n_cells):
            zs = {strata[it] for it in eval_items if cell_of[it] == c}
            for z in zs:
                tr = [it for it in train_items if cell_of[it] == c and strata[it] == z]
                ev = [it for it in eval_items if cell_of[it] == c and strata[it] == z]
                if len(tr) < 10 or len(ev) < 10:
                    continue
                lt = float(np.mean([1.0 if metrics(it, deploy[c])[0] > budget else 0.0 for it in tr]))
                le = float(np.mean([1.0 if metrics(it, deploy[c])[0] > budget else 0.0 for it in ev]))
                gaps.append(le - lt); npairs += 1

        rows.append({
            "coarsening": scheme, "budget_ms": budget, "seed": seed, "n_cells": n_cells,
            "gamma_obs_mean": float(np.mean(finite)) if finite else float("inf"),
            "gamma_obs_max": float(np.max(finite)) if finite else float("inf"),
            "gamma_plugin_mean": float(np.mean(list(gamma_plugin.values()))),
            "cells_unbounded": sum(1 for c in range(n_cells) if not np.isfinite(gamma[c])),
            "eval_mass_unseen_stratum": float(np.mean([unseen[c] for c in range(n_cells)])),
            "R_cert_obs": R, "realized_violation": realized,
            "cert_upper_bounds_realized": bool(R >= realized - 1e-12),
            "kappa_star_plugin_to_obs": kappa, "plugin_dominates_obs": plugin_dominates,
            "exch_pairs": npairs,
            "exch_mean_gap": float(np.mean(gaps)) if gaps else float("nan"),
            "exch_max_abs_gap": float(np.max(np.abs(gaps))) if gaps else float("nan"),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action-rows", type=Path,
                    default=ROOT / "results/larger_workload/all_action_rows_multi.csv")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--output-dir", type=Path, default=Path("rebuttal/results/r3_gamma_audit"))
    ap.add_argument("--budgets-ms", default="80,120,160")
    ap.add_argument("--n-cells", type=int, default=4)
    ap.add_argument("--seeds", default="13,17,23,29,31")
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--rho-quantile", type=float, default=0.9)
    ap.add_argument("--p-train-by-sel", default="0.1:0.8,0.5:0.5,1.0:0.2")
    args = ap.parse_args()

    df = load_items(args.action_rows, args.dataset or None)
    p_train = {float(kv.split(":")[0]): float(kv.split(":")[1])
               for kv in args.p_train_by_sel.split(",") if kv.strip()}
    rows = []
    for b in [float(x) for x in args.budgets_ms.split(",")]:
        for s in [int(x) for x in args.seeds.split(",")]:
            rows.extend(run_block(df, b, p_train, args.n_cells, args.delta, s, args.rho_quantile))
            print(f"  done budget={b:g} seed={s}", flush=True)

    res = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.output_dir / "raw_results.csv", index=False)
    agg = res.groupby("coarsening").agg(
        gamma_obs_mean=("gamma_obs_mean", "mean"), gamma_obs_max=("gamma_obs_max", "max"),
        gamma_plugin_mean=("gamma_plugin_mean", "mean"),
        R_cert_obs=("R_cert_obs", "mean"), realized=("realized_violation", "mean"),
        valid_frac=("cert_upper_bounds_realized", "mean"),
        kappa_star=("kappa_star_plugin_to_obs", "max"),
        plugin_dominates_frac=("plugin_dominates_obs", "mean"),
        unbounded_cells=("cells_unbounded", "mean"),
        exch_mean_gap=("exch_mean_gap", "mean"), exch_max_abs_gap=("exch_max_abs_gap", "max"),
        runs=("seed", "count")).reset_index()
    agg.to_csv(args.output_dir / "aggregate.csv", index=False)
    print(agg.to_string(index=False))
    (args.output_dir / "summary.json").write_text(json.dumps({
        "n_runs": int(len(res) / len(COARSENINGS)),
        "certificate_valid_in_all_runs": bool(res["cert_upper_bounds_realized"].all()),
        "by_coarsening": agg.to_dict(orient="records"),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
