#!/usr/bin/env python3
"""E1 (rebuttal): repair the certificate/implementation mismatch flagged by Reviewer #2.

Reviewer #2 (Availability) observed that `scripts/certified_window_compiler.py` chooses
between the bound plan and the fallback by comparing *calibration*-based certificate
terms, while Proposition 2 states the deployed plan is fixed on the selection split.
The observation is correct: the released implementation lets the calibration split both
select and certify, which breaks Assumption 1(iii).

This script implements three abstain rules on identical splits, seeds and cells, so the
cost of the repair can be read off directly:

  as-released   : abstain if cal-based term_fb < term_plan; eps uses log(2K/delta)  [INVALID]
  union         : same abstain rule, eps uses log(2*(2K)/delta)                     [VALID]
                  (union bound over the 2 candidate plans in each of the K cells, so the
                   bound holds simultaneously for both candidates and hence for whichever
                   the data-dependent rule picks)
  selection     : abstain decided on the SELECTION split only; calibration certifies the
                  already-fixed deployed plan; eps uses log(2K/delta)               [VALID]
                  (this is exactly the policy Proposition 2 describes)

Everything else (cells, plan binding, fallback, Gamma, masses, eval window) is shared, so
differences are attributable to the abstain rule alone.

Outputs raw per-run rows + a paired aggregate to --output-dir.
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
from scripts.certified_window_compiler import (  # noqa: E402
    TELEMETRY_FEATURES, constrained_score, estimate_rho, load_items, telemetry_vector,
)

MODES = ("as-released", "union", "selection")


def eb_eps(losses: list[float], k: int, delta: float) -> float:
    """Empirical-Bernstein one-sided deviation (Maurer & Pontil 2009).

    `k` is the number of simultaneous events the union bound must cover: K for one
    fixed plan per cell, 2K when both candidate plans per cell must be covered.
    """
    n = len(losses)
    if n <= 1:
        return 1.0
    var = float(np.var(np.asarray(losses, dtype=np.float64), ddof=1))
    logt = math.log(2.0 * k / delta)
    return math.sqrt(2.0 * var * logt / n) + 7.0 * logt / (3.0 * (n - 1))


def run_block(df: pd.DataFrame, budget: float, p_train_by_sel: dict[float, float],
              n_cells: int, delta: float, seed: int, rho_quantile: float) -> list[dict]:
    """One (budget, seed) block; returns one row per abstain mode (shared randomness)."""
    rng = np.random.default_rng(seed)
    items = sorted(df["item"].unique())
    by_item = {it: df[df["item"] == it] for it in items}

    train_items, eval_items = [], []
    for it in items:
        s = float(by_item[it].iloc[0]["selectivity"])
        p_tr = p_train_by_sel.get(s, 0.5)
        (train_items if rng.random() < p_tr else eval_items).append(it)
    if len(train_items) < 2 * n_cells or len(eval_items) < n_cells:
        return []

    perm = rng.permutation(len(train_items))
    half = len(train_items) // 2
    sel_items = [train_items[i] for i in perm[:half]]
    cal_items = [train_items[i] for i in perm[half:]]

    phi_fit_items = sel_items + eval_items
    feat_fit = np.vstack([telemetry_vector(by_item[it]) for it in phi_fit_items])
    scaler = StandardScaler().fit(feat_fit)
    km = KMeans(n_clusters=n_cells, n_init=20, random_state=seed).fit(scaler.transform(feat_fit))

    def cell_of(it: str) -> int:
        return int(km.predict(scaler.transform(telemetry_vector(by_item[it]).reshape(1, -1)))[0])

    sel_cell = {it: cell_of(it) for it in sel_items}
    cal_cell = {it: cell_of(it) for it in cal_items}
    eval_cell = {it: cell_of(it) for it in eval_items}
    actions = sorted(df["action"].unique())

    def metrics(it: str, action: str) -> tuple[float, float, float]:
        rows = by_item[it]
        r = rows[rows["action"] == action]
        if len(r) == 0:
            return float("nan"), float("nan"), float("nan")
        r = r.iloc[0]
        lat, nd = float(r["latency_ms"]), float(r["ndcg10"])
        return lat, nd, constrained_score(nd, lat, budget)

    # --- selection-split plan binding (shared by all modes) ---
    plan_by_cell: dict[int, str] = {}
    for c in range(n_cells):
        c_items = [it for it in sel_items if sel_cell[it] == c]
        if not c_items:
            plan_by_cell[c] = "bm25_pre"
            continue
        best_a, best_v = None, -1e9
        for a in actions:
            vals = [metrics(it, a)[2] for it in c_items]
            vals = [v for v in vals if not math.isnan(v)]
            if not vals:
                continue
            mv = float(np.mean(vals))
            if mv > best_v:
                best_v, best_a = mv, a
        plan_by_cell[c] = best_a or "bm25_pre"

    # --- fallback: feasibility-best action on the SELECTION split only ---
    # (the released code screened on the full train window, i.e. selection + calibration;
    #  restricting to the selection split is what Assumption 1(iii) actually requires)
    def sel_violation(a: str) -> float:
        vals = [1.0 if metrics(it, a)[0] > budget else 0.0
                for it in sel_items if not math.isnan(metrics(it, a)[0])]
        return float(np.mean(vals)) if vals else 1.0

    feasible = [a for a in actions if sel_violation(a) <= 1e-9]
    fallback = (max(feasible, key=lambda a: float(np.nanmean([metrics(it, a)[1] for it in sel_items])))
                if feasible else "bm25_pre")

    # --- calibration losses for both candidates (needed by all modes; the *use* differs) ---
    loss_plan: dict[int, list[float]] = {}
    loss_fb: dict[int, list[float]] = {}
    Lhat: dict[int, float] = {}
    Lhat_fb: dict[int, float] = {}
    for c in range(n_cells):
        c_items = [it for it in cal_items if cal_cell[it] == c]
        if not c_items:
            loss_plan[c] = []; loss_fb[c] = []; Lhat[c] = 1.0; Lhat_fb[c] = 1.0
            continue
        loss_plan[c] = [1.0 if metrics(it, plan_by_cell[c])[0] > budget else 0.0 for it in c_items]
        loss_fb[c] = [1.0 if metrics(it, fallback)[0] > budget else 0.0 for it in c_items]
        Lhat[c] = float(np.mean(loss_plan[c]))
        Lhat_fb[c] = float(np.mean(loss_fb[c]))

    # --- selection-split loss estimates (used by the `selection` abstain rule) ---
    sel_loss_plan: dict[int, list[float]] = {}
    sel_loss_fb: dict[int, list[float]] = {}
    for c in range(n_cells):
        c_items = [it for it in sel_items if sel_cell[it] == c]
        sel_loss_plan[c] = [1.0 if metrics(it, plan_by_cell[c])[0] > budget else 0.0 for it in c_items]
        sel_loss_fb[c] = [1.0 if metrics(it, fallback)[0] > budget else 0.0 for it in c_items]

    # --- Gamma (within-cell shift budget), from unlabeled telemetry only ---
    sel_feat = np.vstack([telemetry_vector(by_item[it]) for it in sel_items])
    sel_lab = np.array([sel_cell[it] for it in sel_items])
    eval_feat = np.vstack([telemetry_vector(by_item[it]) for it in eval_items])
    eval_lab = np.array([eval_cell[it] for it in eval_items])
    rho = estimate_rho(sel_feat, eval_feat, sel_lab, eval_lab, n_cells, seed, rho_quantile)

    M = len(eval_items)
    mass = {c: sum(1 for it in eval_items if eval_cell[it] == c) for c in range(n_cells)}

    def realized(policy) -> tuple[float, float]:
        viol, nd = [], []
        for it in eval_items:
            lat, ndcg, _ = metrics(it, policy(it))
            if math.isnan(lat):
                continue
            viol.append(1.0 if lat > budget else 0.0)
            nd.append(ndcg)
        return float(np.mean(viol)), float(np.mean(nd))

    out = []
    for mode in MODES:
        # k = number of simultaneous Hoeffding/EB events the union bound covers
        k_events = 2 * n_cells if mode == "union" else n_cells
        cert_term: dict[int, float] = {}
        deploy: dict[int, str] = {}
        abstained = 0
        for c in range(n_cells):
            eps_plan = eb_eps(loss_plan[c], k_events, delta) if loss_plan[c] else 1.0
            eps_fb = eb_eps(loss_fb[c], k_events, delta) if loss_fb[c] else 1.0
            term_plan = min(rho[c] * (Lhat[c] + eps_plan), 1.0)
            term_fb = min(rho[c] * (Lhat_fb[c] + eps_fb), 1.0)

            if mode in ("as-released", "union"):
                # data-dependent choice using calibration losses
                if term_fb < term_plan:
                    cert_term[c] = term_fb; deploy[c] = fallback; abstained += mass[c]
                else:
                    cert_term[c] = term_plan; deploy[c] = plan_by_cell[c]
            else:  # selection: decide on the selection split, then certify on calibration
                s_eps_plan = eb_eps(sel_loss_plan[c], n_cells, delta) if sel_loss_plan[c] else 1.0
                s_eps_fb = eb_eps(sel_loss_fb[c], n_cells, delta) if sel_loss_fb[c] else 1.0
                s_plan = min(rho[c] * (float(np.mean(sel_loss_plan[c])) + s_eps_plan), 1.0) \
                    if sel_loss_plan[c] else 1.0
                s_fb = min(rho[c] * (float(np.mean(sel_loss_fb[c])) + s_eps_fb), 1.0) \
                    if sel_loss_fb[c] else 1.0
                if s_fb < s_plan:
                    deploy[c] = fallback; abstained += mass[c]
                    cert_term[c] = term_fb
                else:
                    deploy[c] = plan_by_cell[c]
                    cert_term[c] = term_plan

        R_cert = sum((mass[c] / M) * cert_term[c] for c in range(n_cells))
        v_cert, q_cert = realized(lambda it: deploy[eval_cell[it]])
        staticbest = max(actions, key=lambda a: float(np.nanmean([metrics(it, a)[2] for it in train_items])))
        v_static, q_static = realized(lambda it: staticbest)
        out.append({
            "mode": mode, "valid": mode != "as-released",
            "budget_ms": budget, "seed": seed, "n_cells": n_cells,
            "n_eval": M, "n_cal": len(cal_items), "n_sel": len(sel_items),
            "rho_mean": float(np.mean(list(rho.values()))),
            "R_cert_bound": R_cert, "viol_certified": v_cert, "ndcg_certified": q_cert,
            "viol_static": v_static, "ndcg_static": q_static,
            "abstain_frac": abstained / M,
            "deploy_signature": "|".join(deploy[c] for c in range(n_cells)),
            "cert_upper_bounds_realized": bool(R_cert >= v_cert - 1e-12),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action-rows", type=Path,
                    default=ROOT / "results/larger_workload/all_action_rows_multi.csv")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--output-dir", type=Path, default=Path("rebuttal/results/r1_certificate_repair"))
    ap.add_argument("--budgets-ms", default="80,120,160")
    ap.add_argument("--n-cells", default="3,4,6,8,12")
    ap.add_argument("--seeds", default="13,17,23,29,31")
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--rho-quantile", type=float, default=0.9)
    ap.add_argument("--p-train-by-sel", default="0.1:0.8,0.5:0.5,1.0:0.2")
    args = ap.parse_args()

    df = load_items(args.action_rows, args.dataset or None)
    budgets = [float(x) for x in args.budgets_ms.split(",") if x.strip()]
    cells = [int(x) for x in args.n_cells.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    p_train = {float(kv.split(":")[0]): float(kv.split(":")[1])
               for kv in args.p_train_by_sel.split(",") if kv.strip()}

    rows = []
    for K in cells:
        for b in budgets:
            for s in seeds:
                rows.extend(run_block(df, b, p_train, K, args.delta, s, args.rho_quantile))
                print(f"  done K={K} budget={b:g} seed={s}", flush=True)
    res = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.output_dir / "raw_results.csv", index=False)

    agg = res.groupby(["n_cells", "budget_ms", "mode"]).agg(
        R_cert=("R_cert_bound", "mean"), viol=("viol_certified", "mean"),
        ndcg=("ndcg_certified", "mean"), abstain=("abstain_frac", "mean"),
        upper_bounds=("cert_upper_bounds_realized", "mean"), runs=("seed", "count"),
    ).reset_index()
    agg.to_csv(args.output_dir / "aggregate.csv", index=False)

    # Paired repair cost, per run, against the as-released configuration.
    piv = res.pivot_table(index=["n_cells", "budget_ms", "seed"], columns="mode",
                          values=["R_cert_bound", "viol_certified", "ndcg_certified", "abstain_frac"])
    summary = {}
    for m in ("union", "selection"):
        d_cert = (piv[("R_cert_bound", m)] - piv[("R_cert_bound", "as-released")])
        d_viol = (piv[("viol_certified", m)] - piv[("viol_certified", "as-released")])
        d_ndcg = (piv[("ndcg_certified", m)] - piv[("ndcg_certified", "as-released")])
        same_policy = float((res[res["mode"] == m].sort_values(["n_cells", "budget_ms", "seed"])
                             ["deploy_signature"].values ==
                             res[res["mode"] == "as-released"].sort_values(["n_cells", "budget_ms", "seed"])
                             ["deploy_signature"].values).mean())
        summary[m] = {
            "delta_R_cert_mean": float(d_cert.mean()), "delta_R_cert_max": float(d_cert.max()),
            "delta_viol_mean": float(d_viol.mean()), "delta_viol_max": float(d_viol.max()),
            "delta_ndcg_mean": float(d_ndcg.mean()), "delta_ndcg_min": float(d_ndcg.min()),
            "identical_deployed_policy_frac": same_policy,
        }
    summary["n_runs_per_mode"] = int(len(res) / len(MODES))
    summary["upper_bounds_realized_all_modes"] = bool(res["cert_upper_bounds_realized"].all())
    (args.output_dir / "repair_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
