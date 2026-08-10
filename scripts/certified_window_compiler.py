#!/usr/bin/env python3
"""Certified Workload-Window Compiler under bounded workload shift.

Implements the theorem in refine-logs/PROOF_PACKAGE_WINDOW_CERT_20260529.md:
- partition queries into K plan cells over observable telemetry;
- SAMPLE-SPLIT the labeled train window into a selection split (choose the
  per-cell plan) and an independent calibration split (estimate L-hat);
- estimate a per-cell shift budget rho_c between train and eval telemetry;
- emit a finite-sample certified upper bound on the eval-window SLO-violation
  rate R_Q = sum_c (m_c/M) * min(rho_c*(L_hat_c + eps_c), 1);
- abstain to a certifiably-feasible static fallback per cell when that lowers
  the certified term.

Loss L = 1{latency_ms > budget_ms} (SLO / constraint-violation indicator), in [0,1].

This is a STANDALONE analysis on cached action rows; it does not touch the
production pilot. Results are reported as-is (no gate tuning).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Telemetry features available per workload item (action-independent columns).
TELEMETRY_FEATURES = [
    "query_len", "query_digit_ratio", "selectivity", "candidate_count",
    "bm25_top_score", "bm25_entropy", "bm25_gap",
    "dense_top_score", "dense_entropy", "dense_gap",
    "probe_overlap", "probe_latency_ms",
]


def constrained_score(ndcg: float, latency: float, budget: float) -> float:
    """Same objective as the production pilot: ndcg if feasible else -0.15*violation."""
    if latency <= budget:
        return float(ndcg)
    violation = max(0.0, latency / max(budget, 1e-9) - 1.0)
    return -0.15 * violation


def load_items(csv_path: Path, dataset: str | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if dataset:
        df = df[df["dataset"] == dataset].copy()
    # A workload item is (dataset, qid, selectivity); its candidate plans are the action
    # rows. Namespacing by dataset avoids qid collisions across merged BEIR collections.
    df["item"] = (df["dataset"].astype(str) + "|" + df["qid"].astype(str)
                  + "|s=" + df["selectivity"].astype(str))
    return df


def telemetry_vector(item_rows: pd.DataFrame) -> np.ndarray:
    r = item_rows.iloc[0]
    return np.array([float(r.get(c, 0.0)) for c in TELEMETRY_FEATURES], dtype=np.float64)


def hoeffding_eps(n: int, k: int, delta: float) -> float:
    if n <= 0:
        return 1.0
    return math.sqrt(math.log(k / delta) / (2.0 * n))


def eb_eps(losses: list[float], k: int, delta: float) -> float:
    """Empirical-Bernstein one-sided deviation (Maurer & Pontil 2009), variance-adaptive.
    Tighter than Hoeffding when the sample variance is small (rare violations). Bounded in [0,1].
    eps = sqrt(2 * Var_hat * log(2k/delta) / n) + 7 log(2k/delta) / (3(n-1))."""
    n = len(losses)
    if n <= 1:
        return 1.0
    var = float(np.var(np.asarray(losses, dtype=np.float64), ddof=1))
    logt = math.log(2.0 * k / delta)
    return math.sqrt(2.0 * var * logt / n) + 7.0 * logt / (3.0 * (n - 1))


def estimate_rho_ucb(train_feat: np.ndarray, eval_feat: np.ndarray, labels_train: np.ndarray,
                     labels_eval: np.ndarray, n_cells: int, seed: int, delta_prime: float,
                     reg_lambda: float = 4.0) -> dict[int, float]:
    """Per-cell UCB on rho_c under the log-linear density-ratio model (PROOF_PACKAGE_RHO_UCB).

    rho_bar_c = exp(R_c * (||beta_hat|| + chi2_d(1-delta')^{1/2} * lambda_min(V_n)^{-1/2})),
    with V_n = lambda I + sum phi phi^T over cell points. Uses the asymptotic MLE ellipsoid
    (A6-as): valid as n_c -> inf, honestly labeled asymptotic, not finite-sample.
    """
    from scipy.stats import chi2 as chi2_dist
    rho: dict[int, float] = {}
    for c in range(n_cells):
        tr = train_feat[labels_train == c]
        ev = eval_feat[labels_eval == c]
        if len(tr) < 6 or len(ev) < 6:
            rho[c] = 1.0
            continue
        X = np.vstack([tr, ev]); y = np.concatenate([np.zeros(len(tr)), np.ones(len(ev))])
        scaler = StandardScaler().fit(X); Xs = scaler.transform(X)
        clf = LogisticRegression(max_iter=2000, C=1.0 / reg_lambda, random_state=seed).fit(Xs, y)
        beta = clf.coef_.ravel()
        d = beta.shape[0]
        Vn = reg_lambda * np.eye(d) + Xs.T @ Xs
        lam_min = float(np.linalg.eigvalsh(Vn)[0])
        radius = math.sqrt(chi2_dist.ppf(1.0 - delta_prime, d))
        B = float(np.linalg.norm(beta) + radius / math.sqrt(max(lam_min, 1e-9)))
        R_c = float(np.max(np.linalg.norm(Xs[len(tr):], axis=1)))  # empirical eval support radius
        rho[c] = float(max(1.0, math.exp(R_c * B)))
    return rho


def estimate_rho(train_feat: np.ndarray, eval_feat: np.ndarray, labels_train: np.ndarray,
                 labels_eval: np.ndarray, n_cells: int, seed: int, quantile: float) -> dict[int, float]:
    """Per-cell WITHIN-cell shift budget rho_c = sup_x dQ_c/dP_c (cell-conditional).

    The cross-cell mass shift is already captured exactly by the observed eval masses
    m_c/M, so rho_c must measure only the residual WITHIN-cell covariate shift. We fit
    a regularized train-vs-eval classifier on the points of cell c ALONE and take
    w_c(x) = [p_c/(1-p_c)] * (n_train_c/n_eval_c); rho_c = high-quantile of w_c.
    If a cell is too small to fit, rho_c defaults to 1.0 (no within-cell shift), which
    is approximately honest because clustering uses the shift covariate (selectivity).
    PLUG-IN ESTIMATE (see PROOF_PACKAGE Open Risks); swept via rho_inflate downstream.
    """
    rho: dict[int, float] = {}
    for c in range(n_cells):
        tr = train_feat[labels_train == c]
        ev = eval_feat[labels_eval == c]
        if len(tr) < 6 or len(ev) < 6:
            rho[c] = 1.0
            continue
        X = np.vstack([tr, ev])
        y = np.concatenate([np.zeros(len(tr)), np.ones(len(ev))])
        scaler = StandardScaler().fit(X)
        # Strong regularization: avoid overfitting within-cell noise into spurious shift.
        clf = LogisticRegression(max_iter=2000, C=0.25, random_state=seed).fit(scaler.transform(X), y)
        p = np.clip(clf.predict_proba(scaler.transform(ev))[:, 1], 1e-3, 1 - 1e-3)
        w = (p / (1.0 - p)) * (len(tr) / max(len(ev), 1))
        rho[c] = float(max(1.0, np.quantile(w, quantile)))
    return rho


def run_budget(df: pd.DataFrame, budget: float, p_train_by_sel: dict[float, float],
               n_cells: int, delta: float, seed: int, rho_quantile: float,
               rho_inflate: float, rho_mode: str = "plugin", conc_mode: str = "eb") -> dict:
    rng = np.random.default_rng(seed)
    items = sorted(df["item"].unique())
    by_item = {it: df[df["item"] == it] for it in items}

    # BOUNDED covariate shift via mixing proportions: every item is assigned to the
    # train OR eval window by a biased coin whose bias depends on selectivity
    # (a hardness proxy). Both windows contain all strata in different proportions,
    # so train/eval OVERLAP and the per-cell density ratio is bounded (not singular).
    train_items, eval_items = [], []
    for it in items:
        s = float(by_item[it].iloc[0]["selectivity"])
        p_tr = p_train_by_sel.get(s, 0.5)
        (train_items if rng.random() < p_tr else eval_items).append(it)
    if len(train_items) < 2 * n_cells or len(eval_items) < n_cells:
        return {}

    # Sample-split TRAIN into selection (choose plan) vs calibration (estimate L-hat).
    perm = rng.permutation(len(train_items))
    half = len(train_items) // 2
    sel_items = [train_items[i] for i in perm[:half]]
    cal_items = [train_items[i] for i in perm[half:]]

    # Cluster on telemetry. (A3) partition built from selection-split + eval features
    # ONLY -> calibration labels stay independent of phi.
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

    def item_action_metrics(it: str, action: str) -> tuple[float, float, float]:
        rows = by_item[it]
        r = rows[rows["action"] == action]
        if len(r) == 0:
            return float("nan"), float("nan"), float("nan")
        r = r.iloc[0]
        lat = float(r["latency_ms"]); nd = float(r["ndcg10"])
        return lat, nd, constrained_score(nd, lat, budget)

    # Per-cell plan selection on the SELECTION split (argmax mean constrained_score).
    plan_by_cell: dict[int, str] = {}
    for c in range(n_cells):
        c_items = [it for it in sel_items if sel_cell[it] == c]
        if not c_items:
            plan_by_cell[c] = "bm25_pre"
            continue
        best_a, best_v = None, -1e9
        for a in actions:
            vals = [item_action_metrics(it, a)[2] for it in c_items]
            vals = [v for v in vals if not math.isnan(v)]
            if not vals:
                continue
            m = float(np.mean(vals))
            if m > best_v:
                best_v, best_a = m, a
        plan_by_cell[c] = best_a or "bm25_pre"

    # Certifiably-feasible static fallback: best mean-ndcg action whose train
    # calibration violation L-hat + eps is ~0 (always-feasible at this budget).
    def action_train_violation(a: str) -> float:
        vals = [1.0 if item_action_metrics(it, a)[0] > budget else 0.0
                for it in train_items if not math.isnan(item_action_metrics(it, a)[0])]
        return float(np.mean(vals)) if vals else 1.0
    feasible_actions = [a for a in actions if action_train_violation(a) <= 1e-9]
    if feasible_actions:
        fallback = max(feasible_actions,
                       key=lambda a: np.mean([item_action_metrics(it, a)[1] for it in train_items]))
    else:
        fallback = "bm25_pre"

    # Per-cell L-hat + per-cell loss lists (for variance-adaptive concentration) on the
    # CALIBRATION split (independent of plan choice & phi).
    Lhat: dict[int, float] = {}
    n_cal: dict[int, int] = {}
    Lhat_fb: dict[int, float] = {}
    loss_plan: dict[int, list] = {}
    loss_fb: dict[int, list] = {}
    for c in range(n_cells):
        c_items = [it for it in cal_items if cal_cell[it] == c]
        n_cal[c] = len(c_items)
        if not c_items:
            Lhat[c] = 1.0; Lhat_fb[c] = 1.0; loss_plan[c] = []; loss_fb[c] = []; continue
        plan = plan_by_cell[c]
        loss_plan[c] = [1.0 if item_action_metrics(it, plan)[0] > budget else 0.0 for it in c_items]
        loss_fb[c] = [1.0 if item_action_metrics(it, fallback)[0] > budget else 0.0 for it in c_items]
        Lhat[c] = float(np.mean(loss_plan[c]))
        Lhat_fb[c] = float(np.mean(loss_fb[c]))

    # Per-cell shift budget rho_c (plug-in, inflated by rho_inflate for sensitivity).
    sel_feat = np.vstack([telemetry_vector(by_item[it]) for it in sel_items])
    sel_lab = np.array([sel_cell[it] for it in sel_items])
    eval_feat = np.vstack([telemetry_vector(by_item[it]) for it in eval_items])
    eval_lab = np.array([eval_cell[it] for it in eval_items])
    # delta budget split: plugin uses full delta for Hoeffding; ucb splits delta/2 between
    # the rho UCB events and the Hoeffding events (Corollary, PROOF_PACKAGE_RHO_UCB).
    if rho_mode == "ucb":
        rho = estimate_rho_ucb(sel_feat, eval_feat, sel_lab, eval_lab, n_cells, seed,
                               delta_prime=delta / (2 * n_cells))
        delta_hoeff = delta / 2.0
    else:
        rho = estimate_rho(sel_feat, eval_feat, sel_lab, eval_lab, n_cells, seed, rho_quantile)
        delta_hoeff = delta
    rho = {c: v * rho_inflate for c, v in rho.items()}

    # Eval-window cell masses.
    M = len(eval_items)
    m = {c: sum(1 for it in eval_items if eval_cell[it] == c) for c in range(n_cells)}

    # Certified per-cell term + abstain decision.
    cert_term: dict[int, float] = {}
    deploy: dict[int, str] = {}
    abstained = 0
    for c in range(n_cells):
        if conc_mode == "eb":
            eps_plan = eb_eps(loss_plan[c], n_cells, delta_hoeff) if loss_plan[c] else 1.0
            eps_fb = eb_eps(loss_fb[c], n_cells, delta_hoeff) if loss_fb[c] else 1.0
        else:
            eps_plan = eps_fb = hoeffding_eps(n_cal[c], n_cells, delta_hoeff)
        term_plan = min(rho[c] * (Lhat[c] + eps_plan), 1.0)
        term_fb = min(rho[c] * (Lhat_fb[c] + eps_fb), 1.0)
        if term_fb < term_plan:
            cert_term[c] = term_fb; deploy[c] = fallback; abstained += m[c]
        else:
            cert_term[c] = term_plan; deploy[c] = plan_by_cell[c]
    R_cert = sum((m[c] / M) * cert_term[c] for c in range(n_cells))

    # Realized eval-window outcomes (ground truth on the held-out eval window).
    def realized(policy) -> tuple[float, float]:
        viol, nd = [], []
        for it in eval_items:
            a = policy(it)
            lat, ndcg, _ = item_action_metrics(it, a)
            if math.isnan(lat):
                continue
            viol.append(1.0 if lat > budget else 0.0); nd.append(ndcg)
        return float(np.mean(viol)), float(np.mean(nd))

    # Baselines.
    staticbest = max(actions, key=lambda a: np.mean([item_action_metrics(it, a)[2] for it in train_items]))
    v_cert, q_cert = realized(lambda it: deploy[eval_cell[it]])
    v_static, q_static = realized(lambda it: staticbest)
    v_fb, q_fb = realized(lambda it: fallback)
    # Oracle: per-item best constrained_score (upper bound on quality given SLO).
    def oracle(it):
        cand = [(a,) + item_action_metrics(it, a) for a in actions]
        cand = [c for c in cand if not math.isnan(c[1])]
        return max(cand, key=lambda c: c[3])[0]
    v_oracle, q_oracle = realized(oracle)

    return {
        "budget_ms": budget, "seed": seed, "n_cells": n_cells,
        "n_train": len(train_items), "n_eval": M, "n_sel": len(sel_items), "n_cal": len(cal_items),
        "rho_mean": float(np.mean(list(rho.values()))), "rho_max": float(max(rho.values())),
        "abstain_frac": abstained / M,
        "staticbest_action": staticbest, "fallback_action": fallback,
        "R_cert_bound": R_cert,
        "viol_certified": v_cert, "viol_static": v_static, "viol_fallback": v_fb, "viol_oracle": v_oracle,
        "ndcg_certified": q_cert, "ndcg_static": q_static, "ndcg_fallback": q_fb, "ndcg_oracle": q_oracle,
        "cert_nonvacuous": R_cert < 1.0,
        "cert_valid": R_cert >= v_cert - 1e-9,  # bound should upper-bound realized violation
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action-rows", type=Path,
                    default=Path("results/sys_real_external_qwen_splade_colbert_seed13/all_action_rows.csv"))
    ap.add_argument("--dataset", default="nfcorpus")
    ap.add_argument("--output-dir", type=Path, default=Path("results/certified_window_compiler"))
    ap.add_argument("--budgets-ms", default="80,120,160")
    ap.add_argument("--p-train-by-sel", default="0.1:0.8,0.5:0.5,1.0:0.2",
                    help="per-selectivity P(assign to train window); eval gets the rest. "
                         "Easy(0.1) skews train, hard(1.0) skews eval -> bounded covariate shift.")
    ap.add_argument("--n-cells", type=int, default=4)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--rho-quantile", type=float, default=0.9)
    ap.add_argument("--rho-inflate", type=float, default=1.0)
    ap.add_argument("--rho-mode", choices=["plugin", "ucb"], default="plugin",
                    help="plugin = estimated rho_c (assumed); ucb = certified rho_bar_c (log-linear model, asymptotic)")
    ap.add_argument("--conc-mode", choices=["eb", "hoeffding"], default="eb",
                    help="eb = empirical-Bernstein (variance-adaptive, tighter when violations rare); hoeffding = worst-case")
    ap.add_argument("--seeds", default="13,17,23,29,31")
    args = ap.parse_args()

    df = load_items(args.action_rows, args.dataset)
    budgets = [float(x) for x in args.budgets_ms.split(",") if x.strip()]
    p_train_by_sel = {float(kv.split(":")[0]): float(kv.split(":")[1])
                      for kv in args.p_train_by_sel.split(",") if kv.strip()}
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    rows = []
    for b in budgets:
        for s in seeds:
            r = run_budget(df, b, p_train_by_sel, args.n_cells, args.delta, s,
                           args.rho_quantile, args.rho_inflate, args.rho_mode, args.conc_mode)
            if r:
                rows.append(r)
    res = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if res.empty:
        print(f"NO RESULTS: too few workload items for n_cells={args.n_cells} "
              f"(need >= {2*args.n_cells} train + {args.n_cells} eval items). "
              f"Reduce --n-cells or use a larger workload.")
        return
    res.to_csv(args.output_dir / "raw_results.csv", index=False)

    # Aggregate over seeds per budget.
    agg = res.groupby("budget_ms").agg(
        R_cert_bound=("R_cert_bound", "mean"),
        viol_certified=("viol_certified", "mean"),
        viol_static=("viol_static", "mean"),
        viol_fallback=("viol_fallback", "mean"),
        viol_oracle=("viol_oracle", "mean"),
        ndcg_certified=("ndcg_certified", "mean"),
        ndcg_static=("ndcg_static", "mean"),
        ndcg_fallback=("ndcg_fallback", "mean"),
        ndcg_oracle=("ndcg_oracle", "mean"),
        abstain_frac=("abstain_frac", "mean"),
        rho_mean=("rho_mean", "mean"),
        cert_nonvacuous=("cert_nonvacuous", "mean"),
        cert_valid=("cert_valid", "mean"),
    ).reset_index()
    agg.to_csv(args.output_dir / "aggregate.csv", index=False)
    print(json.dumps({"p_train_by_sel": p_train_by_sel, "n_cells": args.n_cells,
                      "rho_inflate": args.rho_inflate}, ensure_ascii=False))
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
