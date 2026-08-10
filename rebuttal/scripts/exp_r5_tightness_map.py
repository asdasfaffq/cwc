#!/usr/bin/env python3
"""E5 (rebuttal): where is the certificate tight enough to serve a real SLA?

Reviewer #2 W3/D3: "a bound around 0.30 is difficult to use for tight SLOs ... it would be
useful to report settings where the certificate is tight enough to satisfy realistic SLA
thresholds, not only where it upper-bounds the realized violation."

The bound decomposes as R_cert ~ sum_c (m_c/M) * Gamma_c * (L_hat_c + eps(n_c)), so it has
a *floor* Gamma * L_hat that no amount of calibration data removes, plus a statistical term
that shrinks like 1/sqrt(n_c). Two separate things therefore have to hold for an SLA level
tau to be certifiable:

    (F) floor condition       Gamma * L_hat  <  tau      (the deployed plan must actually
                                                          be that safe under shift)
    (S) sample-size condition n_c  >~  (Gamma * c(delta) / (tau - Gamma*L_hat))^2

This script measures both. It sweeps the labeled-window size by subsampling the workload,
records R_cert and its floor/statistical split, and reports for each SLA level tau the
smallest measured n_c at which R_cert <= tau -- plus, when the sweep does not reach tau,
the n_c the fitted 1/sqrt(n) law implies (clearly labelled as an extrapolation).

Gamma is the observable-coarsening budget of E3 (measured, not assumed).
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_r3_gamma_audit import eb_eps, observable_gamma, strata_labels  # noqa: E402


def run_block(df, items, budget, p_train_by_sel, n_cells, delta, seed, coarsening):
    rng = np.random.default_rng(seed)
    _items, TEL, LAT, ND, SEL, actions = build_tables(df)
    train_items, eval_items = [], []
    for it in items:
        s = SEL[it]
        (train_items if rng.random() < p_train_by_sel.get(s, 0.5) else eval_items).append(it)
    if len(train_items) < 4 * n_cells or len(eval_items) < 2 * n_cells:
        return None

    perm = rng.permutation(len(train_items))
    half = len(train_items) // 2
    sel_items = [train_items[i] for i in perm[:half]]
    cal_items = [train_items[i] for i in perm[half:]]
    fit = sel_items + eval_items
    scaler = StandardScaler().fit(np.vstack([TEL[it] for it in fit]))
    km = KMeans(n_clusters=n_cells, n_init=10, random_state=seed).fit(
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

    losses, Lhat, ncal = {}, {}, {}
    for c in range(n_cells):
        ci = [it for it in cal_items if cell_of[it] == c]
        ncal[c] = len(ci)
        losses[c] = [1.0 if metrics(it, deploy[c])[0] > budget else 0.0 for it in ci]
        Lhat[c] = float(np.mean(losses[c])) if losses[c] else 1.0

    M = len(eval_items)
    mass = {c: sum(1 for it in eval_items if cell_of[it] == c) for c in range(n_cells)}
    strata = strata_labels(TEL, SEL, items, coarsening, ref_items=items)
    gamma, _ = observable_gamma(cell_of, strata, train_items, eval_items, n_cells)

    R, floor, stat = 0.0, 0.0, 0.0
    for c in range(n_cells):
        eps = eb_eps(losses[c], n_cells, delta) if losses[c] else 1.0
        g = gamma[c]
        w = mass[c] / M
        if not np.isfinite(g):
            R += w; floor += w; continue
        R += w * min(g * (Lhat[c] + eps), 1.0)
        floor += w * min(g * Lhat[c], 1.0)
        stat += w * min(g * eps, 1.0)
    realized = float(np.mean([1.0 if metrics(it, deploy[cell_of[it]])[0] > budget else 0.0
                              for it in eval_items
                              if not math.isnan(metrics(it, deploy[cell_of[it]])[0])]))
    ndcg = float(np.nanmean([metrics(it, deploy[cell_of[it]])[1] for it in eval_items]))
    return {"budget_ms": budget, "seed": seed, "n_cells": n_cells, "delta": delta,
            "n_items": len(items), "n_cal_mean": float(np.mean(list(ncal.values()))),
            "n_cal_min": int(min(ncal.values())),
            "gamma_mean": float(np.mean([g for g in gamma.values() if np.isfinite(g)]) or 1.0),
            "R_cert": R, "R_floor": floor, "R_statistical": stat,
            "Lhat_mean": float(np.mean(list(Lhat.values()))),
            "realized_violation": realized, "ndcg10": ndcg,
            "valid": bool(R >= realized - 1e-12)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action-rows", type=Path,
                    default=ROOT / "results/larger_workload_5ds/all_action_rows_5ds.csv")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--output-dir", type=Path, default=Path("rebuttal/results/r5_tightness_map"))
    ap.add_argument("--budget-ms", type=float, default=80.0)
    ap.add_argument("--fractions", default="0.1,0.2,0.35,0.5,0.75,1.0")
    ap.add_argument("--n-cells", default="2,3,4")
    ap.add_argument("--deltas", default="0.1,0.05")
    ap.add_argument("--seeds", default="13,17,23,29,31")
    ap.add_argument("--taus", default="0.20,0.10,0.05,0.01")
    ap.add_argument("--coarsening", default="sel")
    ap.add_argument("--p-train-by-sel", default="0.1:0.8,0.5:0.5,1.0:0.2")
    args = ap.parse_args()

    df = load_items(args.action_rows, args.dataset or None)
    all_items = sorted(df["item"].unique())
    p_train = {float(kv.split(":")[0]): float(kv.split(":")[1])
               for kv in args.p_train_by_sel.split(",") if kv.strip()}

    rows = []
    for frac in [float(x) for x in args.fractions.split(",")]:
        for K in [int(x) for x in args.n_cells.split(",")]:
            for delta in [float(x) for x in args.deltas.split(",")]:
                for seed in [int(x) for x in args.seeds.split(",")]:
                    rs = np.random.default_rng(1000 + seed)
                    n = max(4 * K + 2, int(round(frac * len(all_items))))
                    sub = sorted(rs.choice(all_items, size=min(n, len(all_items)), replace=False))
                    sdf = df[df["item"].isin(sub)]
                    r = run_block(sdf, sub, args.budget_ms, p_train, K, delta, seed, args.coarsening)
                    if r:
                        r["fraction"] = frac
                        rows.append(r)
            print(f"  done frac={frac} K={K}", flush=True)

    res = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.output_dir / "raw_results.csv", index=False)
    agg = res.groupby(["fraction", "n_cells", "delta"]).agg(
        n_cal_mean=("n_cal_mean", "mean"), gamma_mean=("gamma_mean", "mean"),
        R_cert=("R_cert", "mean"), R_floor=("R_floor", "mean"),
        R_statistical=("R_statistical", "mean"), Lhat=("Lhat_mean", "mean"),
        realized=("realized_violation", "mean"), ndcg=("ndcg10", "mean"),
        valid=("valid", "mean")).reset_index()
    agg.to_csv(args.output_dir / "aggregate.csv", index=False)
    print(agg.to_string(index=False))

    # For each SLA level: the smallest configuration actually measured that certifies it,
    # and -- if none does -- the 1/sqrt(n) extrapolation from the measured statistical term.
    report = {}
    for tau in [float(x) for x in args.taus.split(",")]:
        ok = agg[agg["R_cert"] <= tau].sort_values("n_cal_mean")
        entry: dict = {"tau": tau, "achieved": bool(len(ok) > 0)}
        if len(ok):
            b = ok.iloc[0]
            entry.update({"n_cal_mean": float(b["n_cal_mean"]), "K": int(b["n_cells"]),
                          "delta": float(b["delta"]), "R_cert": float(b["R_cert"]),
                          "realized": float(b["realized"]), "ndcg": float(b["ndcg"])})
        else:
            best = agg.sort_values("R_cert").iloc[0]
            floor = float(best["R_floor"])
            entry.update({"best_R_cert": float(best["R_cert"]), "floor": floor,
                          "K": int(best["n_cells"]), "delta": float(best["delta"]),
                          "n_cal_mean": float(best["n_cal_mean"])})
            if floor >= tau:
                entry["blocked_by"] = "floor"
                entry["note"] = ("Gamma*L_hat already exceeds tau: no calibration size can "
                                 "certify this level for this deployed plan")
            else:
                scale = (float(best["R_statistical"]) / max(tau - floor, 1e-9)) ** 2
                entry["blocked_by"] = "sample_size"
                entry["extrapolated_n_cal"] = float(best["n_cal_mean"] * scale)
                entry["note"] = "1/sqrt(n) extrapolation from the measured statistical term"
        report[str(tau)] = entry
    (args.output_dir / "sla_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
