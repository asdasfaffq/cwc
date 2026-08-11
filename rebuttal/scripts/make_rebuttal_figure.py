#!/usr/bin/env python3
"""One figure summarising the evidence added for the reviewer response.

(a) End-to-end serving at the binding SLO: realized SLO violation vs throughput, measured by
    executing every plan on a real 100,785-document hybrid stack (R1-W2, R2-D4, R3-D2).
(b) Certificate tightness vs labeled-window size, with the SLA levels it supports (R2-W3/D3).
(c) Drift dose-response: what workload-mix drift does to fixed recipes (R3-D1).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
plt.rcParams.update({
    "pdf.fonttype": 42, "font.size": 8.5, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.7, "legend.frameon": False,
})
C = {"CWC": "#c0392b", "StaticBest-cal": "#2c6fbb", "CostGreedy-cal": "#5b8db8",
     "Static-BestQuality": "#7f8c8d", "Static-Fallback": "#95a5a6",
     "Oracle-perItem": "#27ae60"}


def panel_e2e(ax, path: Path, budget: float) -> None:
    d = pd.read_csv(path)
    d = d[np.isclose(d.budget_ms, budget)]
    shown = d[d.policy != "Oracle-perItem"]
    lo, hi = shown.ndcg10.min(), shown.ndcg10.max()
    for _, r in shown.iterrows():
        # marker area encodes retrieval quality, so a cheap-but-worse plan cannot look
        # dominant just by sitting in the fast, feasible corner
        frac = (r.ndcg10 - lo) / max(hi - lo, 1e-9)
        ax.scatter(r.throughput_qps, r.slo_violation, s=40 + 190 * frac, zorder=3,
                   color=C.get(r.policy, "#888"),
                   marker="*" if r.policy == "CWC" else "o",
                   edgecolor="white", linewidth=0.7)
        ax.annotate(f"{r.policy.replace('-cal', '').replace('Static-', '')}\n"
                    f"nDCG {r.ndcg10:.3f}",
                    (r.throughput_qps, r.slo_violation), textcoords="offset points",
                    xytext=(7, 3), fontsize=6.6, color=C.get(r.policy, "#555"))
    ax.set_xlabel("throughput (queries/s)")
    ax.set_ylabel("realized SLO violation")
    ax.set_title(f"(a) live execution, {budget:g} ms SLO, 100,785-doc stack\n"
                 f"marker area = nDCG@10",
                 loc="left", fontsize=8.5)
    ax.set_xlim(0, shown.throughput_qps.max() * 1.35)
    ax.set_ylim(-0.03, max(0.45, d.slo_violation.max() * 1.15))


def panel_tightness(ax, path: Path) -> None:
    a = pd.read_csv(path)
    a = a[(a.n_cells == 2) & np.isclose(a.delta, 0.1)].sort_values("n_cal_mean")
    ax.plot(a.n_cal_mean, a.R_cert, "o-", color="#c0392b", lw=1.4, ms=4.5,
            label="certified bound $R_{cert}$")
    ax.plot(a.n_cal_mean, a.realized, "s--", color="#2c6fbb", lw=1.1, ms=3.5,
            label="realized violation")
    for tau, style in [(0.20, ":"), (0.10, "-."), (0.05, "--")]:
        ax.axhline(tau, color="#999", lw=0.7, ls=style)
        ax.annotate(f"SLA {tau:g}", (a.n_cal_mean.max(), tau), fontsize=6.5,
                    color="#777", va="bottom", ha="right")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("labeled queries per cell $n_c$")
    ax.set_ylabel("violation rate")
    ax.set_title("(b) the bound is usable at a 5% SLA\nmeasured $\\Gamma$, valid at every point",
                 loc="left", fontsize=8.5)
    ax.legend(fontsize=7, loc="lower left")


def panel_drift(ax, path: Path, budget: float) -> None:
    a = pd.read_csv(path)
    a = a[np.isclose(a.budget_ms, budget)]
    show = ["StaticBest-cal", "Fixed-rerank@100", "SOTA-Qwen3", "CWC", "Dense-only"]
    col = {"StaticBest-cal": "#2c6fbb", "Fixed-rerank@100": "#7f8c8d",
           "SOTA-Qwen3": "#8e6fbb", "CWC": "#c0392b", "Dense-only": "#95a5a6"}
    for pol in show:
        s = a[a.policy == pol].sort_values("drift_d")
        if s.empty:
            continue
        ax.plot(s.drift_d, s.violation, "o-", ms=3.5, lw=1.4 if pol == "CWC" else 1.0,
                color=col[pol], label=pol)
    ax.set_xlabel("workload-mix drift $d$")
    ax.set_ylabel("SLO violation")
    ax.set_title("(c) fixed recipes are brittle\nmeasured, not asserted", loc="left",
                 fontsize=8.5)
    ax.legend(fontsize=6.6, loc="upper left")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", type=Path,
                    default=ROOT / "rebuttal/results/r2_end2end_10seed/closed_loop_aggregate.csv")
    ap.add_argument("--e2e-budget", type=float, default=100.0)
    ap.add_argument("--tightness", type=Path,
                    default=ROOT / "rebuttal/results/r5_tightness_map/aggregate.csv")
    ap.add_argument("--drift", type=Path,
                    default=ROOT / "rebuttal/results/r6_drift_motivation/aggregate.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "rebuttal/figures_legible/rebuttal_evidence")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.1))
    panel_e2e(axes[0], args.e2e, args.e2e_budget)
    panel_tightness(axes[1], args.tightness)
    panel_drift(axes[2], args.drift, 80.0)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(args.out) + ".png", dpi=200, bbox_inches="tight")
    print("wrote", args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
