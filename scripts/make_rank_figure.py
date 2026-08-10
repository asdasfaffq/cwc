#!/usr/bin/env python3
"""Generate the headline rank figure for the paper from the verified significance CSV.

Reads refine-logs/RANK_GATE_SIGNIFICANCE_20260531.csv and per-block ranks from
results/rg_final/rank_gate_cwc/rank_gate_input.csv, draws average rank (lower=better)
with standard-error bars, CWC highlighted, SOTA operators marked, and Holm-significance
annotations. Saves paper/figures/rank_gate.pdf.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SIG = ROOT / "refine-logs/RANK_GATE_SIGNIFICANCE_20260531.csv"
INP = ROOT / "results/rg_final/rank_gate_cwc/rank_gate_input.csv"
OUT = ROOT / "paper/figures/rank_gate.pdf"

CAND = "CWC-certified"
SOTA = {"Static-SPLADE", "Static-Qwen3", "Static-E5", "Static-ColBERTv2"}
LABEL = {
    "CWC-certified": "CWC (ours)", "Static-SPLADE": "SPLADE*", "Static-Qwen3": "Qwen3*",
    "Static-E5": "E5*", "Static-ColBERTv2": "ColBERTv2*",
    "StaticBest-cal": "StaticBest-cal", "CostGreedy-cal": "CostGreedy-cal",
}


def per_block_rank_se(inp: Path) -> dict[str, float]:
    df = pd.read_csv(inp)
    piv = df.pivot_table(index="block", columns="method", values="mean_score").dropna()
    # rank within each block: higher score -> rank 1
    ranks = piv.rank(axis=1, ascending=False, method="average")
    return {m: float(ranks[m].std(ddof=1) / np.sqrt(len(ranks))) for m in ranks.columns}


def main() -> None:
    sig = pd.read_csv(SIG).sort_values("avg_rank").reset_index(drop=True)
    se = per_block_rank_se(INP)
    methods = list(sig.method)
    ranks = list(sig.avg_rank)
    errs = [se.get(m, 0.0) for m in methods]

    def color(m):
        if m == CAND: return "#1b5e20"        # deep green
        if m in SOTA: return "#1565c0"         # blue (SOTA)
        return "#9e9e9e"                       # grey (calibrated selectors)

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ypos = np.arange(len(methods))[::-1]       # best (rank1) on top
    bars = ax.barh(ypos, ranks, xerr=errs, height=0.62,
                   color=[color(m) for m in methods],
                   error_kw=dict(ecolor="#444", capsize=3, lw=1), zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([LABEL.get(m, m) for m in methods], fontsize=10)
    ax.set_xlabel("Average rank over 45 blocks (lower is better)", fontsize=10)
    ax.set_xlim(0, max(ranks) + 1.6)
    ax.invert_yaxis()
    ax.grid(axis="x", ls=":", alpha=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # annotate Holm significance vs CWC
    for y, m, r, e in zip(ypos, methods, ranks, errs):
        if m == CAND:
            txt = f"{r:.2f}  (rank-1)"
        else:
            p = float(sig.loc[sig.method == m, "holm_p"].iloc[0])
            mark = "$\\ast\\ast$" if p < 1e-3 else ("$\\ast$" if p < 0.05 else "n.s.")
            txt = f"{r:.2f}   Holm $p$={p:.1e} {mark}"
        ax.text(r + e + 0.12, y, txt, va="center", fontsize=8.2, color="#222")

    # legend
    from matplotlib.patches import Patch
    leg = [Patch(fc="#1b5e20", label="CWC (ours)"),
           Patch(fc="#1565c0", label="SOTA operator ($\\ast$)"),
           Patch(fc="#9e9e9e", label="calibrated selector")]
    ax.legend(handles=leg, fontsize=8, loc="lower right", frameon=False)
    ax.set_title("CWC ranks first and beats all four SOTA operators (Holm-corrected)",
                 fontsize=10.5, pad=8)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print("saved", OUT)


if __name__ == "__main__":
    main()
