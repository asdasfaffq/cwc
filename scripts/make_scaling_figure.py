#!/usr/bin/env python3
"""Certificate tightening figure: R_cert vs per-cell calibration count n_c.
Real data only: 3-ds same-config subsample (NC_SWEEP, K=4) + 5-ds K-sweep (cert_5ds_k*).
Shows the ~1/sqrt(n_c) shrinkage and the 5-ds bound dipping below static realized violation.
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

# 3-dataset, fixed K=4, subsampled workload (same config) -> clean n_c sweep
nc = pd.read_csv(ROOT/"refine-logs/NC_SWEEP_20260601.csv")
x3 = (nc["n_cal"]/4).values          # per-cell calibration count
y3 = nc["R_cert"].values

# 5-dataset scaling probe, K-sweep (n_cal=1376 total)
x5, y5 = [], []
for K in [2,3,4,6]:
    d = pd.read_csv(ROOT/f"results/cert_5ds_k{K}/raw_results.csv")
    r = d[d.budget_ms==80]
    x5.append(r.n_cal.mean()/K); y5.append(r.R_cert_bound.mean())
x5, y5 = np.array(x5), np.array(y5)
static5 = 0.063   # 5-ds static realized violation
static3 = 0.112   # 3-ds canonical static realized violation

plt.rcParams.update({"font.size":9,"font.family":"serif","axes.linewidth":0.7})
fig, ax = plt.subplots(figsize=(3.3,2.45))
# 1/sqrt(n) guides (faint), anchored to each series' largest-n point
ng = np.linspace(30, 760, 100)
ax.plot(ng, y3.min()*np.sqrt(x3.max()/ng), color="0.7", ls=":", lw=0.9, zorder=1)
ax.plot(ng, y5.min()*np.sqrt(x5.max()/ng), color="0.7", ls=":", lw=0.9, zorder=1,
        label=r"$\propto 1/\sqrt{n_c}$ guide")
ax.plot(x3, y3, "o-", color="#c0392b", ms=5, lw=1.3, label="3-dataset (subsample, $K{=}4$)")
ax.plot(x5, y5, "s-", color="#2c3e50", ms=5, lw=1.3, label="5-dataset probe ($K$-sweep)")
ax.axhline(static5, color="#2c3e50", ls="--", lw=0.9)
ax.axhline(static3, color="#c0392b", ls="--", lw=0.9)
# place labels in the empty lower-left region (left points are high, so this band is clear)
ax.text(34, static3+0.008, "static (3-ds) 0.112", ha="left", va="bottom", fontsize=6.8, color="#c0392b")
ax.text(34, static5-0.030, "static (5-ds) 0.063", ha="left", va="bottom", fontsize=6.8, color="#2c3e50")
ax.set_xscale("log")
ax.set_xlabel(r"per-cell calibration count $n_c$")
ax.set_ylabel(r"certified bound $R_{\mathrm{cert}}$")
ax.set_ylim(0, 0.72)
ax.set_xticks([40,60,100,200,400,700])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(fontsize=6.8, frameon=False, loc="upper right")
fig.tight_layout(pad=0.3)
out = ROOT/"paper/figures/scaling_curve.pdf"
fig.savefig(out, bbox_inches="tight"); print("wrote", out)
