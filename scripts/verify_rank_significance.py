#!/usr/bin/env python3
"""Clean, self-contained Wilcoxon+Holm verification of the CWC rank gate.
Writes a CSV so results are read structurally, not via stdout."""
import sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

inp = sys.argv[1]
out = sys.argv[2]
cand = "CWC-certified"
df = pd.read_csv(inp)
piv = df.pivot_table(index="block", columns="method", values="mean_score")
others = [m for m in piv.columns if m != cand]
rows = []
for m in others:
    d = (piv[cand] - piv[m]).dropna()
    try:
        _, p = wilcoxon(d, alternative="greater")
    except Exception:
        p = 1.0
    rows.append({"method": m, "mean_delta": float(piv[cand].mean() - piv[m].mean()),
                 "wilcoxon_p": float(p), "win_blocks": int((d > 0).sum()), "n_blocks": int(len(d))})
ps = np.array([r["wilcoxon_p"] for r in rows])
order = np.argsort(ps)
mtot = len(ps)
holm = [0.0] * mtot
prev = 0.0
for rank, idx in enumerate(order):
    adj = min(1.0, (mtot - rank) * ps[idx])
    adj = max(adj, prev)
    prev = adj
    holm[idx] = adj
# average ranks
ranks = {m: [] for m in piv.columns}
for _, row in piv.iterrows():
    o = row.sort_values(ascending=False)
    for rk, (m, _) in enumerate(o.items(), 1):
        ranks[m].append(rk)
avg = {m: float(np.mean(v)) for m, v in ranks.items()}
for r, h in zip(rows, holm):
    r["holm_p"] = float(h)
    r["avg_rank"] = avg[r["method"]]
    r["sig_0.05"] = bool(h < 0.05)
res = pd.DataFrame(rows).sort_values("avg_rank")
res.loc[len(res)] = {"method": cand, "mean_delta": 0.0, "wilcoxon_p": np.nan, "win_blocks": 0,
                     "n_blocks": int(len(piv)), "holm_p": np.nan, "avg_rank": avg[cand], "sig_0.05": False}
res = res.sort_values("avg_rank")
res.to_csv(out, index=False)
summary = {"candidate": cand, "cand_avg_rank": avg[cand], "n_blocks": int(len(piv)),
           "n_baselines": len(others), "holm_sig_beats": int(sum(1 for h in holm if h < 0.05))}
pd.DataFrame([summary]).to_csv(out.replace(".csv", "_summary.csv"), index=False)
