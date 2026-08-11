#!/usr/bin/env python3
"""Paired significance tests on the end-to-end serving results.

Uses the same test family as the paper's gates: a one-sided Wilcoxon signed-rank test on
per-seed paired differences, Holm-corrected across the baselines compared at a budget.
The direction of each test is fixed in advance by what the method claims:
  SLO violation  -- CWC lower
  p95 latency    -- CWC lower
  throughput     -- CWC higher
  nDCG@10        -- two-sided (we claim no quality win, only a bounded quality cost)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

LOWER_IS_BETTER = {"slo_violation": True, "p95_ms": True, "throughput_qps": False}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path,
                    default=Path("rebuttal/results/r2_end2end_10seed/closed_loop_raw.csv"))
    ap.add_argument("--candidate", default="CWC")
    ap.add_argument("--output", type=Path,
                    default=Path("rebuttal/results/r2_end2end_10seed/significance.json"))
    args = ap.parse_args()

    d = pd.read_csv(args.raw)
    report = {}
    for b in sorted(d.budget_ms.unique()):
        sub = d[d.budget_ms == b]
        cand = sub[sub.policy == args.candidate].sort_values("seed")
        baselines = [p for p in sub.policy.unique()
                     if p not in (args.candidate, "Oracle-perItem")]
        block = {"n_seeds": int(len(cand)), "baselines": {}}
        for metric in ["slo_violation", "p95_ms", "throughput_qps", "ndcg10"]:
            raw_p = {}
            for base in baselines:
                bl = sub[sub.policy == base].sort_values("seed")
                if len(bl) != len(cand):
                    continue
                x, y = cand[metric].values, bl[metric].values
                if not np.any(x - y):
                    raw_p[base] = 1.0
                    continue
                if metric == "ndcg10":
                    alt = "two-sided"
                else:
                    alt = "less" if LOWER_IS_BETTER[metric] else "greater"
                raw_p[base] = float(stats.wilcoxon(x, y, alternative=alt).pvalue)
            adj = holm(raw_p)
            for base in raw_p:
                bl = sub[sub.policy == base].sort_values("seed")
                e = block["baselines"].setdefault(base, {})
                e[metric] = {
                    "candidate_mean": float(cand[metric].mean()),
                    "candidate_sd": float(cand[metric].std(ddof=1)),
                    "baseline_mean": float(bl[metric].mean()),
                    "baseline_sd": float(bl[metric].std(ddof=1)),
                    "wilcoxon_p": raw_p[base], "holm_p": adj[base],
                    "seeds_favouring_candidate": int(sum(
                        (cand[metric].values < bl[metric].values) if metric != "throughput_qps"
                        else (cand[metric].values > bl[metric].values))),
                }
        report[f"budget_{b:g}ms"] = block

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    for bk, block in report.items():
        print(f"\n=== {bk} (n={block['n_seeds']} seeds) ===")
        for base, metrics in block["baselines"].items():
            print(f"-- vs {base}")
            for m, v in metrics.items():
                print(f"   {m:16s} {v['candidate_mean']:.4f}+-{v['candidate_sd']:.4f} vs "
                      f"{v['baseline_mean']:.4f}+-{v['baseline_sd']:.4f}  "
                      f"Holm p={v['holm_p']:.4g}  ({v['seeds_favouring_candidate']}/"
                      f"{block['n_seeds']} seeds)")


if __name__ == "__main__":
    main()
