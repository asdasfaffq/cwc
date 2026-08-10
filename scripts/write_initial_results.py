#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.results.read_text(encoding="utf-8"))
    dataset = data.get("dataset", {})
    completed = sorted(k for k in data if k.startswith("R"))
    lines = [
        "# Initial Experiment Results",
        "",
        "**Plan**: `refine-logs/EXPERIMENT_PLAN.md`",
        f"**Dataset**: `{dataset.get('name')}` source `{dataset.get('source')}`",
        "",
        "## Results by Milestone",
        "",
        "### M0-M2 Pilot",
        "",
        "| Run | Status / Key Metric |",
        "|---|---|",
    ]
    for run_id in completed:
        lines.append(f"| {run_id} | {summarize(data[run_id])} |")
    ready = "NO"
    if "R012" in data and data["R012"].get("mean_regret", 1e9) < data.get("R009", {}).get("mean_regret", -1e9):
        ready = "PARTIAL"
    lines += [
        "",
        "## Summary",
        "",
        f"- Completed pilot runs: {len(completed)}",
        f"- Ready for `/auto-review-loop`: {ready}",
        "- SOTA-family experiments R018-R023 are not run yet.",
        "",
        "## Next Step",
        "",
        "Proceed with R013-R017 only if R012 beats R009 query-only routing on regret or action accuracy.",
    ]
    if dataset.get("source") == "tiny_fallback":
        lines += [
            "",
            "## Warning",
            "",
            "This run used the tiny fallback dataset. It validates code only and must not be used as paper evidence.",
        ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(result) -> str:
    if isinstance(result, dict):
        if "status" in result:
            return str(result["status"])
        if "ndcg10" in result:
            return f"nDCG@10={float(result['ndcg10']):.4f}, Recall@100={float(result['recall100']):.4f}, p95={float(result['latency_p95_ms']):.2f}ms"
        if "action_accuracy" in result:
            mae = result.get("mae_utility")
            mae_text = f", MAE={float(mae):.4f}" if mae is not None else ""
            return f"action_acc={float(result['action_accuracy']):.3f}, mean_regret={float(result['mean_regret']):.4f}{mae_text}"
        if "mean_regret" in result:
            extra = ""
            if "selected_static_plan" in result:
                extra = f", selected={result['selected_static_plan']}"
            return f"mean_regret={float(result['mean_regret']):.4f}{extra}"
        if "probe_latency_p95_ms" in result:
            return f"probe_p95={float(result['probe_latency_p95_ms']):.2f}ms"
    return "`" + json.dumps(result, ensure_ascii=False)[:300] + "`"


if __name__ == "__main__":
    main()
