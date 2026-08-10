#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results = json.loads(args.results.read_text(encoding="utf-8"))
    completed = {key for key in results if re.fullmatch(r"R\d{3}", key)}
    lines = args.tracker.read_text(encoding="utf-8").splitlines()
    new_lines = []
    for line in lines:
        match = re.match(r"\| (R\d{3}) \|", line)
        if match and match.group(1) in completed:
            parts = [part.strip() for part in line.strip().strip("|").split("|")]
            if len(parts) >= 8:
                parts[7] = "DONE"
                parts[8] = summarize(results[match.group(1)])
                line = "| " + " | ".join(parts) + " |"
        new_lines.append(line)
    output = args.output or args.tracker
    output.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def summarize(result) -> str:
    if isinstance(result, dict):
        if "status" in result:
            return str(result["status"])
        if "ndcg10" in result:
            return f"nDCG@10={float(result['ndcg10']):.4f}"
        if "action_accuracy" in result:
            return f"acc={float(result['action_accuracy']):.3f}, regret={float(result['mean_regret']):.4f}"
    return "completed"


if __name__ == "__main__":
    main()

