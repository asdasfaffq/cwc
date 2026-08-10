#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run workload compiler ablation gates from cached action rows.")
    parser.add_argument("--action-rows", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--budgets-ms", default="40,80")
    parser.add_argument("--selectivities", default="1.0,0.5,0.1")
    parser.add_argument("--seeds", default="13,17,23,29,31")
    parser.add_argument("--test-size", type=float, default=0.5)
    parser.add_argument("--scope", choices=["all", "core", "pre", "core_pre"], default="all")
    parser.add_argument("--k-values", default="2,3,4,5,6,8,10,12")
    parser.add_argument("--modes", default="transductive,train_only,random,query_only")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    k_values = [int(item) for item in args.k_values.split(",") if item.strip()]
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]

    variants: list[tuple[str, int]] = [("transductive", k) for k in k_values]
    for mode in modes:
        if mode != "transductive":
            variants.append((mode, 6))

    rows = []
    for mode, k in variants:
        name = f"{mode}_k{k}_scope_{args.scope}"
        out_dir = args.output_root / name
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "rerun_sys_gate_from_actions.py"),
            "--action-rows",
            str(args.action_rows),
            "--output-dir",
            str(out_dir),
            "--test-size",
            str(args.test_size),
            "--budgets-ms",
            args.budgets_ms,
            "--selectivities",
            args.selectivities,
            "--seeds",
            args.seeds,
            "--workload-compiler",
            "--workload-compiler-clusters",
            str(k),
            "--workload-compiler-scope",
            args.scope,
            "--workload-compiler-mode",
            mode,
        ]
        print(f"[run] {name}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        payload = json.loads((out_dir / "rank_gate.json").read_text(encoding="utf-8"))
        report_metrics = aggregate_metrics(out_dir / "sys_summary.csv")
        row = {
            "variant": name,
            "mode": mode,
            "k": k,
            "scope": args.scope,
            "pass_gate": payload["pass_gate"],
            "failures": len(payload["failures"]),
            "candidate_avg_rank": payload["average_ranks"].get("ParetoProbe-Sys", ""),
            "queryonly_avg_rank": payload["average_ranks"].get("QueryOnly-RF", ""),
            "costgreedy_avg_rank": payload["average_ranks"].get("CostGreedy-cal", ""),
            "staticbest_avg_rank": payload["average_ranks"].get("StaticBest-cal", ""),
            **report_metrics,
            "output_dir": str(out_dir),
        }
        rows.append(row)

    write_csv(args.output_root / "summary.csv", rows)
    write_markdown(args.output_root / "SUMMARY.md", rows)


def aggregate_metrics(path: Path) -> dict[str, float]:
    rows = [row for row in csv.DictReader(path.open("r", encoding="utf-8", newline="")) if row["method"] == "ParetoProbe-Sys"]
    out = {}
    for col in ["mean_score", "satisfaction_rate", "mean_latency_ms", "mean_regret"]:
        vals = [float(row[col]) for row in rows]
        out[col] = sum(vals) / len(vals) if vals else 0.0
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    ranked = sorted(rows, key=lambda row: (not truthy(row["pass_gate"]), int(row["failures"]), float(row["candidate_avg_rank"])))
    lines = [
        "# Workload Compiler Ablations",
        "",
        "| Variant | Pass | Failures | Avg rank | Mean score | Satisfaction | Latency | Regret |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        lines.append(
            "| `{variant}` | `{pass_gate}` | {failures} | {candidate_avg_rank:.4f} | {mean_score:.4f} | "
            "{satisfaction_rate:.3f} | {mean_latency_ms:.2f} | {mean_regret:.4f} |".format(
                variant=row["variant"],
                pass_gate=row["pass_gate"],
                failures=int(row["failures"]),
                candidate_avg_rank=float(row["candidate_avg_rank"]),
                mean_score=float(row["mean_score"]),
                satisfaction_rate=float(row["satisfaction_rate"]),
                mean_latency_ms=float(row["mean_latency_ms"]),
                mean_regret=float(row["mean_regret"]),
            )
        )
    lines.append("")
    lines.append("## Output Directories")
    lines.append("")
    for row in ranked:
        lines.append(f"- `{row['variant']}`: `{row['output_dir']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def truthy(value) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


if __name__ == "__main__":
    main()
