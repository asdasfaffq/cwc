#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run auto-k and feature-family workload compiler ablations.")
    parser.add_argument("--action-rows", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--budgets-ms", default="40,80")
    parser.add_argument("--selectivities", default="1.0,0.5,0.1")
    parser.add_argument("--seeds", default="13,17,23,29,31")
    parser.add_argument("--test-size", type=float, default=0.5)
    parser.add_argument("--scope", choices=["all", "core", "pre", "core_pre"], default="all")
    parser.add_argument("--k-candidates", default="2,3,4,5,6,8,10,12")
    parser.add_argument("--fixed-k", type=int, default=6)
    parser.add_argument("--feature-sets", default="full,no_bm25,no_dense,no_overlap,no_probe,metadata_only,query_only")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    variants = [
        {
            "name": f"fixed_k{args.fixed_k}_features_full",
            "k": args.fixed_k,
            "feature_set": "full",
            "k_selection": "fixed",
        },
        {
            "name": "auto_silhouette_features_full",
            "k": args.fixed_k,
            "feature_set": "full",
            "k_selection": "silhouette",
        },
        {
            "name": "auto_elbow_features_full",
            "k": args.fixed_k,
            "feature_set": "full",
            "k_selection": "elbow",
        },
    ]
    for feature_set in [item.strip() for item in args.feature_sets.split(",") if item.strip()]:
        if feature_set == "full":
            continue
        variants.append(
            {
                "name": f"fixed_k{args.fixed_k}_features_{feature_set}",
                "k": args.fixed_k,
                "feature_set": feature_set,
                "k_selection": "fixed",
            }
        )

    rows = []
    for variant in variants:
        out_dir = args.output_root / variant["name"]
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
            str(variant["k"]),
            "--workload-compiler-scope",
            args.scope,
            "--workload-compiler-mode",
            "transductive",
            "--workload-compiler-feature-set",
            variant["feature_set"],
            "--workload-compiler-k-selection",
            variant["k_selection"],
            "--workload-compiler-k-candidates",
            args.k_candidates,
        ]
        print(f"[run] {variant['name']}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        payload = json.loads((out_dir / "rank_gate.json").read_text(encoding="utf-8"))
        metrics = aggregate_metrics(out_dir / "sys_summary.csv")
        selected_ks = selected_k_counts(out_dir / "sys_summary.csv")
        rows.append(
            {
                "variant": variant["name"],
                "k_selection": variant["k_selection"],
                "feature_set": variant["feature_set"],
                "requested_k": variant["k"],
                "selected_k_counts": json.dumps(selected_ks, sort_keys=True),
                "pass_gate": payload["pass_gate"],
                "failures": len(payload["failures"]),
                "candidate_avg_rank": payload["average_ranks"].get("ParetoProbe-Sys", ""),
                "queryonly_avg_rank": payload["average_ranks"].get("QueryOnly-RF", ""),
                "costgreedy_avg_rank": payload["average_ranks"].get("CostGreedy-cal", ""),
                "staticbest_avg_rank": payload["average_ranks"].get("StaticBest-cal", ""),
                "holm_queryonly": payload["pairwise_holm"].get("QueryOnly-RF", {}).get("p_holm", ""),
                "holm_costgreedy": payload["pairwise_holm"].get("CostGreedy-cal", {}).get("p_holm", ""),
                "holm_staticbest": payload["pairwise_holm"].get("StaticBest-cal", {}).get("p_holm", ""),
                **metrics,
                "output_dir": str(out_dir),
            }
        )

    write_csv(args.output_root / "summary.csv", rows)
    write_markdown(args.output_root / "SUMMARY.md", rows)


def aggregate_metrics(path: Path) -> dict[str, float]:
    rows = [row for row in csv.DictReader(path.open("r", encoding="utf-8", newline="")) if row["method"] == "ParetoProbe-Sys"]
    out = {}
    for col in ["mean_score", "satisfaction_rate", "mean_latency_ms", "mean_regret"]:
        vals = [float(row[col]) for row in rows]
        out[col] = sum(vals) / len(vals) if vals else 0.0
    return out


def selected_k_counts(path: Path) -> dict[str, int]:
    counts = Counter()
    for row in csv.DictReader(path.open("r", encoding="utf-8", newline="")):
        if row["method"] == "ParetoProbe-Sys" and row.get("selected_k"):
            counts[str(int(float(row["selected_k"])))] += 1
    return dict(counts)


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
        "# Workload Compiler Auto-k and Feature Ablations",
        "",
        "| Variant | Pass | Failures | Avg rank | Mean score | Regret | Selected k | Holm vs QueryOnly |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in ranked:
        lines.append(
            "| `{variant}` | `{pass_gate}` | {failures} | {candidate_avg_rank:.4f} | {mean_score:.4f} | "
            "{mean_regret:.4f} | `{selected_k_counts}` | {holm_queryonly} |".format(
                variant=row["variant"],
                pass_gate=row["pass_gate"],
                failures=int(row["failures"]),
                candidate_avg_rank=float(row["candidate_avg_rank"]),
                mean_score=float(row["mean_score"]),
                mean_regret=float(row["mean_regret"]),
                selected_k_counts=row["selected_k_counts"],
                holm_queryonly=format_float(row["holm_queryonly"]),
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


def format_float(value) -> str:
    if value == "":
        return ""
    return f"{float(value):.5g}"


if __name__ == "__main__":
    main()
