#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.paretoprobe_sys_pilot import METHOD_CANDIDATE, evaluate_paretoprobe_sys, run_block, write_csv, write_report
from scripts.statistical_rank_gate import evaluate_gate, render_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun ParetoProbe-Sys selector/rank gate from cached action rows.")
    parser.add_argument("--action-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", default="")
    parser.add_argument("--budgets-ms", default="40,80")
    parser.add_argument("--selectivities", default="1.0,0.5")
    parser.add_argument("--seeds", default="13")
    parser.add_argument("--test-size", type=float, default=0.5)
    parser.add_argument("--selector-margin", type=float, default=0.005)
    parser.add_argument("--foundation-prior-weight", type=float, default=0.0)
    parser.add_argument("--rank1-delta", action="store_true")
    parser.add_argument("--workload-compiler", action="store_true")
    parser.add_argument("--workload-compiler-clusters", type=int, default=6)
    parser.add_argument("--workload-compiler-scope", choices=["all", "core", "pre", "core_pre"], default="all")
    parser.add_argument(
        "--workload-compiler-feature-set",
        choices=["full", "query_only", "metadata_only", "no_bm25", "no_dense", "no_overlap", "no_probe"],
        default="full",
    )
    parser.add_argument(
        "--workload-compiler-k-selection",
        choices=["fixed", "silhouette", "elbow"],
        default="fixed",
    )
    parser.add_argument("--workload-compiler-k-candidates", default="2,3,4,5,6,8,10,12")
    parser.add_argument(
        "--workload-compiler-mode",
        choices=[
            "transductive",
            "train_only",
            "random",
            "query_only",
            "counterfactual",
            "plan_frontier",
            "certified",
            "compositional_guard",
            "component_shrink",
            "calibrated",
            "calibrated_static_query",
            "calibrated_static_query_rank",
            "calibrated_rank",
        ],
        default="transductive",
    )
    parser.add_argument("--workload-compiler-cert-support-threshold", type=float, default=0.70)
    parser.add_argument("--workload-compiler-cert-bootstrap-iters", type=int, default=100)
    args = parser.parse_args()

    rows = read_csv(args.action_rows)
    wanted_datasets = {item.strip() for item in args.datasets.split(",") if item.strip()}
    if wanted_datasets:
        rows = [row for row in rows if row["dataset"] in wanted_datasets]

    budgets = [float(item) for item in args.budgets_ms.split(",") if item.strip()]
    selectivities = [float(item) for item in args.selectivities.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    evaluate_paretoprobe_sys.rank1_delta = bool(args.rank1_delta)
    evaluate_paretoprobe_sys.selector_margin = float(args.selector_margin)
    evaluate_paretoprobe_sys.foundation_prior_weight = float(args.foundation_prior_weight)
    evaluate_paretoprobe_sys.workload_compiler = bool(args.workload_compiler)
    evaluate_paretoprobe_sys.workload_compiler_clusters = int(args.workload_compiler_clusters)
    evaluate_paretoprobe_sys.workload_compiler_scope = str(args.workload_compiler_scope)
    evaluate_paretoprobe_sys.workload_compiler_feature_set = str(args.workload_compiler_feature_set)
    evaluate_paretoprobe_sys.workload_compiler_k_selection = str(args.workload_compiler_k_selection)
    evaluate_paretoprobe_sys.workload_compiler_k_candidates = str(args.workload_compiler_k_candidates)
    evaluate_paretoprobe_sys.workload_compiler_mode = str(args.workload_compiler_mode)
    evaluate_paretoprobe_sys.workload_compiler_cert_support_threshold = float(args.workload_compiler_cert_support_threshold)
    evaluate_paretoprobe_sys.workload_compiler_cert_bootstrap_iters = int(args.workload_compiler_cert_bootstrap_iters)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    rank_rows = []
    datasets = sorted({row["dataset"] for row in rows})
    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for selectivity in selectivities:
            sel_rows = [row for row in dataset_rows if abs(float(row["selectivity"]) - selectivity) <= 1e-9]
            if not sel_rows:
                continue
            for budget in budgets:
                for seed in seeds:
                    block_summary, block_rank = run_block(sel_rows, dataset, selectivity, budget, seed, args.test_size)
                    summary_rows.extend(block_summary)
                    rank_rows.extend(block_rank)

    write_csv(args.output_dir / "sys_summary.csv", summary_rows)
    write_csv(args.output_dir / "rank_gate_input.csv", rank_rows)
    gate = evaluate_gate(
        [{key: str(value) for key, value in row.items()} for row in rank_rows],
        candidate=METHOD_CANDIDATE,
        alpha=0.05,
        min_baselines=6,
    )
    gate_payload = {
        "pass_gate": gate.pass_gate,
        "candidate": gate.candidate,
        "blocks": gate.blocks,
        "methods": gate.methods,
        "average_ranks": gate.average_ranks,
        "friedman_p": gate.friedman_p,
        "pairwise_holm": gate.pairwise_holm,
        "failures": gate.failures,
    }
    (args.output_dir / "rank_gate.json").write_text(json.dumps(gate_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "RANK_GATE.md").write_text(render_markdown(gate), encoding="utf-8")
    write_report(args.output_dir / "REPORT.md", summary_rows, gate)
    print(json.dumps({"pass_gate": gate.pass_gate, "average_ranks": gate.average_ranks, "failures": gate.failures}, indent=2, ensure_ascii=False))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key.startswith("action_") and value == "":
                row[key] = "0.0"
    return rows


if __name__ == "__main__":
    main()
