#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.paretoprobe_sys_pilot import (  # noqa: E402
    by_qid,
    evaluate_chosen,
    scoped_action_rows,
    workload_feature_columns,
    workload_feature_vector,
    with_budget,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute plan-frontier certificate and stability statistics.")
    parser.add_argument("--action-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budgets-ms", default="40,80")
    parser.add_argument("--selectivities", default="1.0,0.5,0.1")
    parser.add_argument("--seeds", default="13,17,23,29,31")
    parser.add_argument("--test-size", type=float, default=0.5)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--scope", choices=["all", "core", "pre", "core_pre"], default="all")
    parser.add_argument("--feature-set", default="full")
    parser.add_argument("--bootstrap-iters", type=int, default=200)
    parser.add_argument("--margin-threshold", type=float, default=0.02)
    parser.add_argument("--support-threshold", type=float, default=0.70)
    parser.add_argument("--frontier-epsilon", type=float, default=0.01)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.action_rows)
    budgets = [float(item) for item in args.budgets_ms.split(",") if item.strip()]
    selectivities = [float(item) for item in args.selectivities.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]

    cell_rows: list[dict] = []
    query_rows: list[dict] = []
    block_rows: list[dict] = []
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for selectivity in selectivities:
            sel_rows = [row for row in dataset_rows if abs(float(row["selectivity"]) - selectivity) <= 1e-9]
            if not sel_rows:
                continue
            for budget_ms in budgets:
                for seed in seeds:
                    cells, queries, block = analyze_block(sel_rows, dataset, selectivity, budget_ms, seed, args)
                    cell_rows.extend(cells)
                    query_rows.extend(queries)
                    block_rows.append(block)

    slice_rows = aggregate_slices(block_rows, cell_rows, query_rows)
    paper_rows = aggregate_paper_table(block_rows, cell_rows, query_rows)
    effect_rows = aggregate_certificate_effect(query_rows)

    write_csv(args.output_dir / "cell_certificate.csv", cell_rows)
    write_csv(args.output_dir / "query_certificate.csv", query_rows)
    write_csv(args.output_dir / "block_certificate_summary.csv", block_rows)
    write_csv(args.output_dir / "slice_certificate_summary.csv", slice_rows)
    write_csv(args.output_dir / "paper_certificate_table.csv", paper_rows)
    write_csv(args.output_dir / "certificate_effect_table.csv", effect_rows)
    write_markdown(args.output_dir / "PLAN_FRONTIER_CERTIFICATE.md", args, paper_rows, slice_rows, effect_rows)
    print(json.dumps({"paper_table": paper_rows, "output_dir": str(args.output_dir)}, indent=2, ensure_ascii=False))


def analyze_block(raw_rows: list[dict], dataset: str, selectivity: float, budget_ms: float, seed: int, args) -> tuple[list[dict], list[dict], dict]:
    block_rows = [with_budget(row, budget_ms) for row in raw_rows]
    qids = sorted({row["qid"] for row in block_rows})
    train_qids, eval_qids_split = train_test_split(qids, test_size=args.test_size, random_state=seed)
    train_set = set(train_qids)
    eval_set = set(eval_qids_split)
    train_rows = [row for row in block_rows if row["qid"] in train_set]
    eval_rows = [row for row in block_rows if row["qid"] in eval_set]

    train_by_qid = by_qid(train_rows)
    eval_by_qid = by_qid(eval_rows)
    train_order = list(train_by_qid)
    eval_order = list(eval_by_qid)
    all_keys = [("train", qid) for qid in train_order] + [("eval", qid) for qid in eval_order]
    feature_cols = workload_feature_columns("transductive", args.feature_set)
    features = []
    for split, qid in all_keys:
        rows = train_by_qid[qid] if split == "train" else eval_by_qid[qid]
        features.append(workload_feature_vector(rows, feature_cols))
    x = StandardScaler().fit_transform(np.asarray(features, dtype=np.float64))
    n_clusters = max(1, min(args.clusters, len(all_keys)))
    labels = np.zeros(len(all_keys), dtype=np.int64) if n_clusters == 1 else KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit_predict(x)
    label_by_key = {key: int(label) for key, label in zip(all_keys, labels, strict=True)}

    block_id = f"{dataset}|budget={budget_ms:g}ms|selectivity={selectivity:g}|seed={seed}"
    fallback_action = best_mean_score_action(train_rows, args.scope)
    cell_out: list[dict] = []
    query_out: list[dict] = []
    chosen_rows: list[dict] = []

    for label in sorted(set(label_by_key.values())):
        cell_train_qids = [qid for qid in train_order if label_by_key[("train", qid)] == label]
        cell_eval_qids = [qid for qid in eval_order if label_by_key[("eval", qid)] == label]
        scoped_train = []
        for qid in cell_train_qids:
            scoped_train.extend(scoped_action_rows(train_by_qid[qid], args.scope))
        if scoped_train:
            stats = action_stats(scoped_train)
            chosen_action = best_action_from_stats(stats)
        else:
            stats = action_stats(scoped_action_rows(train_rows, args.scope))
            chosen_action = fallback_action

        ordered = sorted(stats.values(), key=lambda row: (row["mean_score"], row["satisfaction"], -row["mean_latency"], row["action"]), reverse=True)
        best = ordered[0]
        second = ordered[1] if len(ordered) > 1 else None
        margin = float(best["mean_score"] - second["mean_score"]) if second else float(best["mean_score"])
        frontier_width = sum(1 for row in ordered if best["mean_score"] - row["mean_score"] <= args.frontier_epsilon)
        support, support_entropy, support_counts = bootstrap_plan_support(
            train_by_qid=train_by_qid,
            qids=cell_train_qids,
            scope=args.scope,
            selected_action=chosen_action,
            iterations=args.bootstrap_iters,
            seed=seed * 1000 + label,
        )

        cell_chosen = []
        for qid in cell_eval_qids:
            candidates = eval_by_qid[qid]
            selected = next((row for row in candidates if row["action"] == chosen_action), None)
            selected = selected if selected is not None else next((row for row in candidates if row["action"] == fallback_action), candidates[0])
            oracle = max(candidates, key=lambda row: float(row["constrained_score"]))
            regret = float(oracle["constrained_score"]) - float(selected["constrained_score"])
            risk_flag = int(is_risky_cell(margin, support, args))
            query_out.append(
                {
                    "block": block_id,
                    "dataset": dataset,
                    "selectivity": selectivity,
                    "budget_ms": budget_ms,
                    "seed": seed,
                    "cell": label,
                    "qid": qid,
                    "chosen_action": chosen_action,
                    "oracle_action": oracle["action"],
                    "chosen_score": selected["constrained_score"],
                    "oracle_score": oracle["constrained_score"],
                    "regret": regret,
                    "feasible": selected["feasible"],
                    "latency_ms": selected["latency_ms"],
                    "risk_flag": risk_flag,
                }
            )
            cell_chosen.append(selected)
            chosen_rows.append(selected)

        cell_metrics = evaluate_chosen(cell_chosen, eval_rows) if cell_chosen else empty_metrics()
        risk_flag = int(is_risky_cell(margin, support, args))
        cell_out.append(
            {
                "block": block_id,
                "dataset": dataset,
                "selectivity": selectivity,
                "budget_ms": budget_ms,
                "seed": seed,
                "cell": label,
                "train_queries": len(cell_train_qids),
                "eval_queries": len(cell_eval_qids),
                "chosen_action": chosen_action,
                "second_action": second["action"] if second else "",
                "best_train_score": best["mean_score"],
                "second_train_score": second["mean_score"] if second else "",
                "plan_margin": margin,
                "frontier_width_eps": frontier_width,
                "bootstrap_support": support,
                "bootstrap_entropy": support_entropy,
                "bootstrap_top_counts": json.dumps(support_counts, sort_keys=True),
                "risk_flag": risk_flag,
                "mean_eval_score": cell_metrics["mean_score"],
                "mean_eval_regret": cell_metrics["mean_regret"],
                "oracle_action_accuracy": cell_metrics["oracle_action_accuracy"],
                "satisfaction_rate": cell_metrics["satisfaction_rate"],
                "mean_latency_ms": cell_metrics["mean_latency_ms"],
            }
        )

    block_metrics = evaluate_chosen(chosen_rows, eval_rows)
    regrets = [float(row["regret"]) for row in query_out if row["block"] == block_id]
    block = {
        "block": block_id,
        "dataset": dataset,
        "selectivity": selectivity,
        "budget_ms": budget_ms,
        "seed": seed,
        "cells": len(cell_out),
        "eval_queries": len(eval_order),
        "distinct_compiled_actions": len({row["chosen_action"] for row in cell_out}),
        "mean_plan_margin": weighted_mean(cell_out, "plan_margin", "eval_queries"),
        "p10_plan_margin": percentile([float(row["plan_margin"]) for row in cell_out], 10),
        "min_plan_margin": min(float(row["plan_margin"]) for row in cell_out),
        "mean_bootstrap_support": weighted_mean(cell_out, "bootstrap_support", "eval_queries"),
        "risky_cell_share": mean(float(row["risk_flag"]) for row in cell_out),
        "flagged_query_share": mean(float(row["risk_flag"]) for row in query_out if row["block"] == block_id),
        "mean_frontier_width": weighted_mean(cell_out, "frontier_width_eps", "eval_queries"),
        "mean_score": block_metrics["mean_score"],
        "mean_regret": block_metrics["mean_regret"],
        "p90_regret": percentile(regrets, 90),
        "satisfaction_rate": block_metrics["satisfaction_rate"],
        "mean_latency_ms": block_metrics["mean_latency_ms"],
    }
    return cell_out, query_out, block


def action_stats(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for action in sorted({row["action"] for row in rows}):
        action_rows = [row for row in rows if row["action"] == action]
        out[action] = {
            "action": action,
            "mean_score": mean(float(row["constrained_score"]) for row in action_rows),
            "satisfaction": mean(float(row["feasible"]) for row in action_rows),
            "mean_latency": mean(float(row["latency_ms"]) for row in action_rows),
        }
    return out


def best_action_from_stats(stats: dict[str, dict]) -> str:
    actions = sorted(stats)
    return max(actions, key=lambda action: stats[action]["mean_score"])


def best_mean_score_action(rows: list[dict], scope: str) -> str:
    scoped = scoped_action_rows(rows, scope)
    if not scoped:
        scoped = rows
    stats = action_stats(scoped)
    return best_action_from_stats(stats)


def bootstrap_plan_support(
    train_by_qid: dict[str, list[dict]],
    qids: list[str],
    scope: str,
    selected_action: str,
    iterations: int,
    seed: int,
) -> tuple[float, float, dict[str, int]]:
    if not qids:
        return 0.0, 0.0, {}
    rng = np.random.default_rng(seed)
    counts: Counter[str] = Counter()
    for _ in range(iterations):
        sample = rng.choice(qids, size=len(qids), replace=True)
        rows = []
        for qid in sample:
            rows.extend(scoped_action_rows(train_by_qid[str(qid)], scope))
        counts[best_mean_score_action(rows, "all")] += 1
    support = counts[selected_action] / max(1, iterations)
    probs = np.asarray([count / max(1, iterations) for count in counts.values()], dtype=np.float64)
    entropy = float(-(probs * np.log2(np.maximum(probs, 1e-12))).sum()) if probs.size else 0.0
    return float(support), entropy, dict(counts.most_common())


def is_risky_cell(margin: float, support: float, args) -> bool:
    margin_risk = args.margin_threshold > 0.0 and margin < args.margin_threshold
    support_risk = support < args.support_threshold
    return margin_risk or support_risk


def aggregate_slices(block_rows: list[dict], cell_rows: list[dict], query_rows: list[dict]) -> list[dict]:
    out = []
    keys = sorted({(float(row["selectivity"]), float(row["budget_ms"])) for row in block_rows}, key=lambda item: (-item[0], item[1]))
    for selectivity, budget_ms in keys:
        blocks = [row for row in block_rows if float(row["selectivity"]) == selectivity and float(row["budget_ms"]) == budget_ms]
        cells = [row for row in cell_rows if float(row["selectivity"]) == selectivity and float(row["budget_ms"]) == budget_ms]
        queries = [row for row in query_rows if float(row["selectivity"]) == selectivity and float(row["budget_ms"]) == budget_ms]
        out.append(summarize_group(f"s={selectivity:g}, B={budget_ms:g}ms", blocks, cells, queries))
    return out


def aggregate_paper_table(block_rows: list[dict], cell_rows: list[dict], query_rows: list[dict]) -> list[dict]:
    weak = lambda row: math.isclose(float(row["selectivity"]), 0.5) and math.isclose(float(row["budget_ms"]), 80.0)
    groups = [
        ("All 30 blocks", block_rows, cell_rows, query_rows),
        ("Certificate-targeted weak slice: s=0.5, B=80ms", [row for row in block_rows if weak(row)], [row for row in cell_rows if weak(row)], [row for row in query_rows if weak(row)]),
        ("Other slices", [row for row in block_rows if not weak(row)], [row for row in cell_rows if not weak(row)], [row for row in query_rows if not weak(row)]),
    ]
    return [summarize_group(name, blocks, cells, queries) for name, blocks, cells, queries in groups]


def aggregate_certificate_effect(query_rows: list[dict]) -> list[dict]:
    out = []
    for flag, label in [(0, "Not flagged"), (1, "Flagged by certificate")]:
        rows = [row for row in query_rows if int(float(row["risk_flag"])) == flag]
        regrets = [float(row["regret"]) for row in rows]
        out.append(
            {
                "group": label,
                "queries": len(rows),
                "query_share": len(rows) / max(1, len(query_rows)),
                "mean_regret": mean(regrets),
                "p90_regret": percentile(regrets, 90),
                "satisfaction_rate": mean(float(row["feasible"]) for row in rows),
            }
        )
    return out


def summarize_group(name: str, blocks: list[dict], cells: list[dict], queries: list[dict]) -> dict:
    return {
        "group": name,
        "blocks": len(blocks),
        "cells": len(cells),
        "queries": len(queries),
        "mean_plan_margin": weighted_mean(cells, "plan_margin", "eval_queries"),
        "p10_plan_margin": percentile([float(row["plan_margin"]) for row in cells], 10),
        "min_plan_margin": min_or_blank(float(row["plan_margin"]) for row in cells),
        "mean_bootstrap_support": weighted_mean(cells, "bootstrap_support", "eval_queries"),
        "risky_cell_share": mean(float(row["risk_flag"]) for row in cells),
        "flagged_query_share": mean(float(row["risk_flag"]) for row in queries),
        "mean_frontier_width": weighted_mean(cells, "frontier_width_eps", "eval_queries"),
        "distinct_actions_per_block": mean(float(row["distinct_compiled_actions"]) for row in blocks),
        "mean_regret": mean(float(row["regret"]) for row in queries),
        "p90_regret": percentile([float(row["regret"]) for row in queries], 90),
        "satisfaction_rate": mean(float(row["feasible"]) for row in queries),
    }


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key.startswith("action_") and value == "":
                row[key] = "0.0"
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, args, paper_rows: list[dict], slice_rows: list[dict], effect_rows: list[dict]) -> None:
    lines = [
        "# Plan-Frontier Certificate and Stability Statistics",
        "",
        "This report computes certificate statistics for the passing transductive workload compiler using cached Qwen/SPLADE/ColBERT action rows.",
        "",
        "## Certificate Rule",
        "",
        (
            f"- A cell is flagged as risky when plan margin `< {args.margin_threshold:g}` or bootstrap support `< {args.support_threshold:g}`."
            if args.margin_threshold > 0.0
            else f"- A cell is flagged as risky when bootstrap support `< {args.support_threshold:g}`; margin is reported as a continuous certificate statistic."
        ),
        f"- Plan margin is the training constrained-score gap between the compiled action and the second-best action in the same cell.",
        f"- Bootstrap support is the fraction of `{args.bootstrap_iters}` resampled train-cell workloads that reselect the same action.",
        f"- Frontier width counts actions within `{args.frontier_epsilon:g}` constrained-score units of the best cell action.",
        "",
        "## Paper Table",
        "",
        "| Group | Blocks | Cells | Queries | Margin | Bootstrap support | Risky cells | Flagged queries | Frontier width | Regret | p90 regret | Satisfaction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paper_rows:
        lines.append(
            "| {group} | {blocks} | {cells} | {queries} | {mean_plan_margin:.4f} | {mean_bootstrap_support:.3f} | "
            "{risky_cell_share:.3f} | {flagged_query_share:.3f} | {mean_frontier_width:.2f} | {mean_regret:.4f} | {p90_regret:.4f} | {satisfaction_rate:.3f} |".format(
                **format_row(row)
            )
        )
    lines.extend(
        [
            "",
            "## Slice Table",
            "",
            "| Slice | Blocks | Cells | Queries | Margin | Bootstrap support | Risky cells | Flagged queries | Frontier width | Regret | p90 regret | Satisfaction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in slice_rows:
        lines.append(
            "| {group} | {blocks} | {cells} | {queries} | {mean_plan_margin:.4f} | {mean_bootstrap_support:.3f} | "
            "{risky_cell_share:.3f} | {flagged_query_share:.3f} | {mean_frontier_width:.2f} | {mean_regret:.4f} | {p90_regret:.4f} | {satisfaction_rate:.3f} |".format(
                **format_row(row)
            )
        )
    lines.extend(
        [
            "",
            "## Certificate Effect",
            "",
            "| Group | Queries | Query share | Regret | p90 regret | Satisfaction |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in effect_rows:
        lines.append(
            "| {group} | {queries} | {query_share:.3f} | {mean_regret:.4f} | {p90_regret:.4f} | {satisfaction_rate:.3f} |".format(
                **format_row(row)
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `cell_certificate.csv`: one row per compiled cell.",
            "- `query_certificate.csv`: one row per evaluation query with chosen/oracle action and regret.",
            "- `block_certificate_summary.csv`: one row per selectivity-budget-seed block.",
            "- `slice_certificate_summary.csv`: grouped slice table.",
            "- `paper_certificate_table.csv`: compact table for the manuscript.",
            "- `certificate_effect_table.csv`: regret separation for flagged and non-flagged evaluation queries.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_row(row: dict) -> dict:
    out = dict(row)
    for key, value in list(out.items()):
        if key == "group":
            continue
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            pass
    for key in ["blocks", "cells", "queries"]:
        if key in out:
            out[key] = int(float(out[key]))
    return out


def empty_metrics() -> dict:
    return {
        "mean_score": 0.0,
        "mean_regret": 0.0,
        "oracle_action_accuracy": 0.0,
        "satisfaction_rate": 0.0,
        "mean_latency_ms": 0.0,
    }


def weighted_mean(rows: list[dict], value_key: str, weight_key: str) -> float:
    pairs = [(float(row[value_key]), float(row.get(weight_key, 1.0))) for row in rows]
    denom = sum(weight for _, weight in pairs)
    if denom <= 0:
        return mean(value for value, _ in pairs)
    return sum(value * weight for value, weight in pairs) / denom


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def mean(values) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else 0.0


def min_or_blank(values) -> float:
    vals = list(values)
    return float(min(vals)) if vals else 0.0


if __name__ == "__main__":
    main()
