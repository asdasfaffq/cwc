#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


@dataclass
class GateResult:
    pass_gate: bool
    candidate: str
    blocks: int
    methods: list[str]
    average_ranks: dict[str, float]
    friedman_p: float | None
    pairwise_holm: dict[str, dict[str, float | bool]]
    failures: list[str]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether a candidate method is statistically rank #1 across experiment blocks."
    )
    parser.add_argument("--input", type=Path, required=True, help="CSV with columns: block,method,score[,higher_is_better]")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-baselines", type=int, default=6)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    rows = read_rows(args.input)
    result = evaluate_gate(rows, args.candidate, args.alpha, args.min_baselines)
    payload = {
        "pass_gate": result.pass_gate,
        "candidate": result.candidate,
        "blocks": result.blocks,
        "methods": result.methods,
        "average_ranks": result.average_ranks,
        "friedman_p": result.friedman_p,
        "pairwise_holm": result.pairwise_holm,
        "failures": result.failures,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    sys.exit(0 if result.pass_gate else 2)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    required = {"block", "method", "score"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return rows


def evaluate_gate(rows: list[dict[str, str]], candidate: str, alpha: float, min_baselines: int) -> GateResult:
    by_block: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_block.setdefault(row["block"], []).append(row)

    methods = sorted({row["method"] for row in rows})
    failures: list[str] = []
    if candidate not in methods:
        failures.append(f"Candidate `{candidate}` is absent from the result table.")
    if len(methods) - 1 < min_baselines:
        failures.append(f"Only {len(methods) - 1} baselines found; need at least {min_baselines}.")

    complete_blocks = []
    rank_matrix: dict[str, list[float]] = {method: [] for method in methods}
    for block, block_rows in sorted(by_block.items()):
        block_methods = {row["method"] for row in block_rows}
        if block_methods != set(methods):
            continue
        higher = parse_bool(block_rows[0].get("higher_is_better", "1"))
        scores = np.asarray([float(next(row["score"] for row in block_rows if row["method"] == method)) for method in methods])
        oriented = -scores if higher else scores
        ranks = rankdata(oriented, method="average")
        for method, rank in zip(methods, ranks, strict=True):
            rank_matrix[method].append(float(rank))
        complete_blocks.append(block)

    if not complete_blocks:
        failures.append("No complete blocks contain all methods.")

    average_ranks = {
        method: float(np.mean(ranks)) if ranks else math.inf for method, ranks in rank_matrix.items()
    }
    best_rank = min(average_ranks.values()) if average_ranks else math.inf
    if candidate in average_ranks and not math.isclose(average_ranks[candidate], best_rank):
        failures.append(
            f"Candidate average rank is {average_ranks[candidate]:.4f}, not rank #1 ({best_rank:.4f})."
        )

    friedman_p = None
    if len(methods) >= 3 and len(complete_blocks) >= 2:
        _, friedman_p = friedmanchisquare(*[rank_matrix[method] for method in methods])
        friedman_p = float(friedman_p)

    pairwise_raw: list[tuple[str, float]] = []
    if candidate in rank_matrix:
        cand_ranks = np.asarray(rank_matrix[candidate], dtype=np.float64)
        for method in methods:
            if method == candidate:
                continue
            other_ranks = np.asarray(rank_matrix[method], dtype=np.float64)
            if len(cand_ranks) < 2 or np.allclose(cand_ranks, other_ranks):
                p_value = 1.0
            else:
                _, p_value = wilcoxon(cand_ranks, other_ranks, alternative="less", zero_method="wilcox")
            pairwise_raw.append((method, float(p_value)))

    pairwise_holm = holm_adjust(pairwise_raw, alpha)
    for method, stats in pairwise_holm.items():
        if not bool(stats["reject"]):
            failures.append(f"Candidate is not significantly better-ranked than `{method}` after Holm correction.")

    return GateResult(
        pass_gate=not failures,
        candidate=candidate,
        blocks=len(complete_blocks),
        methods=methods,
        average_ranks=average_ranks,
        friedman_p=friedman_p,
        pairwise_holm=pairwise_holm,
        failures=failures,
    )


def parse_bool(text: str) -> bool:
    return str(text).strip().lower() not in {"0", "false", "no", "lower", "min"}


def holm_adjust(raw: list[tuple[str, float]], alpha: float) -> dict[str, dict[str, float | bool]]:
    ordered = sorted(raw, key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, dict[str, float | bool]] = {}
    running_max = 0.0
    for idx, (method, p_value) in enumerate(ordered):
        factor = m - idx
        adj = min(1.0, max(running_max, p_value * factor))
        running_max = adj
        adjusted[method] = {
            "p": p_value,
            "p_holm": adj,
            "reject": adj < alpha,
        }
    return adjusted


def render_markdown(result: GateResult) -> str:
    lines = [
        "# Statistical Rank Gate",
        "",
        f"Candidate: `{result.candidate}`",
        f"Pass: `{result.pass_gate}`",
        f"Complete blocks: `{result.blocks}`",
        "",
        "## Average Ranks",
        "",
        "| Method | Average Rank |",
        "|---|---:|",
    ]
    for method, rank in sorted(result.average_ranks.items(), key=lambda item: item[1]):
        lines.append(f"| `{method}` | {rank:.4f} |")
    lines += [
        "",
        "## Pairwise Holm Tests",
        "",
        "| Baseline | p | p_holm | Reject |",
        "|---|---:|---:|---|",
    ]
    for method, stats in sorted(result.pairwise_holm.items()):
        lines.append(
            f"| `{method}` | {float(stats['p']):.4g} | {float(stats['p_holm']):.4g} | `{stats['reject']}` |"
        )
    if result.failures:
        lines += ["", "## Failures", ""]
        lines.extend(f"- {failure}" for failure in result.failures)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
