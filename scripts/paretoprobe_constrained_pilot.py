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
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.statistical_rank_gate import evaluate_gate, render_markdown


METHOD_CANDIDATE = "ParetoProbe-C"
STATIC_PLANS = ["bm25_full", "dense_full", "rrf", "weighted_0.5"]
ALL_PLANS = ["bm25_probe", "bm25_full", "dense_probe", "dense_full", "rrf", "weighted_0.5"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Constrained ParetoProbe-C pilot for R031/R032.")
    parser.add_argument(
        "--result-dirs",
        nargs="+",
        type=Path,
        default=[ROOT / "results" / "pilot", ROOT / "results" / "nfcorpus_bge", ROOT / "results" / "fiqa_bge"],
    )
    parser.add_argument("--budget-ms", default="20,40,80")
    parser.add_argument("--seeds", default="13,17,23")
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage level for conformal lower bounds.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "constrained_pilot")
    parser.add_argument("--latency-penalty", type=float, default=0.15)
    args = parser.parse_args()

    budgets = [float(item) for item in args.budget_ms.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    rank_rows: list[dict] = []
    for result_dir in args.result_dirs:
        result = load_result_dir(result_dir)
        for budget in budgets:
            for seed in seeds:
                block_rows, block_rank_rows = run_block(
                    result,
                    budget_ms=budget,
                    seed=seed,
                    alpha=args.alpha,
                    latency_penalty=args.latency_penalty,
                )
                summary_rows.extend(block_rows)
                rank_rows.extend(block_rank_rows)

    write_csv(args.output_dir / "constrained_summary.csv", summary_rows)
    write_csv(args.output_dir / "rank_gate_input.csv", rank_rows)
    gate = evaluate_gate(
        [{k: str(v) for k, v in row.items()} for row in rank_rows],
        candidate=METHOD_CANDIDATE,
        alpha=0.05,
        min_baselines=6,
    )
    (args.output_dir / "rank_gate.json").write_text(
        json.dumps(
            {
                "pass_gate": gate.pass_gate,
                "candidate": gate.candidate,
                "blocks": gate.blocks,
                "methods": gate.methods,
                "average_ranks": gate.average_ranks,
                "friedman_p": gate.friedman_p,
                "pairwise_holm": gate.pairwise_holm,
                "failures": gate.failures,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RANK_GATE.md").write_text(render_markdown(gate), encoding="utf-8")
    write_report(args.output_dir / "REPORT.md", summary_rows, gate, budgets, seeds)
    print(
        json.dumps(
            {
                "summary": str(args.output_dir / "constrained_summary.csv"),
                "rank_gate": str(args.output_dir / "RANK_GATE.md"),
                "pass_gate": gate.pass_gate,
                "average_ranks": gate.average_ranks,
                "failures": gate.failures,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def load_result_dir(result_dir: Path) -> dict:
    results_path = result_dir / "results.json"
    feature_path = result_dir / "feature_rows.csv"
    if not results_path.exists() or not feature_path.exists():
        raise FileNotFoundError(f"Missing results.json or feature_rows.csv under {result_dir}")
    meta = json.loads(results_path.read_text(encoding="utf-8"))
    with feature_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {
        "name": meta.get("dataset", {}).get("name", result_dir.name),
        "source": meta.get("dataset", {}).get("source", "unknown"),
        "dir": str(result_dir),
        "rows": rows,
    }


def run_block(result: dict, budget_ms: float, seed: int, alpha: float, latency_penalty: float):
    rows = [with_constrained_score(row, budget_ms, latency_penalty) for row in result["rows"]]
    qids = sorted({row["qid"] for row in rows})
    if len(qids) < 8:
        raise ValueError(f"Need at least 8 queries for constrained split, got {len(qids)} in {result['dir']}")
    train_qids, test_qids = train_test_split(qids, test_size=0.8, random_state=seed)
    train_set = set(train_qids)
    test_set = set(test_qids)
    train_rows = [row for row in rows if row["qid"] in train_set]
    test_rows = [row for row in rows if row["qid"] in test_set]

    methods: dict[str, dict] = {}
    for plan in STATIC_PLANS:
        methods[f"Static-{plan}"] = evaluate_static_plan(test_rows, plan)

    best_plan = best_static_plan(train_rows, STATIC_PLANS)
    methods["StaticBest-cal"] = evaluate_static_plan(test_rows, best_plan)
    methods["StaticBest-cal"]["selected_plan"] = best_plan

    query_cols = ["query_len", "query_digit_ratio"] + plan_cols(rows)
    full_cols = feature_cols() + plan_cols(rows)
    methods["QueryOnly-RF"] = evaluate_model_choice(train_rows, test_rows, query_cols, seed)
    methods["ScalarTelemetry-RF"] = evaluate_model_choice(train_rows, test_rows, full_cols, seed)
    methods["NoUncertainty-ParetoProbe"] = evaluate_model_choice(train_rows, test_rows, full_cols, seed)
    methods[METHOD_CANDIDATE] = evaluate_conformal_dual_choice(
        train_rows,
        test_rows,
        full_cols,
        seed,
        alpha,
        latency_penalty=latency_penalty,
    )

    block = f"{result['name']}|budget={budget_ms:g}ms|selectivity=1.0|seed={seed}"
    summary_rows = []
    rank_rows = []
    for method, metrics in methods.items():
        summary_rows.append(
            {
                "dataset": result["name"],
                "block": block,
                "budget_ms": budget_ms,
                "seed": seed,
                "method": method,
                **metrics,
            }
        )
        rank_rows.append(
            {
                "block": block,
                "method": method,
                "score": metrics["mean_score"],
                "higher_is_better": 1,
            }
        )
    return summary_rows, rank_rows


def with_constrained_score(row: dict, budget_ms: float, latency_penalty: float) -> dict:
    out = dict(row)
    latency = float(out["latency_ms"])
    ndcg = float(out["ndcg10"])
    recall = float(out["recall100"])
    violation = max(0.0, latency / max(budget_ms, 1e-9) - 1.0)
    feasible = latency <= budget_ms
    out["feasible"] = 1.0 if feasible else 0.0
    out["budget_ms"] = budget_ms
    out["constrained_score"] = ndcg if feasible else -latency_penalty * violation
    out["quality_score"] = 0.7 * ndcg + 0.3 * recall
    return out


def feature_cols() -> list[str]:
    return [
        "query_len",
        "query_digit_ratio",
        "bm25_top_score",
        "bm25_entropy",
        "bm25_gap",
        "dense_top_score",
        "dense_entropy",
        "dense_gap",
        "probe_overlap",
        "probe_latency_ms",
    ]


def plan_cols(rows: list[dict]) -> list[str]:
    return sorted(col for col in rows[0] if col.startswith("plan_"))


def best_static_plan(rows: list[dict], plans: list[str]) -> str:
    means = {
        plan: mean(float(row["constrained_score"]) for row in rows if row["plan"] == plan)
        for plan in plans
    }
    return max(means, key=means.get)


def evaluate_static_plan(rows: list[dict], plan: str) -> dict:
    return evaluate_chosen_rows([next(row for row in candidates if row["plan"] == plan) for candidates in by_qid(rows).values()], rows)


def evaluate_model_choice(train_rows: list[dict], test_rows: list[dict], cols: list[str], seed: int) -> dict:
    model = fit_model(train_rows, cols, seed)
    preds = predict_mean(model, test_rows, cols)
    return evaluate_predictions(test_rows, preds)


def evaluate_conformal_choice(
    train_rows: list[dict],
    test_rows: list[dict],
    cols: list[str],
    seed: int,
    alpha: float,
) -> dict:
    qids = sorted({row["qid"] for row in train_rows})
    if len(qids) >= 8:
        model_qids, cal_qids = train_test_split(qids, test_size=0.5, random_state=seed + 101)
        model_rows = [row for row in train_rows if row["qid"] in set(model_qids)]
        cal_rows = [row for row in train_rows if row["qid"] in set(cal_qids)]
    else:
        model_rows = train_rows
        cal_rows = train_rows

    model = fit_model(model_rows, cols, seed)
    cal_pred = predict_mean(model, cal_rows, cols)
    qhat_by_plan = conformal_qhat(cal_rows, cal_pred, alpha)
    test_pred = predict_mean(model, test_rows, cols)
    conservative = np.asarray(
        [pred - qhat_by_plan.get(row["plan"], qhat_by_plan["__global__"]) for row, pred in zip(test_rows, test_pred, strict=True)]
    )
    out = evaluate_predictions(test_rows, conservative)
    out["conformal_alpha"] = alpha
    out["mean_qhat"] = mean(v for k, v in qhat_by_plan.items() if k != "__global__")
    return out


def evaluate_conformal_dual_choice(
    train_rows: list[dict],
    test_rows: list[dict],
    cols: list[str],
    seed: int,
    alpha: float,
    latency_penalty: float,
) -> dict:
    qids = sorted({row["qid"] for row in train_rows})
    if len(qids) >= 8:
        model_qids, cal_qids = train_test_split(qids, test_size=0.5, random_state=seed + 211)
        model_rows = [row for row in train_rows if row["qid"] in set(model_qids)]
        cal_rows = [row for row in train_rows if row["qid"] in set(cal_qids)]
    else:
        model_rows = train_rows
        cal_rows = train_rows

    split_quality = fit_target_model(model_rows, cols, "ndcg10", seed)
    split_latency = fit_target_model(model_rows, cols, "latency_ms", seed + 1)
    q_pred_cal = predict_mean(split_quality, cal_rows, cols)
    l_pred_cal = predict_mean(split_latency, cal_rows, cols)

    candidates = []
    for a in sorted({alpha, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8}):
        q_lower = conformal_lower_qhat(cal_rows, q_pred_cal, "ndcg10", a)
        l_upper = conformal_upper_qhat(cal_rows, l_pred_cal, "latency_ms", a)
        cal_score = dual_scores(cal_rows, q_pred_cal, l_pred_cal, q_lower, l_upper, latency_penalty)
        cal_eval = evaluate_predictions(cal_rows, cal_score)
        candidates.append((cal_eval["mean_score"], a, q_lower, l_upper))
    _, best_alpha, q_lower, l_upper = max(candidates, key=lambda item: item[0])

    final_quality = fit_target_model(train_rows, cols, "ndcg10", seed)
    final_latency = fit_target_model(train_rows, cols, "latency_ms", seed + 1)
    q_pred_test = predict_mean(final_quality, test_rows, cols)
    l_pred_test = predict_mean(final_latency, test_rows, cols)
    test_score = dual_scores(test_rows, q_pred_test, l_pred_test, q_lower, l_upper, latency_penalty)
    out = evaluate_predictions(test_rows, test_score)
    out["conformal_alpha"] = best_alpha
    out["mean_quality_qhat"] = mean(v for k, v in q_lower.items() if k != "__global__")
    out["mean_latency_qhat"] = mean(v for k, v in l_upper.items() if k != "__global__")
    out["selector"] = "dual_quality_latency_conformal"
    return out


def evaluate_paretoprobe_c_portfolio(
    train_rows: list[dict],
    test_rows: list[dict],
    cols: list[str],
    seed: int,
    alpha: float,
    latency_penalty: float,
) -> dict:
    qids = sorted({row["qid"] for row in train_rows})
    if len(qids) >= 8:
        model_qids, cal_qids = train_test_split(qids, test_size=0.5, random_state=seed + 307)
        model_rows = [row for row in train_rows if row["qid"] in set(model_qids)]
        cal_rows = [row for row in train_rows if row["qid"] in set(cal_qids)]
    else:
        model_rows = train_rows
        cal_rows = train_rows

    candidates = []
    for family in ["rf", "et"]:
        split_score = fit_score_model(model_rows, cols, seed, family)
        cal_pred = predict_mean(split_score, cal_rows, cols)
        final_score = fit_score_model(train_rows, cols, seed, family)
        test_pred = predict_mean(final_score, test_rows, cols)

        candidates.append(make_candidate("mean", family, cal_rows, cal_pred, test_rows, test_pred))
        bias = plan_bias(cal_rows, cal_pred)
        candidates.append(make_candidate("plan_bias", family, cal_rows, apply_plan_bias(cal_rows, cal_pred, bias), test_rows, apply_plan_bias(test_rows, test_pred, bias)))
        for a in sorted({alpha, 0.05, 0.1, 0.2, 0.3, 0.5}):
            qhat = conformal_qhat(cal_rows, cal_pred, a)
            cal_lcb = apply_lcb(cal_rows, cal_pred, qhat)
            test_lcb = apply_lcb(test_rows, test_pred, qhat)
            candidates.append(make_candidate(f"score_lcb_alpha={a}", family, cal_rows, cal_lcb, test_rows, test_lcb))

    dual_candidates = dual_portfolio_candidates(model_rows, cal_rows, train_rows, test_rows, cols, seed, alpha, latency_penalty)
    candidates.extend(dual_candidates)

    best = max(candidates, key=lambda item: (item["cal_eval"]["mean_score"], -item["cal_eval"]["mean_regret"]))
    out = dict(best["test_eval"])
    out["selector"] = best["name"]
    out["calibration_score"] = best["cal_eval"]["mean_score"]
    out["calibration_regret"] = best["cal_eval"]["mean_regret"]
    return out


def make_candidate(
    name: str,
    family: str,
    cal_rows: list[dict],
    cal_score: np.ndarray,
    test_rows: list[dict],
    test_score: np.ndarray,
) -> dict:
    return {
        "name": f"{family}:{name}",
        "cal_eval": evaluate_predictions(cal_rows, cal_score),
        "test_eval": evaluate_predictions(test_rows, test_score),
    }


def dual_portfolio_candidates(
    model_rows: list[dict],
    cal_rows: list[dict],
    train_rows: list[dict],
    test_rows: list[dict],
    cols: list[str],
    seed: int,
    alpha: float,
    latency_penalty: float,
) -> list[dict]:
    out = []
    split_quality = fit_target_model(model_rows, cols, "ndcg10", seed)
    split_latency = fit_target_model(model_rows, cols, "latency_ms", seed + 1)
    final_quality = fit_target_model(train_rows, cols, "ndcg10", seed)
    final_latency = fit_target_model(train_rows, cols, "latency_ms", seed + 1)
    q_pred_cal = predict_mean(split_quality, cal_rows, cols)
    l_pred_cal = predict_mean(split_latency, cal_rows, cols)
    q_pred_test = predict_mean(final_quality, test_rows, cols)
    l_pred_test = predict_mean(final_latency, test_rows, cols)

    zero = {"__global__": 0.0, **{plan: 0.0 for plan in ALL_PLANS}}
    out.append(
        make_candidate(
            "dual_mean",
            "rf",
            cal_rows,
            dual_scores(cal_rows, q_pred_cal, l_pred_cal, zero, zero, latency_penalty),
            test_rows,
            dual_scores(test_rows, q_pred_test, l_pred_test, zero, zero, latency_penalty),
        )
    )
    for a in sorted({alpha, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8}):
        q_lower = conformal_lower_qhat(cal_rows, q_pred_cal, "ndcg10", a)
        l_upper = conformal_upper_qhat(cal_rows, l_pred_cal, "latency_ms", a)
        out.append(
            make_candidate(
                f"dual_conformal_alpha={a}",
                "rf",
                cal_rows,
                dual_scores(cal_rows, q_pred_cal, l_pred_cal, q_lower, l_upper, latency_penalty),
                test_rows,
                dual_scores(test_rows, q_pred_test, l_pred_test, q_lower, l_upper, latency_penalty),
            )
        )
    return out


def fit_score_model(rows: list[dict], cols: list[str], seed: int, family: str):
    x = np.asarray([[float(row.get(col, 0.0)) for col in cols] for row in rows], dtype=np.float64)
    y = np.asarray([float(row["constrained_score"]) for row in rows], dtype=np.float64)
    if family == "et":
        model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed)
    else:
        model = RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=seed)
    model.fit(x, y)
    return model


def plan_bias(rows: list[dict], pred: np.ndarray) -> dict[str, float]:
    vals: dict[str, list[float]] = {}
    all_vals = []
    for row, y_pred in zip(rows, pred, strict=True):
        residual = float(row["constrained_score"]) - float(y_pred)
        vals.setdefault(row["plan"], []).append(residual)
        all_vals.append(residual)
    out = {"__global__": mean(all_vals)}
    for plan, plan_vals in vals.items():
        out[plan] = mean(plan_vals)
    return out


def apply_plan_bias(rows: list[dict], pred: np.ndarray, bias: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [float(y_pred) + bias.get(row["plan"], bias["__global__"]) for row, y_pred in zip(rows, pred, strict=True)],
        dtype=np.float64,
    )


def apply_lcb(rows: list[dict], pred: np.ndarray, qhat_by_plan: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [float(y_pred) - qhat_by_plan.get(row["plan"], qhat_by_plan["__global__"]) for row, y_pred in zip(rows, pred, strict=True)],
        dtype=np.float64,
    )


def fit_target_model(rows: list[dict], cols: list[str], target: str, seed: int) -> RandomForestRegressor:
    x = np.asarray([[float(row.get(col, 0.0)) for col in cols] for row in rows], dtype=np.float64)
    y = np.asarray([float(row[target]) for row in rows], dtype=np.float64)
    model = RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=seed)
    model.fit(x, y)
    return model


def conformal_lower_qhat(rows: list[dict], pred: np.ndarray, target: str, alpha: float) -> dict[str, float]:
    residuals: dict[str, list[float]] = {}
    global_residuals = []
    for row, y_pred in zip(rows, pred, strict=True):
        residual = max(0.0, float(y_pred) - float(row[target]))
        residuals.setdefault(row["plan"], []).append(residual)
        global_residuals.append(residual)
    out = {"__global__": quantile_conformal(global_residuals, alpha)}
    for plan, vals in residuals.items():
        out[plan] = quantile_conformal(vals, alpha)
    return out


def conformal_upper_qhat(rows: list[dict], pred: np.ndarray, target: str, alpha: float) -> dict[str, float]:
    residuals: dict[str, list[float]] = {}
    global_residuals = []
    for row, y_pred in zip(rows, pred, strict=True):
        residual = max(0.0, float(row[target]) - float(y_pred))
        residuals.setdefault(row["plan"], []).append(residual)
        global_residuals.append(residual)
    out = {"__global__": quantile_conformal(global_residuals, alpha)}
    for plan, vals in residuals.items():
        out[plan] = quantile_conformal(vals, alpha)
    return out


def dual_scores(
    rows: list[dict],
    quality_pred: np.ndarray,
    latency_pred: np.ndarray,
    quality_qhat: dict[str, float],
    latency_qhat: dict[str, float],
    latency_penalty: float,
) -> np.ndarray:
    scores = []
    for row, q_pred, l_pred in zip(rows, quality_pred, latency_pred, strict=True):
        plan = row["plan"]
        budget_ms = infer_budget_from_row(row)
        quality_lcb = max(0.0, float(q_pred) - quality_qhat.get(plan, quality_qhat["__global__"]))
        latency_ucb = max(0.0, float(l_pred) + latency_qhat.get(plan, latency_qhat["__global__"]))
        violation = max(0.0, latency_ucb / max(budget_ms, 1e-9) - 1.0)
        scores.append(quality_lcb if latency_ucb <= budget_ms else -latency_penalty * violation)
    return np.asarray(scores, dtype=np.float64)


def infer_budget_from_row(row: dict) -> float:
    # The current pilot embeds the budget in constrained_score, so keep an explicit per-row copy.
    return float(row.get("budget_ms", 0.0))


def fit_model(rows: list[dict], cols: list[str], seed: int) -> RandomForestRegressor:
    x = np.asarray([[float(row.get(col, 0.0)) for col in cols] for row in rows], dtype=np.float64)
    y = np.asarray([float(row["constrained_score"]) for row in rows], dtype=np.float64)
    model = RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=seed)
    model.fit(x, y)
    return model


def predict_mean(model: RandomForestRegressor, rows: list[dict], cols: list[str]) -> np.ndarray:
    x = np.asarray([[float(row.get(col, 0.0)) for col in cols] for row in rows], dtype=np.float64)
    return model.predict(x)


def conformal_qhat(rows: list[dict], pred: np.ndarray, alpha: float) -> dict[str, float]:
    residuals: dict[str, list[float]] = {}
    global_residuals = []
    for row, y_pred in zip(rows, pred, strict=True):
        y = float(row["constrained_score"])
        residual = max(0.0, float(y_pred) - y)
        residuals.setdefault(row["plan"], []).append(residual)
        global_residuals.append(residual)
    out = {"__global__": quantile_conformal(global_residuals, alpha)}
    for plan, vals in residuals.items():
        out[plan] = quantile_conformal(vals, alpha)
    return out


def quantile_conformal(values: list[float], alpha: float) -> float:
    if not values:
        return 0.0
    vals = np.asarray(sorted(values), dtype=np.float64)
    n = len(vals)
    idx = min(n - 1, int(math.ceil((n + 1) * (1.0 - alpha))) - 1)
    return float(vals[max(0, idx)])


def evaluate_predictions(rows: list[dict], predictions: np.ndarray) -> dict:
    chosen = []
    for candidates in by_qid_pred(rows, predictions).values():
        chosen.append(max(candidates, key=lambda item: item[1])[0])
    return evaluate_chosen_rows(chosen, rows)


def evaluate_chosen_rows(chosen_rows: list[dict], all_rows: list[dict]) -> dict:
    oracle_by_qid = {
        qid: max(candidates, key=lambda row: float(row["constrained_score"]))
        for qid, candidates in by_qid(all_rows).items()
    }
    regrets = []
    correct = 0
    plan_counts = Counter()
    for chosen in chosen_rows:
        oracle = oracle_by_qid[chosen["qid"]]
        regrets.append(float(oracle["constrained_score"]) - float(chosen["constrained_score"]))
        correct += oracle["plan"] == chosen["plan"]
        plan_counts[chosen["plan"]] += 1
    n = max(1, len(chosen_rows))
    return {
        "queries": len(chosen_rows),
        "mean_score": mean(float(row["constrained_score"]) for row in chosen_rows),
        "mean_ndcg10": mean(float(row["ndcg10"]) for row in chosen_rows),
        "mean_recall100": mean(float(row["recall100"]) for row in chosen_rows),
        "satisfaction_rate": mean(float(row["feasible"]) for row in chosen_rows),
        "mean_regret": mean(regrets),
        "oracle_action_accuracy": correct / n,
        "plan_distribution": json.dumps(dict(sorted(plan_counts.items())), sort_keys=True),
    }


def by_qid(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(row["qid"], []).append(row)
    return out


def by_qid_pred(rows: list[dict], pred: np.ndarray) -> dict[str, list[tuple[dict, float]]]:
    out: dict[str, list[tuple[dict, float]]] = {}
    for row, y_pred in zip(rows, pred, strict=True):
        out.setdefault(row["qid"], []).append((row, float(y_pred)))
    return out


def mean(values) -> float:
    vals = list(values)
    return float(np.mean(vals)) if vals else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = sorted({field for row in rows for field in row})
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary_rows: list[dict], gate, budgets: list[float], seeds: list[int]) -> None:
    by_method: dict[str, list[float]] = {}
    by_method_sat: dict[str, list[float]] = {}
    by_method_regret: dict[str, list[float]] = {}
    for row in summary_rows:
        by_method.setdefault(row["method"], []).append(float(row["mean_score"]))
        by_method_sat.setdefault(row["method"], []).append(float(row["satisfaction_rate"]))
        by_method_regret.setdefault(row["method"], []).append(float(row["mean_regret"]))

    lines = [
        "# ParetoProbe-C Constrained Pilot",
        "",
        f"Budgets: `{', '.join(f'{b:g}ms' for b in budgets)}`",
        f"Seeds: `{', '.join(str(seed) for seed in seeds)}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Method | Mean constrained score | Satisfaction | Regret | Avg rank |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in sorted(by_method, key=lambda item: gate.average_ranks.get(item, math.inf)):
        lines.append(
            f"| `{method}` | {mean(by_method[method]):.4f} | {mean(by_method_sat[method]):.3f} | "
            f"{mean(by_method_regret[method]):.4f} | {gate.average_ranks.get(method, math.inf):.3f} |"
        )
    lines += [
        "",
        "## Gate",
        "",
        f"Pass: `{gate.pass_gate}`",
    ]
    if gate.failures:
        lines += ["", "Failures:"]
        lines.extend(f"- {failure}" for failure in gate.failures)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
