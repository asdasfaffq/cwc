#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paretoprobe.data import RetrievalDataset, load_beir_dataset, tokenize
from paretoprobe.metrics import aggregate, ndcg_at_k, recall_at_k
from paretoprobe.retrieval import (
    BM25Index,
    DenseSVDIndex,
    SearchResult,
    TransformerDenseIndex,
    rrf_fusion,
    score_entropy,
    weighted_fusion,
)


@dataclass
class RunConfig:
    dataset: str
    split: str
    max_docs: int
    max_queries: int
    top_k: int
    probe_k: int
    dense_backend: str
    dense_dim: int
    embedding_model: str
    embedding_batch_size: int
    embedding_max_length: int
    allow_download: bool
    allow_tiny_fallback: bool
    seed: int


def main() -> None:
    parser = argparse.ArgumentParser(description="ParetoProbe pilot harness for R001-R012.")
    parser.add_argument("--project-dir", type=Path, default=ROOT)
    parser.add_argument("--dataset", default="scifact", choices=["scifact", "nfcorpus", "fiqa"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-docs", type=int, default=6000)
    parser.add_argument("--max-queries", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--probe-k", type=int, default=20)
    parser.add_argument("--dense-backend", default="svd", choices=["svd", "transformer"])
    parser.add_argument("--dense-dim", type=int, default=128)
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-tiny-fallback", action="store_true")
    parser.add_argument("--runs", default="R001-R012", help="Comma-separated run ids or ranges, e.g. R001,R002,R005-R008")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "pilot")
    args = parser.parse_args()

    config = RunConfig(
        dataset=args.dataset,
        split=args.split,
        max_docs=args.max_docs,
        max_queries=args.max_queries,
        top_k=args.top_k,
        probe_k=args.probe_k,
        dense_backend=args.dense_backend,
        dense_dim=args.dense_dim,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        embedding_max_length=args.embedding_max_length,
        allow_download=not args.no_download,
        allow_tiny_fallback=not args.no_tiny_fallback,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = expand_runs(args.runs)

    dataset = load_beir_dataset(
        name=config.dataset,
        data_dir=args.project_dir / "data",
        split=config.split,
        max_docs=config.max_docs,
        max_queries=config.max_queries,
        allow_download=config.allow_download,
        allow_tiny_fallback=config.allow_tiny_fallback,
    )
    doc_ids = list(dataset.corpus)
    texts = [dataset.corpus[docid] for docid in doc_ids]
    bm25 = BM25Index(doc_ids, texts)
    dense = build_dense_index(doc_ids, texts, config, args.output_dir / "embedding_cache")

    all_results: dict[str, dict] = {}
    all_results["config"] = asdict(config)
    all_results["dataset"] = dataset_summary(dataset)

    search_cache: dict[str, dict[str, SearchResult]] = {}
    if any(
        run in selected
        for run in [
            "R002",
            "R003",
            "R004",
            "R005",
            "R006",
            "R008",
            "R009",
            "R010",
            "R011",
            "R012",
            "R013",
            "R014",
            "R015",
            "R016",
            "R017",
        ]
    ):
        search_cache = execute_base_searches(dataset, bm25, dense, config)

    if "R001" in selected:
        all_results["R001"] = run_r001(dataset)
    if "R002" in selected:
        all_results["R002"] = evaluate_plan(dataset, search_cache, "bm25_full", config.top_k)
    if "R003" in selected:
        all_results["R003"] = evaluate_plan(dataset, search_cache, "dense_full", config.top_k)
    if "R004" in selected:
        all_results["R004"] = run_r004(search_cache)
    if any(run in selected for run in ["R005", "R006", "R008"]):
        plan_rows = build_plan_rows(dataset, search_cache, config.top_k)
        write_csv(args.output_dir / "plan_rows.csv", plan_rows)
        if "R005" in selected:
            all_results["R005"] = summarize_plan_rows(plan_rows, ["bm25_full", "dense_full", "rrf", "weighted_0.5"])
        if "R006" in selected:
            all_results["R006"] = summarize_plan_rows(plan_rows, ["dense_probe", "dense_full"])
        if "R008" in selected:
            all_results["R008"] = summarize_frontier(plan_rows)
    if "R007" in selected:
        all_results["R007"] = {
            "status": "SKIPPED",
            "reason": "Optional reranker actions are not implemented in the local pilot harness.",
        }
    if any(run in selected for run in ["R009", "R010", "R011", "R012", "R013", "R014", "R015", "R016", "R017"]):
        plan_rows = build_plan_rows(dataset, search_cache, config.top_k)
        feature_rows = build_feature_rows(dataset, search_cache, plan_rows)
        write_csv(args.output_dir / "feature_rows.csv", feature_rows)
        predictor_results = run_predictor_experiments(feature_rows, config.seed)
        main_results = run_main_method_experiments(feature_rows, config.seed)
        predictor_results.update(main_results)
        for run in ["R009", "R010", "R011", "R012", "R013", "R014", "R015", "R016", "R017"]:
            if run in selected:
                all_results[run] = predictor_results[run]

    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_markdown(args.output_dir / "SUMMARY.md", all_results)
    print(json.dumps({"result_path": str(result_path), "runs": sorted(selected), "dataset": all_results["dataset"]}, indent=2))


def expand_runs(spec: str) -> set[str]:
    runs: set[str] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            s = int(start[1:])
            e = int(end[1:])
            prefix = start[0]
            for idx in range(s, e + 1):
                runs.add(f"{prefix}{idx:03d}")
        else:
            runs.add(part)
    return runs


def dataset_summary(dataset: RetrievalDataset) -> dict:
    return {
        "name": dataset.name,
        "source": dataset.source,
        "docs": len(dataset.corpus),
        "queries": len(dataset.queries),
        "qrels": sum(len(v) for v in dataset.qrels.values()),
    }


def build_dense_index(doc_ids: list[str], texts: list[str], config: RunConfig, cache_dir: Path):
    if config.dense_backend == "svd":
        return DenseSVDIndex(doc_ids, texts, dim=config.dense_dim)
    query_prefix = ""
    if "bge-" in config.embedding_model.lower():
        query_prefix = "Represent this sentence for searching relevant passages: "
    return TransformerDenseIndex(
        doc_ids,
        texts,
        model_name=config.embedding_model,
        batch_size=config.embedding_batch_size,
        max_length=config.embedding_max_length,
        cache_dir=cache_dir,
        query_prefix=query_prefix,
    )


def run_r001(dataset: RetrievalDataset) -> dict:
    return {
        "status": "PASS" if dataset.corpus and dataset.queries and dataset.qrels else "FAIL",
        **dataset_summary(dataset),
        "warning": "tiny_fallback is for code sanity only, not paper evidence" if dataset.source == "tiny_fallback" else "",
    }


def execute_base_searches(
    dataset: RetrievalDataset,
    bm25: BM25Index,
    dense: DenseSVDIndex,
    config: RunConfig,
) -> dict[str, dict[str, SearchResult]]:
    cache: dict[str, dict[str, SearchResult]] = {}
    for qid, query in dataset.queries.items():
        bm25_probe = bm25.search(query, config.probe_k)
        bm25_full = bm25.search(query, config.top_k)
        dense_probe = dense.search(query, config.probe_k)
        dense_full = dense.search(query, config.top_k)
        rrf = rrf_fusion([bm25_full, dense_full], config.top_k)
        weighted = weighted_fusion(bm25_full, dense_full, config.top_k, dense_weight=0.5)
        cache[qid] = {
            "bm25_probe": bm25_probe,
            "bm25_full": bm25_full,
            "dense_probe": dense_probe,
            "dense_full": dense_full,
            "rrf": rrf,
            "weighted_0.5": weighted,
        }
    return cache


def evaluate_plan(dataset: RetrievalDataset, cache: dict[str, dict[str, SearchResult]], plan: str, top_k: int) -> dict:
    rows = []
    for qid, rels in dataset.qrels.items():
        result = cache[qid][plan]
        rows.append(
            {
                "qid": qid,
                "ndcg10": ndcg_at_k(result.ranking, rels, 10),
                "recall100": recall_at_k(result.ranking, rels, min(100, top_k)),
                "latency_ms": result.latency_ms,
            }
        )
    return summarize_metric_rows(rows)


def run_r004(cache: dict[str, dict[str, SearchResult]]) -> dict:
    latencies = [
        cache[qid]["bm25_probe"].latency_ms + cache[qid]["dense_probe"].latency_ms
        for qid in cache
    ]
    return {
        "status": "PASS" if latencies else "FAIL",
        "probe_latency_p50_ms": percentile(latencies, 50),
        "probe_latency_p95_ms": percentile(latencies, 95),
        "queries": len(latencies),
    }


def build_plan_rows(dataset: RetrievalDataset, cache: dict[str, dict[str, SearchResult]], top_k: int) -> list[dict]:
    rows = []
    for qid, rels in dataset.qrels.items():
        for plan, result in cache[qid].items():
            if plan.endswith("_probe"):
                # Probe plans are evaluated too, but with their shorter ranking.
                pass
            ndcg10 = ndcg_at_k(result.ranking, rels, 10)
            recall100 = recall_at_k(result.ranking, rels, min(100, top_k))
            utility = ndcg10 - 0.002 * result.latency_ms
            rows.append(
                {
                    "qid": qid,
                    "plan": plan,
                    "ndcg10": ndcg10,
                    "recall100": recall100,
                    "latency_ms": result.latency_ms,
                    "utility": utility,
                }
            )
    return rows


def summarize_plan_rows(rows: list[dict], plans: list[str]) -> dict:
    out = {}
    for plan in plans:
        plan_rows = [row for row in rows if row["plan"] == plan]
        out[plan] = summarize_metric_rows(plan_rows)
    return out


def summarize_metric_rows(rows: list[dict]) -> dict:
    latencies = [float(row["latency_ms"]) for row in rows]
    return {
        "queries": len(rows),
        "ndcg10": aggregate([float(row["ndcg10"]) for row in rows]),
        "recall100": aggregate([float(row["recall100"]) for row in rows]),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
    }


def summarize_frontier(rows: list[dict]) -> dict:
    by_qid: dict[str, list[dict]] = {}
    for row in rows:
        by_qid.setdefault(row["qid"], []).append(row)
    regrets = {}
    plan_counts = {}
    for qid, q_rows in by_qid.items():
        best = max(float(row["utility"]) for row in q_rows)
        best_plan = max(q_rows, key=lambda row: float(row["utility"]))["plan"]
        plan_counts[best_plan] = plan_counts.get(best_plan, 0) + 1
        for row in q_rows:
            plan = row["plan"]
            regrets.setdefault(plan, []).append(best - float(row["utility"]))
    return {
        "oracle_best_plan_counts": plan_counts,
        "mean_regret_by_plan": {plan: aggregate(vals) for plan, vals in regrets.items()},
        "nontrivial_plan_diversity": len(plan_counts),
    }


def build_feature_rows(
    dataset: RetrievalDataset,
    cache: dict[str, dict[str, SearchResult]],
    plan_rows: list[dict],
) -> list[dict]:
    by_qid = {qid: {} for qid in dataset.qrels}
    for qid, query in dataset.queries.items():
        bm25_probe = cache[qid]["bm25_probe"]
        dense_probe = cache[qid]["dense_probe"]
        bm25_docs = set(bm25_probe.ranking)
        dense_docs = set(dense_probe.ranking)
        overlap = len(bm25_docs & dense_docs) / max(1, len(bm25_docs | dense_docs))
        toks = tokenize(query)
        by_qid[qid] = {
            "query_len": len(toks),
            "query_digit_ratio": sum(any(ch.isdigit() for ch in tok) for tok in toks) / max(1, len(toks)),
            "bm25_top_score": first_score(bm25_probe),
            "bm25_entropy": score_entropy(bm25_probe.scores),
            "bm25_gap": score_gap(bm25_probe),
            "dense_top_score": first_score(dense_probe),
            "dense_entropy": score_entropy(dense_probe.scores),
            "dense_gap": score_gap(dense_probe),
            "probe_overlap": overlap,
            "probe_latency_ms": bm25_probe.latency_ms + dense_probe.latency_ms,
        }

    rows = []
    plans = sorted({row["plan"] for row in plan_rows})
    for row in plan_rows:
        features = dict(by_qid[row["qid"]])
        features.update({f"plan_{plan}": 1.0 if row["plan"] == plan else 0.0 for plan in plans})
        features.update(row)
        rows.append(features)
    return rows


def run_predictor_experiments(rows: list[dict], seed: int) -> dict[str, dict]:
    feature_sets = {
        "R009": ["query_len", "query_digit_ratio"],
        "R010": ["query_len", "query_digit_ratio", "bm25_top_score", "bm25_entropy", "bm25_gap"],
        "R011": ["query_len", "query_digit_ratio", "dense_top_score", "dense_entropy", "dense_gap"],
        "R012": [
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
        ],
    }
    plan_cols = sorted(k for k in rows[0] if k.startswith("plan_"))
    qids = sorted({str(row["qid"]) for row in rows})
    train_qids, test_qids = train_test_split(qids, test_size=0.8, random_state=seed)
    train_set = set(train_qids)
    test_set = set(test_qids)
    out = {}
    for run_id, base_features in feature_sets.items():
        cols = base_features + plan_cols
        train_rows = [row for row in rows if row["qid"] in train_set]
        test_rows = [row for row in rows if row["qid"] in test_set]
        model = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=seed)
        x_train = np.asarray([[float(row.get(col, 0.0)) for col in cols] for row in train_rows])
        y_train = np.asarray([float(row["utility"]) for row in train_rows])
        x_test = np.asarray([[float(row.get(col, 0.0)) for col in cols] for row in test_rows])
        y_test = np.asarray([float(row["utility"]) for row in test_rows])
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        choice = evaluate_plan_choice(test_rows, pred)
        out[run_id] = {
            "train_queries": len(train_set),
            "eval_queries": len(test_set),
            "mae_utility": float(mean_absolute_error(y_test, pred)),
            **choice,
        }
    return out


def run_main_method_experiments(rows: list[dict], seed: int) -> dict[str, dict]:
    qids = sorted({str(row["qid"]) for row in rows})
    train_qids, test_qids = train_test_split(qids, test_size=0.8, random_state=seed)
    train_set = set(train_qids)
    test_set = set(test_qids)
    train_rows = [row for row in rows if row["qid"] in train_set]
    test_rows = [row for row in rows if row["qid"] in test_set]
    plan_cols = sorted(k for k in rows[0] if k.startswith("plan_"))
    full_features = [
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
    ] + plan_cols
    query_features = ["query_len", "query_digit_ratio"] + plan_cols

    fixed_plans = ["bm25_full", "dense_full", "rrf", "weighted_0.5"]
    best_fixed = best_static_plan(train_rows, fixed_plans)
    r013 = evaluate_static_plan(test_rows, best_fixed)
    r013["selected_static_plan"] = best_fixed

    query_model = fit_forest(train_rows, query_features, seed)
    query_pred = predict_forest_mean(query_model, test_rows, query_features)
    r014 = evaluate_plan_choice(test_rows, query_pred)

    scalar_model = fit_forest(train_rows, full_features, seed)
    scalar_pred = predict_forest_mean(scalar_model, test_rows, full_features)
    r015 = evaluate_plan_choice(test_rows, scalar_pred)
    r016 = dict(r015)
    r016["note"] = "Point-estimate ParetoProbe without uncertainty intervals."

    robust_pred = predict_forest_lcb(scalar_model, test_rows, full_features, beta=0.25)
    r017 = evaluate_plan_choice(test_rows, robust_pred)
    r017["note"] = "Robust selector uses random-forest lower confidence bound as an interval proxy."

    return {
        "R013": r013,
        "R014": r014,
        "R015": r015,
        "R016": r016,
        "R017": r017,
    }


def fit_forest(rows: list[dict], cols: list[str], seed: int) -> RandomForestRegressor:
    x = np.asarray([[float(row.get(col, 0.0)) for col in cols] for row in rows])
    y = np.asarray([float(row["utility"]) for row in rows])
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=seed)
    model.fit(x, y)
    return model


def predict_forest_mean(model: RandomForestRegressor, rows: list[dict], cols: list[str]) -> np.ndarray:
    x = np.asarray([[float(row.get(col, 0.0)) for col in cols] for row in rows])
    return model.predict(x)


def predict_forest_lcb(model: RandomForestRegressor, rows: list[dict], cols: list[str], beta: float) -> np.ndarray:
    x = np.asarray([[float(row.get(col, 0.0)) for col in cols] for row in rows])
    tree_preds = np.asarray([tree.predict(x) for tree in model.estimators_])
    return tree_preds.mean(axis=0) - beta * tree_preds.std(axis=0)


def best_static_plan(rows: list[dict], plans: list[str]) -> str:
    means = {}
    for plan in plans:
        vals = [float(row["utility"]) for row in rows if row["plan"] == plan]
        means[plan] = aggregate(vals)
    return max(means, key=means.get)


def evaluate_static_plan(rows: list[dict], plan: str) -> dict:
    by_qid: dict[str, list[dict]] = {}
    for row in rows:
        by_qid.setdefault(str(row["qid"]), []).append(row)
    regrets = []
    chosen_utils = []
    for candidates in by_qid.values():
        true_best = max(candidates, key=lambda row: float(row["utility"]))
        chosen = next(row for row in candidates if row["plan"] == plan)
        regrets.append(float(true_best["utility"]) - float(chosen["utility"]))
        chosen_utils.append(float(chosen["utility"]))
    return {
        "queries": len(by_qid),
        "mean_regret": aggregate(regrets),
        "mean_utility": aggregate(chosen_utils),
    }


def evaluate_plan_choice(rows: list[dict], predictions: np.ndarray) -> dict:
    by_qid: dict[str, list[tuple[dict, float]]] = {}
    for row, pred in zip(rows, predictions, strict=True):
        by_qid.setdefault(str(row["qid"]), []).append((row, float(pred)))
    regrets = []
    correct = 0
    for qid, candidates in by_qid.items():
        true_best = max(candidates, key=lambda item: float(item[0]["utility"]))
        pred_best = max(candidates, key=lambda item: item[1])
        if true_best[0]["plan"] == pred_best[0]["plan"]:
            correct += 1
        regrets.append(float(true_best[0]["utility"]) - float(pred_best[0]["utility"]))
    return {
        "action_accuracy": correct / max(1, len(by_qid)),
        "mean_regret": aggregate(regrets),
        "queries": len(by_qid),
    }


def first_score(result: SearchResult) -> float:
    if not result.ranking:
        return 0.0
    return float(result.scores.get(result.ranking[0], 0.0))


def score_gap(result: SearchResult) -> float:
    if len(result.ranking) < 2:
        return first_score(result)
    return float(result.scores.get(result.ranking[0], 0.0) - result.scores.get(result.ranking[1], 0.0))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(path: Path, results: dict[str, dict]) -> None:
    lines = [
        "# ParetoProbe Pilot Summary",
        "",
        f"Dataset: `{results['dataset']['name']}` source `{results['dataset']['source']}`",
        "",
        "| Run | Key Result |",
        "|---|---|",
    ]
    for run_id in sorted(k for k in results if k.startswith("R")):
        lines.append(f"| {run_id} | `{json.dumps(results[run_id], ensure_ascii=False)[:500]}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
