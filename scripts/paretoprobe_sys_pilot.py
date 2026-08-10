#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paretoprobe.data import RetrievalDataset, load_beir_dataset, tokenize
from paretoprobe.metrics import ndcg_at_k, recall_at_k
from paretoprobe.retrieval import BM25Index, SearchResult, TransformerDenseIndex, rrf_fusion, score_entropy, weighted_fusion
from scripts.statistical_rank_gate import evaluate_gate, render_markdown


METHOD_CANDIDATE = "ParetoProbe-Sys"
RERANK_DEPTHS = [20, 50, 100]


class CrossEncoderReranker:
    def __init__(self, model_name: str, batch_size: int):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Real reranking requires torch and transformers.") from exc
        self.torch = torch
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def rerank(
        self,
        query: str,
        corpus: dict[str, str],
        bm25: SearchResult,
        dense: SearchResult,
        selectivity: float,
        top_k: int,
        depth: int,
        metadata: dict[str, str] | None,
    ) -> SearchResult:
        start_docs = []
        seen = set()
        for docid in bm25.ranking[:depth] + dense.ranking[:depth]:
            if docid in seen or not passes_filter(docid, selectivity, metadata):
                continue
            seen.add(docid)
            start_docs.append(docid)
        scores = self.score(query, [corpus.get(docid, "") for docid in start_docs])
        score_map = {docid: float(score) for docid, score in zip(start_docs, scores, strict=True)}
        ranking = [docid for docid, _ in sorted(score_map.items(), key=lambda item: item[1], reverse=True)[:top_k]]
        latency = bm25.latency_ms + dense.latency_ms + 0.85 * len(start_docs)
        return SearchResult(ranking, score_map, latency)

    def score(self, query: str, docs: list[str]) -> list[float]:
        out: list[float] = []
        with self.torch.no_grad():
            for start in range(0, len(docs), self.batch_size):
                batch_docs = docs[start : start + self.batch_size]
                encoded = self.tokenizer(
                    [query] * len(batch_docs),
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=384,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits
                if logits.ndim == 2 and logits.shape[1] > 1:
                    vals = logits[:, -1]
                else:
                    vals = logits.reshape(-1)
                out.extend(vals.detach().cpu().numpy().astype(float).tolist())
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description="ParetoProbe-Sys action-space pilot for filtered/rerank physical planning.")
    parser.add_argument("--datasets", default="scifact,nfcorpus,fiqa")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-docs", type=int, default=12000)
    parser.add_argument("--max-queries", type=int, default=220)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--probe-k", type=int, default=20)
    parser.add_argument("--overfetch-k", type=int, default=1000)
    parser.add_argument("--budgets-ms", default="20,40,80,150")
    parser.add_argument("--selectivities", default="1.0,0.5,0.1,0.01")
    parser.add_argument("--seeds", default="13,17,23")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--extra-dense-models", default="")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-max-length", type=int, default=256)
    parser.add_argument("--external-ranking-dir", type=Path)
    parser.add_argument("--metadata-file", type=Path)
    parser.add_argument("--real-rerank", action="store_true")
    parser.add_argument("--rank1-delta", action="store_true")
    parser.add_argument("--selector-margin", type=float, default=0.005)
    parser.add_argument("--foundation-prior-weight", type=float, default=0.0)
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
        choices=["transductive", "train_only", "random", "query_only", "compositional_guard", "component_shrink"],
        default="transductive",
    )
    parser.add_argument("--rerank-depths", default="20,50,100")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "sys_pilot")
    parser.add_argument("--min-test-queries", type=int, default=30)
    parser.add_argument("--test-size", type=float, default=0.5)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    budgets = [float(item) for item in args.budgets_ms.split(",") if item.strip()]
    selectivities = [float(item) for item in args.selectivities.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    args.rerank_depths = [int(item) for item in args.rerank_depths.split(",") if item.strip()]
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
    metadata = load_metadata(args.metadata_file) if args.metadata_file else None

    all_action_rows = []
    summary_rows = []
    rank_rows = []
    for dataset_name in datasets:
        dataset = load_beir_dataset(
            name=dataset_name,
            data_dir=ROOT / "data",
            split=args.split,
            max_docs=args.max_docs,
            max_queries=args.max_queries,
            allow_download=not args.no_download,
            allow_tiny_fallback=False,
        )
        external_rankings = load_external_rankings(args.external_ranking_dir, dataset.name) if args.external_ranking_dir else {}
        dataset_dir = args.output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        action_rows = build_action_rows(dataset, args, dataset_dir, external_rankings, metadata)
        write_csv(dataset_dir / "action_rows.csv", action_rows)
        all_action_rows.extend(action_rows)

        for selectivity in selectivities:
            sel_rows = [row for row in action_rows if math.isclose(float(row["selectivity"]), selectivity)]
            qids = sorted({row["qid"] for row in sel_rows})
            if len(qids) < args.min_test_queries:
                continue
            for budget_ms in budgets:
                for seed in seeds:
                    block_summary, block_rank = run_block(sel_rows, dataset.name, selectivity, budget_ms, seed, args.test_size)
                    summary_rows.extend(block_summary)
                    rank_rows.extend(block_rank)

    write_csv(args.output_dir / "all_action_rows.csv", all_action_rows)
    write_csv(args.output_dir / "sys_summary.csv", summary_rows)
    write_csv(args.output_dir / "rank_gate_input.csv", rank_rows)
    gate = evaluate_gate(
        [{k: str(v) for k, v in row.items()} for row in rank_rows],
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
    print(
        json.dumps(
            {
                "action_rows": str(args.output_dir / "all_action_rows.csv"),
                "summary": str(args.output_dir / "sys_summary.csv"),
                "rank_gate": str(args.output_dir / "RANK_GATE.md"),
                "pass_gate": gate.pass_gate,
                "average_ranks": gate.average_ranks,
                "failures": gate.failures,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def build_action_rows(
    dataset: RetrievalDataset,
    args,
    dataset_dir: Path,
    external_rankings: dict[str, dict[str, SearchResult]],
    metadata: dict[str, str] | None,
) -> list[dict]:
    doc_ids = list(dataset.corpus)
    texts = [dataset.corpus[docid] for docid in doc_ids]
    bm25 = BM25Index(doc_ids, texts)
    dense_models = [args.embedding_model] + [item.strip() for item in args.extra_dense_models.split(",") if item.strip()]
    dense_indexes = {}
    for model_name in dense_models:
        dense_indexes[dense_action_name(model_name)] = TransformerDenseIndex(
            doc_ids,
            texts,
            model_name=model_name,
            batch_size=args.embedding_batch_size,
            max_length=args.embedding_max_length,
            cache_dir=cache_dir_for(dataset.name, dataset_dir),
            query_prefix=query_prefix_for_model(model_name),
            doc_prefix=doc_prefix_for_model(model_name),
            pooling=pooling_for_model(model_name),
        )
    dense = dense_indexes[dense_action_name(args.embedding_model)]
    reranker = CrossEncoderReranker(args.reranker_model, args.reranker_batch_size) if args.real_rerank else None

    rows = []
    selectivities = [float(item) for item in args.selectivities.split(",") if item.strip()]
    for qid, query in dataset.queries.items():
        if qid not in dataset.qrels:
            continue
        probe = probe_features(query, bm25, dense, args.probe_k)
        bm25_deep = bm25.search(query, args.overfetch_k)
        dense_deeps = {name: index.search(query, args.overfetch_k) for name, index in dense_indexes.items()}
        dense_deep = dense_deeps[dense_action_name(args.embedding_model)]
        base_results = build_base_results(bm25_deep, dense_deep, args.top_k)
        q_external = {name: by_qid[qid] for name, by_qid in external_rankings.items() if qid in by_qid}
        for selectivity in selectivities:
            filtered_rels = {
                docid: score for docid, score in dataset.qrels[qid].items() if passes_filter(docid, selectivity, metadata)
            }
            if not filtered_rels:
                continue
            for action, result in build_filtered_actions(
                query=query,
                corpus=dataset.corpus,
                bm25_deep=bm25_deep,
                dense_deep=dense_deep,
                dense_deeps=dense_deeps,
                base_results=base_results,
                external_results=q_external,
                reranker=reranker,
                rerank_depths=args.rerank_depths,
                selectivity=selectivity,
                top_k=args.top_k,
                metadata=metadata,
            ).items():
                ndcg10 = ndcg_at_k(result.ranking, filtered_rels, 10)
                recall100 = recall_at_k(result.ranking, filtered_rels, min(100, args.top_k))
                row = {
                    "dataset": dataset.name,
                    "qid": qid,
                    "query_len": len(tokenize(query)),
                    "query_digit_ratio": digit_ratio(query),
                    "selectivity": selectivity,
                    "action": action,
                    "ndcg10": ndcg10,
                    "recall100": recall100,
                    "latency_ms": result.latency_ms,
                    "candidate_count": len(result.ranking),
                    **probe,
                    **action_features(action),
                }
                rows.append(row)
    return rows


def cache_dir_for(dataset_name: str, fallback_dir: Path) -> Path:
    existing = ROOT / "results" / f"{dataset_name}_bge" / "embedding_cache"
    if existing.exists():
        return existing
    return fallback_dir / "embedding_cache"


def dense_action_name(model_name: str) -> str:
    short = model_name.rsplit("/", 1)[-1]
    return "dense_" + safe_feature_name(short)


def query_prefix_for_model(model_name: str) -> str:
    lower = model_name.lower()
    if "e5" in lower:
        return "query: "
    if "bge-m3" in lower or "bge_m3" in lower:
        return ""
    if "bge" in lower:
        return "Represent this sentence for searching relevant passages: "
    return ""


def doc_prefix_for_model(model_name: str) -> str:
    return "passage: " if "e5" in model_name.lower() else ""


def pooling_for_model(model_name: str) -> str:
    return "cls" if "bge" in model_name.lower() else "mean"


def probe_features(query: str, bm25: BM25Index, dense: TransformerDenseIndex, probe_k: int) -> dict:
    bm25_probe = bm25.search(query, probe_k)
    dense_probe = dense.search(query, probe_k)
    bm25_docs = set(bm25_probe.ranking)
    dense_docs = set(dense_probe.ranking)
    union = bm25_docs | dense_docs
    return {
        "bm25_top_score": first_score(bm25_probe),
        "bm25_entropy": score_entropy(bm25_probe.scores),
        "bm25_gap": score_gap(bm25_probe),
        "dense_top_score": first_score(dense_probe),
        "dense_entropy": score_entropy(dense_probe.scores),
        "dense_gap": score_gap(dense_probe),
        "probe_overlap": len(bm25_docs & dense_docs) / max(1, len(union)),
        "probe_latency_ms": bm25_probe.latency_ms + dense_probe.latency_ms,
    }


def build_base_results(bm25_deep: SearchResult, dense_deep: SearchResult, top_k: int) -> dict[str, SearchResult]:
    bm25_top = truncate_result(bm25_deep, top_k)
    dense_top = truncate_result(dense_deep, top_k)
    return {
        "bm25": bm25_top,
        "dense": dense_top,
        "rrf": rrf_fusion([bm25_top, dense_top], top_k),
        "weighted": weighted_fusion(bm25_top, dense_top, top_k, dense_weight=0.5),
    }


def build_filtered_actions(
    query: str,
    corpus: dict[str, str],
    bm25_deep: SearchResult,
    dense_deep: SearchResult,
    dense_deeps: dict[str, SearchResult],
    base_results: dict[str, SearchResult],
    external_results: dict[str, SearchResult],
    reranker: "CrossEncoderReranker | None",
    rerank_depths: list[int],
    selectivity: float,
    top_k: int,
    metadata: dict[str, str] | None,
) -> dict[str, SearchResult]:
    actions = {
        "bm25_post": filter_result(base_results["bm25"], selectivity, top_k, metadata, extra_latency_ms=0.05),
        "dense_post": filter_result(base_results["dense"], selectivity, top_k, metadata, extra_latency_ms=0.05),
        "rrf_post": filter_result(base_results["rrf"], selectivity, top_k, metadata, extra_latency_ms=0.05),
        "weighted_post": filter_result(base_results["weighted"], selectivity, top_k, metadata, extra_latency_ms=0.05),
        "bm25_pre": filter_result(bm25_deep, selectivity, top_k, metadata, latency_scale=max(selectivity, 0.02)),
        "dense_pre": filter_result(dense_deep, selectivity, top_k, metadata, latency_scale=max(selectivity, 0.02)),
    }
    for dense_name, dense_result in dense_deeps.items():
        if dense_name == "dense_" + safe_feature_name("bge-small-en-v1.5"):
            continue
        dense_top = truncate_result(dense_result, top_k)
        actions[f"{dense_name}_post"] = filter_result(dense_top, selectivity, top_k, metadata, extra_latency_ms=0.05)
        actions[f"{dense_name}_pre"] = filter_result(dense_result, selectivity, top_k, metadata, latency_scale=max(selectivity, 0.02))
        actions[f"rrf_{dense_name}_post"] = filter_result(
            rrf_fusion([base_results["bm25"], dense_top], top_k),
            selectivity,
            top_k,
            metadata,
            extra_latency_ms=0.05,
        )
        actions[f"rrf_{dense_name}_pre"] = rrf_fusion([actions["bm25_pre"], actions[f"{dense_name}_pre"]], top_k)
        actions[f"weighted_{dense_name}_post"] = filter_result(
            weighted_fusion(base_results["bm25"], dense_top, top_k, dense_weight=0.5),
            selectivity,
            top_k,
            metadata,
            extra_latency_ms=0.05,
        )
        actions[f"weighted_{dense_name}_pre"] = weighted_fusion(
            actions["bm25_pre"],
            actions[f"{dense_name}_pre"],
            top_k,
            dense_weight=0.5,
        )
    for name, result in external_results.items():
        actions[f"external_{name}_post"] = filter_result(result, selectivity, top_k, metadata, extra_latency_ms=0.05)
        actions[f"external_{name}_pre"] = filter_result(result, selectivity, top_k, metadata, latency_scale=max(selectivity, 0.02))
    pre_bm25 = actions["bm25_pre"]
    pre_dense = actions["dense_pre"]
    actions["rrf_pre"] = rrf_fusion([pre_bm25, pre_dense], top_k)
    actions["weighted_pre"] = weighted_fusion(pre_bm25, pre_dense, top_k, dense_weight=0.5)

    for depth in rerank_depths:
        if reranker is None:
            actions[f"rerank_depth_{depth}"] = rerank_candidates(query, corpus, pre_bm25, pre_dense, selectivity, top_k, depth, metadata)
        else:
            actions[f"ce_rerank_depth_{depth}"] = reranker.rerank(query, corpus, pre_bm25, pre_dense, selectivity, top_k, depth, metadata)
    return actions


def filter_result(
    result: SearchResult,
    selectivity: float,
    top_k: int,
    metadata: dict[str, str] | None,
    extra_latency_ms: float = 0.0,
    latency_scale: float = 1.0,
) -> SearchResult:
    ranking = [docid for docid in result.ranking if passes_filter(docid, selectivity, metadata)][:top_k]
    scores = {docid: result.scores[docid] for docid in ranking if docid in result.scores}
    latency = result.latency_ms * latency_scale + extra_latency_ms
    return SearchResult(ranking, scores, latency)


def rerank_candidates(
    query: str,
    corpus: dict[str, str],
    bm25: SearchResult,
    dense: SearchResult,
    selectivity: float,
    top_k: int,
    depth: int,
    metadata: dict[str, str] | None,
) -> SearchResult:
    bm25_norm = normalize_score_map(bm25.scores)
    dense_norm = normalize_score_map(dense.scores)
    candidates = []
    seen = set()
    for docid in bm25.ranking[:depth] + dense.ranking[:depth]:
        if docid in seen or not passes_filter(docid, selectivity, metadata):
            continue
        seen.add(docid)
        candidates.append(docid)
    q_tokens = set(tokenize(query))
    scores = {}
    for docid in candidates:
        d_tokens = set(tokenize(corpus.get(docid, "")))
        lexical = len(q_tokens & d_tokens) / max(1, len(q_tokens))
        scores[docid] = 0.45 * bm25_norm.get(docid, 0.0) + 0.45 * dense_norm.get(docid, 0.0) + 0.10 * lexical
    ranking = [docid for docid, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]]
    latency = bm25.latency_ms + dense.latency_ms + 0.18 * len(candidates)
    return SearchResult(ranking, scores, latency)


def run_block(rows: list[dict], dataset: str, selectivity: float, budget_ms: float, seed: int, test_size: float):
    block_rows = [with_budget(row, budget_ms) for row in rows]
    qids = sorted({row["qid"] for row in block_rows})
    train_qids, test_qids = train_test_split(qids, test_size=test_size, random_state=seed)
    train_rows = [row for row in block_rows if row["qid"] in set(train_qids)]
    test_rows = [row for row in block_rows if row["qid"] in set(test_qids)]

    methods: dict[str, dict] = {}
    static_actions = sorted({row["action"] for row in block_rows})
    for action in static_actions:
        methods[f"Static-{action}"] = evaluate_static(test_rows, action)
    best_static = best_action(train_rows, static_actions)
    methods["StaticBest-cal"] = evaluate_static(test_rows, best_static)
    methods["StaticBest-cal"]["selected_action"] = best_static
    methods["QueryOnly-RF"] = evaluate_model(train_rows, test_rows, query_features(block_rows), seed)
    methods["ScalarTelemetry-RF"] = evaluate_model(train_rows, test_rows, full_features(block_rows), seed)
    methods["CostGreedy-cal"] = evaluate_cost_greedy(train_rows, test_rows, budget_ms)
    methods["AgentIR-Cascade-proxy"] = evaluate_agentir_cascade_proxy(train_rows, test_rows)
    methods["RCTMARS-SoftRouter-proxy"] = evaluate_soft_signature_router(train_rows, test_rows, seed)
    methods["CARROT-CostQualityLCB-proxy"] = evaluate_cost_quality_lcb_router(train_rows, test_rows, seed, budget_ms)
    methods[METHOD_CANDIDATE] = evaluate_paretoprobe_sys(train_rows, test_rows, full_features(block_rows), seed, budget_ms)

    block = f"{dataset}|budget={budget_ms:g}ms|selectivity={selectivity:g}|seed={seed}"
    summary_rows = []
    rank_rows = []
    for method, metrics in methods.items():
        row = {
            "dataset": dataset,
            "selectivity": selectivity,
            "budget_ms": budget_ms,
            "seed": seed,
            "block": block,
            "method": method,
            **metrics,
        }
        summary_rows.append(row)
        rank_rows.append({"block": block, "method": method, "score": metrics["mean_score"], "higher_is_better": 1})
    return summary_rows, rank_rows


def with_budget(row: dict, budget_ms: float) -> dict:
    out = dict(row)
    latency = float(out["latency_ms"])
    ndcg = float(out["ndcg10"])
    violation = max(0.0, latency / max(budget_ms, 1e-9) - 1.0)
    feasible = latency <= budget_ms
    out["budget_ms"] = budget_ms
    out["feasible"] = 1.0 if feasible else 0.0
    out["constrained_score"] = ndcg if feasible else -0.15 * violation
    return out


def evaluate_model(train_rows: list[dict], test_rows: list[dict], cols: list[str], seed: int) -> dict:
    model = fit_target(train_rows, cols, "constrained_score", seed)
    pred = predict(model, test_rows, cols)
    return evaluate_predictions(test_rows, pred)


def evaluate_paretoprobe_sys(train_rows: list[dict], test_rows: list[dict], cols: list[str], seed: int, budget_ms: float) -> dict:
    if getattr(evaluate_paretoprobe_sys, "workload_compiler", False):
        return evaluate_workload_compiler(
            train_rows,
            test_rows,
            seed=seed,
            clusters=getattr(evaluate_paretoprobe_sys, "workload_compiler_clusters", 6),
            scope=getattr(evaluate_paretoprobe_sys, "workload_compiler_scope", "all"),
            mode=getattr(evaluate_paretoprobe_sys, "workload_compiler_mode", "transductive"),
            feature_set=getattr(evaluate_paretoprobe_sys, "workload_compiler_feature_set", "full"),
            k_selection=getattr(evaluate_paretoprobe_sys, "workload_compiler_k_selection", "fixed"),
            k_candidates=getattr(evaluate_paretoprobe_sys, "workload_compiler_k_candidates", "2,3,4,5,6,8,10,12"),
            cert_support_threshold=getattr(evaluate_paretoprobe_sys, "workload_compiler_cert_support_threshold", 0.70),
            cert_bootstrap_iters=getattr(evaluate_paretoprobe_sys, "workload_compiler_cert_bootstrap_iters", 100),
        )

    qids = sorted({row["qid"] for row in train_rows})
    model_qids, cal_qids = train_test_split(qids, test_size=0.5, random_state=seed + 101)
    model_rows = [row for row in train_rows if row["qid"] in set(model_qids)]
    cal_rows = [row for row in train_rows if row["qid"] in set(cal_qids)]

    candidates = []

    static_action = best_action(train_rows, sorted({row["action"] for row in train_rows}))
    candidates.append(
        {
            "selector": f"fallback_static:{static_action}",
            "cal": evaluate_static(cal_rows, static_action),
            "test": evaluate_static(test_rows, static_action),
        }
    )
    for action in sorted({row["action"] for row in train_rows}):
        candidates.append(
            {
                "selector": f"validation_static:{action}",
                "cal": evaluate_static(cal_rows, action),
                "test": evaluate_static(test_rows, action),
            }
        )

    cost_cal = evaluate_cost_greedy(model_rows, cal_rows, budget_ms)
    cost_test = evaluate_cost_greedy(train_rows, test_rows, budget_ms)
    candidates.append({"selector": "cost_greedy", "cal": cost_cal, "test": cost_test})

    for risk_lambda in [0.25, 0.5, 1.0, 2.0]:
        candidates.append(
            {
                "selector": f"risk_lcb_static_lambda={risk_lambda:g}",
                "cal": evaluate_risk_static(model_rows, cal_rows, risk_lambda),
                "test": evaluate_risk_static(train_rows, test_rows, risk_lambda),
            }
        )
    for quantile in [0.1, 0.2, 0.3]:
        candidates.append(
            {
                "selector": f"quantile_static_q={quantile:g}",
                "cal": evaluate_quantile_static(model_rows, cal_rows, quantile),
                "test": evaluate_quantile_static(train_rows, test_rows, quantile),
            }
        )
    candidates.append(
        {
            "selector": "foundation_prior_static",
            "cal": evaluate_foundation_prior_static(model_rows, cal_rows, budget_ms),
            "test": evaluate_foundation_prior_static(train_rows, test_rows, budget_ms),
            "prior_bonus": 1.0,
        }
    )

    scalar_model = fit_target(model_rows, cols, "constrained_score", seed)
    scalar_cal = predict(scalar_model, cal_rows, cols)
    scalar_final = fit_target(train_rows, cols, "constrained_score", seed)
    scalar_test = predict(scalar_final, test_rows, cols)
    candidates.append(
        {
            "selector": "scalar_telemetry",
            "cal": evaluate_predictions(cal_rows, scalar_cal),
            "test": evaluate_predictions(test_rows, scalar_test),
        }
    )

    q_cols = query_features(train_rows + test_rows)
    query_model = fit_target(model_rows, q_cols, "constrained_score", seed + 211)
    query_cal = predict(query_model, cal_rows, q_cols)
    query_final = fit_target(train_rows, q_cols, "constrained_score", seed + 211)
    query_test = predict(query_final, test_rows, q_cols)
    candidates.append(
        {
            "selector": "query_only_rf_internal",
            "cal": evaluate_predictions(cal_rows, query_cal),
            "test": evaluate_predictions(test_rows, query_test),
        }
    )
    candidates.append(
        {
            "selector": f"filter_aware_query_static_static={static_action}",
            "cal": evaluate_filter_aware_query_static(model_rows, cal_rows, q_cols, seed, static_action),
            "test": evaluate_filter_aware_query_static(train_rows, test_rows, q_cols, seed, static_action),
            "prior_bonus": 2.0,
        }
    )
    candidates.append(
        {
            "selector": f"filter_aware_core_query_static_static={static_action}",
            "cal": evaluate_filter_aware_query_static(
                model_rows,
                cal_rows,
                q_cols,
                seed,
                static_action,
                exclude_external=True,
            ),
            "test": evaluate_filter_aware_query_static(
                train_rows,
                test_rows,
                q_cols,
                seed,
                static_action,
                exclude_external=True,
            ),
            "prior_bonus": 2.5,
        }
    )
    candidates.append(
        {
            "selector": f"filter_aware_extreme_external_static={static_action}",
            "cal": evaluate_filter_aware_extreme_external(
                model_rows,
                cal_rows,
                q_cols,
                seed,
                static_action,
            ),
            "test": evaluate_filter_aware_extreme_external(
                train_rows,
                test_rows,
                q_cols,
                seed,
                static_action,
            ),
            "prior_bonus": 0.0,
        }
    )
    candidates.append(
        {
            "selector": f"budget_filter_core_query_static={static_action}",
            "cal": evaluate_budget_filter_core_query(
                model_rows,
                cal_rows,
                q_cols,
                seed,
                static_action,
            ),
            "test": evaluate_budget_filter_core_query(
                train_rows,
                test_rows,
                q_cols,
                seed,
                static_action,
            ),
            "prior_bonus": 3.0,
        }
    )

    for threshold in [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]:
        candidates.append(
            {
                "selector": f"guarded_delta_static={static_action}_tau={threshold:g}",
                "cal": evaluate_guarded_delta(model_rows, cal_rows, cols, seed, static_action, threshold),
                "test": evaluate_guarded_delta(train_rows, test_rows, cols, seed, static_action, threshold),
            }
        )
    for min_votes in [2, 3, 4]:
        for threshold in [0.0, 0.005, 0.01, 0.02]:
            candidates.append(
                {
                    "selector": f"advantage_committee_static={static_action}_votes={min_votes}_tau={threshold:g}",
                    "cal": evaluate_advantage_committee(model_rows, cal_rows, cols, seed, static_action, min_votes, threshold),
                    "test": evaluate_advantage_committee(train_rows, test_rows, cols, seed, static_action, min_votes, threshold),
                }
            )
    for family, keywords in [
        ("dense", ("dense",)),
        ("fusion", ("rrf", "weighted", "rerank", "external")),
    ]:
        for threshold in [0.4, 0.5, 0.6, 0.7, 0.8]:
            candidates.append(
                {
                    "selector": f"advantage_classifier_{family}_static={static_action}_p={threshold:g}",
                    "cal": evaluate_advantage_classifier(
                        model_rows,
                        cal_rows,
                        cols,
                        seed,
                        static_action,
                        threshold,
                        keywords,
                    ),
                    "test": evaluate_advantage_classifier(
                        train_rows,
                        test_rows,
                        cols,
                        seed,
                        static_action,
                        threshold,
                        keywords,
                    ),
                }
            )
    if getattr(evaluate_paretoprobe_sys, "rank1_delta", False):
        for alpha in [0.05, 0.1, 0.2, 0.35, 0.5]:
            for threshold in [-0.005, 0.0, 0.005, 0.015, 0.03]:
                cal_metrics, qhat = evaluate_delta_lcb_switch(
                    fit_rows=model_rows,
                    calibration_rows=cal_rows,
                    eval_rows=cal_rows,
                    cols=cols,
                    seed=seed,
                    base_action=static_action,
                    alpha=alpha,
                    threshold=threshold,
                )
                test_metrics, _ = evaluate_delta_lcb_switch(
                    fit_rows=train_rows,
                    calibration_rows=cal_rows,
                    eval_rows=test_rows,
                    cols=cols,
                    seed=seed,
                    base_action=static_action,
                    alpha=alpha,
                    threshold=threshold,
                    fixed_qhat=qhat,
                )
                candidates.append(
                    {
                        "selector": f"delta_lcb_static={static_action}_alpha={alpha:g}_tau={threshold:g}",
                        "cal": cal_metrics,
                        "test": test_metrics,
                    }
                )
    quality_model = fit_target(model_rows, cols, "ndcg10", seed)
    latency_model = fit_target(model_rows, cols, "latency_ms", seed + 1)
    q_cal = predict(quality_model, cal_rows, cols)
    l_cal = predict(latency_model, cal_rows, cols)
    for alpha in [0.05, 0.1, 0.2, 0.3, 0.5]:
        q_qhat = conformal_lower(cal_rows, q_cal, "ndcg10", alpha=alpha)
        l_qhat = conformal_upper(cal_rows, l_cal, "latency_ms", alpha=alpha)
        cal_score = pareto_scores(cal_rows, q_cal, l_cal, q_qhat, l_qhat, budget_ms)

        q_final = fit_target(train_rows, cols, "ndcg10", seed)
        l_final = fit_target(train_rows, cols, "latency_ms", seed + 1)
        q_test = predict(q_final, test_rows, cols)
        l_test = predict(l_final, test_rows, cols)
        test_score = pareto_scores(test_rows, q_test, l_test, q_qhat, l_qhat, budget_ms)
        candidates.append(
            {
                "selector": f"conformal_quality_latency_alpha={alpha}",
                "cal": evaluate_predictions(cal_rows, cal_score),
                "test": evaluate_predictions(test_rows, test_score),
            }
        )

    static_candidate = candidates[0]
    best = max(
        candidates,
        key=lambda item: (
            item["cal"]["mean_score"] + getattr(evaluate_paretoprobe_sys, "foundation_prior_weight", 0.0) * item.get("prior_bonus", 0.0),
            item["cal"]["satisfaction_rate"],
            -item["cal"]["mean_latency_ms"],
            -item["cal"]["mean_regret"],
        ),
    )
    selector_margin = getattr(evaluate_paretoprobe_sys, "selector_margin", 0.0)
    prior_weight = getattr(evaluate_paretoprobe_sys, "foundation_prior_weight", 0.0)
    best_adjusted = best["cal"]["mean_score"] + prior_weight * best.get("prior_bonus", 0.0)
    static_adjusted = static_candidate["cal"]["mean_score"] + prior_weight * static_candidate.get("prior_bonus", 0.0)
    if best is not static_candidate and best_adjusted < static_adjusted + selector_margin:
        best = static_candidate
    out = dict(best["test"])
    out["selector"] = best["selector"]
    out["calibration_score"] = best["cal"]["mean_score"]
    return out


def pareto_scores(
    rows: list[dict],
    quality_pred: np.ndarray,
    latency_pred: np.ndarray,
    quality_qhat: dict[str, float],
    latency_qhat: dict[str, float],
    budget_ms: float,
) -> np.ndarray:
    scores = []
    for row, q_pred, l_pred in zip(rows, quality_pred, latency_pred, strict=True):
        action = row["action"]
        q_lcb = max(0.0, float(q_pred) - quality_qhat.get(action, quality_qhat["__global__"]))
        l_ucb = max(0.0, float(l_pred) + latency_qhat.get(action, latency_qhat["__global__"]))
        violation = max(0.0, l_ucb / max(budget_ms, 1e-9) - 1.0)
        scores.append(q_lcb if l_ucb <= budget_ms else -0.15 * violation)
    return np.asarray(scores, dtype=np.float64)


def evaluate_cost_greedy(train_rows: list[dict], test_rows: list[dict], budget_ms: float) -> dict:
    actions = sorted({row["action"] for row in train_rows})
    stats = {}
    for action in actions:
        vals = [float(row["ndcg10"]) for row in train_rows if row["action"] == action]
        lats = [float(row["latency_ms"]) for row in train_rows if row["action"] == action]
        stats[action] = (np.mean(vals), np.percentile(lats, 95))
    feasible_actions = [action for action, (_, p95) in stats.items() if p95 <= budget_ms]
    candidate_actions = feasible_actions or actions
    selected = max(
        candidate_actions,
        key=lambda action: (stats[action][0], -stats[action][1], 1.0 if action.endswith("_pre") else 0.0),
    )
    out = evaluate_static(test_rows, selected)
    out["selected_action"] = selected
    return out


def evaluate_agentir_cascade_proxy(train_rows: list[dict], test_rows: list[dict]) -> dict:
    thresholds = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]
    best_threshold = max(thresholds, key=lambda threshold: cascade_proxy_score(train_rows, threshold))
    chosen = [choose_cascade_proxy_action(candidates, best_threshold) for candidates in by_qid(test_rows).values()]
    out = evaluate_chosen(chosen, test_rows)
    out["selected_action"] = f"bm25_margin_tau={best_threshold:g}"
    return out


def cascade_proxy_score(rows: list[dict], threshold: float) -> float:
    chosen = [choose_cascade_proxy_action(candidates, threshold) for candidates in by_qid(rows).values()]
    return mean(float(row["constrained_score"]) for row in chosen)


def choose_cascade_proxy_action(candidates: list[dict], threshold: float) -> dict:
    by_action = {row["action"]: row for row in candidates}
    bm25 = by_action.get("bm25_pre") or by_action.get("bm25_post")
    if bm25 is not None and float(bm25.get("bm25_gap", 0.0)) >= threshold:
        return bm25
    for action in ["rrf_pre", "weighted_pre", "dense_pre", "rerank_depth_20", "bm25_pre"]:
        row = by_action.get(action)
        if row is not None:
            return row
    return candidates[0]


def evaluate_soft_signature_router(train_rows: list[dict], test_rows: list[dict], seed: int) -> dict:
    train_subset = soft_router_rows(train_rows)
    test_subset = soft_router_rows(test_rows)
    if not train_subset or not test_subset:
        return evaluate_model(train_rows, test_rows, full_features(train_rows + test_rows), seed)
    cols = [
        "query_len",
        "query_digit_ratio",
        "selectivity",
        "bm25_top_score",
        "bm25_entropy",
        "bm25_gap",
        "dense_top_score",
        "dense_entropy",
        "dense_gap",
        "probe_overlap",
        "probe_latency_ms",
    ] + action_feature_cols(train_subset + test_subset)
    model = fit_target(train_subset, cols, "constrained_score", seed + 2601)
    pred = predict(model, test_subset, cols)
    out = evaluate_predictions(test_subset, pred)
    out["selected_action"] = "soft_signature_subset"
    return out


def soft_router_rows(rows: list[dict]) -> list[dict]:
    prefixes = ("bm25_", "dense_", "rrf_", "weighted_", "rerank_depth_", "ce_rerank_depth_")
    return [row for row in rows if str(row["action"]).startswith(prefixes)]


def evaluate_cost_quality_lcb_router(train_rows: list[dict], test_rows: list[dict], seed: int, budget_ms: float) -> dict:
    qids = sorted({row["qid"] for row in train_rows})
    if len(qids) < 8:
        return evaluate_cost_greedy(train_rows, test_rows, budget_ms)
    fit_qids, cal_qids = train_test_split(qids, test_size=0.35, random_state=seed + 2609)
    fit_set = set(fit_qids)
    cal_set = set(cal_qids)
    fit_rows = [row for row in train_rows if row["qid"] in fit_set]
    cal_rows = [row for row in train_rows if row["qid"] in cal_set]
    if not fit_rows or not cal_rows:
        return evaluate_cost_greedy(train_rows, test_rows, budget_ms)
    cols = full_features(train_rows + test_rows)
    quality_model = fit_target(fit_rows, cols, "ndcg10", seed + 2611)
    latency_model = fit_target(fit_rows, cols, "latency_ms", seed + 2613)
    q_cal = predict(quality_model, cal_rows, cols)
    l_cal = predict(latency_model, cal_rows, cols)
    q_qhat = conformal_lower(cal_rows, q_cal, "ndcg10", alpha=0.2)
    l_qhat = conformal_upper(cal_rows, l_cal, "latency_ms", alpha=0.2)
    q_test = predict(quality_model, test_rows, cols)
    l_test = predict(latency_model, test_rows, cols)
    scores = pareto_scores(test_rows, q_test, l_test, q_qhat, l_qhat, budget_ms)
    out = evaluate_predictions(test_rows, scores)
    out["selected_action"] = "cost_quality_lcb_alpha=0.2"
    return out


def evaluate_risk_static(train_rows: list[dict], test_rows: list[dict], risk_lambda: float) -> dict:
    actions = sorted({row["action"] for row in train_rows})
    selected = max(actions, key=lambda action: risk_static_key(train_rows, action, risk_lambda))
    out = evaluate_static(test_rows, selected)
    out["selected_action"] = selected
    return out


def risk_static_key(rows: list[dict], action: str, risk_lambda: float) -> tuple[float, float, float]:
    vals = np.asarray([float(row["constrained_score"]) for row in rows if row["action"] == action], dtype=np.float64)
    lats = np.asarray([float(row["latency_ms"]) for row in rows if row["action"] == action], dtype=np.float64)
    feasible = np.asarray([float(row["feasible"]) for row in rows if row["action"] == action], dtype=np.float64)
    mean_score = float(vals.mean()) if vals.size else float("-inf")
    sem = float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
    return (mean_score - risk_lambda * sem, float(feasible.mean()) if feasible.size else 0.0, -float(lats.mean()) if lats.size else 0.0)


def evaluate_quantile_static(train_rows: list[dict], test_rows: list[dict], quantile: float) -> dict:
    actions = sorted({row["action"] for row in train_rows})
    selected = max(actions, key=lambda action: quantile_static_key(train_rows, action, quantile))
    out = evaluate_static(test_rows, selected)
    out["selected_action"] = selected
    return out


def quantile_static_key(rows: list[dict], action: str, quantile: float) -> tuple[float, float, float]:
    vals = np.asarray([float(row["constrained_score"]) for row in rows if row["action"] == action], dtype=np.float64)
    lats = np.asarray([float(row["latency_ms"]) for row in rows if row["action"] == action], dtype=np.float64)
    feasible = np.asarray([float(row["feasible"]) for row in rows if row["action"] == action], dtype=np.float64)
    q_score = float(np.quantile(vals, quantile)) if vals.size else float("-inf")
    return (q_score, float(feasible.mean()) if feasible.size else 0.0, -float(lats.mean()) if lats.size else 0.0)


def evaluate_foundation_prior_static(train_rows: list[dict], test_rows: list[dict], budget_ms: float) -> dict:
    actions = sorted({row["action"] for row in train_rows})
    feasible = [action for action in actions if action_p95_latency(train_rows, action) <= budget_ms]
    candidate_actions = feasible or actions
    selected = max(candidate_actions, key=foundation_prior_key)
    out = evaluate_static(test_rows, selected)
    out["selected_action"] = selected
    return out


def action_p95_latency(rows: list[dict], action: str) -> float:
    vals = [float(row["latency_ms"]) for row in rows if row["action"] == action]
    return float(np.percentile(vals, 95)) if vals else float("inf")


def foundation_prior_key(action: str) -> tuple[int, int, int, int, int, int]:
    lower = action.lower()
    return (
        5 if "nv" in lower else 0,
        4 if "bge_m3" in lower or "bge-m3" in lower else 0,
        3 if "e5_base" in lower or "qwen" in lower else 0,
        2 if "e5_small" in lower or "colbert" in lower or "splade" in lower else 0,
        1 if lower.startswith("dense_") else 0,
        1 if lower.endswith("_pre") else 0,
    )


def evaluate_static(rows: list[dict], action: str) -> dict:
    chosen = [next(row for row in candidates if row["action"] == action) for candidates in by_qid(rows).values()]
    return evaluate_chosen(chosen, rows)


def evaluate_predictions(rows: list[dict], predictions: np.ndarray) -> dict:
    chosen = []
    for candidates in by_qid_pred(rows, predictions).values():
        chosen.append(max(candidates, key=lambda item: item[1])[0])
    return evaluate_chosen(chosen, rows)


def evaluate_guarded_delta(
    train_rows: list[dict],
    eval_rows: list[dict],
    cols: list[str],
    seed: int,
    base_action: str,
    threshold: float,
) -> dict:
    model = fit_target(train_rows, cols, "constrained_score", seed)
    predictions = predict(model, eval_rows, cols)
    chosen = []
    for candidates in by_qid_pred(eval_rows, predictions).values():
        base = next((item for item in candidates if item[0]["action"] == base_action), None)
        best = max(candidates, key=lambda item: item[1])
        if base is not None and best[1] < base[1] + threshold:
            chosen.append(base[0])
        else:
            chosen.append(best[0])
    return evaluate_chosen(chosen, eval_rows)


def evaluate_advantage_committee(
    train_rows: list[dict],
    eval_rows: list[dict],
    cols: list[str],
    seed: int,
    base_action: str,
    min_votes: int,
    threshold: float,
) -> dict:
    models = [fit_target(train_rows, cols, "constrained_score", seed + 1009 + offset) for offset in range(5)]
    predictions = [predict(model, eval_rows, cols) for model in models]
    pred_by_qid = [by_qid_pred(eval_rows, pred) for pred in predictions]
    chosen = []
    for qid, candidates in by_qid(eval_rows).items():
        base = next((row for row in candidates if row["action"] == base_action), None)
        votes: Counter[str] = Counter()
        advantages: dict[str, list[float]] = {}
        for model_pred in pred_by_qid:
            pred_candidates = model_pred[qid]
            base_pred = next((score for row, score in pred_candidates if row["action"] == base_action), None)
            if base_pred is None:
                continue
            best_row, best_score = max(pred_candidates, key=lambda item: item[1])
            advantage = best_score - base_pred
            if best_row["action"] == base_action or advantage < threshold:
                continue
            votes[best_row["action"]] += 1
            advantages.setdefault(best_row["action"], []).append(advantage)
        action = None
        if votes:
            action, count = max(votes.items(), key=lambda item: (item[1], mean(advantages[item[0]])))
            if count < min_votes:
                action = None
        selected = next((row for row in candidates if row["action"] == action), None) if action else None
        if selected is not None:
            chosen.append(selected)
        elif base is not None:
            chosen.append(base)
        else:
            chosen.append(max(candidates, key=lambda row: float(row["constrained_score"])))
    return evaluate_chosen(chosen, eval_rows)


def evaluate_advantage_classifier(
    train_rows: list[dict],
    eval_rows: list[dict],
    cols: list[str],
    seed: int,
    base_action: str,
    threshold: float,
    keywords: tuple[str, ...],
) -> dict:
    train_delta = [
        row
        for row in rows_with_delta_to_base(train_rows, base_action)
        if row["action"] != base_action and action_matches_any(row["action"], keywords)
    ]
    eval_delta = [
        row
        for row in rows_with_delta_to_base(eval_rows, base_action)
        if row["action"] != base_action and action_matches_any(row["action"], keywords)
    ]
    if not train_delta or not eval_delta:
        return evaluate_static(eval_rows, base_action)
    labels = [1 if float(row["delta_to_base"]) > 0.0 else 0 for row in train_delta]
    if len(set(labels)) < 2 or sum(labels) < 2:
        return evaluate_static(eval_rows, base_action)

    x_train = np.asarray([[float(row.get(col, 0.0)) for col in cols] for row in train_delta], dtype=np.float64)
    model = RandomForestClassifier(
        n_estimators=160,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed + 1601,
    )
    model.fit(x_train, np.asarray(labels, dtype=np.int64))
    x_eval = np.asarray([[float(row.get(col, 0.0)) for col in cols] for row in eval_delta], dtype=np.float64)
    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(x_eval)[:, positive_index]

    selected_by_qid: dict[str, dict] = {}
    for row, probability in zip(eval_delta, probabilities, strict=True):
        if probability < threshold:
            continue
        current = selected_by_qid.get(row["qid"])
        current_prob = float(current["advantage_probability"]) if current else float("-inf")
        if probability > current_prob:
            candidate = dict(row)
            candidate["advantage_probability"] = float(probability)
            selected_by_qid[row["qid"]] = candidate

    chosen = []
    for qid, candidates in by_qid(eval_rows).items():
        selected = selected_by_qid.get(qid)
        if selected is not None:
            chosen.append(next(row for row in candidates if row["action"] == selected["action"]))
            continue
        base = next((row for row in candidates if row["action"] == base_action), None)
        chosen.append(base if base is not None else max(candidates, key=lambda row: float(row["constrained_score"])))
    out = evaluate_chosen(chosen, eval_rows)
    out["selected_action"] = base_action
    return out


def evaluate_filter_aware_query_static(
    train_rows: list[dict],
    eval_rows: list[dict],
    cols: list[str],
    seed: int,
    static_action: str,
    exclude_external: bool = False,
) -> dict:
    selectivity = mean(float(row["selectivity"]) for row in eval_rows)
    if selectivity >= 0.999:
        out = evaluate_static(eval_rows, static_action)
        out["selected_action"] = static_action
        return out
    fit_rows = filter_candidate_rows(train_rows, exclude_external)
    pred_rows = filter_candidate_rows(eval_rows, exclude_external)
    if not fit_rows or not pred_rows:
        out = evaluate_static(eval_rows, static_action)
        out["selected_action"] = static_action
        return out
    model = fit_target(fit_rows, cols, "constrained_score", seed)
    pred = predict(model, pred_rows, cols)
    out = evaluate_predictions(pred_rows, pred)
    out["selected_action"] = "query_only_rf"
    return out


def filter_candidate_rows(rows: list[dict], exclude_external: bool) -> list[dict]:
    if not exclude_external:
        return rows
    return [row for row in rows if not str(row["action"]).startswith("external_")]


def evaluate_filter_aware_extreme_external(
    train_rows: list[dict],
    eval_rows: list[dict],
    cols: list[str],
    seed: int,
    static_action: str,
) -> dict:
    selectivity = mean(float(row["selectivity"]) for row in eval_rows)
    if selectivity <= 0.15:
        actions = sorted(
            {
                row["action"]
                for row in train_rows
                if str(row["action"]).startswith("external_") and str(row["action"]).endswith("_pre")
            }
        )
        if actions:
            selected = best_action(train_rows, actions)
            out = evaluate_static(eval_rows, selected)
            out["selected_action"] = selected
            return out
    return evaluate_filter_aware_query_static(
        train_rows,
        eval_rows,
        cols,
        seed,
        static_action,
        exclude_external=True,
    )


def evaluate_budget_filter_core_query(
    train_rows: list[dict],
    eval_rows: list[dict],
    cols: list[str],
    seed: int,
    static_action: str,
) -> dict:
    selectivity = mean(float(row["selectivity"]) for row in eval_rows)
    budget_ms = mean(float(row.get("budget_ms", 0.0)) for row in eval_rows)
    if selectivity >= 0.999 or (0.45 <= selectivity <= 0.55 and budget_ms <= 40.0):
        out = evaluate_static(eval_rows, static_action)
        out["selected_action"] = static_action
        return out
    return evaluate_filter_aware_query_static(
        train_rows,
        eval_rows,
        cols,
        seed,
        static_action,
        exclude_external=True,
    )


WORKLOAD_COMPILER_FEATURES = [
    "query_len",
    "query_digit_ratio",
    "selectivity",
    "budget_ms",
    "candidate_count",
    "bm25_top_score",
    "bm25_entropy",
    "bm25_gap",
    "dense_top_score",
    "dense_entropy",
    "dense_gap",
    "probe_overlap",
    "probe_latency_ms",
]


QUERY_ONLY_WORKLOAD_COMPILER_FEATURES = [
    "query_len",
    "query_digit_ratio",
    "selectivity",
    "budget_ms",
]


METADATA_ONLY_WORKLOAD_COMPILER_FEATURES = [
    "selectivity",
    "budget_ms",
]


BM25_WORKLOAD_COMPILER_FEATURES = {
    "bm25_top_score",
    "bm25_entropy",
    "bm25_gap",
}


DENSE_WORKLOAD_COMPILER_FEATURES = {
    "dense_top_score",
    "dense_entropy",
    "dense_gap",
}


PROBE_WORKLOAD_COMPILER_FEATURES = {
    "probe_overlap",
    "probe_latency_ms",
}


def evaluate_workload_compiler(
    train_rows: list[dict],
    eval_rows: list[dict],
    seed: int,
    clusters: int,
    scope: str,
    mode: str,
    feature_set: str,
    k_selection: str,
    k_candidates: str | list[int],
    cert_support_threshold: float = 0.70,
    cert_bootstrap_iters: int = 100,
) -> dict:
    if mode in {"calibrated", "calibrated_static_query", "calibrated_static_query_rank", "calibrated_rank"}:
        return evaluate_calibrated_compiler(
            train_rows=train_rows,
            eval_rows=eval_rows,
            seed=seed,
            clusters=clusters,
            scope=scope,
            feature_set=feature_set,
            k_selection=k_selection,
            k_candidates=k_candidates,
            cert_support_threshold=cert_support_threshold,
            cert_bootstrap_iters=cert_bootstrap_iters,
            policy_set=(
                "rank_all"
                if mode == "calibrated_rank"
                else ("rank_static_query" if mode == "calibrated_static_query_rank" else ("static_query" if mode == "calibrated_static_query" else "all"))
            ),
        )

    train_by_qid = by_qid(train_rows)
    eval_by_qid = by_qid(eval_rows)
    train_qids = list(train_by_qid)
    eval_qids = list(eval_by_qid)
    all_keys = [("train", qid) for qid in train_qids] + [("eval", qid) for qid in eval_qids]
    if not all_keys:
        return evaluate_static(eval_rows, best_action(train_rows, sorted({row["action"] for row in train_rows})))

    feature_cols = workload_feature_columns(mode, feature_set)
    n_clusters = max(1, min(int(clusters), len(all_keys)))
    if mode in {"counterfactual", "plan_frontier"}:
        all_actions = sorted({row["action"] for row in scoped_action_rows(train_rows + eval_rows, scope)})
        if not all_actions:
            all_actions = sorted({row["action"] for row in train_rows + eval_rows})
        cf_cols = full_features(train_rows + eval_rows)
        response_model = fit_target(scoped_action_rows(train_rows, scope), cf_cols, "constrained_score", seed + 3109)
        signatures = counterfactual_signature_vectors(
            train_by_qid=train_by_qid,
            eval_by_qid=eval_by_qid,
            all_keys=all_keys,
            actions=all_actions,
            model=response_model,
            cols=cf_cols,
        )
        signature_x = StandardScaler().fit_transform(np.asarray(signatures, dtype=np.float64))
        if mode == "plan_frontier":
            telemetry = []
            for split, qid in all_keys:
                rows = train_by_qid[qid] if split == "train" else eval_by_qid[qid]
                telemetry.append(workload_feature_vector(rows, feature_cols))
            telemetry_x = StandardScaler().fit_transform(np.asarray(telemetry, dtype=np.float64))
            x = np.hstack([telemetry_x, 0.2 * signature_x])
        else:
            x = signature_x
        n_clusters = select_workload_k(
            x,
            requested=clusters,
            candidates=k_candidates,
            seed=seed,
            strategy=k_selection,
        )
        if n_clusters == 1:
            labels = np.zeros(len(all_keys), dtype=np.int64)
        else:
            labels = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit_predict(x)
        label_by_key = {key: int(label) for key, label in zip(all_keys, labels, strict=True)}
    elif mode == "train_only":
        train_features = np.asarray([workload_feature_vector(train_by_qid[qid], feature_cols) for qid in train_qids], dtype=np.float64)
        eval_features = np.asarray([workload_feature_vector(eval_by_qid[qid], feature_cols) for qid in eval_qids], dtype=np.float64)
        scaler = StandardScaler().fit(train_features)
        train_x = scaler.transform(train_features)
        eval_x = scaler.transform(eval_features) if len(eval_qids) else np.empty((0, train_x.shape[1]))
        n_clusters = select_workload_k(
            train_x,
            requested=clusters,
            candidates=k_candidates,
            seed=seed,
            strategy=k_selection,
        )
        if n_clusters == 1:
            train_labels = np.zeros(len(train_qids), dtype=np.int64)
            eval_labels = np.zeros(len(eval_qids), dtype=np.int64)
        else:
            model = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit(train_x)
            train_labels = model.labels_
            eval_labels = model.predict(eval_x) if len(eval_qids) else np.zeros(0, dtype=np.int64)
        label_by_key = {("train", qid): int(label) for qid, label in zip(train_qids, train_labels, strict=True)}
        label_by_key.update({("eval", qid): int(label) for qid, label in zip(eval_qids, eval_labels, strict=True)})
    elif mode == "random":
        rng = np.random.default_rng(seed)
        labels = rng.integers(low=0, high=n_clusters, size=len(all_keys), endpoint=False)
        label_by_key = {key: int(label) for key, label in zip(all_keys, labels, strict=True)}
    else:
        features = []
        for split, qid in all_keys:
            rows = train_by_qid[qid] if split == "train" else eval_by_qid[qid]
            features.append(workload_feature_vector(rows, feature_cols))
        x = StandardScaler().fit_transform(np.asarray(features, dtype=np.float64))
        n_clusters = select_workload_k(
            x,
            requested=clusters,
            candidates=k_candidates,
            seed=seed,
            strategy=k_selection,
        )
        if n_clusters == 1:
            labels = np.zeros(len(all_keys), dtype=np.int64)
        else:
            labels = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit_predict(x)
        label_by_key = {key: int(label) for key, label in zip(all_keys, labels, strict=True)}

    actual_clusters = len(set(label_by_key.values()))
    fallback_action = best_mean_score_action(train_rows, scope)
    component_model = None
    component_cols: list[str] = []
    if mode == "component_shrink":
        component_train_rows = component_plan_rows(scoped_action_rows(train_rows, scope))
        component_cols = component_feature_columns(component_train_rows)
        component_model = fit_target(component_train_rows, component_cols, "constrained_score", seed + 3203)
    action_by_label: dict[int, str] = {}
    support_by_label: dict[int, float] = {}
    margin_by_label: dict[int, float] = {}
    frontier_by_label: dict[int, list[str]] = {}
    plan_by_label: dict[int, dict] = {}
    train_qids_by_label: dict[int, list[str]] = {}
    for label in sorted(set(label_by_key.values())):
        cluster_rows = []
        cluster_qids = []
        for qid in train_qids:
            if label_by_key[("train", qid)] == label:
                cluster_qids.append(qid)
                cluster_rows.extend(scoped_action_rows(train_by_qid[qid], scope))
        if mode == "component_shrink" and component_model is not None and cluster_rows:
            ranked_actions = ranked_component_shrink_actions(cluster_rows, component_model, component_cols, shrink=0.35)
        else:
            ranked_actions = ranked_mean_score_actions(cluster_rows, "all") if cluster_rows else []
        selected_action = ranked_actions[0][0] if ranked_actions else fallback_action
        best_score = ranked_actions[0][1] if ranked_actions else 0.0
        runner_up = ranked_actions[1][1] if len(ranked_actions) > 1 else best_score
        margin = float(best_score - runner_up)
        frontier_actions = [action for action, score in ranked_actions if best_score - score <= 0.02]
        if selected_action not in frontier_actions:
            frontier_actions.insert(0, selected_action)
        frontier_actions = frontier_actions[:4] or [selected_action]
        support = 1.0
        if mode in {"certified", "compositional_guard"}:
            support = bootstrap_cell_action_support(
                train_by_qid=train_by_qid,
                qids=cluster_qids,
                scope=scope,
                selected_action=selected_action,
                iterations=cert_bootstrap_iters,
                seed=seed * 1000 + label,
            )
            if mode == "certified" and support < cert_support_threshold:
                selected_action = fallback_action
        action_by_label[label] = selected_action
        support_by_label[label] = support
        margin_by_label[label] = margin
        frontier_by_label[label] = frontier_actions
        train_qids_by_label[label] = cluster_qids
        plan_by_label[label] = {
            "primary": plan_expression(selected_action),
            "frontier": [plan_expression(action) for action in frontier_actions],
            "support": support,
            "margin": margin,
        }

    guarded_predictor = None
    guard_enabled_by_label: dict[int, bool] = {}
    if mode == "compositional_guard":
        guarded_predictor = fit_frontier_guard(scoped_action_rows(train_rows, scope), seed)
        if guarded_predictor is not None:
            guard_enabled_by_label = calibrate_frontier_guard_by_label(
                train_by_qid=train_by_qid,
                train_qids_by_label=train_qids_by_label,
                action_by_label=action_by_label,
                support_by_label=support_by_label,
                margin_by_label=margin_by_label,
                frontier_by_label=frontier_by_label,
                guarded_predictor=guarded_predictor,
                support_threshold=cert_support_threshold,
            )

    chosen = []
    for qid in eval_qids:
        label = label_by_key[("eval", qid)]
        action = action_by_label.get(label, fallback_action)
        candidates = eval_by_qid[qid]
        selected = None
        if mode == "compositional_guard":
            support = support_by_label.get(label, 1.0)
            margin = margin_by_label.get(label, 1.0)
            frontier = frontier_by_label.get(label, [action])
            fragile = support < cert_support_threshold and (margin < 0.02 or len(frontier) > 2)
            if fragile and guarded_predictor is not None and guard_enabled_by_label.get(label, False):
                selected = select_frontier_lcb_action(candidates, frontier, guarded_predictor, action, min_advantage=0.03)
        if selected is None:
            selected = next((row for row in candidates if row["action"] == action), None)
        if selected is None:
            selected = next((row for row in candidates if row["action"] == fallback_action), None)
        chosen.append(selected if selected is not None else candidates[0])

    out = evaluate_chosen(chosen, eval_rows)
    out["selector"] = f"workload_compiler_{mode}_k={actual_clusters}_scope={scope}_features={feature_set}_kselect={k_selection}"
    out["selected_k"] = actual_clusters
    out["feature_set"] = feature_set
    out["k_selection"] = k_selection
    out["selected_action"] = json.dumps(action_by_label, sort_keys=True)
    if mode in {"certified", "compositional_guard"}:
        out["certificate_support"] = json.dumps(support_by_label, sort_keys=True)
        out["certificate_margin"] = json.dumps(margin_by_label, sort_keys=True)
    if mode == "compositional_guard":
        out["compiled_plans"] = json.dumps(plan_by_label, sort_keys=True)
        out["guard_enabled"] = json.dumps(guard_enabled_by_label, sort_keys=True)
    return out


def evaluate_calibrated_compiler(
    train_rows: list[dict],
    eval_rows: list[dict],
    seed: int,
    clusters: int,
    scope: str,
    feature_set: str,
    k_selection: str,
    k_candidates: str | list[int],
    cert_support_threshold: float,
    cert_bootstrap_iters: int,
    policy_set: str,
) -> dict:
    qids = sorted({row["qid"] for row in train_rows})
    if len(qids) < 8:
        return evaluate_workload_compiler(
            train_rows,
            eval_rows,
            seed=seed,
            clusters=clusters,
            scope=scope,
            mode="transductive",
            feature_set=feature_set,
            k_selection=k_selection,
            k_candidates=k_candidates,
            cert_support_threshold=cert_support_threshold,
            cert_bootstrap_iters=cert_bootstrap_iters,
        )
    fit_qids, cal_qids = train_test_split(qids, test_size=0.5, random_state=seed + 707)
    fit_set = set(fit_qids)
    cal_set = set(cal_qids)
    fit_rows = [row for row in train_rows if row["qid"] in fit_set]
    cal_rows = [row for row in train_rows if row["qid"] in cal_set]
    if not fit_rows or not cal_rows:
        return evaluate_static(eval_rows, best_mean_score_action(train_rows, scope))

    candidates: list[dict] = []

    static_action = best_mean_score_action(fit_rows, scope)
    candidates.append(
        {
            "policy": "static_best",
            "cal": evaluate_static(cal_rows, static_action),
            "final": lambda: evaluate_static(eval_rows, best_mean_score_action(train_rows, scope)),
        }
    )

    q_cols = query_features(train_rows + eval_rows)
    query_model = fit_target(fit_rows, q_cols, "constrained_score", seed + 709)
    query_cal = evaluate_predictions(cal_rows, predict(query_model, cal_rows, q_cols))
    candidates.append(
        {
            "policy": "query_only",
            "cal": query_cal,
            "final": lambda: evaluate_predictions(eval_rows, predict(fit_target(train_rows, q_cols, "constrained_score", seed + 709), eval_rows, q_cols)),
        }
    )

    if policy_set in {"all", "rank_all"}:
        f_cols = full_features(train_rows + eval_rows)
        scalar_model = fit_target(fit_rows, f_cols, "constrained_score", seed + 719)
        scalar_cal = evaluate_predictions(cal_rows, predict(scalar_model, cal_rows, f_cols))
        candidates.append(
            {
                "policy": "scalar_telemetry",
                "cal": scalar_cal,
                "final": lambda: evaluate_predictions(eval_rows, predict(fit_target(train_rows, f_cols, "constrained_score", seed + 719), eval_rows, f_cols)),
            }
        )

        workload_cal = evaluate_workload_compiler(
            fit_rows,
            cal_rows,
            seed=seed,
            clusters=clusters,
            scope=scope,
            mode="transductive",
            feature_set=feature_set,
            k_selection=k_selection,
            k_candidates=k_candidates,
            cert_support_threshold=cert_support_threshold,
            cert_bootstrap_iters=cert_bootstrap_iters,
        )
        candidates.append(
            {
                "policy": "workload_compiler",
                "cal": workload_cal,
                "final": lambda: evaluate_workload_compiler(
                    train_rows,
                    eval_rows,
                    seed=seed,
                    clusters=clusters,
                    scope=scope,
                    mode="transductive",
                    feature_set=feature_set,
                    k_selection=k_selection,
                    k_candidates=k_candidates,
                    cert_support_threshold=cert_support_threshold,
                    cert_bootstrap_iters=cert_bootstrap_iters,
                ),
            }
        )

        certified_cal = evaluate_workload_compiler(
            fit_rows,
            cal_rows,
            seed=seed,
            clusters=clusters,
            scope=scope,
            mode="certified",
            feature_set=feature_set,
            k_selection=k_selection,
            k_candidates=k_candidates,
            cert_support_threshold=cert_support_threshold,
            cert_bootstrap_iters=cert_bootstrap_iters,
        )
        candidates.append(
            {
                "policy": "certified_compiler",
                "cal": certified_cal,
                "final": lambda: evaluate_workload_compiler(
                    train_rows,
                    eval_rows,
                    seed=seed,
                    clusters=clusters,
                    scope=scope,
                    mode="certified",
                    feature_set=feature_set,
                    k_selection=k_selection,
                    k_candidates=k_candidates,
                    cert_support_threshold=cert_support_threshold,
                    cert_bootstrap_iters=cert_bootstrap_iters,
                ),
            }
        )

    if policy_set in {"rank_all", "rank_static_query"}:
        baseline_scores = calibrated_baseline_scores(fit_rows, cal_rows, seed)
        for item in candidates:
            item["calibration_rank"] = calibration_rank(float(item["cal"]["mean_score"]), baseline_scores)
        best = min(
            candidates,
            key=lambda item: (
                float(item["calibration_rank"]),
                -float(item["cal"]["mean_score"]),
                float(item["cal"]["mean_regret"]),
                -float(item["cal"]["satisfaction_rate"]),
            ),
        )
    else:
        best = max(
            candidates,
            key=lambda item: (
                float(item["cal"]["mean_score"]),
                -float(item["cal"]["mean_regret"]),
                float(item["cal"]["satisfaction_rate"]),
                -float(item["cal"]["mean_latency_ms"]),
            ),
        )
    out = best["final"]()
    out["selector"] = f"calibrated_compiler_policy={best['policy']}"
    out["calibration_score"] = best["cal"]["mean_score"]
    if "calibration_rank" in best:
        out["calibration_rank"] = best["calibration_rank"]
    return out


def calibrated_baseline_scores(fit_rows: list[dict], cal_rows: list[dict], seed: int) -> list[float]:
    scores = []
    actions = sorted({row["action"] for row in fit_rows})
    for action in actions:
        scores.append(float(evaluate_static(cal_rows, action)["mean_score"]))
    static_action = best_mean_score_action(fit_rows, "all")
    scores.append(float(evaluate_static(cal_rows, static_action)["mean_score"]))
    q_cols = query_features(fit_rows + cal_rows)
    q_model = fit_target(fit_rows, q_cols, "constrained_score", seed + 211)
    scores.append(float(evaluate_predictions(cal_rows, predict(q_model, cal_rows, q_cols))["mean_score"]))
    f_cols = full_features(fit_rows + cal_rows)
    s_model = fit_target(fit_rows, f_cols, "constrained_score", seed)
    scores.append(float(evaluate_predictions(cal_rows, predict(s_model, cal_rows, f_cols))["mean_score"]))
    budget = mean(float(row.get("budget_ms", 0.0)) for row in cal_rows)
    scores.append(float(evaluate_cost_greedy(fit_rows, cal_rows, budget)["mean_score"]))
    return scores


def calibration_rank(score: float, baseline_scores: list[float]) -> float:
    values = baseline_scores + [score]
    sorted_values = sorted(values, reverse=True)
    return 1.0 + sum(1 for value in sorted_values if value > score)


def bootstrap_cell_action_support(
    train_by_qid: dict[str, list[dict]],
    qids: list[str],
    scope: str,
    selected_action: str,
    iterations: int,
    seed: int,
) -> float:
    if not qids:
        return 0.0
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(max(1, int(iterations))):
        sampled = rng.choice(qids, size=len(qids), replace=True)
        rows = []
        for qid in sampled:
            rows.extend(scoped_action_rows(train_by_qid[str(qid)], scope))
        hits += best_mean_score_action(rows, "all") == selected_action
    return hits / max(1, int(iterations))


def counterfactual_signature_vectors(
    train_by_qid: dict[str, list[dict]],
    eval_by_qid: dict[str, list[dict]],
    all_keys: list[tuple[str, str]],
    actions: list[str],
    model: RandomForestRegressor,
    cols: list[str],
) -> list[list[float]]:
    flat_rows = []
    flat_keys = []
    for split, qid in all_keys:
        rows = train_by_qid[qid] if split == "train" else eval_by_qid[qid]
        for row in rows:
            if row["action"] in actions:
                flat_rows.append(row)
                flat_keys.append((split, qid, row["action"]))
    pred_by_key: dict[tuple[str, str, str], float] = {}
    if flat_rows:
        for key, pred in zip(flat_keys, predict(model, flat_rows, cols), strict=True):
            pred_by_key[key] = float(pred)

    vectors: list[list[float]] = []
    for split, qid in all_keys:
        rows = train_by_qid[qid] if split == "train" else eval_by_qid[qid]
        row_by_action = {row["action"]: row for row in rows}
        available = [action for action in actions if action in row_by_action]
        if not available:
            vectors.append([0.0 for _ in actions] + [0.0, 0.0, 0.0])
            continue
        pred_by_action = {action: pred_by_key.get((split, qid, action), 0.0) for action in available}
        values = np.asarray([pred_by_action.get(action, min(pred_by_action.values())) for action in actions], dtype=np.float64)
        max_value = float(values.max())
        centered = values - max_value
        sorted_values = np.sort(values)[::-1]
        margin = float(sorted_values[0] - sorted_values[1]) if len(sorted_values) > 1 else 0.0
        spread = float(values.std()) if len(values) > 1 else 0.0
        feasible_share = mean(float(row_by_action[action].get("feasible", 0.0)) for action in available)
        vectors.append(centered.tolist() + [margin, spread, feasible_share])
    return vectors


def workload_feature_columns(mode: str, feature_set: str) -> list[str]:
    if mode == "query_only" or feature_set == "query_only":
        return list(QUERY_ONLY_WORKLOAD_COMPILER_FEATURES)
    if feature_set == "metadata_only":
        return list(METADATA_ONLY_WORKLOAD_COMPILER_FEATURES)
    cols = list(WORKLOAD_COMPILER_FEATURES)
    if feature_set == "no_bm25":
        return [col for col in cols if col not in BM25_WORKLOAD_COMPILER_FEATURES]
    if feature_set == "no_dense":
        return [col for col in cols if col not in DENSE_WORKLOAD_COMPILER_FEATURES]
    if feature_set == "no_overlap":
        return [col for col in cols if col != "probe_overlap"]
    if feature_set == "no_probe":
        return [col for col in cols if col not in PROBE_WORKLOAD_COMPILER_FEATURES]
    return cols


def select_workload_k(
    x: np.ndarray,
    requested: int,
    candidates: str | list[int],
    seed: int,
    strategy: str,
) -> int:
    if strategy == "fixed":
        return max(1, min(int(requested), len(x)))
    valid = [
        k
        for k in parse_k_candidates(candidates)
        if 1 < k < len(x)
    ]
    if not valid:
        return max(1, min(int(requested), len(x)))
    if strategy == "silhouette":
        scored = []
        for k in valid:
            labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(x)
            if len(set(labels)) <= 1 or len(set(labels)) >= len(x):
                continue
            scored.append((float(silhouette_score(x, labels)), -k, k))
        if scored:
            return max(scored)[2]
    if strategy == "elbow":
        inertias = []
        for k in valid:
            model = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(x)
            inertias.append((k, float(model.inertia_)))
        if len(inertias) == 1:
            return inertias[0][0]
        x1, y1 = float(inertias[0][0]), math.log(max(inertias[0][1], 1e-12))
        x2, y2 = float(inertias[-1][0]), math.log(max(inertias[-1][1], 1e-12))
        denom = math.hypot(y2 - y1, x2 - x1)
        if denom <= 0:
            return inertias[0][0]
        distances = []
        for k, inertia in inertias:
            y0 = math.log(max(inertia, 1e-12))
            distance = abs((y2 - y1) * k - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / denom
            distances.append((distance, -k, k))
        return max(distances)[2]
    return max(1, min(int(requested), len(x)))


def parse_k_candidates(value: str | list[int]) -> list[int]:
    if isinstance(value, str):
        return [int(item) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def workload_feature_vector(rows: list[dict], cols: list[str]) -> list[float]:
    return [float(rows[0].get(col, 0.0)) for col in cols]


def scoped_action_rows(rows: list[dict], scope: str) -> list[dict]:
    scoped = []
    for row in rows:
        action = str(row["action"])
        if scope == "core" and action.startswith("external_"):
            continue
        if scope == "pre" and not action.endswith("_pre"):
            continue
        if scope == "core_pre" and (action.startswith("external_") or not action.endswith("_pre")):
            continue
        scoped.append(row)
    return scoped or rows


def best_scoped_action(rows: list[dict], scope: str) -> str:
    scoped = scoped_action_rows(rows, scope)
    actions = sorted({row["action"] for row in scoped})
    if not actions:
        actions = sorted({row["action"] for row in rows})
    return best_action(scoped or rows, actions)


def best_mean_score_action(rows: list[dict], scope: str) -> str:
    scoped = scoped_action_rows(rows, scope)
    actions = sorted({row["action"] for row in scoped})
    if not actions:
        actions = sorted({row["action"] for row in rows})
        scoped = rows
    return max(actions, key=lambda action: mean(float(row["constrained_score"]) for row in scoped if row["action"] == action))


def ranked_mean_score_actions(rows: list[dict], scope: str) -> list[tuple[str, float]]:
    scoped = scoped_action_rows(rows, scope)
    actions = sorted({row["action"] for row in scoped})
    if not actions:
        scoped = rows
        actions = sorted({row["action"] for row in scoped})
    scored = [
        (action, mean(float(row["constrained_score"]) for row in scoped if row["action"] == action))
        for action in actions
    ]
    return sorted(scored, key=lambda item: (item[1], action_plan_complexity_bonus(item[0])), reverse=True)


def ranked_component_shrink_actions(
    rows: list[dict],
    model: RandomForestRegressor,
    cols: list[str],
    shrink: float,
) -> list[tuple[str, float]]:
    component_rows = component_plan_rows(rows)
    pred = predict(model, component_rows, cols)
    pred_by_action: dict[str, list[float]] = {}
    for row, value in zip(component_rows, pred, strict=True):
        pred_by_action.setdefault(row["action"], []).append(float(value))
    empirical = ranked_mean_score_actions(rows, "all")
    scored = []
    for action, empirical_score in empirical:
        predicted_score = mean(pred_by_action.get(action, [empirical_score]))
        score = (1.0 - shrink) * empirical_score + shrink * predicted_score
        scored.append((action, score))
    return sorted(scored, key=lambda item: (item[1], action_plan_complexity_bonus(item[0])), reverse=True)


def action_plan_complexity_bonus(action: str) -> float:
    components = plan_components(action)
    bonus = 0.0
    if components["fusion"] != "none":
        bonus += 0.02
    if components["rerank_depth"] > 0:
        bonus += 0.01
    if components["filter_placement"] == "pre":
        bonus += 0.005
    return bonus


def plan_components(action: str) -> dict:
    tokens = action.split("_")
    filter_placement = "pre" if action.endswith("_pre") else ("post" if action.endswith("_post") else "none")
    generator = "external" if action.startswith("external_") else tokens[0]
    fusion = "none"
    model = "default"
    rerank_depth = 0.0
    if action.startswith("rrf"):
        fusion = "rrf"
    elif action.startswith("weighted"):
        fusion = "weighted"
    if action.startswith("rerank_depth_"):
        generator = "bm25+dense"
        fusion = "cross_encoder"
        rerank_depth = float(action.rsplit("_", 1)[-1])
    elif action.startswith("ce_rerank_depth_"):
        generator = "bm25+dense"
        fusion = "cross_encoder"
        rerank_depth = float(action.rsplit("_", 1)[-1])
    elif "e5_base_v2" in action:
        model = "e5_base_v2"
    elif "qwen3_embedding_06b" in action:
        model = "qwen3_embedding_06b"
    elif "splade_cocondenser_ensembledistil" in action:
        model = "splade_cocondenser_ensembledistil"
    elif "colbertv2" in action:
        model = "colbertv2"
    return {
        "generator": generator,
        "fusion": fusion,
        "model": model,
        "filter_placement": filter_placement,
        "rerank_depth": rerank_depth,
    }


def plan_expression(action: str) -> dict:
    components = plan_components(action)
    expression = (
        f"Plan(generator={components['generator']}, model={components['model']}, "
        f"fusion={components['fusion']}, filter={components['filter_placement']}, "
        f"rerank_depth={components['rerank_depth']:g})"
    )
    return {"action": action, "expression": expression, **components}


def component_plan_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        item = dict(row)
        components = plan_components(str(row["action"]))
        for key in ["generator", "fusion", "model", "filter_placement"]:
            item[f"plan_{key}_{safe_feature_name(str(components[key]))}"] = 1.0
        item["plan_rerank_depth"] = float(components["rerank_depth"])
        out.append(item)
    return out


def component_feature_columns(rows: list[dict]) -> list[str]:
    base = [
        "query_len",
        "query_digit_ratio",
        "selectivity",
        "candidate_count",
        "bm25_top_score",
        "bm25_entropy",
        "bm25_gap",
        "dense_top_score",
        "dense_entropy",
        "dense_gap",
        "probe_overlap",
        "probe_latency_ms",
    ]
    component_cols = sorted({col for row in rows for col in row if col.startswith("plan_")})
    return base + component_cols


def fit_frontier_guard(train_rows: list[dict], seed: int) -> dict | None:
    if not train_rows:
        return None
    qids = sorted({row["qid"] for row in train_rows})
    if len(qids) >= 8:
        fit_qids, cal_qids = train_test_split(qids, test_size=0.35, random_state=seed + 1701)
        fit_set = set(fit_qids)
        cal_set = set(cal_qids)
        fit_rows = [row for row in train_rows if row["qid"] in fit_set]
        cal_rows = [row for row in train_rows if row["qid"] in cal_set]
    else:
        fit_rows = train_rows
        cal_rows = train_rows
    if not fit_rows or not cal_rows:
        return None
    cols = full_features(train_rows)
    model = fit_target(fit_rows, cols, "constrained_score", seed + 1703)
    cal_pred = predict(model, cal_rows, cols)
    qhat_by_action = conformal_lower(cal_rows, cal_pred, "constrained_score", alpha=0.2)
    return {"model": model, "cols": cols, "qhat": qhat_by_action}


def select_frontier_lcb_action(
    candidates: list[dict],
    frontier: list[str],
    guarded_predictor: dict,
    primary_action: str,
    min_advantage: float,
) -> dict | None:
    frontier_set = set(frontier)
    rows = [row for row in candidates if row["action"] in frontier_set]
    if not rows:
        return None
    pred = predict(guarded_predictor["model"], rows, guarded_predictor["cols"])
    qhat = guarded_predictor["qhat"]
    scored = []
    for row, value in zip(rows, pred, strict=True):
        action = row["action"]
        lower = float(value) - float(qhat.get(action, qhat.get("__global__", 0.0)))
        scored.append((lower, float(row.get("feasible", 0.0)), -float(row.get("latency_ms", 0.0)), row))
    best = max(scored, key=lambda item: item[:3])
    primary = next((item for item in scored if item[3]["action"] == primary_action), None)
    if primary is not None and best[3]["action"] != primary_action and best[0] < primary[0] + min_advantage:
        return primary[3]
    return best[3]


def calibrate_frontier_guard_by_label(
    train_by_qid: dict[str, list[dict]],
    train_qids_by_label: dict[int, list[str]],
    action_by_label: dict[int, str],
    support_by_label: dict[int, float],
    margin_by_label: dict[int, float],
    frontier_by_label: dict[int, list[str]],
    guarded_predictor: dict,
    support_threshold: float,
) -> dict[int, bool]:
    enabled: dict[int, bool] = {}
    for label, qids in train_qids_by_label.items():
        primary_action = action_by_label.get(label)
        frontier = frontier_by_label.get(label, [])
        support = support_by_label.get(label, 1.0)
        margin = margin_by_label.get(label, 1.0)
        fragile = support < support_threshold and (margin < 0.02 or len(frontier) > 2)
        if not fragile or not primary_action or not frontier or len(qids) < 3:
            enabled[label] = False
            continue
        primary_rows = []
        guarded_rows = []
        for qid in qids:
            candidates = train_by_qid.get(qid, [])
            primary = next((row for row in candidates if row["action"] == primary_action), None)
            guarded = select_frontier_lcb_action(candidates, frontier, guarded_predictor, primary_action, min_advantage=0.03)
            if primary is not None and guarded is not None:
                primary_rows.append(primary)
                guarded_rows.append(guarded)
        if not primary_rows or not guarded_rows:
            enabled[label] = False
            continue
        primary_score = mean(float(row["constrained_score"]) for row in primary_rows)
        guarded_score = mean(float(row["constrained_score"]) for row in guarded_rows)
        guarded_latency = mean(float(row["latency_ms"]) for row in guarded_rows)
        primary_latency = mean(float(row["latency_ms"]) for row in primary_rows)
        enabled[label] = guarded_score > primary_score + 0.01 and guarded_latency <= primary_latency + 10.0
    return enabled


def action_matches_any(action: str, keywords: tuple[str, ...]) -> bool:
    lower = action.lower()
    return any(keyword in lower for keyword in keywords)


def evaluate_delta_lcb_switch(
    fit_rows: list[dict],
    calibration_rows: list[dict],
    eval_rows: list[dict],
    cols: list[str],
    seed: int,
    base_action: str,
    alpha: float,
    threshold: float,
    fixed_qhat: float | None = None,
) -> tuple[dict, float]:
    fit_delta = rows_with_delta_to_base(fit_rows, base_action)
    cal_delta = rows_with_delta_to_base(calibration_rows, base_action)
    eval_delta = rows_with_delta_to_base(eval_rows, base_action)
    if not fit_delta or not cal_delta or not eval_delta:
        return evaluate_static(eval_rows, base_action), 0.0
    model = fit_target(fit_delta, cols, "delta_to_base", seed + 503)
    cal_pred = predict(model, cal_delta, cols)
    qhat = fixed_qhat
    if qhat is None:
        residuals = [max(0.0, float(pred) - float(row["delta_to_base"])) for row, pred in zip(cal_delta, cal_pred, strict=True)]
        qhat = conformal_quantile(residuals, alpha)
    eval_pred = predict(model, eval_delta, cols)
    chosen = []
    for candidates in by_qid_pred(eval_delta, eval_pred).values():
        base = next((item for item in candidates if item[0]["action"] == base_action), None)
        best = max(candidates, key=lambda item: item[1] - qhat)
        best_lcb = best[1] - qhat
        if base is not None and best_lcb < threshold:
            chosen.append(base[0])
        else:
            chosen.append(best[0])
    return evaluate_chosen(chosen, eval_rows), float(qhat)


def rows_with_delta_to_base(rows: list[dict], base_action: str) -> list[dict]:
    out = []
    for candidates in by_qid(rows).values():
        base = next((row for row in candidates if row["action"] == base_action), None)
        if base is None:
            continue
        base_score = float(base["constrained_score"])
        for row in candidates:
            item = dict(row)
            item["delta_to_base"] = float(row["constrained_score"]) - base_score
            out.append(item)
    return out


def evaluate_chosen(chosen: list[dict], rows: list[dict]) -> dict:
    oracle = {qid: max(candidates, key=lambda row: float(row["constrained_score"])) for qid, candidates in by_qid(rows).items()}
    regrets = []
    correct = 0
    counts = Counter()
    for row in chosen:
        best = oracle[row["qid"]]
        regrets.append(float(best["constrained_score"]) - float(row["constrained_score"]))
        correct += row["action"] == best["action"]
        counts[row["action"]] += 1
    return {
        "queries": len(chosen),
        "mean_score": mean(float(row["constrained_score"]) for row in chosen),
        "mean_ndcg10": mean(float(row["ndcg10"]) for row in chosen),
        "mean_recall100": mean(float(row["recall100"]) for row in chosen),
        "satisfaction_rate": mean(float(row["feasible"]) for row in chosen),
        "mean_latency_ms": mean(float(row["latency_ms"]) for row in chosen),
        "mean_regret": mean(regrets),
        "oracle_action_accuracy": correct / max(1, len(chosen)),
        "action_distribution": json.dumps(dict(sorted(counts.items())), sort_keys=True),
    }


def best_action(rows: list[dict], actions: list[str]) -> str:
    return max(actions, key=lambda action: action_selection_key(rows, action))


def action_selection_key(rows: list[dict], action: str) -> tuple[float, float, float, float]:
    action_rows = [row for row in rows if row["action"] == action]
    score = mean(float(row["constrained_score"]) for row in action_rows)
    satisfaction = mean(float(row["feasible"]) for row in action_rows)
    latency = mean(float(row["latency_ms"]) for row in action_rows)
    pre_bonus = 1.0 if action.endswith("_pre") else 0.0
    return (score, satisfaction, -latency, pre_bonus)


def fit_target(rows: list[dict], cols: list[str], target: str, seed: int) -> RandomForestRegressor:
    x = np.asarray([[float(row.get(col, 0.0)) for col in cols] for row in rows], dtype=np.float64)
    y = np.asarray([float(row[target]) for row in rows], dtype=np.float64)
    model = RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=seed)
    model.fit(x, y)
    return model


def predict(model: RandomForestRegressor, rows: list[dict], cols: list[str]) -> np.ndarray:
    x = np.asarray([[float(row.get(col, 0.0)) for col in cols] for row in rows], dtype=np.float64)
    return model.predict(x)


def conformal_lower(rows: list[dict], pred: np.ndarray, target: str, alpha: float) -> dict[str, float]:
    residuals: dict[str, list[float]] = {"__global__": []}
    for row, y_pred in zip(rows, pred, strict=True):
        residual = max(0.0, float(y_pred) - float(row[target]))
        residuals["__global__"].append(residual)
        residuals.setdefault(row["action"], []).append(residual)
    return {key: conformal_quantile(vals, alpha) for key, vals in residuals.items()}


def conformal_upper(rows: list[dict], pred: np.ndarray, target: str, alpha: float) -> dict[str, float]:
    residuals: dict[str, list[float]] = {"__global__": []}
    for row, y_pred in zip(rows, pred, strict=True):
        residual = max(0.0, float(row[target]) - float(y_pred))
        residuals["__global__"].append(residual)
        residuals.setdefault(row["action"], []).append(residual)
    return {key: conformal_quantile(vals, alpha) for key, vals in residuals.items()}


def conformal_quantile(values: list[float], alpha: float) -> float:
    if not values:
        return 0.0
    vals = np.asarray(sorted(values), dtype=np.float64)
    idx = min(len(vals) - 1, int(math.ceil((len(vals) + 1) * (1.0 - alpha))) - 1)
    return float(vals[max(0, idx)])


def query_features(rows: list[dict]) -> list[str]:
    return ["query_len", "query_digit_ratio", "selectivity"] + action_feature_cols(rows)


def full_features(rows: list[dict]) -> list[str]:
    return [
        "query_len",
        "query_digit_ratio",
        "selectivity",
        "candidate_count",
        "bm25_top_score",
        "bm25_entropy",
        "bm25_gap",
        "dense_top_score",
        "dense_entropy",
        "dense_gap",
        "probe_overlap",
        "probe_latency_ms",
    ] + action_feature_cols(rows)


def action_feature_cols(rows: list[dict]) -> list[str]:
    return sorted({col for row in rows for col in row if col.startswith("action_")})


def action_features(action: str) -> dict[str, float]:
    families = ["bm25", "dense", "rrf", "weighted", "rerank", "ce_rerank", "external"]
    modes = ["pre", "post"]
    out = {f"action_family_{family}": 1.0 if action.startswith(family) else 0.0 for family in families}
    out.update({f"action_mode_{mode}": 1.0 if action.endswith(mode) or f"_{mode}" in action else 0.0 for mode in modes})
    out["action_is_rerank"] = 1.0 if action.startswith("rerank") or action.startswith("ce_rerank") else 0.0
    out["action_rerank_depth"] = (
        float(action.rsplit("_", 1)[-1])
        if action.startswith("rerank_depth_") or action.startswith("ce_rerank_depth_")
        else 0.0
    )
    out[f"action_id_{safe_feature_name(action)}"] = 1.0
    return out


def safe_feature_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


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


def truncate_result(result: SearchResult, top_k: int) -> SearchResult:
    ranking = result.ranking[:top_k]
    scores = {docid: result.scores[docid] for docid in ranking if docid in result.scores}
    return SearchResult(ranking, scores, result.latency_ms)


def normalize_score_map(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = np.asarray(list(scores.values()), dtype=np.float64)
    lo = float(vals.min())
    hi = float(vals.max())
    if math.isclose(lo, hi):
        return {key: 1.0 for key in scores}
    return {key: (value - lo) / (hi - lo) for key, value in scores.items()}


def load_external_rankings(root: Path | None, dataset: str) -> dict[str, dict[str, SearchResult]]:
    if root is None:
        return {}
    paths = []
    dataset_dir = root / dataset
    if dataset_dir.exists():
        paths.extend(sorted(dataset_dir.glob("*.tsv")))
        paths.extend(sorted(dataset_dir.glob("*.csv")))
        paths.extend(sorted(dataset_dir.glob("*.jsonl")))
    paths.extend(sorted(root.glob(f"{dataset}*.tsv")))
    paths.extend(sorted(root.glob(f"{dataset}*.csv")))
    paths.extend(sorted(root.glob(f"{dataset}*.jsonl")))
    out: dict[str, dict[str, list[tuple[int, str, float, float]]]] = {}
    for path in sorted(set(paths)):
        for row in read_ranking_rows(path):
            operator = clean_operator_name(row.get("operator") or row.get("method") or path.stem)
            qid = str(row["qid"])
            docid = str(row["docid"])
            rank = int(float(row.get("rank", len(out.get(operator, {}).get(qid, [])) + 1)))
            score = float(row.get("score", -rank))
            latency = float(row.get("latency_ms", 0.0))
            out.setdefault(operator, {}).setdefault(qid, []).append((rank, docid, score, latency))
    converted: dict[str, dict[str, SearchResult]] = {}
    for operator, by_qid in out.items():
        converted[operator] = {}
        for qid, items in by_qid.items():
            items = sorted(items, key=lambda item: item[0])
            ranking = [docid for _, docid, _, _ in items]
            scores = {docid: score for _, docid, score, _ in items}
            latency = max((lat for _, _, _, lat in items), default=0.0)
            converted[operator][qid] = SearchResult(ranking, scores, latency)
    return converted


def read_ranking_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8", newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        return list(csv.DictReader(fh, dialect=dialect))


def clean_operator_name(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def load_metadata(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        rows = list(csv.DictReader(fh, dialect=dialect))
    if not rows:
        return {}
    doc_col = "docid" if "docid" in rows[0] else "_id"
    value_col = "bucket" if "bucket" in rows[0] else "metadata"
    return {str(row[doc_col]): str(row[value_col]) for row in rows if row.get(doc_col)}


def passes_filter(docid: str, selectivity: float, metadata: dict[str, str] | None = None) -> bool:
    if selectivity >= 1.0:
        return True
    bucket = metadata_bucket(metadata[docid]) if metadata and docid in metadata else stable_bucket(docid)
    threshold = max(1, min(100, int(round(selectivity * 100))))
    return bucket < threshold


def metadata_bucket(value: str) -> int:
    try:
        numeric = float(value)
        if 0.0 <= numeric <= 1.0:
            return int(numeric * 100)
        return int(numeric) % 100
    except ValueError:
        return stable_bucket(value)


def stable_bucket(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def first_score(result: SearchResult) -> float:
    if not result.ranking:
        return 0.0
    return float(result.scores.get(result.ranking[0], 0.0))


def score_gap(result: SearchResult) -> float:
    if len(result.ranking) < 2:
        return first_score(result)
    return float(result.scores.get(result.ranking[0], 0.0) - result.scores.get(result.ranking[1], 0.0))


def digit_ratio(query: str) -> float:
    toks = tokenize(query)
    return sum(any(ch.isdigit() for ch in tok) for tok in toks) / max(1, len(toks))


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


def write_report(path: Path, summary_rows: list[dict], gate) -> None:
    by_method: dict[str, list[dict]] = {}
    for row in summary_rows:
        by_method.setdefault(row["method"], []).append(row)
    lines = [
        "# ParetoProbe-Sys Action-Space Pilot",
        "",
        f"Complete rank blocks: `{gate.blocks}`",
        f"Rank gate pass: `{gate.pass_gate}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Method | Mean score | Satisfaction | Latency | Regret | Avg rank |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in sorted(by_method, key=lambda item: gate.average_ranks.get(item, math.inf)):
        rows = by_method[method]
        lines.append(
            f"| `{method}` | {mean(float(row['mean_score']) for row in rows):.4f} | "
            f"{mean(float(row['satisfaction_rate']) for row in rows):.3f} | "
            f"{mean(float(row['mean_latency_ms']) for row in rows):.2f} | "
            f"{mean(float(row['mean_regret']) for row in rows):.4f} | "
            f"{gate.average_ranks.get(method, math.inf):.3f} |"
        )
    if gate.failures:
        lines += ["", "## Failures", ""]
        lines.extend(f"- {failure}" for failure in gate.failures)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
