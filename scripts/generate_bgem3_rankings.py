#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paretoprobe.data import load_beir_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate external BGE-M3 dense/sparse/colbert/hybrid rankings.")
    parser.add_argument("--dataset", default="nfcorpus")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-docs", type=int, default=1000)
    parser.add_argument("--max-queries", type=int, default=20)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "external_rankings")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--skip-colbert", action="store_true")
    args = parser.parse_args()

    from FlagEmbedding import BGEM3FlagModel

    dataset = load_beir_dataset(
        name=args.dataset,
        data_dir=ROOT / "data",
        split=args.split,
        max_docs=args.max_docs,
        max_queries=args.max_queries,
        allow_download=not args.no_download,
        allow_tiny_fallback=False,
    )
    doc_ids = list(dataset.corpus)
    docs = [dataset.corpus[docid] for docid in doc_ids]
    qids = [qid for qid in dataset.queries if qid in dataset.qrels]
    queries = [dataset.queries[qid] for qid in qids]

    model = BGEM3FlagModel(args.model, use_fp16=False, batch_size=args.batch_size)
    doc_start = time.perf_counter()
    doc_out = model.encode(
        docs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=not args.skip_colbert,
    )
    doc_time_ms = (time.perf_counter() - doc_start) * 1000.0

    doc_dense = np.asarray(doc_out["dense_vecs"], dtype=np.float32)
    doc_sparse = doc_out["lexical_weights"]
    doc_colbert = doc_out.get("colbert_vecs") if not args.skip_colbert else None

    out_dir = args.output_dir / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bgem3_official.tsv"
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["qid", "docid", "rank", "score", "operator", "latency_ms"],
            delimiter="\t",
        )
        writer.writeheader()
        for qid, query in zip(qids, queries, strict=True):
            q_start = time.perf_counter()
            q_out = model.encode(
                [query],
                batch_size=1,
                max_length=args.max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=not args.skip_colbert,
            )
            q_dense = np.asarray(q_out["dense_vecs"], dtype=np.float32)[0]
            dense_scores = np.asarray(doc_dense @ q_dense, dtype=np.float64)
            sparse_scores = sparse_scores_for_query(q_out["lexical_weights"][0], doc_sparse)
            score_maps = {
                "bge_m3_dense": dense_scores,
                "bge_m3_sparse": sparse_scores,
            }
            if doc_colbert is not None:
                colbert_scores = colbert_scores_for_query(q_out["colbert_vecs"][0], doc_colbert)
                score_maps["bge_m3_colbert"] = colbert_scores
                score_maps["bge_m3_hybrid"] = (
                    normalize(dense_scores) + normalize(sparse_scores) + normalize(colbert_scores)
                ) / 3.0
            else:
                score_maps["bge_m3_hybrid"] = (normalize(dense_scores) + normalize(sparse_scores)) / 2.0
            latency_ms = (time.perf_counter() - q_start) * 1000.0
            for operator, scores in score_maps.items():
                top_idx = top_indices(scores, args.top_k)
                for rank, idx in enumerate(top_idx, start=1):
                    writer.writerow(
                        {
                            "qid": qid,
                            "docid": doc_ids[idx],
                            "rank": rank,
                            "score": float(scores[idx]),
                            "operator": operator,
                            "latency_ms": latency_ms,
                        }
                    )
    print({"output": str(out_path), "docs": len(doc_ids), "queries": len(qids), "doc_encode_ms": doc_time_ms})


def sparse_scores_for_query(q_weights: dict, doc_weights: list[dict]) -> np.ndarray:
    scores = np.zeros(len(doc_weights), dtype=np.float64)
    q_items = [(str(token), float(weight)) for token, weight in q_weights.items()]
    for idx, weights in enumerate(doc_weights):
        total = 0.0
        for token, q_weight in q_items:
            total += q_weight * float(weights.get(token, 0.0))
        scores[idx] = total
    return scores


def colbert_scores_for_query(q_vecs: np.ndarray, doc_vecs: list[np.ndarray]) -> np.ndarray:
    q = np.asarray(q_vecs, dtype=np.float32)
    scores = np.zeros(len(doc_vecs), dtype=np.float64)
    for idx, d_vecs in enumerate(doc_vecs):
        d = np.asarray(d_vecs, dtype=np.float32)
        if q.size == 0 or d.size == 0:
            continue
        sims = q @ d.T
        scores[idx] = float(np.max(sims, axis=1).sum())
    return scores


def normalize(scores: np.ndarray) -> np.ndarray:
    vals = np.asarray(scores, dtype=np.float64)
    lo = float(vals.min())
    hi = float(vals.max())
    if np.isclose(hi, lo):
        return np.ones_like(vals)
    return (vals - lo) / (hi - lo)


def top_indices(scores: np.ndarray, top_k: int) -> list[int]:
    if scores.size == 0 or top_k <= 0:
        return []
    top_k = min(top_k, scores.size)
    idx = np.argpartition(-scores, top_k - 1)[:top_k]
    idx = idx[np.argsort(-scores[idx])]
    return idx.tolist()


if __name__ == "__main__":
    main()
