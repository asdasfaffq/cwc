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

from paretoprobe.data import RetrievalDataset, load_beir_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate external ranking files from real HF retrieval/reranking models.")
    parser.add_argument("--dataset", default="nfcorpus")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-docs", type=int, default=500)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--kind", choices=["dense", "splade", "colbert", "causal_yesno"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--candidate-k", type=int, default=80)
    parser.add_argument("--pooling", choices=["mean", "cls", "last"], default="mean")
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--doc-prefix", default="")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "external_rankings_real")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    dataset = load_beir_dataset(
        name=args.dataset,
        data_dir=ROOT / "data",
        split=args.split,
        max_docs=args.max_docs,
        max_queries=args.max_queries,
        allow_download=not args.no_download,
        allow_tiny_fallback=False,
    )

    if args.kind == "dense":
        rows = generate_dense(dataset, args)
    elif args.kind == "splade":
        rows = generate_splade(dataset, args)
    elif args.kind == "colbert":
        rows = generate_colbert(dataset, args)
    else:
        rows = generate_causal_yesno(dataset, args)

    out_dir = args.output_dir / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_name(args.operator)}.tsv"
    write_rows(out_path, rows)
    print({"output": str(out_path), "rows": len(rows), "operator": args.operator})


def generate_dense(dataset: RetrievalDataset, args) -> list[dict]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=args.trust_remote_code).to(device)
    model.eval()

    doc_ids = list(dataset.corpus)
    docs = [args.doc_prefix + dataset.corpus[docid] for docid in doc_ids]
    doc_emb = encode_dense(model, tokenizer, docs, args.batch_size, args.max_length, args.pooling, device)

    rows: list[dict] = []
    for qid, query in query_items(dataset):
        start = time.perf_counter()
        q_emb = encode_dense(model, tokenizer, [args.query_prefix + query], 1, args.max_length, args.pooling, device)[0]
        scores = np.asarray(doc_emb @ q_emb, dtype=np.float64)
        latency_ms = (time.perf_counter() - start) * 1000.0
        rows.extend(ranking_rows(qid, doc_ids, scores, args.operator, args.top_k, latency_ms))
    return rows


def encode_dense(model, tokenizer, texts: list[str], batch_size: int, max_length: int, pooling: str, device: str) -> np.ndarray:
    import torch

    vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state
            if pooling == "cls":
                pooled = hidden[:, 0]
            elif pooling == "last":
                lengths = encoded["attention_mask"].sum(dim=1) - 1
                pooled = hidden[torch.arange(hidden.shape[0], device=device), lengths]
            else:
                mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy().astype(np.float32))
    return np.vstack(vectors) if vectors else np.zeros((0, 1), dtype=np.float32)


def generate_splade(dataset: RetrievalDataset, args) -> list[dict]:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForMaskedLM.from_pretrained(args.model, trust_remote_code=args.trust_remote_code).to(device)
    model.eval()

    doc_ids = list(dataset.corpus)
    docs = [args.doc_prefix + dataset.corpus[docid] for docid in doc_ids]
    doc_vec = encode_splade(model, tokenizer, docs, args.batch_size, args.max_length, device)

    rows: list[dict] = []
    for qid, query in query_items(dataset):
        start = time.perf_counter()
        q_vec = encode_splade(model, tokenizer, [args.query_prefix + query], 1, args.max_length, device)[0]
        scores = np.asarray(doc_vec @ q_vec, dtype=np.float64)
        latency_ms = (time.perf_counter() - start) * 1000.0
        rows.extend(ranking_rows(qid, doc_ids, scores, args.operator, args.top_k, latency_ms))
    return rows


def encode_splade(model, tokenizer, texts: list[str], batch_size: int, max_length: int, device: str) -> np.ndarray:
    import torch

    vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            weights = torch.log1p(torch.relu(logits)) * encoded["attention_mask"].unsqueeze(-1)
            pooled = torch.max(weights, dim=1).values
            vectors.append(pooled.cpu().numpy().astype(np.float32))
    return np.vstack(vectors) if vectors else np.zeros((0, 1), dtype=np.float32)


def generate_colbert(dataset: RetrievalDataset, args) -> list[dict]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=args.trust_remote_code).to(device)
    model.eval()

    doc_ids = list(dataset.corpus)
    docs = [args.doc_prefix + dataset.corpus[docid] for docid in doc_ids]
    doc_vecs = encode_tokens(model, tokenizer, docs, args.batch_size, args.max_length, device)

    rows: list[dict] = []
    for qid, query in query_items(dataset):
        start = time.perf_counter()
        q_vec = encode_tokens(model, tokenizer, [args.query_prefix + query], 1, args.max_length, device)[0]
        scores = np.asarray([colbert_score(q_vec, d_vec) for d_vec in doc_vecs], dtype=np.float64)
        latency_ms = (time.perf_counter() - start) * 1000.0
        rows.extend(ranking_rows(qid, doc_ids, scores, args.operator, args.top_k, latency_ms))
    return rows


def encode_tokens(model, tokenizer, texts: list[str], batch_size: int, max_length: int, device: str) -> list[np.ndarray]:
    import torch

    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=2)
            masks = encoded["attention_mask"].bool()
            for item, mask in zip(hidden, masks, strict=True):
                out.append(item[mask].cpu().numpy().astype(np.float32))
    return out


def colbert_score(q_vec: np.ndarray, d_vec: np.ndarray) -> float:
    if q_vec.size == 0 or d_vec.size == 0:
        return 0.0
    sims = q_vec @ d_vec.T
    return float(np.max(sims, axis=1).sum())


def generate_causal_yesno(dataset: RetrievalDataset, args) -> list[dict]:
    import torch
    from paretoprobe.retrieval import BM25Index
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=args.trust_remote_code).to(device)
    model.eval()
    yes_id = first_token_id(tokenizer, " yes")
    no_id = first_token_id(tokenizer, " no")

    doc_ids = list(dataset.corpus)
    docs = [dataset.corpus[docid] for docid in doc_ids]
    bm25 = BM25Index(doc_ids, docs)

    rows: list[dict] = []
    with torch.no_grad():
        for qid, query in query_items(dataset):
            start = time.perf_counter()
            candidates = bm25.search(query, args.candidate_k).ranking
            scores = {}
            for docid in candidates:
                prompt = relevance_prompt(query, dataset.corpus[docid])
                encoded = tokenizer(prompt, truncation=True, max_length=args.max_length, return_tensors="pt")
                encoded = {key: value.to(device) for key, value in encoded.items()}
                logits = model(**encoded).logits[:, -1, :]
                scores[docid] = float(logits[0, yes_id] - logits[0, no_id])
            latency_ms = (time.perf_counter() - start) * 1000.0
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[: args.top_k]
            for rank, (docid, score) in enumerate(ranked, start=1):
                rows.append(
                    {
                        "qid": qid,
                        "docid": docid,
                        "rank": rank,
                        "score": score,
                        "operator": args.operator,
                        "latency_ms": latency_ms,
                    }
                )
    return rows


def relevance_prompt(query: str, document: str) -> str:
    return (
        "You are a retrieval reranker. Answer yes or no.\n"
        f"Query: {query}\n"
        f"Document: {document[:1600]}\n"
        "Is the document relevant to the query? Answer:"
    )


def first_token_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        ids = tokenizer(text.strip(), add_special_tokens=False)["input_ids"]
    return int(ids[0])


def query_items(dataset: RetrievalDataset) -> list[tuple[str, str]]:
    return [(qid, dataset.queries[qid]) for qid in dataset.queries if qid in dataset.qrels]


def ranking_rows(
    qid: str,
    doc_ids: list[str],
    scores: np.ndarray,
    operator: str,
    top_k: int,
    latency_ms: float,
) -> list[dict]:
    rows = []
    for rank, idx in enumerate(top_indices(scores, top_k), start=1):
        rows.append(
            {
                "qid": qid,
                "docid": doc_ids[idx],
                "rank": rank,
                "score": float(scores[idx]),
                "operator": operator,
                "latency_ms": latency_ms,
            }
        )
    return rows


def top_indices(scores: np.ndarray, top_k: int) -> list[int]:
    if scores.size == 0 or top_k <= 0:
        return []
    top_k = min(top_k, scores.size)
    idx = np.argpartition(-scores, top_k - 1)[:top_k]
    idx = idx[np.argsort(-scores[idx])]
    return idx.tolist()


def safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["qid", "docid", "rank", "score", "operator", "latency_ms"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
