from __future__ import annotations

import json
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


BEIR_URLS = {
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "nfcorpus": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
    "fiqa": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
    "arguana": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip",
    "scidocs": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip",
}


@dataclass
class RetrievalDataset:
    name: str
    source: str
    corpus: dict[str, str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


def download_beir_dataset(name: str, data_dir: Path) -> Path:
    if name not in BEIR_URLS:
        raise ValueError(f"Unsupported BEIR dataset: {name}")
    data_dir.mkdir(parents=True, exist_ok=True)
    target_dir = data_dir / name
    if (target_dir / "corpus.jsonl").exists():
        return target_dir

    zip_path = data_dir / f"{name}.zip"
    url = BEIR_URLS[name]
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    if not target_dir.exists():
        candidates = [p for p in data_dir.iterdir() if p.is_dir() and p.name.lower() == name]
        if candidates:
            target_dir = candidates[0]
    if not (target_dir / "corpus.jsonl").exists():
        raise FileNotFoundError(f"Downloaded {name}, but corpus.jsonl was not found under {target_dir}")
    return target_dir


def load_beir_dataset(
    name: str,
    data_dir: Path,
    split: str = "test",
    max_docs: int | None = None,
    max_queries: int | None = None,
    allow_download: bool = True,
    allow_tiny_fallback: bool = True,
) -> RetrievalDataset:
    dataset_dir = data_dir / name
    try:
        if allow_download and not (dataset_dir / "corpus.jsonl").exists():
            dataset_dir = download_beir_dataset(name, data_dir)
        dataset = _read_beir_dir(name, dataset_dir, split, max_docs, max_queries)
        dataset.source = "beir"
        return dataset
    except Exception:
        if not allow_tiny_fallback:
            raise
        return tiny_dataset(max_docs=max_docs, max_queries=max_queries)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_beir_dir(
    name: str,
    dataset_dir: Path,
    split: str,
    max_docs: int | None,
    max_queries: int | None,
) -> RetrievalDataset:
    corpus_rows = _read_jsonl(dataset_dir / "corpus.jsonl")
    query_rows = _read_jsonl(dataset_dir / "queries.jsonl")
    qrels_path = dataset_dir / "qrels" / f"{split}.tsv"
    if not qrels_path.exists():
        qrels_files = sorted((dataset_dir / "qrels").glob("*.tsv"))
        if not qrels_files:
            raise FileNotFoundError(f"No qrels TSV found under {dataset_dir / 'qrels'}")
        qrels_path = qrels_files[0]

    all_corpus = {
        str(row["_id"]): " ".join(
            part for part in [str(row.get("title", "")), str(row.get("text", ""))] if part
        ).strip()
        for row in corpus_rows
    }
    all_queries = {str(row["_id"]): str(row.get("text", "")).strip() for row in query_rows}

    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 3 or parts[0].lower().startswith("query"):
                continue
            qid, docid, score_text = parts[0], parts[1], parts[2]
            try:
                score = int(float(score_text))
            except ValueError:
                continue
            if score > 0 and qid in all_queries and docid in all_corpus:
                qrels.setdefault(qid, {})[docid] = score

    query_ids = [qid for qid in all_queries if qid in qrels]
    if max_queries is not None:
        query_ids = query_ids[:max_queries]
    qrels = {qid: qrels[qid] for qid in query_ids}
    queries = {qid: all_queries[qid] for qid in query_ids}

    relevant_doc_ids = {docid for rels in qrels.values() for docid in rels}
    doc_ids = list(relevant_doc_ids)
    if max_docs is None:
        doc_ids = list(all_corpus)
    else:
        for docid in all_corpus:
            if len(doc_ids) >= max_docs:
                break
            if docid not in relevant_doc_ids:
                doc_ids.append(docid)
    corpus = {docid: all_corpus[docid] for docid in doc_ids if docid in all_corpus}
    qrels = {
        qid: {docid: score for docid, score in rels.items() if docid in corpus}
        for qid, rels in qrels.items()
    }
    qrels = {qid: rels for qid, rels in qrels.items() if rels}
    queries = {qid: queries[qid] for qid in queries if qid in qrels}
    return RetrievalDataset(name=name, source="beir", corpus=corpus, queries=queries, qrels=qrels)


def tiny_dataset(max_docs: int | None = None, max_queries: int | None = None) -> RetrievalDataset:
    corpus = {
        "d1": "database query optimization chooses efficient physical plans under latency constraints",
        "d2": "hybrid search combines sparse lexical retrieval and dense vector retrieval",
        "d3": "pareto optimization balances quality latency storage and compute cost",
        "d4": "neural rerankers improve ranking quality but add serving latency",
        "d5": "vector indexes use approximate nearest neighbor search for embeddings",
        "d6": "bm25 is a sparse lexical retrieval baseline for information retrieval",
        "d7": "query performance prediction estimates retrieval quality without labels",
        "d8": "filtered vector search depends on metadata selectivity and pre filtering",
    }
    queries = {
        "q1": "hybrid sparse dense retrieval",
        "q2": "query optimization latency physical plan",
        "q3": "pareto quality latency cost",
        "q4": "filtered vector search metadata",
    }
    qrels = {
        "q1": {"d2": 2, "d6": 1},
        "q2": {"d1": 2, "d4": 1},
        "q3": {"d3": 2},
        "q4": {"d8": 2, "d5": 1},
    }
    if max_docs is not None:
        keep_docs = set(list(corpus)[:max_docs])
        corpus = {k: v for k, v in corpus.items() if k in keep_docs}
        qrels = {q: {d: s for d, s in rels.items() if d in corpus} for q, rels in qrels.items()}
    if max_queries is not None:
        keep_queries = set(list(queries)[:max_queries])
        queries = {k: v for k, v in queries.items() if k in keep_queries}
        qrels = {q: rels for q, rels in qrels.items() if q in queries and rels}
    return RetrievalDataset(name="tiny", source="tiny_fallback", corpus=corpus, queries=queries, qrels=qrels)


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())
