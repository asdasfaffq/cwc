from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from .data import tokenize


@dataclass
class SearchResult:
    ranking: list[str]
    scores: dict[str, float]
    latency_ms: float


class BM25Index:
    def __init__(self, doc_ids: list[str], texts: list[str], k1: float = 1.2, b: float = 0.75):
        self.doc_ids = doc_ids
        self.vectorizer = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        self.x = self.vectorizer.fit_transform(texts).astype(np.float64).tocsr()
        self.k1 = k1
        self.b = b
        self.doc_len = np.asarray(self.x.sum(axis=1)).ravel()
        self.avgdl = float(np.mean(self.doc_len)) if len(self.doc_len) else 1.0
        df = np.asarray((self.x > 0).sum(axis=0)).ravel()
        n_docs = max(1, self.x.shape[0])
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))
        self.vocab = self.vectorizer.vocabulary_

    def search(self, query: str, top_k: int) -> SearchResult:
        start = time.perf_counter()
        term_ids = [self.vocab[t] for t in tokenize(query) if t in self.vocab]
        if not term_ids:
            return SearchResult([], {}, (time.perf_counter() - start) * 1000)
        term_ids = sorted(set(term_ids))
        scores = np.zeros(self.x.shape[0], dtype=np.float64)
        denom_norm = self.k1 * (1 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9))
        for term_id in term_ids:
            col = self.x[:, term_id].tocoo()
            tf = col.data
            partial = self.idf[term_id] * (tf * (self.k1 + 1)) / (tf + denom_norm[col.row])
            scores[col.row] += partial
        ranking_idx = top_indices(scores, top_k)
        ranking = [self.doc_ids[i] for i in ranking_idx if scores[i] > 0]
        score_map = {self.doc_ids[i]: float(scores[i]) for i in ranking_idx if scores[i] > 0}
        return SearchResult(ranking, score_map, (time.perf_counter() - start) * 1000)


class DenseSVDIndex:
    def __init__(self, doc_ids: list[str], texts: list[str], dim: int = 128, max_features: int = 50000):
        self.doc_ids = doc_ids
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            lowercase=False,
            token_pattern=None,
            sublinear_tf=True,
            ngram_range=(1, 2),
            max_features=max_features,
        )
        tfidf = self.vectorizer.fit_transform(texts)
        n_features = tfidf.shape[1]
        n_docs = tfidf.shape[0]
        self.mode = "tfidf"
        self.svd: TruncatedSVD | None = None
        if n_features > 2 and n_docs > 2:
            n_components = min(dim, n_features - 1, n_docs - 1)
            if n_components >= 2:
                self.svd = TruncatedSVD(n_components=n_components, random_state=13)
                emb = self.svd.fit_transform(tfidf)
                self.doc_emb = normalize(emb)
                self.mode = f"svd{n_components}"
            else:
                self.doc_emb = normalize(tfidf).tocsr()
        else:
            self.doc_emb = normalize(tfidf).tocsr()

    def _encode_query(self, query: str):
        q = self.vectorizer.transform([query])
        if self.svd is not None:
            return normalize(self.svd.transform(q))
        return normalize(q).tocsr()

    def search(self, query: str, top_k: int) -> SearchResult:
        start = time.perf_counter()
        q = self._encode_query(query)
        scores = np.asarray(q @ self.doc_emb.T).ravel()
        ranking_idx = top_indices(scores, top_k)
        ranking = [self.doc_ids[i] for i in ranking_idx if scores[i] > 0]
        score_map = {self.doc_ids[i]: float(scores[i]) for i in ranking_idx if scores[i] > 0}
        return SearchResult(ranking, score_map, (time.perf_counter() - start) * 1000)


class TransformerDenseIndex:
    def __init__(
        self,
        doc_ids: list[str],
        texts: list[str],
        model_name: str,
        batch_size: int = 32,
        max_length: int = 256,
        cache_dir: Path | None = None,
        query_prefix: str = "",
        doc_prefix: str = "",
        pooling: str = "mean",
    ):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "TransformerDenseIndex requires torch and transformers. "
                "Use --dense-backend svd or install the missing packages."
            ) from exc

        self.doc_ids = doc_ids
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.pooling = pooling
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        cache_path = self._cache_path(cache_dir, texts)
        if cache_path is not None and cache_path.exists():
            self.doc_emb = np.load(cache_path)
        else:
            self.doc_emb = self._encode([self.doc_prefix + text for text in texts])
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, self.doc_emb)

    def _cache_path(self, cache_dir: Path | None, texts: list[str]) -> Path | None:
        if cache_dir is None:
            return None
        import hashlib

        h = hashlib.sha1()
        h.update(self.model_name.encode("utf-8"))
        h.update(str(self.max_length).encode("utf-8"))
        h.update(self.doc_prefix.encode("utf-8"))
        h.update(self.pooling.encode("utf-8"))
        for docid, text in zip(self.doc_ids, texts, strict=True):
            h.update(docid.encode("utf-8"))
            h.update(b"\0")
            h.update(text[:256].encode("utf-8", errors="ignore"))
            h.update(b"\0")
        return cache_dir / f"docemb_{h.hexdigest()[:16]}.npy"

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        with self.torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(**encoded)
                token_embeddings = outputs.last_hidden_state
                if self.pooling == "cls":
                    pooled = token_embeddings[:, 0]
                else:
                    mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                    summed = (token_embeddings * mask).sum(dim=1)
                    counts = mask.sum(dim=1).clamp(min=1e-9)
                    pooled = summed / counts
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(pooled.cpu().numpy().astype(np.float32))
        if not vectors:
            return np.zeros((0, 1), dtype=np.float32)
        return np.vstack(vectors)

    def search(self, query: str, top_k: int) -> SearchResult:
        start = time.perf_counter()
        q = self._encode([self.query_prefix + query])
        scores = np.asarray(q @ self.doc_emb.T).ravel()
        ranking_idx = top_indices(scores, top_k)
        ranking = [self.doc_ids[i] for i in ranking_idx]
        score_map = {self.doc_ids[i]: float(scores[i]) for i in ranking_idx}
        return SearchResult(ranking, score_map, (time.perf_counter() - start) * 1000)


def top_indices(scores: np.ndarray, top_k: int) -> list[int]:
    if scores.size == 0 or top_k <= 0:
        return []
    top_k = min(top_k, scores.size)
    idx = np.argpartition(-scores, top_k - 1)[:top_k]
    idx = idx[np.argsort(-scores[idx])]
    return idx.tolist()


def rrf_fusion(results: list[SearchResult], top_k: int, k: int = 60) -> SearchResult:
    start = time.perf_counter()
    fused: dict[str, float] = {}
    latency = 0.0
    for result in results:
        latency += result.latency_ms
        for rank, docid in enumerate(result.ranking, start=1):
            fused[docid] = fused.get(docid, 0.0) + 1.0 / (k + rank)
    ranking = [docid for docid, _ in sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]]
    return SearchResult(ranking, fused, latency + (time.perf_counter() - start) * 1000)


def weighted_fusion(
    bm25: SearchResult,
    dense: SearchResult,
    top_k: int,
    dense_weight: float = 0.5,
) -> SearchResult:
    start = time.perf_counter()
    bm25_norm = normalize_scores(bm25.scores)
    dense_norm = normalize_scores(dense.scores)
    docs = set(bm25_norm) | set(dense_norm)
    scores = {
        docid: (1.0 - dense_weight) * bm25_norm.get(docid, 0.0) + dense_weight * dense_norm.get(docid, 0.0)
        for docid in docs
    }
    ranking = [docid for docid, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]]
    latency = bm25.latency_ms + dense.latency_ms + (time.perf_counter() - start) * 1000
    return SearchResult(ranking, scores, latency)


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = np.asarray(list(scores.values()), dtype=np.float64)
    lo = float(vals.min())
    hi = float(vals.max())
    if math.isclose(hi, lo):
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def score_entropy(scores: dict[str, float]) -> float:
    if not scores:
        return 0.0
    vals = np.asarray([max(0.0, v) for v in scores.values()], dtype=np.float64)
    total = float(vals.sum())
    if total <= 0:
        return 0.0
    p = vals / total
    return float(-(p * np.log(p + 1e-12)).sum())
