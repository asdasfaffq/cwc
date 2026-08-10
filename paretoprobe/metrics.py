from __future__ import annotations

import math
from collections.abc import Sequence


def dcg(relevances: Sequence[int]) -> float:
    return sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def ndcg_at_k(ranking: Sequence[str], rels: dict[str, int], k: int) -> float:
    gains = [rels.get(docid, 0) for docid in ranking[:k]]
    ideal = sorted(rels.values(), reverse=True)[:k]
    denom = dcg(ideal)
    if denom <= 0:
        return 0.0
    return dcg(gains) / denom


def recall_at_k(ranking: Sequence[str], rels: dict[str, int], k: int) -> float:
    if not rels:
        return 0.0
    retrieved = set(ranking[:k])
    relevant = {docid for docid, rel in rels.items() if rel > 0}
    return len(retrieved & relevant) / max(1, len(relevant))


def mrr_at_k(ranking: Sequence[str], rels: dict[str, int], k: int) -> float:
    for idx, docid in enumerate(ranking[:k], start=1):
        if rels.get(docid, 0) > 0:
            return 1.0 / idx
    return 0.0


def aggregate(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

