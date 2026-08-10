#!/usr/bin/env python3
"""E2 (rebuttal): end-to-end serving experiment on a real hybrid-search stack.

Answers R1-W2 ("integration into a real system, end-to-end latency and/or throughput"),
R2-D4 ("actually execute the selected plans and measure total serving latency, rather
than relying only on cached or imported action rows") and R3-D2 ("how does plan choice
translate into overall gains in throughput or latency").

Nothing here is replayed from cached action rows: every reported latency is wall-clock
time of a real execution against real indexes.

Stack (real engines):
  lexical   : Okapi BM25 over the full BEIR collection, evaluated as a sparse
              term-weight matrix (the same scoring an inverted index performs; this
              replaces rank_bm25's per-term dense Python loop, which would have
              inflated every lexical measurement by an implementation artifact)
  vector ANN: hnswlib HNSW over intfloat/e5-base-v2 embeddings (GPU-encoded)
  fusion    : reciprocal-rank fusion
  rerank    : cross-encoder/ms-marco-MiniLM-L-6-v2 at depth 20 / 50
  filtering : metadata predicate at selectivity s, placed PRE (constrained ANN traversal
              via hnswlib's filter callback / masked lexical scoring) or POST (retrieve
              then drop) -- the placement axis the paper's plan catalog models.

Phases
  A  measure : execute every plan on every workload item once; record wall-clock latency
               and nDCG@10. Only TRAIN-window rows are ever shown to the compiler.
  B  replay  : closed-loop live replay of the EVAL window. Each policy executes for real;
               CWC pays its telemetry probe + routing inside the measured path.
  C  load    : open-loop Poisson arrivals into a single-server queue, executed live, to
               measure response-time tails and sustainable throughput under load.
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hnswlib  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from paretoprobe.data import load_beir_dataset  # noqa: E402
from paretoprobe.metrics import ndcg_at_k  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TOPK = 100
PLANS = ["bm25_pre", "bm25_post", "dense_pre", "dense_post",
         "rrf_pre", "rrf_post", "rerank20_pre", "rerank50_pre"]


def now() -> float:
    return time.perf_counter()


class SparseBM25:
    """Okapi BM25 (k1=1.5, b=0.75) stored as a document-term weight matrix.

    Scoring a query is a sum of the query terms' posting columns -- algorithmically the
    same work an inverted index does, without the interpreted per-term loop of rank_bm25.
    """

    def __init__(self, tokenized: list[list[str]], k1: float = 1.5, b: float = 0.75):
        vocab: dict[str, int] = {}
        rows, cols, tfs = [], [], []
        dl = np.zeros(len(tokenized), dtype=np.float64)
        for d, toks in enumerate(tokenized):
            dl[d] = len(toks)
            counts: dict[int, int] = {}
            for w in toks:
                t = vocab.setdefault(w, len(vocab))
                counts[t] = counts.get(t, 0) + 1
            for t, c in counts.items():
                rows.append(d); cols.append(t); tfs.append(c)
        n, V = len(tokenized), len(vocab)
        tf = sp.csr_matrix((np.asarray(tfs, dtype=np.float64), (rows, cols)), shape=(n, V))
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
        avgdl = dl.mean()
        norm = k1 * (1.0 - b + b * dl / avgdl)
        tf = tf.tocoo()
        w = (tf.data * (k1 + 1.0)) / (tf.data + norm[tf.row]) * idf[tf.col]
        self.W = sp.csc_matrix((w, (tf.row, tf.col)), shape=(n, V))
        self.vocab = vocab
        self.n = n

    def scores(self, query: str) -> np.ndarray:
        cols = [self.vocab[w] for w in query.lower().split() if w in self.vocab]
        if not cols:
            return np.zeros(self.n, dtype=np.float64)
        return np.asarray(self.W[:, cols].sum(axis=1)).ravel()


def mean_embed(texts, tok, model, bs=128, prefix=""):
    out = []
    for i in range(0, len(texts), bs):
        b = [prefix + t for t in texts[i:i + bs]]
        enc = tok(b, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEV)
        with torch.no_grad():
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            v = torch.nn.functional.normalize(v, dim=-1)
        out.append(v.cpu().numpy())
    return np.vstack(out).astype(np.float32)


class Stack:
    """Real hybrid-search engine with a metadata predicate and pre/post filter placement."""

    def __init__(self, dataset: str, max_queries: int, seed: int, corpus_datasets=None):
        ds = load_beir_dataset(dataset, ROOT / "data", split="test", allow_download=False)
        qrels = ds.qrels
        qids = [q for q in ds.queries if qrels.get(q)]
        rng = random.Random(seed)
        rng.shuffle(qids)
        qids = sorted(qids[:max_queries])
        self.qids = qids
        self.queries = [ds.queries[q] for q in qids]
        self.qrels = {q: qrels.get(q, {}) for q in qids}
        # The searched collection may be a union of BEIR corpora: the query set's own
        # documents plus distractors from other collections. A larger index is what makes
        # constrained (pre-filter) ANN traversal genuinely expensive, i.e. what makes the
        # filter-placement decision a real one rather than a toy.
        corpus: dict[str, str] = {f"{dataset}::{k}": str(v).strip() for k, v in ds.corpus.items()}
        self.qrels = {q: {f"{dataset}::{d}": r for d, r in rels.items()}
                      for q, rels in self.qrels.items()}
        for extra in (corpus_datasets or []):
            if extra == dataset:
                continue
            ex = load_beir_dataset(extra, ROOT / "data", split="test", allow_download=False)
            for k, v in ex.corpus.items():
                corpus[f"{extra}::{k}"] = str(v).strip()
        self.doc_ids = list(corpus.keys())
        self.docs = [corpus[d] for d in self.doc_ids]
        n = len(self.docs)
        print(f"[stack] queries={dataset} ({len(qids)}), collection={n} docs "
              f"(+{sorted(set(corpus_datasets or []) - {dataset})}), device={DEV}", flush=True)

        tok = AutoTokenizer.from_pretrained("intfloat/e5-base-v2")
        enc = AutoModel.from_pretrained("intfloat/e5-base-v2").to(DEV).eval()
        t = now(); doc_emb = mean_embed(self.docs, tok, enc, prefix="passage: ")
        self.t_embed = now() - t
        self.q_emb = mean_embed(self.queries, tok, enc, prefix="query: ")
        del enc
        torch.cuda.empty_cache()

        t = now()
        self.index = hnswlib.Index(space="cosine", dim=doc_emb.shape[1])
        self.index.init_index(max_elements=n, ef_construction=200, M=16)
        self.index.add_items(doc_emb, np.arange(n))
        self.index.set_ef(96)
        self.index.set_num_threads(1)
        self.t_hnsw = now() - t

        t = now()
        self.bm25 = SparseBM25([d.lower().split() for d in self.docs])
        self.t_bm25 = now() - t
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=DEV, max_length=256)

        # Deterministic metadata predicate: doc keeps attribute value in [0,1).
        rs = np.random.default_rng(20260810)
        self.attr = rs.random(n)
        self.masks: dict[float, np.ndarray] = {}

    def add_selectivities(self, sels) -> None:
        for s in sels:
            self.masks[float(s)] = (self.attr < float(s)) if s < 1.0 else np.ones_like(self.attr, bool)

    # ---- primitive operators -------------------------------------------------
    def _bm25_scores(self, i: int) -> np.ndarray:
        return self.bm25.scores(self.queries[i])

    def _bm25_rank(self, i: int, s: float, pre: bool, scores=None, k=TOPK) -> np.ndarray:
        sc = self._bm25_scores(i) if scores is None else scores
        if pre:
            sc = np.where(self.masks[s], sc, -np.inf)
            return np.argsort(-sc)[:k]
        cand = np.argsort(-sc)[:k]
        return cand[self.masks[s][cand]]

    def _dense_rank(self, i: int, s: float, pre: bool, k=TOPK, cached=None) -> np.ndarray:
        # `cached` is the probe's UNFILTERED candidate list: reusable by the post-filter
        # arm (which is exactly "retrieve then drop"), never by the constrained-traversal
        # pre-filter arm, whose whole point is to search inside the predicate.
        if not pre and cached is not None and len(cached) >= k:
            return cached[:k][self.masks[s][cached[:k]]]
        if pre and s < 1.0:
            # Constrained ANN traversal. A fixed ef collapses recall once the predicate is
            # selective, so -- as production filtered-ANN implementations do -- we widen the
            # search list in proportion to 1/s. This is what makes pre-filtering the
            # high-quality/high-cost arm of the placement decision.
            mask = self.masks[s]
            n_allowed = int(mask.sum())
            k_eff = int(min(k, n_allowed))
            if k_eff <= 0:
                return np.empty(0, dtype=np.int64)
            self.index.set_ef(int(min(4000, max(k_eff, 96, 96.0 / s))))
            lbl, _ = self.index.knn_query(self.q_emb[i:i + 1], k=k_eff,
                                          filter=lambda idx: bool(mask[idx]))
            self.index.set_ef(96)
            return lbl[0]
        lbl, _ = self.index.knn_query(self.q_emb[i:i + 1], k=k)
        cand = lbl[0]
        return cand if pre else cand[self.masks[s][cand]]

    @staticmethod
    def _rrf(a: np.ndarray, b: np.ndarray, k=TOPK) -> np.ndarray:
        rr: dict[int, float] = {}
        for r, d in enumerate(a):
            rr[int(d)] = rr.get(int(d), 0.0) + 1.0 / (60 + r)
        for r, d in enumerate(b):
            rr[int(d)] = rr.get(int(d), 0.0) + 1.0 / (60 + r)
        return np.array(sorted(rr, key=rr.get, reverse=True)[:k], dtype=np.int64)

    def _rerank(self, i: int, cand: np.ndarray, depth: int) -> np.ndarray:
        cand = cand[:depth]
        if len(cand) == 0:
            return cand
        pairs = [[self.queries[i], self.docs[c]] for c in cand]
        sc = self.reranker.predict(pairs, batch_size=len(pairs), show_progress_bar=False)
        return cand[np.argsort(-np.asarray(sc))]

    # ---- physical plans ------------------------------------------------------
    def execute(self, plan: str, i: int, s: float, probe: dict | None = None) -> np.ndarray:
        """Run one physical plan. `probe` lets a routed plan reuse telemetry artifacts,
        exactly as a real optimizer would; baselines are called with probe=None."""
        bs = probe.get("bm25_scores") if probe else None
        dc = probe.get("dense_unfiltered") if probe else None
        if plan == "bm25_pre":
            return self._bm25_rank(i, s, True, bs)
        if plan == "bm25_post":
            return self._bm25_rank(i, s, False, bs)
        if plan == "dense_pre":
            return self._dense_rank(i, s, True)
        if plan == "dense_post":
            return self._dense_rank(i, s, False, cached=dc)
        if plan in ("rrf_pre", "rrf_post"):
            pre = plan.endswith("_pre")
            return self._rrf(self._bm25_rank(i, s, pre, bs),
                             self._dense_rank(i, s, pre, cached=None if pre else dc))
        if plan.startswith("rerank"):
            depth = int(plan.replace("rerank", "").split("_")[0])
            return self._rerank(i, self._dense_rank(i, s, True), depth)
        raise ValueError(plan)

    def ndcg(self, i: int, ranking: np.ndarray) -> float:
        rels = self.qrels[self.qids[i]]
        return ndcg_at_k([self.doc_ids[int(d)] for d in ranking], rels, 10)

    # ---- telemetry probe (what CWC reads before routing) ---------------------
    def probe(self, i: int, s: float) -> tuple[np.ndarray, dict]:
        """Telemetry probe. Its two retrieval artifacts (lexical scores and the
        pre-filtered dense candidate list) are handed to the routed plan, so the probe is
        shared work rather than pure overhead -- what a real optimizer does."""
        bs = self._bm25_scores(i)
        top = np.sort(bs)[-50:][::-1]
        p = top / (top.sum() + 1e-9)
        lbl, dist = self.index.knn_query(self.q_emb[i:i + 1], k=TOPK)
        dsc = 1.0 - dist[0][:50]
        bl = np.argsort(-bs)[:50]
        overlap = len(set(lbl[0][:50].tolist()) & set(bl.tolist())) / 50.0
        feat = np.array([
            len(self.queries[i].split()),
            sum(ch.isdigit() for ch in self.queries[i]) / max(1, len(self.queries[i])),
            s, float(self.masks[s].sum()),
            float(top[0]), float(-(p * np.log(p + 1e-12)).sum()), float(top[0] - top[1]),
            float(dsc[0]), float(-(np.abs(dsc) / (np.abs(dsc).sum() + 1e-9) *
                                   np.log(np.abs(dsc) / (np.abs(dsc).sum() + 1e-9) + 1e-12)).sum()),
            float(dsc[0] - dsc[1]), overlap,
        ], dtype=np.float64)
        return feat, {"bm25_scores": bs, "dense_unfiltered": lbl[0]}


def constrained_score(nd: float, lat: float, budget: float) -> float:
    if lat <= budget:
        return float(nd)
    return -0.15 * max(0.0, lat / max(budget, 1e-9) - 1.0)


def compile_cwc(train_items, eval_items, feats, lat, nd, budget, n_cells, seed, delta=0.1):
    """Repaired CWC compile: selection-split binding + selection-split abstain (Prop. 2),
    calibration split certifies only the already-fixed deployed plan."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_items))
    half = len(train_items) // 2
    sel = [train_items[i] for i in perm[:half]]
    cal = [train_items[i] for i in perm[half:]]

    fit = sel + eval_items
    scaler = StandardScaler().fit(np.vstack([feats[i] for i in fit]))
    km = KMeans(n_clusters=n_cells, n_init=20, random_state=seed).fit(
        scaler.transform(np.vstack([feats[i] for i in fit])))

    def cell(it):
        return int(km.predict(scaler.transform(feats[it].reshape(1, -1)))[0])

    sel_c = {it: cell(it) for it in sel}
    cal_c = {it: cell(it) for it in cal}

    plan_by_cell, sviol = {}, {}
    for c in range(n_cells):
        ci = [it for it in sel if sel_c[it] == c]
        if not ci:
            plan_by_cell[c] = "bm25_pre"
            continue
        best, bv = "bm25_pre", -1e9
        for a in PLANS:
            v = float(np.mean([constrained_score(nd[(it, a)], lat[(it, a)], budget) for it in ci]))
            if v > bv:
                bv, best = v, a
        plan_by_cell[c] = best
    for a in PLANS:
        sviol[a] = float(np.mean([1.0 if lat[(it, a)] > budget else 0.0 for it in sel]))
    feasible = [a for a in PLANS if sviol[a] <= 1e-9]
    fallback = (max(feasible, key=lambda a: float(np.mean([nd[(it, a)] for it in sel])))
                if feasible else min(PLANS, key=lambda a: sviol[a]))

    # abstain on the selection split only
    deploy = {}
    for c in range(n_cells):
        ci = [it for it in sel if sel_c[it] == c]
        if not ci:
            deploy[c] = fallback
            continue
        vp = float(np.mean([1.0 if lat[(it, plan_by_cell[c])] > budget else 0.0 for it in ci]))
        vf = float(np.mean([1.0 if lat[(it, fallback)] > budget else 0.0 for it in ci]))
        deploy[c] = fallback if vf < vp else plan_by_cell[c]

    # certificate on the calibration split (deployed plan already fixed)
    Lhat, ncal = {}, {}
    for c in range(n_cells):
        ci = [it for it in cal if cal_c[it] == c]
        ncal[c] = len(ci)
        Lhat[c] = (float(np.mean([1.0 if lat[(it, deploy[c])] > budget else 0.0 for it in ci]))
                   if ci else 1.0)
    return {"cell": cell, "deploy": deploy, "fallback": fallback, "Lhat": Lhat, "ncal": ncal,
            "plan_by_cell": plan_by_cell, "n_cells": n_cells}


def pctl(v, q):
    return float(np.percentile(np.asarray(v, dtype=np.float64), q)) if len(v) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fiqa")
    ap.add_argument("--max-queries", type=int, default=400)
    ap.add_argument("--corpus-datasets", default="",
                    help="comma-separated BEIR collections to union into the searched index")
    ap.add_argument("--selectivities", default="1.0,0.5,0.1,0.02")
    ap.add_argument("--n-cells", type=int, default=4)
    ap.add_argument("--seeds", default="13,17,23")
    ap.add_argument("--load-seed", type=int, default=13)
    ap.add_argument("--load-requests", type=int, default=300)
    ap.add_argument("--load-rhos", default="0.6,0.8,1.0")
    ap.add_argument("--n-budgets", type=int, default=5)
    ap.add_argument("--skip-closed-loop", action="store_true")
    ap.add_argument("--load-budget-index", type=int, default=1,
                    help="which SLO of the grid the open-loop test runs at; the "
                         "load test is only informative at an SLO where routing "
                         "actually changes the plan mix")
    ap.add_argument("--budgets-ms", default="",
                    help="explicit SLO grid in ms; overrides the quantile rule so a "
                         "rerun uses the same grid even though measured latencies "
                         "move slightly between runs")
    ap.add_argument("--output-dir", type=Path, default=Path("rebuttal/results/r2_end2end"))
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    sels = [float(x) for x in args.selectivities.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    st = Stack(args.dataset, args.max_queries, seed=13,
               corpus_datasets=[x for x in args.corpus_datasets.split(",") if x.strip()])
    st.add_selectivities(sels)
    items = [(i, s) for i in range(len(st.qids)) for s in sels]

    # ---------------- Phase A: measure every plan on every item ----------------
    print("[A] measuring plan latency/quality (real execution)", flush=True)
    for w in range(3):  # warm up kernels/caches
        for a in PLANS:
            st.execute(a, w, 1.0)
        st.probe(w, 1.0)
    lat, nd, feats, probe_ms = {}, {}, {}, []
    t0 = now()
    for n_done, (i, s) in enumerate(items):
        t = now(); f, _ = st.probe(i, s); probe_ms.append((now() - t) * 1e3)
        feats[(i, s)] = f
        for a in PLANS:
            t = now(); r = st.execute(a, i, s); dt = (now() - t) * 1e3
            lat[((i, s), a)] = dt
            nd[((i, s), a)] = st.ndcg(i, r)
        if (n_done + 1) % 100 == 0:
            print(f"    {n_done+1}/{len(items)} items ({now()-t0:.0f}s)", flush=True)
    feats = {k: v for k, v in feats.items()}
    plan_lat = {a: float(np.mean([lat[(it, a)] for it in items])) for a in PLANS}
    plan_nd = {a: float(np.mean([nd[(it, a)] for it in items])) for a in PLANS}
    print("[A] mean plan latency (ms):", json.dumps({k: round(v, 2) for k, v in plan_lat.items()}))
    print("[A] mean plan nDCG@10   :", json.dumps({k: round(v, 4) for k, v in plan_nd.items()}))
    profile = {}
    for s in sels:
        sub = [it for it in items if it[1] == s]
        profile[str(s)] = {a: {"lat_ms": round(float(np.mean([lat[(it, a)] for it in sub])), 3),
                               "ndcg10": round(float(np.mean([nd[(it, a)] for it in sub])), 4)}
                           for a in PLANS}
        print(f"[A] s={s}: " + "  ".join(
            f"{a}={profile[str(s)][a]['lat_ms']:.2f}ms/{profile[str(s)][a]['ndcg10']:.3f}"
            for a in PLANS), flush=True)

    # SLO grid fixed a priori, with no reference to any policy's outcome: log-spaced
    # between the 20th and 95th percentile of the pooled measured plan latency, floored at
    # twice the p95 telemetry-probe cost (an SLO below the router's own cost is not a
    # meaningful operating point for *any* routing method, so it is out of scope rather
    # than a result).
    pooled = np.array([lat[(it, a)] for it in items for a in PLANS])
    if args.budgets_ms.strip():
        budgets = [float(x) for x in args.budgets_ms.split(",") if x.strip()]
    else:
        lo = max(float(np.percentile(pooled, 20)), 2.0 * pctl(probe_ms, 95))
        hi = float(np.percentile(pooled, 95))
        budgets = [round(float(x), 2) for x in np.geomspace(lo, hi, args.n_budgets)]
    print(f"[A] probe p95={pctl(probe_ms, 95):.2f} ms -> SLO grid {budgets} ms", flush=True)

    # ---------------- Phase B: closed-loop live replay of the eval window -------
    p_train = {1.0: 0.8, 0.5: 0.5, 0.1: 0.35, 0.02: 0.2}  # cheap strata skew train, hard strata skew eval
    rows, cert_rows = [], []
    for budget in ([] if args.skip_closed_loop else budgets):
        for seed in seeds:
            rng = np.random.default_rng(seed)
            train, ev = [], []
            for it in items:
                (train if rng.random() < p_train[it[1]] else ev).append(it)
            comp = compile_cwc(train, ev, feats, lat, nd, budget, args.n_cells, seed)

            # baselines chosen on the train window only
            staticbest = max(PLANS, key=lambda a: float(np.mean(
                [constrained_score(nd[(it, a)], lat[(it, a)], budget) for it in train])))
            p95 = {a: pctl([lat[(it, a)] for it in train], 95) for a in PLANS}
            feas = [a for a in PLANS if p95[a] <= budget] or PLANS
            costgreedy = max(feas, key=lambda a: float(np.mean([nd[(it, a)] for it in train])))
            best_quality = max(PLANS, key=lambda a: float(np.mean([nd[(it, a)] for it in train])))

            policies = {
                "CWC": None,
                "StaticBest-cal": staticbest,
                "CostGreedy-cal": costgreedy,
                "Static-BestQuality": best_quality,
                "Static-Fallback": comp["fallback"],
                "Oracle-perItem": "__oracle__",
            }
            for pname, fixed in policies.items():
                lats, nds, plans_used, probe_t, route_t = [], [], [], [], []
                t_start = now()
                for (i, s) in ev:
                    t = now()
                    if fixed == "__oracle__":
                        # not deployable: uses this item's measured outcomes for every plan
                        a = max(PLANS, key=lambda x: constrained_score(
                            nd[((i, s), x)], lat[((i, s), x)], budget))
                        r = st.execute(a, i, s)
                    elif fixed is None:
                        f, pr = st.probe(i, s)
                        t_p = now()
                        a = comp["deploy"][comp["cell"]((i, s))]
                        t_r = now()
                        r = st.execute(a, i, s, probe=pr)
                        probe_t.append((t_p - t) * 1e3); route_t.append((t_r - t_p) * 1e3)
                    else:
                        a = fixed
                        r = st.execute(a, i, s)
                    e2e = (now() - t) * 1e3
                    lats.append(e2e); nds.append(st.ndcg(i, r)); plans_used.append(a)
                wall = now() - t_start
                rows.append({
                    "phase": "closed_loop", "dataset": args.dataset, "budget_ms": budget,
                    "seed": seed, "policy": pname, "n_eval": len(ev),
                    "mean_ms": float(np.mean(lats)), "p50_ms": pctl(lats, 50),
                    "p95_ms": pctl(lats, 95), "p99_ms": pctl(lats, 99),
                    "slo_violation": float(np.mean([1.0 if x > budget else 0.0 for x in lats])),
                    "ndcg10": float(np.mean(nds)), "throughput_qps": len(ev) / wall,
                    "distinct_plans": len(set(plans_used)),
                    "probe_ms": float(np.mean(probe_t)) if probe_t else 0.0,
                    "route_ms": float(np.mean(route_t)) if route_t else 0.0,
                })
                print(f"  [B] b={budget} seed={seed} {pname:20s} "
                      f"p95={rows[-1]['p95_ms']:7.1f} viol={rows[-1]['slo_violation']:.3f} "
                      f"nDCG={rows[-1]['ndcg10']:.4f} qps={rows[-1]['throughput_qps']:.1f}", flush=True)
            cert_rows.append({"budget_ms": budget, "seed": seed,
                              "fallback": comp["fallback"],
                              "deployed": "|".join(comp["deploy"][c] for c in range(args.n_cells)),
                              "n_cal_min": min(comp["ncal"].values())})

    import pandas as pd
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out / "closed_loop_raw.csv", index=False)
    agg = (df.groupby(["budget_ms", "policy"]).agg(
        mean_ms=("mean_ms", "mean"), p95_ms=("p95_ms", "mean"), p99_ms=("p99_ms", "mean"),
        slo_violation=("slo_violation", "mean"), ndcg10=("ndcg10", "mean"),
        throughput_qps=("throughput_qps", "mean")).reset_index()
           if not df.empty else pd.DataFrame())
    if not agg.empty:
        agg.to_csv(out / "closed_loop_aggregate.csv", index=False)
        print(agg.to_string(index=False), flush=True)

    # ---------------- Phase C: open-loop load test ------------------------------
    print("[C] open-loop Poisson load test (real execution behind a single-server queue)",
          flush=True)
    budget = budgets[args.load_budget_index]
    seed = args.load_seed
    rng = np.random.default_rng(seed)
    train, ev = [], []
    for it in items:
        (train if rng.random() < p_train[it[1]] else ev).append(it)
    comp = compile_cwc(train, ev, feats, lat, nd, budget, args.n_cells, seed)
    staticbest = max(PLANS, key=lambda a: float(np.mean(
        [constrained_score(nd[(it, a)], lat[(it, a)], budget) for it in train])))
    p95t = {a: pctl([lat[(it, a)] for it in train], 95) for a in PLANS}
    feas = [a for a in PLANS if p95t[a] <= budget] or PLANS
    costgreedy = max(feas, key=lambda a: float(np.mean([nd[(it, a)] for it in train])))
    best_quality = max(PLANS, key=lambda a: float(np.mean([nd[(it, a)] for it in train])))
    load_policies = {"CWC": None, "StaticBest-cal": staticbest,
                     "CostGreedy-cal": costgreedy, "Static-BestQuality": best_quality}

    # service-rate capacity per policy, from the measured closed-loop mean service time
    svc = {}
    for pname, fixed in load_policies.items():
        s_ms = []
        for (i, s) in ev[:120]:
            t = now()
            if fixed is None:
                f, pr = st.probe(i, s)
                a = comp["deploy"][comp["cell"]((i, s))]
                st.execute(a, i, s, probe=pr)
            else:
                st.execute(fixed, i, s)
            s_ms.append((now() - t) * 1e3)
        svc[pname] = float(np.mean(s_ms))
    # Reference arrival rate: the median policy capacity. Anchoring on the slowest policy
    # leaves every other policy idle and discriminates nothing; anchoring on CWC's own
    # capacity would build in a bias. Every policy's own capacity is reported alongside.
    caps = sorted(1000.0 / v for v in svc.values())
    mu_min = float(np.median(caps))
    print("[C] mean service ms:", {k: round(v, 2) for k, v in svc.items()},
          "-> capacities " + str(sorted(round(1000.0 / v, 1) for v in svc.values()))
          + f" qps, reference (median) {mu_min:.1f} qps", flush=True)

    load_rows = []
    order = list(rng.permutation(len(ev)))
    req = [ev[order[i % len(ev)]] for i in range(args.load_requests)]
    for rho in [float(x) for x in args.load_rhos.split(",")]:
        lam = rho * mu_min
        gaps = np.random.default_rng(seed).exponential(1.0 / lam, size=len(req))
        for pname, fixed in load_policies.items():
            q: "queue.Queue" = queue.Queue()
            done = []

            def producer():
                # Sleep once to each absolute arrival instant. A spin-wait here contends with
                # the server thread for the GIL and inflates every measured response time --
                # a harness artifact rather than queueing.
                t0 = now()
                acc = 0.0
                for gidx, item in enumerate(req):
                    acc += gaps[gidx]
                    delay = t0 + acc - now()
                    if delay > 0:
                        time.sleep(delay)
                    q.put((item, now()))
                q.put(None)

            def server():
                while True:
                    got = q.get()
                    if got is None:
                        break
                    (i, s), arr = got
                    if fixed is None:
                        f, pr = st.probe(i, s)
                        a = comp["deploy"][comp["cell"]((i, s))]
                        st.execute(a, i, s, probe=pr)
                    else:
                        st.execute(fixed, i, s)
                    done.append((now() - arr) * 1e3)

            th_p = threading.Thread(target=producer)
            th_s = threading.Thread(target=server)
            t0 = now(); th_p.start(); th_s.start(); th_p.join(); th_s.join(); wall = now() - t0
            load_rows.append({
                "phase": "open_loop", "dataset": args.dataset, "budget_ms": budget,
                "offered_load_rho": rho, "arrival_qps": lam, "policy": pname,
                "completed": len(done), "goodput_qps": len(done) / wall,
                "resp_p50_ms": pctl(done, 50), "resp_p95_ms": pctl(done, 95),
                "resp_p99_ms": pctl(done, 99),
                "resp_slo_violation": float(np.mean([1.0 if x > budget else 0.0 for x in done])),
            })
            print(f"  [C] rho={rho} {pname:20s} p95={load_rows[-1]['resp_p95_ms']:8.1f} "
                  f"p99={load_rows[-1]['resp_p99_ms']:8.1f} "
                  f"viol={load_rows[-1]['resp_slo_violation']:.3f} "
                  f"goodput={load_rows[-1]['goodput_qps']:.1f}", flush=True)

    pd.DataFrame(load_rows).to_csv(out / "open_loop_raw.csv", index=False)
    meta = {
        "dataset": args.dataset, "n_docs": len(st.docs), "n_queries": len(st.qids),
        "n_items": len(items), "device": DEV, "plans": PLANS, "budgets_ms": budgets,
        "index_build_s": {"doc_embed": round(st.t_embed, 1), "hnsw": round(st.t_hnsw, 1),
                          "bm25": round(st.t_bm25, 1)},
        "mean_plan_latency_ms": plan_lat, "mean_plan_ndcg10": plan_nd,
        "per_selectivity_profile": profile,
        "telemetry_probe_ms_mean": float(np.mean(probe_ms)),
        "telemetry_probe_ms_p95": pctl(probe_ms, 95),
        "mean_service_ms_open_loop": svc, "reference_capacity_qps": mu_min,
        "cwc_deployments": cert_rows,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("budgets_ms", "telemetry_probe_ms_mean",
                                           "reference_capacity_qps")}, indent=2))


if __name__ == "__main__":
    main()
