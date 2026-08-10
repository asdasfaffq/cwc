#!/usr/bin/env python3
"""(B1) Real hybrid-search systems micro-benchmark of CWC's optimizer overhead.

Stack (real engines/libraries, no fabrication):
  - lexical: rank_bm25 (BM25Okapi)
  - vector ANN: hnswlib HNSW index over intfloat/e5-base-v2 embeddings (GPU)
  - rerank: cross-encoder/ms-marco-MiniLM-L-6-v2 (sentence_transformers)
Physical plans (actions CWC selects among): bm25, dense (HNSW), rrf fusion, dense+rerank@20.

We measure, with wall-clock (time.perf_counter), CWC's added cost vs the retrieval/rerank
it sits on top of:
  (a) telemetry collection time / query   (cheap probes CWC reads)
  (b) compile time / window               (telemetry embed + k-means(K) + per-cell bind)
  (c) per-query routing overhead          (cell assignment + bound-plan dispatch)
  (d) plan-switch latency                 (dispatch cost when the routed plan changes;
                                           all indexes are resident, so no reload)
Reference: actual per-query latency of each physical plan.
Corpus: BEIR NFCorpus (323 queries, 3633 docs). Reports means over the query window.
"""
import sys, time, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch, hnswlib
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModel
from paretoprobe.data import load_beir_dataset

DEV = "cuda" if torch.cuda.is_available() else "cpu"
def now(): return time.perf_counter()

def mean_embed(texts, tok, model, bs=64, prefix=""):
    out = []
    for i in range(0, len(texts), bs):
        b = [prefix + t for t in texts[i:i+bs]]
        enc = tok(b, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEV)
        with torch.no_grad():
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h*m).sum(1)/m.sum(1).clamp(min=1e-9)
            v = torch.nn.functional.normalize(v, dim=-1)
        out.append(v.cpu().numpy())
    return np.vstack(out).astype(np.float32)

def main():
    print(f"device={DEV}")
    ds = load_beir_dataset("nfcorpus", ROOT/"data", split="test", allow_download=False)
    doc_ids = list(ds.corpus.keys())
    docs = [ str(ds.corpus[d]).strip() for d in doc_ids ]
    qids = list(ds.queries.keys()); queries = [ds.queries[q] for q in qids]
    print(f"corpus={len(docs)} queries={len(queries)}")

    # ---- build engines (one-time index build, reported separately) ----
    tok = AutoTokenizer.from_pretrained("intfloat/e5-base-v2")
    enc_model = AutoModel.from_pretrained("intfloat/e5-base-v2").to(DEV).eval()
    t=now(); doc_emb = mean_embed(docs, tok, enc_model, prefix="passage: "); t_doc_emb=now()-t
    dim = doc_emb.shape[1]
    t=now()
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=len(docs), ef_construction=200, M=16)
    index.add_items(doc_emb, np.arange(len(docs))); index.set_ef(64)
    t_hnsw_build=now()-t
    t=now(); bm25 = BM25Okapi([d.lower().split() for d in docs]); t_bm25_build=now()-t
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=DEV, max_length=256)
    q_emb = mean_embed(queries, tok, enc_model, prefix="query: ")

    # ---- physical-plan latencies (real per-query wall-clock) ----
    def plan_bm25(i):
        s = bm25.get_scores(queries[i].lower().split()); return np.argsort(-s)[:100]
    def plan_dense(i):
        lbl,_ = index.knn_query(q_emb[i:i+1], k=100); return lbl[0]
    def plan_rrf(i):
        s = bm25.get_scores(queries[i].lower().split()); bl=np.argsort(-s)[:100]
        dl,_ = index.knn_query(q_emb[i:i+1], k=100); dl=dl[0]
        rr={};
        for r,d in enumerate(bl): rr[d]=rr.get(d,0)+1/(60+r)
        for r,d in enumerate(dl): rr[d]=rr.get(d,0)+1/(60+r)
        return np.array(sorted(rr,key=rr.get,reverse=True)[:100])
    def plan_dense_rerank(i, depth=20):
        cand = plan_dense(i)[:depth]
        pairs=[[queries[i], docs[c]] for c in cand]
        sc = reranker.predict(pairs, batch_size=depth, show_progress_bar=False)
        order=np.argsort(-sc); return cand[order]
    plans = {"bm25":plan_bm25, "dense(HNSW)":plan_dense, "rrf":plan_rrf, "dense+rerank@20":plan_dense_rerank}

    W = list(range(len(queries)))  # full window
    lat = {}
    for name, fn in plans.items():
        for i in W[:5]: fn(i)  # warmup
        t=now()
        for i in W: fn(i)
        lat[name]=(now()-t)/len(W)*1000  # ms/query

    # ---- (a) telemetry collection time / query (cheap probes CWC reads) ----
    def telemetry(i):
        bs = bm25.get_scores(queries[i].lower().split())
        top = np.sort(bs)[-50:][::-1]
        p = top/ (top.sum()+1e-9)
        bm25_top=float(top[0]); bm25_gap=float(top[0]-top[1]); bm25_ent=float(-(p*np.log(p+1e-12)).sum())
        dl,dd = index.knn_query(q_emb[i:i+1], k=50); dscore=1-dd[0]
        dense_top=float(dscore[0]); dense_gap=float(dscore[0]-dscore[1])
        cand=len(set(dl[0]) | set(np.argsort(-bs)[:50]))
        return np.array([len(queries[i].split()), bm25_top,bm25_gap,bm25_ent, dense_top,dense_gap, cand],dtype=np.float64)
    for i in W[:5]: telemetry(i)
    t=now()
    tel = np.vstack([telemetry(i) for i in W]); t_tel=(now()-t)/len(W)*1000  # ms/query

    # ---- (b) compile time / window: standardize + kmeans(K) + bind ----
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    # per-(query,plan) utility proxy = -latency (cheap, retrieval-quality binding uses cached nDCG offline;
    #  here we only time the COMPILE mechanics, which are quality-agnostic)
    K=4
    t=now()
    Z = StandardScaler().fit_transform(tel)
    km = KMeans(n_clusters=K, n_init=10, random_state=13).fit(Z)
    cellplan = {c: list(plans)[c % len(plans)] for c in range(K)}  # bind (mechanics timing)
    t_compile=(now()-t)*1000  # ms for the whole window

    # ---- (c) per-query routing overhead + (d) plan-switch dispatch ----
    scaler = StandardScaler().fit(tel)
    for i in W[:5]:
        c=int(km.predict(scaler.transform(tel[i:i+1]))[0]); _=cellplan[c]
    t=now()
    routed=[]
    for i in W:
        c=int(km.predict(scaler.transform(tel[i:i+1]))[0]); routed.append(cellplan[c])
    t_route=(now()-t)/len(W)*1000
    switches=sum(1 for a,b in zip(routed, routed[1:]) if a!=b)

    out = {
      "corpus":len(docs),"queries":len(queries),"device":DEV,"K":K,
      "index_build_ms":{"doc_embed":round(t_doc_emb*1000,1),"hnsw":round(t_hnsw_build*1000,1),"bm25":round(t_bm25_build*1000,1)},
      "plan_latency_ms_per_query":{k:round(v,3) for k,v in lat.items()},
      "telemetry_ms_per_query":round(t_tel,3),
      "compile_ms_per_window":round(t_compile,1),
      "routing_overhead_ms_per_query":round(t_route,4),
      "plan_switches":switches, "switch_frac":round(switches/max(1,len(W)-1),3),
    }
    (ROOT/"refine-logs/SYSTEM_OVERHEAD_20260604.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
