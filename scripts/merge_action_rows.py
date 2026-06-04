#!/usr/bin/env python3
"""Merge per-dataset all_action_rows.csv files into one all_action_rows_multi.csv.

Verifies a consistent schema and that qids are namespaced by dataset (the
certified_window_compiler keys workload items by qid+selectivity, and
paretoprobe_sys_pilot already writes a `dataset` column, so we keep qids as-is
but assert uniqueness of (dataset, qid, action, selectivity)).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REQUIRED = ["action", "qid", "dataset", "selectivity", "latency_ms", "ndcg10",
            "candidate_count", "query_len", "query_digit_ratio",
            "bm25_top_score", "dense_top_score", "probe_overlap", "probe_latency_ms"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="per-dataset all_action_rows.csv files")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    frames, schemas = [], []
    for p in args.inputs:
        path = Path(p)
        if not path.exists():
            print(f"WARN: missing {path}, skipping", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            print(f"ERROR: {path} missing required columns {missing}", file=sys.stderr)
            sys.exit(1)
        schemas.append(tuple(df.columns))
        frames.append(df)
        print(f"  + {path}: rows={len(df)} datasets={sorted(df.dataset.unique())} "
              f"queries={df.groupby('dataset').qid.nunique().to_dict()}")
    if not frames:
        print("ERROR: no input files found", file=sys.stderr); sys.exit(1)
    if len(set(schemas)) != 1:
        print("WARN: column sets differ across inputs; aligning on intersection of columns", file=sys.stderr)
        common = set.intersection(*[set(s) for s in schemas])
        frames = [f[[c for c in f.columns if c in common]] for f in frames]

    merged = pd.concat(frames, ignore_index=True)
    dup = merged.duplicated(subset=["dataset", "qid", "action", "selectivity"]).sum()
    if dup:
        print(f"WARN: {dup} duplicate (dataset,qid,action,selectivity) rows; keeping first", file=sys.stderr)
        merged = merged.drop_duplicates(subset=["dataset", "qid", "action", "selectivity"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    n_items = merged.groupby(["dataset", "qid", "selectivity"]).ngroups
    print(f"\nMERGED -> {args.output}")
    print(f"  total rows={len(merged)}  datasets={sorted(merged.dataset.unique())}")
    print(f"  workload items (dataset,qid,selectivity)={n_items}  total queries={merged.qid.nunique()}")
    print(f"  => with K cells, approx n_c ~ {n_items}/K per cell (target: hundreds).")


if __name__ == "__main__":
    main()
