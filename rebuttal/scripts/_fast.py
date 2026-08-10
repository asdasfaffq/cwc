"""Single-pass lookup tables for the rebuttal experiments.

The original analysis scripts resolved every (item, action) pair with a boolean mask over
the whole action-row frame, which is O(n_items * n_rows) per block and dominated runtime
on the merged multi-dataset workloads. These helpers build the same lookups in one pass.
Values are identical to the mask-based path (first matching row wins, as before).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.certified_window_compiler import TELEMETRY_FEATURES


def build_tables(df: pd.DataFrame):
    """Return (items, telemetry, lat, ndcg, selectivity, actions).

    telemetry : item -> np.ndarray of TELEMETRY_FEATURES (from the item's first row)
    lat/ndcg  : (item, action) -> float
    """
    wanted = ["item", "action", "latency_ms", "ndcg10", "selectivity"] + TELEMETRY_FEATURES
    cols, seen = [], set()
    for c in wanted:  # dedupe: `selectivity` is both a key and a telemetry feature
        if c in df.columns and c not in seen:
            cols.append(c); seen.add(c)
    sub = df[cols]
    lat: dict[tuple[str, str], float] = {}
    ndcg: dict[tuple[str, str], float] = {}
    telemetry: dict[str, np.ndarray] = {}
    selectivity: dict[str, float] = {}
    feat_idx = [sub.columns.get_loc(c) if c in sub.columns else None for c in TELEMETRY_FEATURES]
    ci = {c: sub.columns.get_loc(c) for c in sub.columns}
    for row in sub.itertuples(index=False, name=None):
        it = row[ci["item"]]
        key = (it, row[ci["action"]])
        if key not in lat:
            lat[key] = float(row[ci["latency_ms"]])
            ndcg[key] = float(row[ci["ndcg10"]])
        if it not in telemetry:
            telemetry[it] = np.array(
                [float(row[j]) if j is not None and row[j] == row[j] else 0.0 for j in feat_idx],
                dtype=np.float64)
            selectivity[it] = float(row[ci["selectivity"]])
    items = sorted(telemetry)
    actions = sorted(df["action"].unique())
    return items, telemetry, lat, ndcg, selectivity, actions
