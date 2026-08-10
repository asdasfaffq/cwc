#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic metadata filter buckets from BEIR corpus fields.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.corpus)
    features = []
    years = []
    lengths = []
    for row in rows:
        text = " ".join(str(row.get(key, "")) for key in ["title", "text"]).strip()
        year = extract_year(text)
        length = len(TOKEN_RE.findall(text))
        if year is not None:
            years.append(year)
        lengths.append(length)
        features.append((str(row["_id"]), year, length))

    year_min = min(years) if years else 0
    year_max = max(years) if years else 1
    length_sorted = sorted(lengths)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["docid", "bucket", "year", "length"])
        writer.writeheader()
        for docid, year, length in features:
            if year is not None and year_max > year_min:
                recency = (year_max - year) / (year_max - year_min)
                bucket = int(max(0, min(99, round(recency * 99))))
            else:
                bucket = percentile_bucket(length_sorted, length)
            writer.writerow({"docid": docid, "bucket": bucket, "year": year or "", "length": length})
    print({"output": str(args.output), "documents": len(features), "year_coverage": len(years)})


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def extract_year(text: str) -> int | None:
    years = [int(item) for item in YEAR_RE.findall(text)]
    return max(years) if years else None


def percentile_bucket(sorted_values: list[int], value: int) -> int:
    if not sorted_values:
        return 99
    lo = 0
    hi = len(sorted_values)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_values[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    percentile = lo / max(1, len(sorted_values))
    return int(max(0, min(99, round(percentile * 99))))


if __name__ == "__main__":
    main()
