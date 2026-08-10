#!/usr/bin/env bash
# Orchestrate larger-workload action-row generation for the certified window-compiler.
#
# Usage:
#   SMOKE=1 bash scripts/run_larger_workload.sh         # CPU plumbing check (tiny, no external rankers)
#   bash scripts/run_larger_workload.sh                 # full run (needs GPU for external rankers)
#   DATASETS="scifact fiqa" bash scripts/run_larger_workload.sh   # override dataset list
#
# Reads configs/larger_workload.json for models/operators. Emits a merged
# all_action_rows_multi.csv that scripts/certified_window_compiler.py consumes.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

SMOKE="${SMOKE:-0}"
GPU="${GPU:-0}"
EXT_DIR="external_rankings_multi"
OUT_DIR="results/larger_workload"
MERGED="$OUT_DIR/all_action_rows_multi.csv"
SEEDS="${SEEDS:-13,17,23,29,31}"
SELECTIVITIES="1.0,0.5,0.1"
BUDGETS="40,80,120,160"
BASE_EMB="intfloat/e5-base-v2"

if [[ "$SMOKE" == "1" ]]; then
  DATASETS="${DATASETS:-scifact}"; MAXQ="${MAXQ:-20}"; MAXD="${MAXD:-2000}"
  SEEDS="13,17"; SKIP_EXTERNAL=1
  OUT_DIR="results/larger_workload_smoke"; MERGED="$OUT_DIR/all_action_rows_multi.csv"
  echo "### SMOKE MODE: CPU plumbing check (datasets=$DATASETS, max_queries=$MAXQ, no external rankers) ###"
else
  DATASETS="${DATASETS:-scifact fiqa nfcorpus}"; MAXQ="${MAXQ:-648}"; MAXD="${MAXD:-0}"
  SKIP_EXTERNAL="${SKIP_EXTERNAL:-0}"
  echo "### FULL MODE: datasets=$DATASETS (GPU device $GPU for external rankers) ###"
fi
mkdir -p "$OUT_DIR" "$EXT_DIR"

PER_DATASET_CSVS=()
for d in $DATASETS; do
  echo ">>> [$d] generating action rows"
  ext_arg=()
  if [[ "${SKIP_EXTERNAL:-0}" != "1" ]]; then
    mkdir -p "$EXT_DIR/$d"
    # --- external rankers (GPU) ---
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/generate_external_rankings.py \
      --dataset "$d" --kind dense  --model "Qwen/Qwen3-Embedding-0.6B" \
      --operator qwen3_embedding_06b --pooling last --query-prefix "query: " \
      --trust-remote-code --max-queries "$MAXQ" --output-dir "$EXT_DIR/$d"
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/generate_external_rankings.py \
      --dataset "$d" --kind splade --model "naver/splade-cocondenser-ensembledistil" \
      --operator splade_cocondenser_ensembledistil --max-queries "$MAXQ" --output-dir "$EXT_DIR/$d"
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/generate_external_rankings.py \
      --dataset "$d" --kind colbert --model "colbert-ir/colbertv2.0" \
      --operator colbertv2 --max-queries "$MAXQ" --output-dir "$EXT_DIR/$d"
    ext_arg=(--external-ranking-dir "$EXT_DIR/$d")
  else
    echo "    (skipping external rankers)"
  fi

  # --- metadata filters (selectivity) ---
  python3 scripts/generate_metadata_filters.py --dataset "$d" \
    --selectivities "$SELECTIVITIES" --output-dir "$OUT_DIR/$d" 2>/dev/null \
    || echo "    (metadata filter step skipped / using pilot defaults)"

  # --- action rows ---
  maxd_arg=(); [[ "$MAXD" != "0" ]] && maxd_arg=(--max-docs "$MAXD")
  python3 scripts/paretoprobe_sys_pilot.py \
    --datasets "$d" --max-queries "$MAXQ" "${maxd_arg[@]}" \
    --budgets-ms "$BUDGETS" --selectivities "$SELECTIVITIES" --seeds "$SEEDS" \
    --embedding-model "$BASE_EMB" "${ext_arg[@]}" \
    --output-dir "$OUT_DIR/$d"
  PER_DATASET_CSVS+=("$OUT_DIR/$d/all_action_rows.csv")
done

echo ">>> merging $(echo ${PER_DATASET_CSVS[@]} | wc -w) per-dataset CSVs"
python3 scripts/merge_action_rows.py --inputs "${PER_DATASET_CSVS[@]}" --output "$MERGED"

echo ">>> certified window-compiler on merged rows (K sweep)"
for K in 3 4 6 8 12; do
  python3 scripts/certified_window_compiler.py \
    --action-rows "$MERGED" --dataset "" \
    --n-cells "$K" --rho-mode plugin \
    --output-dir "results/cwc_multi_k${K}" || true
done
echo "### DONE. Acceptance check: see refine-logs/LARGER_WORKLOAD_PLAN_20260529.md ###"
