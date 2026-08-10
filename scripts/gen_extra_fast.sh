#!/usr/bin/env bash
# FAST re-run: expand workload with arguana + scidocs to push n_c up, but cut the
# ce_rerank combinatorics that hung the full run (budgets 80 only, 3 seeds,
# rerank-depth 20). Keeps ALL queries (n_c depends on #queries, not depth).
# arguana externals already exist -> reuse. Does NOT clobber the canonical 3-ds merge.
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${GPU:-0}"
EXT_DIR="external_rankings_multi"
OUT_DIR="results/larger_workload"
OUT5="results/larger_workload_5ds"
MERGED5="$OUT5/all_action_rows_5ds.csv"
SEEDS="13,17,23"
SEL="1.0,0.5,0.1"
BASE_EMB="intfloat/e5-base-v2"
mkdir -p "$OUT5"

# NOTE: ColBERT external uses a pure-Python O(queries*docs) MaxSim loop that takes
# ~hours per dataset; it is NOT needed for the n_c scaling probe (n_c depends on
# #queries/items, not on the action set), so we SKIP ColBERT for scidocs. arguana
# keeps its already-computed colbert. Asymmetry noted as a scoped-probe caveat.
echo ">>> [scidocs] external rankings (capped corpus 6000 docs, GPU $GPU) -- qwen3+splade only, skip colbert"
mkdir -p "$EXT_DIR/scidocs"
for spec in "dense Qwen/Qwen3-Embedding-0.6B qwen3_embedding_06b" \
            "splade naver/splade-cocondenser-ensembledistil splade_cocondenser_ensembledistil"; do
  set -- $spec; kind=$1; model=$2; op=$3
  if [ -f "$EXT_DIR/scidocs/scidocs/$op.tsv" ]; then echo "    [$op] already present, reuse"; continue; fi
  extra=(); [ "$kind" = dense ] && extra=(--pooling last --query-prefix "query: " --trust-remote-code)
  CUDA_VISIBLE_DEVICES=$GPU python3 scripts/generate_external_rankings.py \
    --dataset scidocs --kind "$kind" --model "$model" --operator "$op" \
    --max-queries 1500 --max-docs 6000 "${extra[@]}" --output-dir "$EXT_DIR/scidocs" \
    || { echo "FAIL ext $kind scidocs"; exit 1; }
done

# action rows: arguana (full corpus, reuse externals) + scidocs (capped 6000)
echo ">>> [arguana] action rows (reuse externals; 1 budget, 3 seeds, depth 20)"
python3 scripts/generate_metadata_filters.py --dataset arguana --selectivities "$SEL" --output-dir "$OUT_DIR/arguana" 2>/dev/null || true
python3 scripts/paretoprobe_sys_pilot.py --datasets arguana --max-queries 1500 \
  --budgets-ms 80 --selectivities "$SEL" --seeds "$SEEDS" --rerank-depths 20 \
  --embedding-model "$BASE_EMB" --external-ranking-dir "$EXT_DIR/arguana" \
  --output-dir "$OUT_DIR/arguana" || { echo "FAIL rows arguana"; exit 1; }

echo ">>> [scidocs] action rows (capped 6000; 1 budget, 3 seeds, depth 20)"
python3 scripts/generate_metadata_filters.py --dataset scidocs --selectivities "$SEL" --output-dir "$OUT_DIR/scidocs" 2>/dev/null || true
python3 scripts/paretoprobe_sys_pilot.py --datasets scidocs --max-queries 1500 --max-docs 6000 \
  --budgets-ms 80 --selectivities "$SEL" --seeds "$SEEDS" --rerank-depths 20 \
  --embedding-model "$BASE_EMB" --external-ranking-dir "$EXT_DIR/scidocs" \
  --output-dir "$OUT_DIR/scidocs" || { echo "FAIL rows scidocs"; exit 1; }

echo ">>> merge FIVE per-dataset CSVs -> $MERGED5"
python3 scripts/merge_action_rows.py --inputs \
  "$OUT_DIR/scifact/all_action_rows.csv" "$OUT_DIR/fiqa/all_action_rows.csv" \
  "$OUT_DIR/nfcorpus/all_action_rows.csv" "$OUT_DIR/arguana/all_action_rows.csv" \
  "$OUT_DIR/scidocs/all_action_rows.csv" --output "$MERGED5" || { echo "FAIL merge"; exit 1; }

echo ">>> certificate K-sweep (plugin) + UCB-Gamma on 5-ds merge (80ms)"
for K in 2 3 4 6; do
  python3 scripts/certified_window_compiler.py --action-rows "$MERGED5" --dataset "" \
    --budgets-ms 80 --n-cells "$K" --conc-mode eb --rho-mode plugin \
    --output-dir "results/cert_5ds_k${K}" || true
done
python3 scripts/certified_window_compiler.py --action-rows "$MERGED5" --dataset "" \
  --budgets-ms 80 --n-cells 4 --conc-mode eb --rho-mode ucb \
  --output-dir "results/cert_5ds_k4_ucb" || true
echo "### DONE gen_extra_fast ###"
