# CWC: Certified Workload-Window Plan Selection for Hybrid Vector Search under SLOs

Reference implementation and full artifact for the paper
**"Certified Workload-Window Plan Selection for Hybrid Vector Search under SLOs"**
(under review, IEEE ICDE 2027, paper 214).

CWC (Certified Workload-window Compiler) treats hybrid-search plan selection as a
query-optimization problem: given a labeled historical workload, an unlabeled incoming
workload window, and telemetry from candidate operators, it partitions queries into
*plan cells*, binds each cell to a physical plan, and **abstains** to a feasibility-best
fallback when the selected plan is fragile. Each deployment decision carries a
**distribution-free, finite-sample certificate** on the evaluation-window SLO-violation rate
under a within-cell shift budget.

> **This is now a complete release.** The submitted version of the paper described a partial
> release with the data-generation pipeline withheld. That was the wrong call: reviewers
> should be able to run everything. This repository now contains the full pipeline, every
> script behind every number in the paper, the six additional experiments run for the
> reviewer response, all of their raw outputs, and compressed action-row artifacts that let
> you reproduce the analysis tables **without a GPU**.

## Layout

| Path | Contents |
|------|----------|
| `scripts/` | Full pipeline: operator execution, external-ranking import, action-row merge, the compiler, the certificate, and all statistical gates |
| `paretoprobe/` | Retrieval/evaluation library used by the pipeline (BEIR loading, metrics, retrieval) |
| `rebuttal/scripts/` | The six experiments run for the reviewer response (E1–E6) |
| `rebuttal/results/` | Every raw CSV/JSON behind every number in the response |
| `rebuttal/figures_legible/` | Regenerated Figure 3 (9 pt type, 1×4 layout) plus per-panel source data |
| `rebuttal/paper_revision_preview/` | The manuscript with every editorial fix applied, compiled |
| `rebuttal/apply_presentation_fixes.py` | The 24 editorial edits, each keyed to the reviewer item it answers |
| `artifacts/` | Compressed action rows (3- and 5-collection workloads, 3.6 MB total) |
| `configs/`, `data/sample/` | Workload configuration and a small runnable NFCorpus sample |

## Reviewer comment → artifact map

| Reviewer item | Answered by | Command |
|---|---|---|
| **R2 availability** — calibration data used both to select and to certify | `rebuttal/scripts/exp_r1_certificate_repair.py` | `python3 rebuttal/scripts/exp_r1_certificate_repair.py` |
| **R1-W2 / R2-D4 / R3-D2** — end-to-end latency and throughput on a real system | `rebuttal/scripts/exp_r2_end2end_serving.py` | see "End-to-end serving" below (needs GPU) |
| **R2-W1 / R2-D1** — how should an operator choose or validate Γ? | `rebuttal/scripts/exp_r3_gamma_audit.py` | `python3 rebuttal/scripts/exp_r3_gamma_audit.py` |
| **R2-W2 / R2-D2** — admission decided before execution | `rebuttal/scripts/exp_r4_prospective_admission.py` | `python3 rebuttal/scripts/exp_r4_prospective_admission.py --pool --action-rows <5ds csv> --n-cells 2 --delta 0.05` |
| **R2-W3 / R2-D3** — where is the certificate tight enough for a real SLA? | `rebuttal/scripts/exp_r5_tightness_map.py` | `python3 rebuttal/scripts/exp_r5_tightness_map.py` |
| **R3-D1** — quantify the brittleness of fixed recipes | `rebuttal/scripts/exp_r6_drift_motivation.py` | `python3 rebuttal/scripts/exp_r6_drift_motivation.py` |
| **R1-D1…D14, R1/R2/R3 formatting and figures** | `rebuttal/apply_presentation_fixes.py`, `rebuttal/scripts/make_fig3_legible.py` | `python3 rebuttal/apply_presentation_fixes.py` |

## Two corrections to our own claims

1. **The certificate/implementation mismatch found by Reviewer #2 was real.** The released
   `certified_window_compiler.py` chose between the bound plan and the fallback by comparing
   *calibration*-based certificate terms, while Proposition 2 requires the deployed plan to be
   fixed on the selection split. `exp_r1_certificate_repair.py` implements both repairs
   (selection-split abstain, and a union bound over the two candidates per cell) and measures
   the cost of each on identical splits. At the headline configuration the certified bound is
   unchanged under the selection-split repair.
2. **Our plug-in Γ̂ was not conservative.** On an observable coarsening the measured within-cell
   shift budget exceeded the plug-in estimate by up to 3.95× in some cell in every run.
   `exp_r3_gamma_audit.py` replaces the estimate with the computed quantity and
   falsification-tests the exchangeability assumption that remains.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers the CPU-only analysis path. The end-to-end serving experiment
additionally needs `torch`, `transformers`, `sentence-transformers` and `hnswlib`.

## Quickstart (CPU only, no GPU, no downloads)

Reproduce the certificate repair and the Γ audit straight from the shipped action rows:

```bash
gunzip -c artifacts/action_rows_3datasets.csv.gz > /tmp/rows_3ds.csv

python3 rebuttal/scripts/exp_r1_certificate_repair.py --action-rows /tmp/rows_3ds.csv
python3 rebuttal/scripts/exp_r3_gamma_audit.py       --action-rows /tmp/rows_3ds.csv
python3 rebuttal/scripts/exp_r6_drift_motivation.py  --action-rows /tmp/rows_3ds.csv
```

and the tightness map plus prospective admission on the 5-collection pool:

```bash
gunzip -c artifacts/action_rows_5datasets.csv.gz > /tmp/rows_5ds.csv

python3 rebuttal/scripts/exp_r5_tightness_map.py --action-rows /tmp/rows_5ds.csv
python3 rebuttal/scripts/exp_r4_prospective_admission.py --pool \
    --action-rows /tmp/rows_5ds.csv --n-cells 2 --delta 0.05 --taus 0.01,0.02,0.05,0.10
```

The core compiler on the bundled 100-query sample:

```bash
python3 -m scripts.certified_window_compiler \
    --action-rows data/sample/nfcorpus_action_rows_sample.csv --dataset nfcorpus
```

On a single 100-query collection the bound is deliberately loose (≈1), reproducing the pilot
regime discussed in the paper; it tightens with per-cell calibration size (see E5).

## End-to-end serving (needs a GPU and the BEIR collections under `data/`)

Nothing in this experiment is replayed from cached action rows: every reported latency is
wall-clock time of a real execution against real indexes (sparse-matrix Okapi BM25, hnswlib
HNSW over E5-base, RRF, a MiniLM cross-encoder, and a metadata predicate placed pre or post).

```bash
python3 rebuttal/scripts/exp_r2_end2end_serving.py \
    --dataset nfcorpus \
    --corpus-datasets nfcorpus,fiqa,scifact,scidocs,arguana \
    --max-queries 200 --seeds 13,17 --n-budgets 4
```

This builds a 100,785-document union collection, measures every plan on every workload item,
replays the served window through each policy, and finishes with an open-loop Poisson load
test behind a single-server queue.

## Datasets

All datasets are public BEIR benchmarks (SciFact, FiQA-2018, NFCorpus, SciDocs, ArguAna),
available from the [BEIR repository](https://github.com/beir-cellar/beir) and the
[Hugging Face Hub](https://huggingface.co/BeIR). We use the standard test splits; no private
data is used.

## Citation

```bibtex
@inproceedings{cwc2027,
  title     = {Certified Workload-Window Plan Selection for Hybrid Vector Search under SLOs},
  author    = {Li, Yanxiao and Wei, Junhao and Zhao, Yifu and Li, Haochen and
               Yang, Xu and Im, Sio-Kei and Wang, Yapeng},
  booktitle = {IEEE International Conference on Data Engineering (ICDE)},
  year      = {2027},
  note      = {Under review}
}
```

## License

See [LICENSE](LICENSE).
