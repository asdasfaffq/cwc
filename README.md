# CWC: Certified Workload-Window Plan Selection for Hybrid Vector Search under SLOs

Reference implementation for the paper
**"Certified Workload-Window Plan Selection for Hybrid Vector Search under SLOs"**
(under review, IEEE ICDE 2027).

CWC (Certified Workload-window Compiler) treats hybrid-search plan selection as a
query-optimization problem: given a labeled historical workload, an unlabeled incoming
workload window, and telemetry from candidate operators, it partitions queries into
*plan cells*, binds each cell to a physical plan, and **abstains** to a feasibility-best
fallback when the selected plan is fragile. Each deployment decision carries a
**distribution-free, finite-sample certificate** on the evaluation-window SLO-violation
rate under a specified within-cell shift budget.

> **Pre-publication note.** This is a *partial* reference release. It contains the core
> compiler, the certificate, and the evaluation/statistical-testing harness so that the
> method and its claims can be inspected and run on a bundled public-data sample. The full
> data-generation pipeline (operator execution, external-ranking import, multi-dataset
> sweeps) and the complete precomputed result artifacts will be released upon publication.

## What is included

| Path | Role (paper section) |
|------|----------------------|
| `scripts/certified_window_compiler.py` | CWC core: telemetry, cell construction, binding, abstain, certificate (§IV–V) |
| `scripts/compute_plan_frontier_certificate.py` | Distribution-free certificate computation (§V) |
| `scripts/rank_gate_cwc.py` | Canonical average-rank gate (§IV, §VI-A) |
| `scripts/statistical_rank_gate.py`, `scripts/verify_rank_significance.py` | Holm-corrected Wilcoxon rank tests (§IV) |
| `scripts/sla_admission_gate.py` | SLA admission-control objective (§VI-D) |
| `scripts/merge_action_rows.py` | Multi-dataset action-row merge utility (§VI-C scaling) |
| `configs/` | Workload / baseline configuration used in the study |
| `data/sample/` | A 100-query NFCorpus action-row sample (derived from public BEIR) for a runnable demo |

## What is *not* included (released upon publication)

- The operator-execution and action-row generation pipeline.
- External-ranking import (SPLADE / ColBERTv2 / Qwen3-Embedding) generation.
- The full multi-dataset precomputed results and figures.

These are withheld while the paper is under review; they are not required to read or run
the core method on the bundled sample.

## Datasets

All datasets in the paper are public BEIR benchmarks (SciFact, FiQA-2018, NFCorpus),
available from the [BEIR repository](https://github.com/beir-cellar/beir) and the
[Hugging Face Hub](https://huggingface.co/BeIR). We use the standard test splits; no
private data is used. The bundled `data/sample/` rows are derived from the public NFCorpus
collection.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

Run the certified compiler on the bundled public-data sample:

```bash
python3 -m scripts.certified_window_compiler \
    --action-rows data/sample/nfcorpus_action_rows_sample.csv --dataset nfcorpus
```

This prints the per-budget certificate table (`R_cert_bound`, realized violation,
nDCG, abstain fraction, `cert_valid`). On a single 100-query collection the bound is
deliberately loose (≈1), reproducing the pilot regime discussed in the paper; the bound
tightens with per-cell calibration size on the larger workloads.

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

See [LICENSE](LICENSE). This pre-publication release is provided for academic inspection
and verification; please cite the paper. A standard open-source license will accompany the
full release upon publication.
