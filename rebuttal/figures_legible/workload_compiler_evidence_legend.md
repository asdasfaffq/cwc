# Figure Legend

**Figure. Workload-level compilation achieves rank-first constrained retrieval and identifies the useful telemetry.**

**a,** Average rank across 30 complete NFCorpus selectivity-budget-seed blocks. Lower is better. Error bars show the standard error of block-level ranks. Holm-adjusted pairwise tests compare the workload compiler against each baseline in the shared rank gate.

**b,** Mean constrained score under the same gate protocol. The score includes ranking quality with latency and metadata-filter penalties; higher is better.

**c,** Sensitivity to the number of workload cells. Filled markers indicate variants that pass the Rank-1 gate. The unsupervised elbow selector chooses k=5 and passes; the silhouette selector is shown as a negative control.

**d,** Feature-family ablation at k=6. Removing retrieval telemetry families causes gate failures, while the full telemetry variant remains first-ranked.

Statistics: n=30 complete blocks. Rank-gate comparisons use the existing Friedman/Holm protocol from `rank_gate.json`.
