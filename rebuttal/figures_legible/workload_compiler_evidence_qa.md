# Figure QA Notes

Core conclusion: workload-level physical-plan compilation ranks first under constrained hybrid retrieval, and ablations indicate that retrieval telemetry is the useful signal.
Figure archetype: quantitative grid.
Backend: Python/matplotlib only.
Final size: 183 mm x 142 mm double-column composite.
Exports:
- svg: `rebuttal/figures_legible/workload_compiler_evidence.svg`
- pdf: `rebuttal/figures_legible/workload_compiler_evidence.pdf`
- tiff: `rebuttal/figures_legible/workload_compiler_evidence.tiff`
- png_preview: `rebuttal/figures_legible/workload_compiler_evidence.png`

Source data:
- main_rank: `rebuttal/figures_legible/source_main_rank.csv`
- score: `rebuttal/figures_legible/source_score.csv`
- k_selection: `rebuttal/figures_legible/source_k_selection.csv`
- feature_ablation: `rebuttal/figures_legible/source_feature_ablation.csv`

Checks:
- editable_svg_text: 107 SVG text nodes
- tiff_resolution: 4632x2467 px, mode=RGBA
- rank_gate_blocks: 30
- rank_gate_pass: True
- backend_exclusivity: All rendering and QA exports were produced with Python/matplotlib/Pillow.

Review risks:
- Current figure supports the one-workload claim; a second dataset should become a separate robustness panel if added.
- NV-Embed and RankZephyr are not in the rendered evidence because their true outputs were not available in the current result files.
- The k-selection panel intentionally shows silhouette failure to avoid cherry-pick framing.
