#!/usr/bin/env python3
"""Apply every presentation fix requested by the three reviewers to the revision preview.

Run from the repository root:  python3 rebuttal/apply_presentation_fixes.py

Each edit is keyed to the reviewer item it answers. The script fails loudly if an
anchor string is missing, so a silent no-op cannot be mistaken for a fix.
"""
from __future__ import annotations

import sys
from pathlib import Path

BASE = Path("rebuttal/paper_revision_preview")
applied: list[str] = []


def edit(relpath: str, old: str, new: str, item: str) -> None:
    p = BASE / relpath
    s = p.read_text()
    if old not in s:
        sys.exit(f"ANCHOR NOT FOUND for {item} in {relpath}:\n  {old[:120]}")
    p.write_text(s.replace(old, new, 1))
    applied.append(item)


def edit_all(relpath: str, old: str, new: str, item: str) -> None:
    p = BASE / relpath
    s = p.read_text()
    if old not in s:
        sys.exit(f"ANCHOR NOT FOUND for {item} in {relpath}:\n  {old[:120]}")
    p.write_text(s.replace(old, new))
    applied.append(item)


# ----------------------------------------------------------------- R1-D1
edit("sections/1_introduction.tex",
     "A production hybrid-search stack chooses among BM25, dense vector search, "
     "learned-sparse retrieval, late interaction, reciprocal-rank fusion, "
     "metadata-filter placement, and neural reranking.",
     "Production hybrid-search stacks combine BM25~\\cite{robertson2009bm25}, dense vector "
     "search~\\cite{wang2022e5}, learned-sparse retrieval~\\cite{formal2021splade}, late "
     "interaction~\\cite{santhanam2021colbertv2}, reciprocal-rank fusion~\\cite{cormack2009rrf}, "
     "metadata-filter placement, and neural reranking~\\cite{pradeep2023rankzephyr}. This is the "
     "operator menu that the retrieval literature and current vector engines expose in general; "
     "we do not model one particular deployment, and the action catalog we actually execute is "
     "listed in Table~\\ref{tab:catalog}.",
     "R1-D1 (citation + scope of the production-stack claim)")

# ----------------------------------------------------------------- R1-D2
edit("sections/1_introduction.tex",
     "On top of this it adds the abstain rule and certificate of Section~\\ref{sec:certificate}.",
     "On top of this it adds an abstain rule and a deployment certificate, both developed later "
     "in the paper.",
     "R1-D2 (forward reference removed)")

# ----------------------------------------------------------------- R1-D3
edit("sections/6_related_work.tex",
     "but differ on three axes that a reviewer should weigh",
     "but differ on three axes that are important to take into account",
     "R1-D3 (wording)")

# ----------------------------------------------------------------- R1-D4
edit("sections/6_related_work.tex",
     "Finally, BEIR systematized zero-shot retrieval evaluation \\cite{thakur2021beir}, but "
     "conventional metrics alone do not capture deployability;",
     "Finally, the BEIR benchmark systematized the zero-shot evaluation of retrieval models "
     "\\cite{thakur2021beir}; its metrics, however, score ranking quality alone and say nothing "
     "about deployability under a latency SLO;",
     "R1-D4 (missing words / unclear sentence)")

# ----------------------------------------------------------------- R1-D5
edit("sections/2_problem.tex",
     "$\\features(\\q,\\W)$ the observable telemetry for query $\\q$;",
     "$\\features(\\q,\\W)$ the observable telemetry for query $\\q$ computed in the context of "
     "the window $\\W\\in\\{\\W_{\\mathrm{tr}},\\W_{\\mathrm{ev}}\\}$ that contains it (the window "
     "enters only through window-level statistics such as filter survival and catalog state);",
     "R1-D5 (define W in the notation list)")

# ----------------------------------------------------------------- R1-D6
edit("sections/2_problem.tex",
     "\\emph{Notation:}",
     "\\emph{On the word ``telemetry''.} We use it for everything the optimizer may read "
     "\\emph{before} committing a plan for the incoming window, which is of two kinds: "
     "(i) quantities computed or estimated up front for the query at hand---query-derived "
     "features, filter-survival estimates, and cheap retrieval-probe outputs; and "
     "(ii) quantities genuinely observed from execution, such as latency, which are available "
     "for the labeled historical window but never for the window being compiled. Compiling the "
     "incoming window uses (i) on that window plus (i)--(ii) on history; no measurement of the "
     "incoming window's own execution is ever used. "
     "\\emph{Notation:}",
     "R1-D6 (telemetry terminology: computed-up-front vs observed)")

# ----------------------------------------------------------------- R1-D9 / R1-D13
# IEEEtran numbers \paragraph as ``a)''. A run-in heading avoids the stray enumerator and
# lets ``Feature families and controls.'' start its own paragraph.
edit("main.tex", "\\input{math_commands}",
     "\\input{math_commands}\n"
     "% Run-in heading: IEEEtran numbers \\paragraph as ``a)'', which produced a stray\n"
     "% enumerator at the head of the results section (R1-D9) and buried a run-in heading\n"
     "% inside a paragraph (R1-D13).\n"
     "\\newcommand{\\parhead}[1]{\\par\\smallskip\\noindent\\textbf{#1}\\ }",
     "R1-D9/D13 (run-in heading macro)")
for f in ["sections/1_introduction.tex", "sections/2_problem.tex", "sections/3_optimizer.tex",
          "sections/3b_certificate.tex", "sections/4_experimental_protocol.tex",
          "sections/5_results_and_analysis.tex", "sections/6_related_work.tex",
          "sections/7_discussion_conclusion.tex"]:
    p = BASE / f
    s = p.read_text()
    if "\\paragraph{" in s:
        p.write_text(s.replace("\\paragraph{", "\\parhead{"))
applied.append("R1-D9 (no stray 'a)' enumerator anywhere)")

edit("sections/5_results_and_analysis.tex",
     "\\emph{Feature families and controls.} Full telemetry passes",
     "\n\n\\parhead{Feature families and controls.} Full telemetry passes",
     "R1-D13 (own paragraph)")

# ----------------------------------------------------------------- R1-D7
edit("sections/4_experimental_protocol.tex",
     "Baselines include static execution of one physical action for the whole workload, "
     "QueryOnly-RF, ScalarTelemetry-RF, CostGreedy-cal, and StaticBest-cal.",
     "Baselines include static execution of one physical action for the whole workload and four "
     "adaptive or calibrated selectors, named by the information each is allowed to use: "
     "\\emph{QueryOnly-RF}, a random forest (RF) trained to pick the action from query-side "
     "features only; \\emph{ScalarTelemetry-RF}, the same random forest given a scalar summary "
     "of the retrieval telemetry instead of the full feature set; \\emph{CostGreedy-cal}, which "
     "picks the highest-nDCG action whose calibrated (``-cal'') training-window $p95$ latency "
     "fits the budget; and \\emph{StaticBest-cal}, the single action with the best calibrated "
     "training-window constrained utility. Throughout, ``SOTA'' denotes the four strong "
     "published retrieval operators of Table~\\ref{tab:catalog} used as static plans.",
     "R1-D7 (expand RF/SOTA and explain baseline prefixes)")

# ----------------------------------------------------------------- R1-D8
edit("sections/5_results_and_analysis.tex",
     "As an initial single-collection study, \\WCR{} passes the 30-block Rank-1 gate",
     "This pilot was run \\emph{first} chronologically and motivated the design; we report it "
     "\\emph{after} the canonical multi-dataset result because it uses a single collection and a "
     "local action catalog, and so is the weaker evidence of the two. Readers who prefer "
     "development order may read this subsection before Section~\\ref{sec:results-rank}. "
     "As an initial single-collection study, \\WCR{} passes the 30-block Rank-1 gate",
     "R1-D8 (explain pilot placement)")

# ----------------------------------------------------------------- R1-D10 / R1-D12
edit("sections/5_results_and_analysis.tex",
     "Table~\\ref{tab:rank} reports average ranks;",
     "Table~\\ref{tab:rank} reports the average rank of every method in the pool;",
     "R1-D10 (make the Table III pointer explicit)")
edit("sections/5_results_and_analysis.tex",
     "We rank the seven methods per block and apply the same one-sided Holm-corrected Wilcoxon "
     "test as in Section~\\ref{sec:results-rank}.",
     "We rank the seven methods per block and apply the same one-sided Holm-corrected Wilcoxon "
     "test as in Section~\\ref{sec:results-rank}. Table~\\ref{tab:admission} reports the "
     "resulting admission utilities, ranks and realized violation rates.",
     "R1-D12 (Table VI now referenced)")

# ----------------------------------------------------------------- R1-D11
edit("sections/5_results_and_analysis.tex",
     "at $\\approx$two orders of magnitude fewer realized violations ($\\approx0.1\\%$ vs.\\ "
     "$\\approx11\\%$ at $80$\\,ms, Table~\\ref{tab:cert-scale}).",
     "at $\\approx$two orders of magnitude fewer realized violations. Both numbers come from one "
     "stated setting---the combined SciFact+FiQA+NFCorpus workload at the binding $80$\\,ms "
     "budget with $K{=}4$, averaged over the five window-assignment seeds---and are the "
     "\\texttt{viol.\\ \\CWC{}} and \\texttt{viol.\\ StaticBest-cal} entries of "
     "Table~\\ref{tab:cert-scale} ($0.001$ vs.\\ $0.112$); no other configuration is pooled into "
     "them.",
     "R1-D11a (provenance of the SLO-violation numbers)")
edit("sections/5_results_and_analysis.tex",
     "Per-query plan latencies span $0.16$\\,ms (dense HNSW) to $13.6$\\,ms (dense$+$rerank@20).",
     "Per-query plan latencies span $0.16$\\,ms (dense HNSW) to $13.6$\\,ms (dense$+$rerank@20); "
     "every latency in this subsection is a wall-clock mean over the $323$ NFCorpus test queries, "
     "measured with \\texttt{time.perf\\_counter} on the single RTX~4090 of "
     "Section~\\ref{sec:protocol} after a five-query warm-up, with all indexes resident.",
     "R1-D11b (provenance of the latency numbers)")

# ----------------------------------------------------------------- R1-D14
edit("sections/5_results_and_analysis.tex",
     "\\emph{Cell count.} The result is not tied to $k{=}6$:",
     "\\emph{Cell count} (Figure~\\ref{fig:evidence}c). The result is not tied to $k{=}6$:",
     "R1-D14a (Fig. 3c referenced)")
edit("sections/5_results_and_analysis.tex",
     "\\parhead{Feature families and controls.} Full telemetry passes",
     "\\parhead{Feature families and controls} (Figure~\\ref{fig:evidence}d). Full telemetry passes",
     "R1-D14b (Fig. 3d referenced)")
edit("sections/5_results_and_analysis.tex",
     "The rank result reflects higher constrained utility, not only a favorable rank statistic:",
     "The rank result reflects higher constrained utility, not only a favorable rank statistic "
     "(Figure~\\ref{fig:evidence}b):",
     "R1-D14c (Fig. 3b referenced)")

# --------------------------------------------- R1/R2/R3 formatting: page-6 text overlap
# The certification contract used a `description` list whose long \textsc labels overrun the
# IEEEtran column and collide with the body text. A run-in itemize keeps the labels inline.
edit("sections/3b_certificate.tex",
     "\\begin{description}\\itemsep1pt\n"
     "\\item[\\textsc{Theorem-valid}.] Proposition~\\ref{prop:cert} holds",
     "\\begin{itemize}\\itemsep1pt\\setlength{\\leftmargini}{1.2em}\n"
     "\\item \\textsc{Theorem-valid}. Proposition~\\ref{prop:cert} holds",
     "R1/R2/R3 (page-6 label/text overlap, part 1)")
edit("sections/3b_certificate.tex",
     "\\item[\\textsc{Headline plug-in}.] The reported",
     "\\item \\textsc{Headline plug-in}. The reported",
     "R1/R2/R3 (page-6 overlap, part 2)")
edit("sections/3b_certificate.tex",
     "\\item[\\textsc{Model-based UCB}.] Remark~\\ref{rem:rho-ucb}",
     "\\item \\textsc{Model-based UCB}. Remark~\\ref{rem:rho-ucb}",
     "R1/R2/R3 (page-6 overlap, part 3)")
edit("sections/3b_certificate.tex", "\\end{description}", "\\end{itemize}",
     "R1/R2/R3 (page-6 overlap, part 4)")

# --------------------------------------------- R1/R2/R3 figure legibility
edit("sections/3_optimizer.tex",
     "\\includegraphics[width=0.84\\textwidth]{cwc_framework}",
     "\\includegraphics[width=\\textwidth]{cwc_framework}",
     "R1/R2/R3 (Figure 1 enlarged to the full two-column width)")
edit("sections/5_results_and_analysis.tex",
     "\\includegraphics[width=0.50\\textwidth]{workload_compiler_evidence.pdf}",
     "\\includegraphics[width=\\textwidth]{workload_compiler_evidence.pdf}",
     "R1/R2/R3 (Figure 3 enlarged from 0.50 to the full two-column width)")

print("Applied fixes:")
for a in applied:
    print("  -", a)
print(f"\n{len(applied)} edits applied to {BASE}")
