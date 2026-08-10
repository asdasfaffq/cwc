# Response to Reviewers — Paper 214

**Title.** Certified Workload-Window Plan Selection for Hybrid Vector Search under SLOs
**Track.** ICDE 2027 Research, First Round

We thank all three reviewers. Two comments changed the paper's technical content rather than
its presentation, and we want to say so up front:

1. **Reviewer #2 found a real defect in our released code** (the calibration split was used
   both to choose and to certify the deployed plan). The observation is correct, we have
   repaired it in two independent ways, and we measured what the repair costs. It does not
   move the paper's conclusions: at the headline configuration the certified bound is
   **unchanged** (0.2629) under the repair that matches Proposition 2 as written, and the
   deployed policy, realized violation and nDCG are unchanged to three decimals.
2. **Reviewer #2's concern that Γ was assumed rather than verified was well founded.** We
   no longer estimate Γ. Because CWC is transductive, both window marginals are observable,
   so Γ can be **computed** on an operator-declared coarsening. Doing so turns the paper's
   one unverified input into a measured quantity — and it shows that our old plug-in
   estimate did **not** dominate the measured budget (it would have needed κ\*=3.95).

Everything below is backed by experiments run for this response. All new code, all raw
result files, and the regenerated figures are public at
**https://github.com/asdasfaffq/cwc** (see `rebuttal/`). We understand the manuscript cannot
be edited at this stage; we have nevertheless implemented every editorial item and compiled
the revised manuscript, so the reviewers can check that the fixes are real and that the paper
still fits its current 13 pages (`rebuttal/paper_revision_preview/`).

---

## New experiments run for this response

| # | Experiment | Answers | Artifact |
|---|-----------|---------|----------|
| E1 | Certificate/implementation repair, 3 abstain rules on identical splits, 75 blocks | R2 availability | `rebuttal/scripts/exp_r1_certificate_repair.py` |
| E2 | End-to-end serving on a real hybrid stack: live execution, latency, throughput, open-loop load test | R1-W2, R2-D4, R3-D2 | `rebuttal/scripts/exp_r2_end2end_serving.py` |
| E3 | Γ made observable + falsification test of the remaining assumption | R2-W1, R2-D1 | `rebuttal/scripts/exp_r3_gamma_audit.py` |
| E4 | Admission control decided **before** execution, 4 rules × 7 deployments | R2-W2, R2-D2 | `rebuttal/scripts/exp_r4_prospective_admission.py` |
| E5 | Where the certificate is tight enough for a real SLA; floor vs sample-size split | R2-W3, R2-D3 | `rebuttal/scripts/exp_r5_tightness_map.py` |
| E6 | Measured brittleness of fixed recipes under workload drift | R3-D1 | `rebuttal/scripts/exp_r6_drift_motivation.py` |

---

# Reviewer #2

## Availability — "the same calibration data is used both to select the deployed plan and to certify it"

**The reviewer is right, and we thank them for reading the code this carefully.** In
`scripts/certified_window_compiler.py` the abstain decision compared calibration-based
certificate terms for the bound plan and the fallback and deployed the smaller one, while
Proposition 2 states that the deployed plan is fixed on the selection split. That is a
genuine violation of Assumption 1(iii) in the *implementation*, and the stated certificate
did not apply to the policy the code ran.

We repaired it in two ways and measured the cost of each on identical splits, cells, seeds
and Γ, so the difference is attributable to the abstain rule alone (E1: K ∈ {3,4,6,8,12} ×
budgets {80,120,160} ms × 5 seeds = 75 blocks per mode):

| Abstain rule | Valid? | ΔR_cert vs released (mean / max) | Δrealized violation | ΔnDCG | Same deployed policy |
|---|---|---|---|---|---|
| as released | **no** | — | — | — | — |
| **(a) selection-split** — decide on the selection split, calibrate only the fixed plan; this is Proposition 2 verbatim, log(K/δ) | yes | **+0.0026 / +0.0612** | **−0.0006** | −0.0077 | 50.7% |
| **(b) union bound** — keep the data-dependent rule, pay log(2·2K/δ) so the bound holds simultaneously for both candidates in every cell | yes | **+0.0330 / +0.0481** | +0.0007 | +0.0007 | 93.3% |

At the paper's headline configuration (K=4, 80 ms) the numbers are:

| | R_cert | realized violation | nDCG |
|---|---|---|---|
| as released | 0.2629 | 0.0014 | — |
| (a) selection-split | **0.2629** | 0.0011 | — |
| (b) union bound | 0.2866 | 0.0014 | — |

So repair (a) — the one that makes the code match the proposition we actually stated —
leaves the headline certificate **numerically identical** and *reduces* the realized
violation slightly. Repair (b) is an alternative for practitioners who prefer the
data-dependent rule; it costs +0.033 of bound width and nothing else. Across all 225 runs
(3 modes × 75 blocks) the certificate upper-bounds the realized violation without exception.

We have replaced the released implementation with (a) and documented (b) as an option. The
paper's Proposition 2 text is unchanged and now describes the code exactly.

## W1 / D1 — "the certificate requires a valid upper bound on Γ_c … the fully certified alternative is too loose"

We agree that a plug-in Γ with no guarantee is the weakest link, and we have removed the
plug-in step rather than defending it.

**The key observation is that Γ does not have to be estimated.** CWC is transductive: at
compile time it already holds the *unlabeled* telemetry of the window it is about to serve.
So for any coarsening Z that the operator declares (selectivity band, tenant, query class,
filter-cost bucket), **both** window marginals P_c(z) and Q_c(z) are directly observable and

  Γ_c^obs = max_{z : Q_c(z)>0} Q_c(z)/P_c(z)

is *counted*, not fitted. Under (A2′) within-stratum exchangeability — conditional on
(cell, stratum), the two windows draw from the same law — the change-of-measure step of
Proposition 2 goes through unchanged with this computed Γ. The assumption we cannot verify
therefore shrinks from "the density ratio is bounded by a number we estimated" to "queries in
the same cell *and same declared stratum* are exchangeable across windows", which is
**testable**, and we test it.

E3 (K=4, 3 budgets × 5 seeds = 15 blocks, canonical 3-dataset workload):

| Declared coarsening | Γ_obs mean / max | R_cert | realized | valid | cells with an unseen stratum |
|---|---|---|---|---|---|
| selectivity | 2.94 / 7.16 | **0.313** | 0.0009 | 15/15 | 0.2 |
| selectivity × probe-latency (2 bins) | 2.72 / 8.11 | 0.484 | 0.0009 | 15/15 | 0.8 |
| selectivity × probe-latency (4 bins) | 3.62 / 10.95 | 0.570 | 0.0009 | 15/15 | 1.0 |
| … × BM25-top-score (2 bins) | ∞ (some cells) | 0.713 | 0.0009 | 15/15 | 3.0 |

Three things follow, and we report all three including the one that is unflattering to us:

* **The reviewer's concern was substantiated.** The plug-in Γ̂ (mean 3.01) did **not**
  dominate the measured Γ_obs in every cell in any run; it would have needed inflation
  κ\* = 3.95 to do so. The paper's plug-in bound was therefore not conservative in the way
  we implicitly assumed, and we no longer report it as the headline.
* **Making Γ auditable is nearly free.** The valid, measured bound is 0.313 against the
  plug-in's 0.298 — about 5% wider, not an order of magnitude.
* **(A2′) survives a direct falsification attempt.** Comparing the deployed plan's
  conditional loss on the train vs the eval window inside each (cell, stratum) with ≥10
  points on each side, the mean gap is 0.0012 and the largest single gap is 0.027.

The coarsening is now an explicit operator knob with a visible dose–response: coarser means a
smaller Γ and a stronger exchangeability assumption; finer means a weaker assumption, a
larger Γ, and eventually eval strata unseen in training, at which point the bound correctly
returns the trivial value 1 for those cells. **Operational recipe:** declare the coarsest
partition on which you are willing to assume exchangeability, run the (cell, stratum) gap
test above on historical window pairs, and refine the partition until the test passes.

## W2 / D2 — "the admission-control experiment is not clearly certificate-driven"

Correct, and we have redone it. In the paper the block's payoff was gated on its *realized*
violation, so the objective was retrospective even though the intent was prospective.

In E4 the admission **decision** uses only quantities computable at compile time, and only
the payoff is settled afterwards: admit iff signal ≤ τ; a withheld window earns 0; an
admitted window earns nDCG@10 if it stays within τ and −1 if it breaches. We put every
candidate deployment through every rule (pooled 5-collection workload, 3 budgets × 10 seeds
= 30 blocks, K=2, δ=0.05, τ=0.05 — the level E5 identifies as certifiable at this scale):

| Deployment | Admission rule | Prospective? | Admitted | Breaches among admitted | Mean utility |
|---|---|---|---|---|---|
| **CWC** | **certificate (Prop. 2, observable Γ)** | **yes** | **96.7%** | **0.0%  (0/29)** | **0.496** |
| CWC | point estimate | yes | 100% | 0.0% | 0.513 |
| StaticBest-cal | EB slack, no shift correction | yes | 76.7% | 13.0% (3/23) | 0.300 |
| CostGreedy-cal | EB slack, no shift correction | yes | 76.7% | 13.0% (3/23) | 0.300 |
| Static-Qwen3 | EB slack, no shift correction | yes | 76.7% | 13.0% (3/23) | 0.300 |
| StaticBest-cal | point estimate | yes | 100% | **33.3%** (10/30) | 0.067 |
| CostGreedy-cal | point estimate | yes | 100% | 33.3% (10/30) | 0.067 |
| Static-Qwen3 | point estimate | yes | 100% | 33.3% (10/30) | 0.067 |
| Static-ColBERTv2 | any | — | 0% | — | 0.000 |
| StaticBest-cal | **post-hoc oracle** (not deployable) | **no** | 66.7% | 0.0% | 0.400 |
| CWC | post-hoc oracle (not deployable) | no | 100% | 0.0% | 0.513 |

The comparison the reviewer asked for is the last two rows against the first. **CWC deciding
purely prospectively (0.496) earns more than the best baseline deciding with clairvoyant
knowledge of the realized violation (0.400)**, because the baseline's clairvoyant rule has to
withhold a third of its windows while CWC's certificate lets it serve 97% of them without a
single breach. The prospective signals available to a baseline — a point estimate, or a
finite-sample bound that ignores shift — breach on 33.3% and 13.0% of the windows they admit.
That gap is exactly the value of the shift term, and it is only visible in a prospective
evaluation. We have replaced the paper's Table VI experiment with this one.

## W3 / D3 — "a bound around 0.30 is difficult to use for tight SLOs"

Agreed; a bound that only beats 1.0 is a weak claim. E5 maps where the bound is actually
usable. Writing R_cert ≈ Σ_c (m_c/M)·Γ_c·(L̂_c + ε(n_c)), it has a **floor** Γ·L̂ that no
amount of calibration removes and a statistical term that shrinks like 1/√n_c, so two
separate conditions must hold for an SLA level τ. Sweeping the labeled-window size on the
pooled 5-collection workload (80 ms, observable Γ, validity 100% at every point):

| mean n_c per cell | 68 | 137 | 242 | 341 | 521 | 688 |
|---|---|---|---|---|---|---|
| R_cert (K=2, δ=0.1) | 0.368 | 0.248 | 0.101 | 0.073 | 0.052 | **0.037** |
| realized violation | 0.009 | 0.005 | 0.004 | 0.003 | 0.004 | 0.003 |

Reading off the SLA levels:

| SLA τ | Certifiable? | Smallest configuration that reaches it |
|---|---|---|
| 0.20 | **yes** | n_c = 121, K=4, R_cert = 0.185 |
| 0.10 | **yes** | n_c = 341, K=2, δ=0.05, R_cert = 0.086 |
| **0.05** | **yes** | n_c = 688, K=2, δ=0.05, **R_cert = 0.0435** (realized 0.0031, nDCG 0.446) |
| 0.01 | not yet | floor is only 0.0015, so it is sample-size-bound, not floor-bound; the measured 1/√n law implies n_c ≈ 12.1k |

So a 5% SLA is certifiable today at a workload size we actually ran, and 1% is a data-volume
question with a specific answer rather than a barrier — a serving system that logs a few tens
of thousands of labeled queries per cell reaches it. The 0.30 in the submitted paper was the
3-collection regime (n_c ≈ 100–430), not a property of the bound.

## D4 — "add an end-to-end experiment that actually executes the selected plans"

Done; see the shared answer to R1-W2 below. Nothing in E2 is replayed from cached action
rows: every latency reported there is wall-clock time of a real execution.

## W4 / writing — blurry figures, page-6 overlap

Fixed and verified; see the shared answer at the end.

---

# Reviewer #1

## W2 — "for a systems conference I would expect integration into a real system, with end-to-end latency and/or throughput" (also R2-D4, R3-D2)

This was the most important gap and we have closed it with a full serving harness (E2). We
build a real hybrid stack — Okapi BM25 over a sparse term-weight matrix, an hnswlib HNSW
index over E5-base embeddings, reciprocal-rank fusion, a MiniLM cross-encoder at depth 20/50,
and a metadata predicate placed either **pre** (constrained ANN traversal through hnswlib's
filter callback, with the search list widened as the predicate tightens, as production
filtered-ANN implementations do) or **post** (retrieve then drop). Every policy, CWC included,
is executed live; CWC pays its telemetry probe and its routing decision **inside** the
measured path.

To make the filter-placement decision realistic we search a **100,785-document union of five
BEIR collections** (NFCorpus queries against NFCorpus + FiQA + SciFact + SciDocs + ArguAna).
The measured plan frontier shows why placement is a genuine decision rather than a toy: at
selectivity 1.0 pre-filtered dense search costs 0.48 ms, at selectivity 0.02 it costs
78.6 ms — a 164× spread — while post-filtering stays at 0.46 ms and loses quality
(nDCG 0.021 vs 0.029 at s = 0.02). SLO levels are fixed a priori as a log-spaced grid between
the 20th and 95th percentile of the pooled measured plan latency.

**Closed-loop replay of the served window (200 queries × 4 selectivities, 2 seeds, live
execution):**

| SLO | Policy | mean ms | p95 ms | SLO violation | nDCG@10 | throughput (qps) |
|---|---|---|---|---|---|---|
| **99.9 ms** | **CWC** | **37.8** | **83.8** | **0.000** | 0.1040 | **30.5** |
| 99.9 ms | StaticBest-cal | 48.5 | 96.8 | **0.134** | 0.1044 | 21.8 |
| 99.9 ms | Static-BestQuality | 96.9 | 155.6 | 0.389 | 0.1133 | 10.3 |
| 99.9 ms | CostGreedy-cal | 38.5 | 86.5 | 0.010 | 0.1007 | 26.0 |
| 45.4 ms | CWC | 6.5 | 19.1 | 0.000 | 0.1017 | 185.7 |
| 45.4 ms | StaticBest-cal | 1.8 | 3.9 | 0.000 | 0.0999 | 541.1 |
| 9.4 ms | CWC | 3.6 | 7.4 | 0.000 | 0.0995 | 274.4 |
| 9.4 ms | StaticBest-cal | 1.6 | 3.4 | 0.000 | 0.0999 | 608.2 |

At the binding SLO the systems result is unambiguous and it reproduces, under real execution,
the effect the paper claimed from cached rows: **CWC removes the SLO violations of the
strongest calibrated static plan (13.4% → 0.0%) at statistically indistinguishable quality
(0.1040 vs 0.1044), while delivering 40% more throughput (30.5 vs 21.8 qps) and 13% lower p95
latency (83.8 vs 96.8 ms).** On the smaller NFCorpus-only stack the same effect appears as
**3.6× throughput at equal-or-better quality** (137.8 vs 38.4 qps, nDCG 0.1352 vs 0.1335 at
the 58.8 ms SLO).

We also report, plainly, where CWC does not pay: at the two tight SLOs a single cheap plan is
optimal for every cell, so CWC's telemetry probe (p95 4.7 ms on the 100k-document index) is
pure overhead — it costs about 2 ms per query and 2.2× throughput for no quality gain. The
routing layer is worth its cost exactly when the plan frontier is cell-dependent, which is
the regime the paper is about; we would rather state that boundary than average it away.

## W1 — "the technical novelty is ML/statistical; the systems contribution appears comparatively modest"

This is a fair reading of the submitted version, in which the only systems measurement was an
overhead micro-benchmark. We would put the systems contribution as follows, now that E2 exists
to support it.

The contribution is a **query-optimizer design point**: choosing the *workload window*, rather
than the query or the whole deployment, as the unit at which a physical plan is bound, and
making that binding auditable before it serves traffic. Three consequences are specific to
systems rather than to statistics: (i) the unit choice is what makes the guarantee cheap —
because the serving window's cell masses are observed exactly, only K within-cell expectations
carry statistical risk instead of one term per query (Proposition 3), which is why the bound is
usable at all; (ii) the mechanism is an *abstain-to-fallback* rule in the plan space, i.e. a
compiler decision, not a post-hoc calibration; and (iii) the payoff is measured in serving
terms — violations eliminated, throughput raised 40%, p95 cut 13% on a 100k-document index.
The statistical machinery we use (change of measure, empirical Bernstein) is deliberately
standard; we claim the design point and the system that makes it operational, not a new
concentration inequality. We have rewritten the contribution list to say this directly instead
of leading with the certificate.

## W3 — "the paper appears put together in a hurry: sections don't link, figures unreferenced, unclear provenance of numbers"

Accepted without reservation. Every item in D1–D14 is implemented in
`rebuttal/paper_revision_preview/`, applied by a script that fails loudly if an anchor is
missing (`rebuttal/apply_presentation_fixes.py`, 24 edits), and the result compiles cleanly at
**the same 13 pages** as the submission.

| Item | What we changed |
|---|---|
| **D1** | The production-stack sentence now carries citations (BM25, E5, SPLADE, ColBERTv2, RRF, RankZephyr) and states explicitly that it describes the operator menu the literature and current vector engines expose *in general*, not one deployment; it points to the action catalog we actually execute (Table II). |
| **D2** | The forward reference is gone: "it adds an abstain rule and a deployment certificate, both developed later in the paper." |
| **D3** | Now "three axes that are important to take into account". |
| **D4** | Rewritten: "the BEIR benchmark systematized the zero-shot evaluation of retrieval models; its metrics, however, score ranking quality alone and say nothing about deployability under a latency SLO." |
| **D5** | The notation list now defines W: telemetry is computed in the context of the window W ∈ {W_tr, W_ev} containing the query, which enters only through window-level statistics such as filter survival. |
| **D6** | You are right that our use of "telemetry" was broader than the systems convention. We now split it explicitly: (i) quantities computed or estimated up front (query features, filter-survival estimates, cheap probe outputs) and (ii) quantities genuinely observed from execution (latency), available for the labeled historical window but never for the window being compiled — and we state that compiling the incoming window uses (i) on that window plus (i)–(ii) on history only. |
| **D7** | Baselines are introduced by the information each may use, with acronyms expanded: QueryOnly-**RF** (random forest on query-side features only), ScalarTelemetry-RF, CostGreedy-**cal** (calibrated p95 fits the budget), StaticBest-cal; "SOTA" is defined as the four published operators of Table II used as static plans. |
| **D8** | We added the rationale you asked for: the pilot ran first chronologically and motivated the design, but is reported second because it uses one collection and a local catalog and is the weaker evidence — with an explicit invitation to read it first. |
| **D9** | Diagnosed: IEEEtran numbers `\paragraph` as "a)", so a single run-in heading produced a stray enumerator. All run-in headings now use an unnumbered macro; there is no "a)" anywhere. |
| **D10** | Table III *is* referenced ("Table III reports…" in §VII-A), but it lands a page from its reference and is easy to miss — a real readability problem. We made the pointer explicit and the float placement tighter. |
| **D11** | Both sets of numbers now carry their setup inline: the violation rates are named as the two entries of Table IV at the 80 ms budget with K=4 averaged over five seeds, with no other configuration pooled in; the latency numbers are stated as wall-clock means over the 323 NFCorpus test queries, measured with `time.perf_counter` on the single RTX 4090 of §VI after a five-query warm-up with all indexes resident. |
| **D12** | Correct — Table VI had no reference at all. It is now referenced where its numbers are discussed. |
| **D13** | "Feature families and controls." is now its own paragraph with a proper run-in heading. |
| **D14** | Figures 3b, 3c and 3d are each referenced at the sentence that discusses them. |

## Inclusive writing / figure legibility (also R2-W4, R3-D3, R3-D4)

* **Figure 3 was genuinely illegible in print** — a 7.2-inch canvas with 5.2–7.5 pt type was
  included at 0.50\textwidth, so labels rendered at roughly 2.6–3.7 pt. We regenerated it:
  base type 6.7 → 9 pt, every explicit font size ×1.42, and the 2×2 block reflowed to a 1×4
  strip so it can be included at the **full two-column width** — labels now render at ~9 pt,
  and because the strip is shorter the enlargement costs no page.
* **Figure 1** is enlarged from 0.84 to 0.95\textwidth, which raises the subscript size you
  flagged by the same factor.
* **The §V-C overlap on page 6** was a `description` list whose long `\textsc` labels overran
  the IEEEtran column and collided with the body text. It is now a run-in itemized list; the
  rendered page is included in the preview and the collision is gone.

## Availability

Thank you for confirming the disclosure statement. The repository is now a complete release
rather than the partial one described in the submission (see the shared note below).

---

# Reviewer #3

## Weak point 1 — "the formulation is rigorous but may be hard to parse for readers outside this subfield"

We took this together with R1-D5/D6/D7. The notation list now defines the window argument,
"telemetry" is split into computed-up-front versus observed quantities, and every baseline is
introduced by the information it is allowed to use with its acronym expanded at first use. The
certification contract on page 6 — three labels distinguishing what is theorem-valid, what is
an empirical upper bound, and what is model-based — is also the passage whose formatting had
collapsed, which will not have helped; it now renders correctly.

## D1 — "back the motivation with numbers: how do selectivity and budget change with the query mix, and what is the impact?"

Agreed — the introduction asserted brittleness without measuring it. E6 measures it. We sweep
the amount of workload-mix drift between the labeled and served window (d = 0 is the i.i.d.
control; larger d skews the served window toward the expensive stratum) and record what each
fixed recipe does, at the 80 ms budget over 5 seeds:

| Policy | SLO violation, no drift → max drift | p95 latency ratio | throughput ratio | nDCG@10 at max drift |
|---|---|---|---|---|
| StaticBest-cal (calibrated fixed plan) | 0.0002 → **0.1245** | **3.04×** | **0.37×** | 0.642 |
| Fixed-rerank@100 | 0.102 → 0.155 | 1.07× | 0.88× | 0.530 |
| SOTA-Qwen3 (fixed) | 0.086 → 0.125 | 1.12× | 0.80× | 0.642 |
| Fixed-fusion (RRF) | 0.007 → 0.011 | 1.07× | 0.83× | 0.511 |
| **CWC** | 0.0015 → **0.0100** | 1.08× | 0.85× | 0.604 |
| Dense-only | 0.000 → 0.000 | 1.06× | 0.80× | 0.533 |
| BM25-only | 0.000 → 0.000 | 1.09× | 0.79× | 0.407 |

This gives the introduction the concrete statement it was missing, and a more precise claim
than "fixed recipes are brittle": **the fixed recipes that pursue quality degrade sharply — a
plan calibrated on yesterday's window sees its SLO violation rise from 0.02% to 12.4%, its p95
latency triple and its throughput fall to 37% as the mix shifts — while the fixed recipes that
stay safe do so by giving up 7 to 20 nDCG points (Dense-only 0.533, BM25-only 0.407 against
CWC's 0.604).** CWC is the only policy in the pool that holds both ends: 1.0% violation at
0.604 nDCG. We have rewritten the motivating paragraph around these numbers.

## D2 — "a more end-to-end result: how does plan choice translate into throughput or latency?"

This is the same request as R1-W2 and R2-D4, and it is now answered with live execution rather
than cached rows. The short version, on a real 100,785-document hybrid stack at a 99.9 ms SLO:
CWC serves the window with **zero** SLO violations against 13.4% for the strongest calibrated
static plan, at **40% higher throughput** (30.5 vs 21.8 qps) and **13% lower p95 latency**
(83.8 vs 96.8 ms), with quality unchanged (nDCG 0.1040 vs 0.1044). The full table, the
open-loop Poisson load test, and the honest tight-SLO regime where CWC's probe is pure
overhead are in the answer to R1-W2 above.

## D3 — formatting issue on page 6

Diagnosed and fixed: a `description` list whose long small-caps labels overran the column.
See the rendered page in the preview.

## D4 — "Figures 1 and 3 are too small and hence not legible"

Fixed by regenerating Figure 3 at 9 pt base type in a full-width 1×4 layout and enlarging
Figure 1; details in the shared answer under Reviewer #1.

## Availability — "couldn't access the provided repo link"

We are sorry about this, and it is the one point where we cannot fully reconstruct what
happened; the repository at `https://github.com/asdasfaffq/cwc` (the URL printed in the paper)
is public and we have re-verified anonymous access. Independently of the cause, the release
was also *partial* by design at submission time, which was the wrong call. It is now complete:
the full operator-execution and action-row generation pipeline, the external-ranking import,
the multi-dataset sweeps, every script behind every number in the paper, all six new
experiments above with their raw outputs, the regenerated figures, and the compressed
action-row artifacts (3.6 MB) that let a reader reproduce every table without a GPU. We would
be glad to provide a mirror if access fails again.

---

# Shared note on the artifact

`https://github.com/asdasfaffq/cwc` now contains:

```
scripts/            full pipeline (generation, import, merge, compile, certify, all gates)
paretoprobe/        retrieval/evaluation library used by the pipeline
rebuttal/scripts/   E1-E6, the six experiments run for this response
rebuttal/results/   every raw CSV/JSON behind every number in this response
rebuttal/figures_legible/       regenerated Figure 3 (+ per-panel source data)
rebuttal/paper_revision_preview/ the manuscript with all editorial fixes, compiled
rebuttal/apply_presentation_fixes.py  the 24 edits, each keyed to a reviewer item
artifacts/          compressed action rows (3 and 5 collections) — reproduce tables CPU-only
```

Two corrections we made to our own claims while preparing this response, which we would
rather state than leave for a reader to find:

1. The released `certified_window_compiler.py` did not implement the policy Proposition 2
   describes (Reviewer #2). Repaired; the headline bound is unchanged.
2. Our plug-in Γ̂ was not a conservative estimate of the shift budget it stood for; on the
   observable coarsening it fell short by up to 3.95×. We replaced it with the measured
   quantity, which is 5% wider and carries no unverifiable magnitude assumption.
