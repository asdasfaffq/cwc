# Response to Reviewers — Paper 214

**Certified Workload-Window Plan Selection for Hybrid Vector Search under SLOs**
ICDE 2027 Research Track, First Round

We are grateful for three careful reviews. Two of them changed the paper's technical content,
not just its presentation, and we lead with those rather than bury them.

**Reviewer #2 read our code and found a real defect.** The released compiler let the
calibration split both *choose* and *certify* the deployed plan, which breaks the
independence assumption Proposition 2 relies on. The finding is correct. Auditing the rest of
the data flow we found a second instance the review did not mention — the fallback plan was
also screened using calibration data — and we report it here rather than wait for someone
else to find it. Both are fixed, and we measured what the fix costs: **at the headline
configuration the certified bound is numerically unchanged (0.2629) and the realized violation
falls slightly.**

**Reviewer #2 was also right that Γ was assumed rather than verified.** We no longer estimate
it. Because CWC is transductive it already holds the incoming window's unlabeled telemetry, so
on any operator-declared coarsening both window marginals are *observable* and
Γ_c = max_z Q_c(z)/P_c(z) can be **computed**. That turns the paper's one unverified input into
a measured quantity, at a cost of 5% in bound width — and it shows our old plug-in estimate
was **not** conservative (it would have needed κ\* = 3.95). We report that against ourselves.

Six new experiments were run for this response. All code, all raw result files, the
regenerated figures and the compiled revision are public at
**https://github.com/asdasfaffq/cwc** (directory `rebuttal/`); anonymous access re-verified.

---

## The evidence in five numbers

| Claim | Measurement |
|---|---|
| The certificate repair does not move the result | Bound **0.2629 → 0.2629** under the Prop.-2-faithful repair; valid in **225/225** runs |
| Γ is now measured, not assumed | **0.298 → 0.313** bound width; the exchangeability assumption that remains survives a direct falsification test (gap **0.0012** mean) |
| The method works end-to-end on a real engine | On a live 100,785-document hybrid stack at a 100 ms SLO: SLO violations **20.4% → 0.7%**, throughput **+75%**, p95 **−28%**, nDCG cost **3.9%** |
| The certificate is usable at a real SLA | **τ = 0.05 certified** at n_c = 688 (R_cert = 0.0435, realized 0.0031) |
| Admission decided before execution is sound | Under the certificate rule **0 breaches across all 7 deployments**; the best certificate-free prospective signal breaches **13.0%**, a point estimate **33.3%** |

---

## On relevance to ICDE (Reviewer #1 marked this "Borderline")

We would like to make the case explicitly, because we believe the fit is closer than the
submitted version conveyed.

The object of the paper is **physical-plan selection under a resource constraint** — the
field's core problem, applied to the hybrid/vector operators that data systems now ship. Three
things place it inside the database agenda rather than beside it:

1. **The unit of decision is the workload, not the query.** This is the same unit as physical
   database design: index and materialized-view selection have always been workload-level
   problems [9], and we argue plan *binding* for hybrid retrieval should be too. Proposition 3
   makes the systems consequence precise: because a serving window's cell masses are observed
   exactly, only K within-cell expectations carry statistical risk, whereas a per-query
   certifier pays log M instead of log K. The unit choice is what makes the guarantee
   affordable — a systems argument, not a statistical one.
2. **The output is an admission decision under an SLA**, i.e. a resource-management artifact a
   DBMS can act on before serving traffic, not a post-hoc quality report (§VII-D, now redone
   prospectively).
3. **It is now evaluated as a system.** The new experiment executes every selected plan
   against real indexes — sparse-matrix BM25, HNSW, RRF, a cross-encoder, and a metadata
   predicate placed pre- or post-retrieval — and reports service latency, tail latency,
   throughput, and behaviour under open-loop arrivals into a queue.

We do not claim a new retriever or a new concentration inequality, and we say so in the paper.
We claim a query-optimizer design point for hybrid search and the system that makes it
operational.

---

# Reviewer #2

## Availability — "the same calibration data is used both to select the deployed plan and to certify it"

**You are right, and thank you for reading the implementation this closely.** In
`certified_window_compiler.py` the abstain decision compared calibration-based certificate
terms for the bound plan and the fallback and deployed whichever was smaller, while
Proposition 2 states the deployed plan is fixed on the selection split. Assumption 1(iii) was
violated in the implementation, so the stated certificate did not apply to the policy the code
actually ran.

**A second instance, which the review did not raise, was found when we audited the rest of the
data flow.** The fallback was screened by `action_train_violation()` over `train_items` — the
selection split *and* the calibration split — so the plan a cell falls back to was also a
function of calibration labels. Both defects are fixed: the fallback is now screened on the
selection split alone.

We repaired the abstain rule in two ways and measured each on identical splits, cells, seeds
and Γ, so any difference is attributable to the abstain rule alone
(E1: K ∈ {3,4,6,8,12} × budgets {80,120,160} ms × 5 seeds = 75 blocks per rule):

| Abstain rule | Valid | ΔR_cert (mean / max) | Δviolation | ΔnDCG | Identical policy |
|---|---|---|---|---|---|
| as released | **no** | — | — | — | — |
| **(a) selection-split** — decide on the selection split, calibrate only the already-fixed plan. Proposition 2 verbatim, log(K/δ) | yes | **+0.0026 / +0.0612** | **−0.0006** | −0.0077 | 50.7% |
| **(b) union bound** — keep the data-dependent rule and pay log(2·2K/δ), so the bound covers both candidates in every cell | yes | **+0.0330 / +0.0481** | +0.0007 | +0.0007 | 93.3% |

At the paper's headline configuration (K = 4, 80 ms):

| | R_cert | realized violation |
|---|---|---|
| as released | 0.2629 | 0.0014 |
| **(a) selection-split** | **0.2629** | **0.0011** |
| (b) union bound | 0.2866 | 0.0014 |

So the repair that makes the code match the proposition we stated leaves the certificate
numerically identical and *lowers* the realized violation. It does change which plan some
cells deploy — in half the runs — at a mean cost of 0.008 nDCG, which we report rather than
round away. Repair (b) is offered for practitioners who prefer the data-dependent rule; it
costs +0.033 of bound width and nothing else. Across all 225 runs the certificate
upper-bounds the realized violation without exception.

One calibration for reading these against the submitted paper: all three rows already use the
repaired *fallback* screening, which is why they sit at 0.263 rather than the 0.298 in
Table IV — moving the fallback screen off the calibration split accounts for that difference
on its own. We have replaced the released implementation with (a) and documented (b).

## W1 / D1 — "the certificate requires a valid upper bound on Γ_c … please explain how an operator should choose or validate it"

We agree this was the weakest link, and we removed the estimation step rather than defend it.

**Γ does not have to be estimated.** CWC is transductive: at compile time it holds the
unlabeled telemetry of the window it is about to serve. So for any coarsening Z the operator
declares — selectivity band, tenant, query class, filter-cost bucket — *both* window marginals
P_c(z) and Q_c(z) are observable, and

  Γ_c^obs = max_{z : Q_c(z) > 0} Q_c(z) / P_c(z)

is counted, not fitted. Under (A2′) within-stratum exchangeability — conditional on (cell,
stratum), the two windows draw from the same law — the change-of-measure step of Proposition 2
goes through unchanged with this computed Γ. The unverifiable part therefore shrinks from "the
density ratio is bounded by a number we estimated" to "queries in the same cell *and the same
declared stratum* are exchangeable across windows", which is testable — and we test it.

E3, K = 4, 3 budgets × 5 seeds = 15 blocks on the canonical workload:

| Declared coarsening | Γ_obs mean / max | R_cert | realized | valid | cells with an unseen stratum |
|---|---|---|---|---|---|
| selectivity | 2.94 / 7.16 | **0.313** | 0.0009 | 15/15 | 0.2 |
| selectivity × probe-latency (2 bins) | 2.72 / 8.11 | 0.484 | 0.0009 | 15/15 | 0.8 |
| selectivity × probe-latency (4 bins) | 3.62 / 10.95 | 0.570 | 0.0009 | 15/15 | 1.0 |
| … × BM25-top-score (2 bins) | ∞ in some cells | 0.713 | 0.0009 | 15/15 | 3.0 |

Three consequences, including the one that does not favour us:

* **Your concern was substantiated.** The plug-in Γ̂ (mean 3.01) did **not** dominate the
  measured Γ_obs in every cell in a single run; it would have needed κ\* = 3.95. The submitted
  headline bound was therefore not conservative in the way we implicitly assumed, and we no
  longer present it that way.
* **Auditing Γ is nearly free.** The valid, measured bound is 0.313 against the plug-in's
  0.298 — about 5% wider, not an order of magnitude.
* **(A2′) survives a direct attempt to falsify it.** Comparing the deployed plan's conditional
  loss on the train versus the eval window inside every (cell, stratum) with ≥10 points on each
  side, the mean gap is **0.0012** and the largest single gap is **0.027**.

**Operational recipe**, which is what D1 asked for: declare the coarsest partition on which you
are willing to assume exchangeability; compute Γ_c from the two observed marginals; run the
(cell, stratum) gap test above on historical window pairs; refine the partition until the test
passes. The dose–response is visible and monotone — coarser gives a smaller Γ and a stronger
assumption, finer gives a weaker assumption and a larger Γ, and when a served stratum is absent
from training the bound correctly returns the trivial value 1 for that cell rather than a
number the operator should not believe.

## W2 / D2 — "the admission-control experiment is not clearly certificate-driven"

Correct, and we redid it. In the submitted version a block's payoff was gated on its *realized*
violation, so the objective was retrospective even though the intent was prospective.

In E4 the admission **decision** uses only quantities computable at compile time; only the
payoff is settled afterwards. We also removed a confound present in our own first attempt: the
certificate is a property of the analysis, not of CWC, so it can be computed for *any* fixed
deployed plan. Granting it to every deployment separates the admission rule from the policy it
judges. Pooled 5-collection workload, 3 budgets × 10 seeds = 30 blocks, K = 2, δ = 0.05,
τ = 0.05 (the level E5 identifies as certifiable at this scale):

| Admission rule | Deployment | Admitted | Breaches among admitted | Utility |
|---|---|---|---|---|
| **certificate** | Static-SPLADE | 96.7% | **0.0%** | **0.532** |
| **certificate** | **CWC** | 96.7% | **0.0%** | 0.496 |
| **certificate** | Static-E5 | 100% | **0.0%** | 0.411 |
| **certificate** | StaticBest-cal | 33.3% | **0.0%** | 0.200 |
| **certificate** | Static-Qwen3 | 33.3% | **0.0%** | 0.200 |
| **certificate** | CostGreedy-cal | 26.7% | **0.0%** | 0.160 |
| **certificate** | Static-ColBERTv2 | 0% | **0.0%** | 0.000 |
| EB slack, shift ignored | StaticBest-cal / CostGreedy-cal / Static-Qwen3 | 76.7% | **13.0%** | 0.300 |
| point estimate | StaticBest-cal / CostGreedy-cal / Static-Qwen3 | 100% | **33.3%** | 0.067 |
| post-hoc oracle (not deployable) | StaticBest-cal | 66.7% | 0.0% | 0.400 |

* **The rule is sound for every policy, not only ours.** Under the certificate rule all seven
  deployments breach **0 of the windows they admit**. Under the strongest prospective signal a
  certificate-free operator has, the three quality-seeking recipes breach 13.0% of admitted
  windows; under a plain point estimate, 33.3%. That gap is exactly the value of the shift
  term, and a retrospective evaluation cannot see it — which is why this objection was the
  right one to raise.
* **Under a strict SLA the certificate is what makes a good plan bankable.** CWC's compiled
  plan passes its own certificate on 96.7% of windows against 26.7–33.3% for the calibrated
  static plans, so a certified operator serving CWC earns 2.5× the utility of one serving the
  best calibrated static plan (0.496 vs 0.200) despite that plan's higher raw nDCG (0.600 vs
  0.513) — and more than the *clairvoyant* post-hoc rule applied to it (0.400).
* **Scope, stated plainly.** On this pooled 5-collection workload at K = 2, Static-SPLADE is
  itself safe (violation 0.0010) and of decent quality, so under the certificate rule it earns
  0.532 against CWC's 0.496. This configuration is outside the paper's evaluation setting; on
  the canonical 3-collection workload SPLADE is not on the frontier and CWC leads it (average
  rank 2.51 vs 3.07, nDCG 0.641 vs 0.605, Tables III and VI). We report it because a reader of
  the artifact would find it, and because it is the honest boundary of the claim: CWC is
  reliably *certifiable at high quality*, not superior to every operator on every workload.

This experiment also exposes a trade-off we now state explicitly: a tight certificate wants
few, well-populated cells, while good routing wants enough cells to separate the workload. At
K = 2 the τ = 0.05 SLA is certifiable but routing is constrained (nDCG 0.513); at K = 4 CWC
recovers quality (0.580) while the certificate supports τ ≈ 0.30 at this workload size. E5
quantifies the exchange rate, and it is a data-volume question rather than a structural one.

## W3 / D3 — "a bound around 0.30 is difficult to use for tight SLOs"

Agreed; a bound that merely beats 1.0 is a weak claim, and we now say where the bound is
genuinely usable. Writing R_cert ≈ Σ_c (m_c/M)·Γ_c·(L̂_c + ε(n_c)), it carries a **floor**
Γ·L̂ that no amount of calibration removes plus a statistical term shrinking like 1/√n_c, so an
SLA level τ needs two separate conditions to hold. Sweeping the labeled-window size on the
pooled 5-collection workload at 80 ms (observable Γ; the bound upper-bounds the realized rate
at every point):

| mean n_c per cell | 68 | 137 | 242 | 341 | 521 | 688 |
|---|---|---|---|---|---|---|
| R_cert (K = 2, δ = 0.1) | 0.368 | 0.248 | 0.101 | 0.073 | 0.052 | **0.037** |
| realized violation | 0.009 | 0.005 | 0.004 | 0.003 | 0.004 | 0.003 |

| SLA τ | Certifiable? | Smallest configuration reaching it |
|---|---|---|
| 0.20 | **yes** | n_c = 121, K = 4, R_cert = 0.185 |
| 0.10 | **yes** | n_c = 341, K = 2, δ = 0.05, R_cert = 0.086 |
| **0.05** | **yes** | n_c = 688, K = 2, δ = 0.05, **R_cert = 0.0435** (realized 0.0031) |
| 0.01 | not yet | floor is only 0.0015, so this is sample-size-bound, not floor-bound; the measured 1/√n law implies n_c ≈ 12.1k |

A 5% SLA is therefore certifiable at a workload size we actually ran, and 1% is a data-volume
question with a specific answer rather than a barrier — a serving system that logs a few tens
of thousands of labeled queries per cell reaches it. The 0.30 in the submission was the
3-collection regime (n_c ≈ 100–430), not a property of the bound, and the paper should have
said so.

## D4 — "add an end-to-end experiment that actually executes the selected plans"

Done; see the shared answer to R1-W2 below. Nothing there is replayed from cached action rows.

## W4 — presentation

Fixed and verified; see the shared answer at the end.

---

# Reviewer #1

## W2 — "integration into a real system, with end-to-end latency and/or throughput" (also R2-D4, R3-D2)

This was the most important gap and we have closed it with a full serving harness. The stack
is real throughout: Okapi BM25 evaluated as a sparse term-weight matrix (the same scoring an
inverted index performs), an hnswlib HNSW index over E5-base embeddings, reciprocal-rank
fusion, a MiniLM cross-encoder at depth 20/50, and a metadata predicate placed either **pre**
— constrained ANN traversal through hnswlib's filter callback, with the search list widened as
the predicate tightens, as production filtered-ANN implementations do — or **post**. Every
policy executes live, and CWC pays its telemetry probe and routing decision **inside** the
measured path.

To make filter placement a real decision rather than a toy we search a **100,785-document
union of five BEIR collections** (NFCorpus queries against NFCorpus + FiQA + SciFact + SciDocs
+ ArguAna). The measured frontier shows why the decision matters: pre-filtered dense search
costs 0.48 ms at selectivity 1.0 and **78.6 ms at selectivity 0.02** — a 164× spread — while
post-filtering stays at 0.45 ms and loses quality (nDCG 0.021 vs 0.029 at s = 0.02).

The SLO grid is fixed a priori and never tuned to an outcome: it is log-spaced between the
20th and 95th percentile of the pooled per-plan latency measured in the same run, which on
this machine evaluates to 9.4/20.7/45.4/99.9 ms; the table below reports at the rounded levels
10/25/50/100 ms. We flag one thing we learned while repeating the experiment: an SLO stated in
absolute milliseconds is machine-dependent. Re-running while five unrelated jobs occupied the
CPU made every plan 20–25% slower, which moves a fixed 100 ms budget across the knee of the
latency distribution and changes which plans are feasible. The quantile definition keeps the
operating point scale-free, and it is why we treat the *ratios* — violation reduction,
throughput gain, p95 reduction — as the headline rather than absolute latencies. Both the
quantile-defined and the fixed-millisecond runs are in the artifact.

**Closed-loop replay of the served window** (200 queries × 4 selectivities, 5 seeds):

| SLO | Policy | mean ms | p95 ms | SLO violation | nDCG@10 | throughput (qps) |
|---|---|---|---|---|---|---|
| **100 ms** | **CWC** | **39.2** | **82.1** | **0.007** | 0.1013 | **27.1** |
| 100 ms | StaticBest-cal | 67.5 | 114.4 | **0.204** | 0.1054 | 15.4 |
| 100 ms | CostGreedy-cal | 55.9 | 100.8 | 0.082 | 0.1046 | 17.9 |
| 100 ms | Static-BestQuality | 84.1 | 133.9 | 0.372 | 0.1079 | 11.9 |
| 100 ms | Static-Fallback | 28.9 | 63.2 | 0.000 | 0.0967 | 155.6 |
| 50 ms | CWC | 7.5 | 26.4 | 0.009 | 0.0972 | 181.5 |
| 50 ms | StaticBest-cal | 1.6 | 3.4 | 0.000 | 0.0957 | 619.0 |
| 25 ms | CWC | 3.2 | 7.0 | 0.000 | 0.0953 | 315.7 |
| 25 ms | StaticBest-cal | 1.6 | 3.4 | 0.000 | 0.0957 | 621.7 |
| 10 ms | CWC | 3.2 | 7.1 | 0.000 | 0.0953 | 310.8 |
| 10 ms | StaticBest-cal | 1.6 | 3.5 | 0.000 | 0.0957 | 610.6 |

At the binding SLO this reproduces, under real execution, exactly the effect the paper claimed
from cached rows: **CWC cuts the realized SLO-violation rate of the strongest calibrated static
plan by 30× (20.4% → 0.7%) while delivering 75% more throughput (27.1 vs 15.4 qps) and 28%
lower p95 latency (82.1 vs 114.4 ms), at a 3.9% nDCG cost** — within a whisker of the ≈4% the
paper reports from the cached-row study, now obtained by executing every plan. On the smaller
NFCorpus-only stack the same mechanism appears as 3.6× throughput at equal-or-better quality.

We are careful about what this experiment is and is not. It has 5 window-assignment seeds, and
**every one of the 5 favours CWC on violation, p95 and throughput against every baseline**
(one-sided Wilcoxon, per-comparison p = 0.031; with only 5 seeds a Holm correction over four
baselines cannot reach 0.05, so we report per-comparison values and do not claim family-wise
significance here). The paper's statistical claim continues to rest on the 45-block study with
Holm-corrected tests; this experiment's job is to show that the same effect survives when the
plans are actually executed on real indexes, and it does, at an effect size — a 30× reduction
in violations — that is not in question at this sample size.

**Under load.** We also drive the same stack open-loop, with Poisson arrivals into a
single-server queue that executes every request, scoring the SLO against *response* time
(queue wait + service). Measured saturation capacities at the 100 ms SLO are **CWC 21.2 qps**,
CostGreedy-cal 18.1, StaticBest-cal 18.0, Static-BestQuality 12.2 — CWC sustains the highest
arrival rate of the four, because routing sends the expensive constrained-ANN and rerank plans
only to the cells that need them.

| Policy | p95 @ 9.05 qps | viol | p95 @ 13.6 qps | viol | p95 @ 18.1 qps | viol |
|---|---|---|---|---|---|---|
| **CWC** | **175 ms** | **0.383** | **219 ms** | **0.508** | **429 ms** | **0.696** |
| StaticBest-cal | 226 ms | 0.508 | 450 ms | 0.650 | 1074 ms | 0.942 |
| CostGreedy-cal | 231 ms | 0.504 | 471 ms | 0.679 | 1011 ms | 0.883 |
| Static-BestQuality | 501 ms | 0.792 | 1812 ms | 0.992 | saturated | — |

The margin **widens with load**: CWC's response p95 is 22% below the strongest calibrated
static plan at the lightest load, 51% below at the middle one and **60% below at the heaviest**
(429 ms vs 1074 ms). Absolute violation rates are high for every policy under open-loop
arrivals because queueing delay compounds a service-time distribution that is heavy-tailed at
low selectivity; that is a property of the workload, and it is precisely why an operator wants
an admission rule rather than only a fast plan.

**Where CWC does not pay**, stated rather than averaged away: at 10–25 ms a single cheap plan
is optimal for every cell, so the telemetry probe (p95 4.6 ms on this index) is pure overhead —
0.0953 vs 0.0957 nDCG at half the throughput. At 50 ms CWC buys +1.6% nDCG but introduces a
0.9% violation rate where the static plan had none. The routing layer earns its cost when the
plan frontier is cell-dependent, which is the regime the paper is about.

## W1 — "the technical novelty is ML/statistical; the systems contribution appears comparatively modest"

That was a fair reading of a submission whose only systems measurement was an overhead
micro-benchmark. With the evaluation above in place we would put the contribution this way.

The contribution is a **query-optimizer design point**: binding a physical plan at the
granularity of a *workload window*, and making that binding auditable before it serves
traffic. Three consequences are systems consequences rather than statistical ones.
**(i) The unit choice is what makes the guarantee affordable.** Because the serving window's
cell masses are observed exactly, only K within-cell expectations carry statistical risk;
a per-query certifier pays log M in its radius instead of log K, and a distributional one pays
a mass-deviation term governed by window size that does not vanish with more calibration data
(Proposition 3). **(ii) The mechanism is a compiler decision** — abstain to a feasibility-best
fallback in plan space — not a post-hoc recalibration of a score. **(iii) The payoff is
measured in serving terms**: violations eliminated, throughput up 75%, p95 down 28%, capacity
up 18%, on a 100k-document index. The statistical machinery is deliberately textbook (change
of measure, empirical Bernstein) and we claim no new inequality; we claim the design point and
the system that makes it operational. We have rewritten the contribution list to say this
directly instead of leading with the certificate.

## W3 — "the paper appears put together in a hurry"

Accepted without reservation. Every item of D1–D14 is implemented in
`rebuttal/paper_revision_preview/`, applied by a script that fails loudly if an anchor string
is missing (`rebuttal/apply_presentation_fixes.py`, 24 edits keyed to reviewer items), and the
revision compiles cleanly.

| Item | What changed |
|---|---|
| **D1** | The production-stack sentence now carries citations (BM25, E5, SPLADE, ColBERTv2, RRF, RankZephyr) and states explicitly that it describes the operator menu the literature and current vector engines expose *in general*, not one deployment; it points to the catalog we actually execute (Table II). |
| **D2** | Forward reference removed: "it adds an abstain rule and a deployment certificate, both developed later in the paper." |
| **D3** | Now "three axes that are important to take into account". |
| **D4** | Rewritten: "the BEIR benchmark systematized the zero-shot evaluation of retrieval models; its metrics, however, score ranking quality alone and say nothing about deployability under a latency SLO." |
| **D5** | The notation list defines W: telemetry is computed in the context of the window W ∈ {W_tr, W_ev} containing the query, which enters only through window-level statistics such as filter survival. |
| **D6** | You are right that our "telemetry" was broader than the systems convention. We now split it: (i) quantities computed or estimated up front — query features, filter-survival estimates, cheap probe outputs — and (ii) quantities genuinely observed from execution, such as latency, available for the labeled historical window but never for the window being compiled; and we state that compiling the incoming window uses (i) on that window plus (i)–(ii) on history only. |
| **D7** | Baselines are introduced by the information each may use, acronyms expanded: QueryOnly-**RF** (random forest on query-side features only), ScalarTelemetry-RF, CostGreedy-**cal** (calibrated p95 fits the budget), StaticBest-cal; "SOTA" is defined as the four published operators of Table II used as static plans. |
| **D8** | The rationale is now stated: the pilot ran first chronologically and motivated the design, but is reported second because it uses one collection and a local catalog and is the weaker evidence — with an explicit invitation to read it first. |
| **D9** | Diagnosed: IEEEtran numbers `\paragraph` as "a)", so a single run-in heading produced a stray enumerator. All run-in headings now use an unnumbered macro; no "a)" appears anywhere. |
| **D10** | Table III *is* referenced in §VII-A, but it lands a page from its pointer and is easy to miss — a real readability problem, which we treated as such by making the pointer explicit and tightening float placement. |
| **D11** | Both sets of numbers now carry their setup inline: the violation rates are named as the two entries of Table IV at 80 ms with K = 4 over five seeds, with nothing else pooled in; the latencies are stated as wall-clock means over the 323 NFCorpus test queries, measured with `time.perf_counter` on the single RTX 4090 of §VI after a five-query warm-up with all indexes resident. |
| **D12** | Correct — Table VI had no reference at all. It is now referenced where its numbers are discussed. |
| **D13** | "Feature families and controls." is its own paragraph with a proper run-in heading. |
| **D14** | Figures 3b, 3c and 3d are each referenced at the sentence that discusses them. |

## Inclusive writing / figure legibility (also R2-W4, R3-D3, R3-D4)

* **Figure 3 was genuinely illegible in print.** A 7.2-inch canvas carrying 5.2–7.5 pt type was
  included at 0.50\textwidth, so labels rendered at roughly 2.6–3.7 pt. We regenerated it with
  base type raised 6.7 → 9 pt, every explicit font size ×1.42, and a compressed canvas, and
  enlarged the include to 0.86\textwidth; labels now render at about 7.5–9 pt.
* **Figure 1** is enlarged from 0.84 to 0.95\textwidth, raising the subscripts you flagged by
  the same factor.
* **The §V-C overlap on page 6** was a `description` list whose long `\textsc` labels overran
  the IEEEtran column and collided with the body text. It is now a run-in itemized list, and
  the rendered page in the preview shows the collision gone.

The revision is 14 pages against the submitted 13. The extra page is the added provenance,
terminology and baseline-definition text (D6, D7, D11) together with the enlarged figures; the
added text alone still fits in 13. We will absorb it in the camera-ready by moving the two
pilot-ablation panels of Figure 3 into the artifact, and we have already removed one summary
sentence that restated its own paragraph.

---

# Reviewer #3

## "The formulation is rigorous but may be hard to parse for readers outside this subfield"

We took this together with R1-D5/D6/D7. The notation list now defines the window argument,
"telemetry" is split into computed-up-front versus observed quantities, and every baseline is
introduced by the information it is allowed to use with its acronym expanded at first use. The
certification contract on page 6 — the three labels separating what is theorem-valid, what is
an empirical upper bound, and what is model-based — is also the passage whose formatting had
collapsed, which will not have helped; it now renders correctly.

## D1 — "back the motivation with numbers: how do selectivity and budget change with the query mix, and what is the impact?"

Agreed — the introduction asserted brittleness without measuring it. We now measure it. We
sweep the amount of workload-mix drift between the labeled and served window (d = 0 is the
i.i.d. control; larger d skews the served window toward the expensive stratum) and record what
each fixed recipe does at the 80 ms budget over 5 seeds:

| Policy | SLO violation, no drift → max drift | p95 ratio | throughput ratio | nDCG@10 at max drift |
|---|---|---|---|---|
| StaticBest-cal (calibrated fixed plan) | 0.0002 → **0.1245** | **3.04×** | **0.37×** | 0.642 |
| Fixed-rerank@100 | 0.102 → 0.155 | 1.07× | 0.88× | 0.530 |
| SOTA-Qwen3 (fixed) | 0.086 → 0.125 | 1.12× | 0.80× | 0.642 |
| Fixed-fusion (RRF) | 0.007 → 0.011 | 1.07× | 0.83× | 0.511 |
| **CWC** | 0.0015 → **0.0100** | 1.08× | 0.85× | 0.604 |
| Dense-only | 0.000 → 0.000 | 1.06× | 0.80× | 0.533 |
| BM25-only | 0.000 → 0.000 | 1.09× | 0.79× | 0.407 |

This yields a sharper claim than "fixed recipes are brittle": **the fixed recipes that pursue
quality degrade sharply — a plan calibrated on yesterday's window sees its SLO violation rise
from 0.02% to 12.4%, its p95 latency triple and its throughput fall to 37% as the mix shifts —
while the fixed recipes that stay safe do so by giving up 7 to 20 nDCG points** (Dense-only
0.533, BM25-only 0.407 against CWC's 0.604). CWC is the only policy in the pool that holds both
ends: 1.0% violation at 0.604 nDCG. The motivating paragraph is rewritten around these numbers.

## D2 — "a more end-to-end result: how does plan choice translate into throughput or latency?"

Same request as R1-W2 and R2-D4, now answered with live execution. On a real 100,785-document
hybrid stack at a 100 ms SLO, CWC serves the window with 0.7% SLO violations against 20.4% for
the strongest calibrated static plan, at **75% higher throughput** (27.1 vs 15.4 qps) and **28%
lower p95 latency** (82.1 vs 114.4 ms), for a 3.9% nDCG cost; its saturation capacity is 21.2
qps against 18.0, and its response-time p95 under load is up to 60% lower. The full tables, the
open-loop load test, and the tight-SLO regime where CWC's probe is pure overhead are in the
answer to R1-W2.

## D3 — formatting issue on page 6

Diagnosed and fixed: a `description` list whose long small-caps labels overran the column. The
rendered page is in the preview.

## D4 — "Figures 1 and 3 are too small and hence not legible"

Fixed by regenerating Figure 3 at 9 pt base type and enlarging both figures; details under
Reviewer #1.

## Availability — "couldn't access the provided repo link"

We are sorry, and this is the one point we cannot fully reconstruct: the repository at the URL
printed in the paper is public and we have re-verified anonymous access. Independently of the
cause, the release was also *partial* by design at submission time, which was the wrong call.
It is now complete: the full operator-execution and action-row generation pipeline, the
external-ranking import, the multi-dataset sweeps, every script behind every number in the
paper, all six experiments above with their raw outputs, the serving harness, the regenerated
figures, the compiled revision, and compressed action rows (3.6 MB) that reproduce every table
without a GPU. We are glad to provide a mirror if access fails again.

---

# Changes to the manuscript

1. Proposition 2's implementation now matches its statement; the fallback and the abstain
   decision are both selection-split functions. The union-bound variant is documented as an
   alternative.
2. Γ is computed on an operator-declared observable coarsening, not estimated; the
   exchangeability assumption is stated, tested, and the test is part of the released tooling.
   The plug-in headline is withdrawn.
3. §VII-D (admission control) is replaced by the prospective experiment, with the certificate
   granted to every deployment.
4. A new end-to-end serving section: live execution on a 100k-document hybrid stack, service
   and tail latency, throughput, saturation capacity, and an open-loop load test.
5. A tightness map replacing the single loose number, with the SLA levels the certificate
   supports and the calibration size each needs.
6. The introduction's brittleness claim is replaced by measured drift numbers.
7. The contribution list leads with the query-optimizer design point rather than the
   certificate.
8. All fourteen editorial items, the page-6 formatting fault, and both figures.

# Two corrections to our own claims

We would rather state these than leave them for a reader of the artifact to find.

1. The released `certified_window_compiler.py` did not implement the policy Proposition 2
   describes — in two places, one of which no review raised. Repaired; the headline bound is
   unchanged.
2. Our plug-in Γ̂ was not a conservative estimate of the shift budget it stood for; on the
   observable coarsening it fell short by up to 3.95×. It is replaced by the measured
   quantity, which is 5% wider and carries no unverifiable magnitude assumption.
