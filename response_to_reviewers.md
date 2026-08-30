# Response to Reviewers

**Where to Query a Blackbox Anomaly Pipeline: Shell Sampling for Label-Free Distillation**
Alexander Apartsin, Yehudit Aperstein

We thank the reviewer for the exceptionally detailed report. It substantially improved the paper. Below we respond to each point; section and figure numbers refer to the revised manuscript. All new experiments are released with the code, and every table and figure is regenerated from raw per-seed results by the scripts named below.

## Summary of changes

- **Reframed the central claim** to what the evidence supports: *shell placement* is the robust, query-efficient lever; within-shell weighting is a narrow, low-dimensional refinement. The abstract, contributions, discussion, and conclusion were rewritten accordingly.
- **Added five experiments** with real data at 5–10 seeds: a matched-cost budget study, a high-dimensional controlled-manifold scaling study (d up to 64), a real-tabular benchmark (10 ADBench datasets), three genuine composite pipelines, and a student-capacity sweep.
- **Corrected every claim** the report flagged as claim/evidence mismatch (query accounting, the shape-probe "prediction," the discontinuity/"fundamental limit" wording, the empirical-CDF gradient).
- **Repositioned novelty** against blackbox model extraction and one-class boundary synthesis (DROCC), and added those references.

## Blocking issues (W1–W4)

**W1 — "fixed query budget" did not count acquisition queries.** Fixed. §5 now defines the budget explicitly: $M$ is the number of *student-training* queries; the pipeline evaluations each sampler spends on selection are counted separately, and Appendix A gives the exact per-sampler complexity ($M$; $C$ for score; $C(1+2d)$ for coordinate variation; $C(1+2K)$ for the $K$-direction estimator). We add a **matched-cost budget study** (§6.7, `figure_budget.png`, `experiment_budget.py`): fixing student-training queries and varying total teacher evaluations, uniform-shell is the most query-efficient sampler, and the variation term never overtakes it once its $O(d)$/$O(K)$ selection cost is charged. We also add the **query-efficient $K$-random-direction estimator** ($O(K)$, dimension-independent), which matches the coordinate estimator (§6.5). We renamed the signal from "gradient magnitude" to **local score variation** throughout (see W8).

**W2 — test anomalies matched to the sampling shell.** Addressed. In the high-dimensional and real-data studies the shell is defined **from normal data only** (leave-one-out k-NN distance scale), never from the anomalies (§5, §6.5). We further add a coverage analysis (§6.5, `figure_coverage.png`) quantifying how much the anomaly support actually overlaps the normal-data shell and how that predicts the gain — see the foundational-questions section below.

**W3 — "shape probe predicts undistillability."** Removed. §4.5 now presents the radial profile only as a **descriptive diagnostic** (growing / saturating / non-monotone), and states explicitly that a one-dimensional radial summary cannot bound the representational difficulty of the full surface (the constant-vs-angular counterexample). The distillability question is instead answered empirically by the capacity sweep (§6.4).

**W4 — "discontinuous max fusion," "not distillable at any budget."** Corrected. We now describe the max-fusion teacher as **continuous but non-smooth** (a sharp ridge), not discontinuous, and identify the empirical-percentile transform as the piecewise source. We replaced "fundamental limit" with an **acquisition-vs-capacity interaction**, backed by a **student-width sweep** (§6.4): fidelity climbs with width for the non-smooth teacher, confirming the failure at the default width is partly capacity, not placement.

## Major issues (W5–W16)

**W5 — one budget only.** Added budget curves at matched total teacher evaluations (§6.7).

**W6 — weak baselines.** Added the **uniform-shell** baseline to every relevant comparison (§6.1, §6.5, §6.7) — the key control isolating allocation from coverage — and space-filling is implicitly covered by uniform-shell in the shell. We discuss model-extraction baselines in related work (§2.1) and position the method against them; a full active-model-extraction comparator on scalar regression is noted as the remaining baseline we did not run.

**W7 — positioning vs one-class boundary synthesis.** Added §2.3 discussion of **DROCC**, Outlier Exposure, and boundary-generation, with the explicit distinction: those synthesize negatives to *train a detector*; we synthesize teacher queries to *reproduce an existing scorer*, and study *where* in the off-manifold region queries are informative.

**W8 — empirical-CDF gradient underspecified.** Fixed. §4.3 defines the monotone rank as a **piecewise-linear quantile map** with an explicit capped tail extension, and renames the acquisition signal to **local score variation** (a finite-scale roughness measure, not a derivative of a step function).

**W9 — normalization invariance.** The rank transform of §4.3 is invariant to strictly monotone rescalings of the raw score up to the tail slope; we state this property where the transform is defined.

**W10 — "adapts" mixture.** Corrected to **hedge**: §4.4 states the equal mixture does not infer the shape, and §6.2 adds a **coefficient sweep** ($\lambda \in \{0,0.25,0.5,0.75,1\}$, `experiment_v2.py`) showing the equal mix is a robust default, not a tuned choice.

**W11 — teacher diversity / Isolation Forest.** Isolation Forest now appears in the composite pipelines (§6.6, P3). The single-teacher spectrum is kNN, KDE, one-class SVM, and autoencoder, chosen to span growing/saturating/non-smooth shapes; the real-data study uses an autoencoder across 10 datasets.

**W12 — high-dimensional evidence.** Added §6.5: a controlled-manifold scaling study (intrinsic $m=5$, ambient $d\in\{8,32,64\}$) and a 10-dataset real benchmark. Shell placement beats normals-only at every dimension and gives a statistically detectable real-data lift (8/10 datasets, Wilcoxon $p\approx0.05$).

**W13 — not actually pipelines.** Added §6.6: three genuine composite pipelines (smooth weighted-mean ensemble; density-gated ensemble; non-smooth Isolation-Forest/kNN/AE max-fusion). Shell placement helps all three; we report honestly that composite fidelity is much lower than single detectors and barely improves with width — an open limitation.

**W14 — capacity asserted, not shown.** Added the width sweep (§6.4) and a composite-pipeline width check (§6.6).

**W15 — Spearman too narrow.** Fidelity remains the primary metric (Spearman); the real-data study also reports standalone AUROC against ground-truth labels used for evaluation only.

**W16 — statistics too weak.** Controlled studies now use 5–10 seeds with standard deviations; the real-data comparison reports a **paired Wilcoxon** test and per-dataset wins.

## Moderate issues (W17–W20)

**W17 — far-field rationale.** §4.2 no longer rests the far-field exclusion on saturation alone; it gives the two-shape argument (saturating: no new information; growing: trivial monotone extrapolation).

**W18 — "plausible anomaly."** §4.2 restricts the plausibility claim to continuous standardized features; §7.1 adds the categorical/bounded-feature caveat.

**W19 — no deployment evidence.** §7.1 adds standardized-CPU numbers: the student is $28$–$41\times$ faster than the composite pipelines and $0.5$–$4$ KB versus $79$ KB–$1.6$ MB for the non-parametric teachers, whose size grows with $N$.

**W20 — "label-free."** The abstract and §3 now state precisely: no anomaly labels for training, acquisition, or model selection; benchmark labels used only for final evaluation.

## Two foundational questions

Two questions clarified the paper's motivation and are now addressed directly.

**Is running over all normal training data (no synthetic sampling) enough?** No, and this is now the stated crux (§4.1): the pipeline's scores on the normals are all "normal" and carry no signal about the anomaly-side ranking, so a student trained on normals alone — at any $N$ — cannot reproduce the teacher off the manifold. The whole normals-only-vs-shell comparison is the evidence.

**When is querying only from normal data sufficient?** Exactly to the extent the anomalies lie near the normal boundary. §6.5 measures this: the fraction of real anomalies inside the normal-data shell correlates $r=+0.44$ with the fidelity gain, and shuttle ($0.1\%$ in shell) is the clean failure. The assumption is thus measurable at evaluation time.

**Why distill at all?** Because the teacher is non-parametric or a heavy pipeline — undeployable. §1 and §7.1 make this explicit with the footprint numbers above.

## What we did not do

We did not run an active-model-extraction baseline adapted to scalar regression, an energy measurement on embedded hardware, or a fully cost-matched real-data budget curve; we note these as future work. We believe the revision resolves the four blocking issues and the major evidence gaps, and we welcome further guidance.
