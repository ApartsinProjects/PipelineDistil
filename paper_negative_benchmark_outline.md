# Paper skeleton — honest negative/benchmark

## Working title
**Space-Filling Is Hard to Beat: A Cautionary Benchmark for Label-Free Distillation of Black-Box Anomaly Scorers**

(Alt: "When Query Design and Target Shaping Don't Help: Distilling Black-Box Anomaly Detectors from Normal Data Only.")

## One-line thesis
For distilling a query-only unsupervised anomaly scorer (trained on normal data only) into a small student, a trivial baseline — space-filling queries + raw-score regression — is a stubbornly strong baseline that a battery of principled, intuitively-motivated interventions fail to beat; we characterize when and why, so practitioners stop paying for machinery that does not help.

## Abstract (draft)
Deploying an unsupervised anomaly-detection pipeline often means distilling a heavy, query-only teacher (kNN/KDE/one-class SVM/Isolation Forest/autoencoder, or a composite) into a small student. Because only normal data exists at training time, the student must be taught the teacher's behavior off the data manifold via synthetic queries. Intuition suggests the queries should be placed cleverly (near the boundary), weighted by informativeness, or given a decision-aligned target. Across 21-24 real tabular datasets, five teacher families, and 5-10 seeds with two-sided paired tests, we find that none of these beat a trivial baseline of space-filling queries with raw-score regression at any operationally meaningful magnitude: near-manifold "shell" placement is dominated by uniform space-filling (p<1e-4); adaptive boundary-seeking sampling matches Sobol at the learnable frontier; rank/quantile "order-canonical" targets, though invariant to monotone score rewrites, reduce top-k alarm fidelity by destroying tail resolution; and a tail-contrastive top-k objective does not improve alarm-set regret. We give the mechanism (teacher smoothness relative to student capacity; a fillable low-dimensional informative region) and delimit the narrow regime where off-manifold sampling does help (a growing single-detector teacher's aggregate score and alarm decision, though not its root-cause attribution). The contribution is a controlled negative result and a benchmark protocol that saves the community from unproductive query-design and target-shaping machinery, plus a falsifiable map of the regime where cleverness could still pay.

## Claims (all validated, wins-only for what is claimed)
1. Space-filling (uniform box) + raw-score regression is a strong baseline: shell placement does not beat it (p=0.56 vs normals-only; box beats shell p<1e-4).
2. Adaptive boundary-seeking (margin x ensemble-disagreement + k-center) does not beat Sobol at the learnable frontier (controlled limit-state sweep, D up to 32, budgets to 2048): tiny (<=2 pt) edge, no query-efficiency win.
3. Order-canonical (rank/quantile) distillation is invariant to monotone score rewrites but HURTS top-k alarm fidelity (median ΔR=-0.025; worse on identity), because uniform rank destroys tail resolution.
4. Top-K Contrastive Distillation does not beat raw-score regression on alarm-set regret (median ΔR=+0.00 even with oracle lambda).
5. Where off-manifold sampling DOES help: a single growing teacher's aggregate score (+0.19, p=3e-4) and alarm decision (+0.11, p=1e-3), but NOT its root-cause attribution (ns).
6. Mechanism + regime map: cleverness only stands to pay when the informative region is not fillable at the budget (high effective dimension / thin boundary) AND when the teacher's off-manifold structure exceeds what the student learns for free from space-filling.

## Section plan
1. Introduction — the label-free distillation setting; the intuition that query design should matter; the surprising negative.
2. Setup — teachers, students, query designs, the alarm-set-regret and fidelity metrics, datasets, protocol (two-sided, paired, seeds).
3. Query placement does not beat space-filling (shell vs box vs normals; adaptive vs Sobol; the controlled crossover sweep = Fig 1/2).
4. Target shaping does not beat raw-score regression (order-canonical; top-k contrastive) — with the tail-resolution mechanism.
5. Where sampling does help, narrowly (multi-output pipeline: score/decision yes, attribution no).
6. Mechanism and regime map — the coverage-difficulty vs budget picture; the negative-control corner we occupy; falsifiable prediction for the positive corner.
7. Related work — anomaly-score distillation (Hong & Kang 2024), model extraction (Knockoff/ActiveThief), active learning / level-set / reliability (Gotovos; AK-MCS), design of computer experiments (Sobol/LHS space-filling), CPS surrogate/falsification (Qin et al. 2022).
8. Limitations + scope — cheap tabular teachers; one reasonable active learner; the positive corner (expensive teacher + un-fillable region) left as future work.

## Figures (all from validated CSVs on disk)
- Fig 1: shell vs box vs normals-only per teacher family (results_realbench_v2) — the headline negative.
- Fig 2: adaptive vs Sobol fidelity-vs-budget at D=2/8/32 (results_regime_crossover + frontier) — no crossover.
- Fig 3: order-canonical and TKCD alarm-set regret vs raw (results_ordercanon, results_tkcd) — target shaping fails.
- Fig 4: multi-output pipeline — score/decision helped, attribution not (results_pipeline_scenario).
- Fig 5: regime-map schematic — coverage difficulty vs budget; where we are (fillable) vs where cleverness could pay.

## Venue
Honest negative/benchmark: a workshop (e.g. an AD or DMKD workshop) or a short/benchmark track. Not a top-tier main track without the positive corner.

## Status of the higher-upside alternative (not taken now)
Expensive-real-teacher pivot (BenchBase+PostgreSQL SLO map, or F16 GCAS): could deliver the positive corner and a stronger paper, but needs new infra and faces close prior work (Qin et al. 2022 for F16). Greenlight separately.
