The project distills a query-only, undeployable unsupervised anomaly detector into a small deployable student by querying the teacher on synthetic/off-manifold inputs and optimizing student–teacher fidelity, and the latest result shows that globally rank-canonical targets hurt the deployed top-k alarm decision because they destroy useful tail resolution. 

RESEARCHPIVOT_STATE

Cycle 3 Research Pivot: Top-K Contrastive Distillation
Verdict

Pivot to decision-aligned, top-k contrastive distillation: distill the teacher's alarm set, not its entire score geometry.

This is materially different from the rejected rank-target pivot. Rank canonicalization forces the whole score distribution into uniform resolution; the proposed objective leaves raw score information intact and spends additional learning capacity only on the comparisons that determine whether a point enters the deployed top-k alarm set.

(a) Hypothesis / reframing

For black-box anomaly scorers deployed as a fixed-budget top-k alarm, global score fidelity is the wrong distillation objective: the operational loss is determined primarily by whether the student preserves the teacher's ordering across the top-k cutoff. A student trained with raw-score regression plus an explicit tail contrastive loss between teacher-top-k points and near-cutoff non-top-k points should reduce alarm-set regret even when ordinary score regression, rank regression, and query-selection changes do not.

The mechanistic claim is therefore not “tail points are more informative queries.” No new queries are required. The claim is: given the same teacher-labelled query set, allocating the loss to the operational tail boundary produces a better deployable surrogate.

Concrete method

Call it Top-K Contrastive Distillation (TKCD).

For each fixed teacher-labelled training query set:

Retain the existing raw teacher score target and the existing raw-score MSE baseline.

Compute the teacher's empirical top-5% cutoff τ on the training queries.

Define:

P: teacher points in the top 5%;

N_tail: teacher points in the 85th–95th percentile.

Train the same small MLP student with

L=L
raw
	​

+λL
top-k
	​


where L_raw is the current raw-score regression loss and

L
top-k
	​

=mean[softplus(m−(f(x
pos
	​

)−f(x
neg
	​

)))]

over randomly paired x_pos ∈ P, x_neg ∈ N_tail.

Use a fixed margin after standardizing the teacher score on the training query set, and tune only λ from a tiny predeclared grid such as {0.1, 0.3, 1.0} using the existing validation split.

The important design choice is the negative set immediately below the operational cutoff. Pairing top-5% points against arbitrary bulk negatives would make the auxiliary task too easy and would not test the claimed mechanism.

(b) Why this could be novel and who would cite it

The project's validated negatives already remove several obvious stories:

near-manifold shell sampling is not special;

adaptive boundary seeking does not materially beat Sobol space filling at the learnable frontier;

attribution-aware sampling fails;

multi-head sampling gains do not appear;

rank/quantile canonicalization is operationally harmful despite perfect monotone invariance. 

RESEARCHPIVOT_STATE

That leaves a sharper contribution: black-box anomaly-detector distillation should be evaluated and trained as decision-set imitation under a fixed alarm budget, rather than as global score approximation. The cycle-2 failure gives the mechanism directly: uniform rank removes resolution exactly where a top-k policy needs it, while the raw score already contains useful tail emphasis.

If the result holds across ADBench datasets and heterogeneous teacher families, the publishable story is not merely “we invented another loss.” It is:

Global fidelity objectives can be systematically misaligned with deployed anomaly decisions; a simple cutoff-aware distillation objective recovers the teacher's alarm set without extra teacher queries.

Likely citing communities:

anomaly detection and anomaly-score deployment;

knowledge distillation / model compression;

black-box model extraction and surrogate modelling;

decision-focused learning / learning-to-rank;

systems using fixed review, inspection, or alert budgets.

Closest prior work to beat

The attached state brief does not provide a bibliography, so it does not support naming specific papers without external research. The closest conceptual prior-art classes that this result would need to distinguish itself from are:

ordinary response/score-matching knowledge distillation;

rank- or order-preserving distillation;

pairwise/listwise learning-to-rank losses;

decision-focused learning objectives;

anomaly-score calibration or threshold-preservation methods.

The novelty bar is therefore not the existence of a pairwise loss. It is the black-box anomaly-distillation formulation, fixed-query setting, top-k alarm-set-regret objective, and empirical demonstration across teacher families that decision-aligned tail supervision beats both score fidelity and global rank fidelity.

If TKCD only gives a tiny improvement, that is not enough for a paper: pairwise ranking losses are too standard for a weak empirical gain to carry the contribution.

(c) Single cheapest confirm-or-kill experiment
Experiment

Reuse exactly the existing cycle-2 smoke infrastructure:

3 ADBench datasets

2 teacher families

3 seeds

same teacher-labelled training queries

same query budget

same train/validation/test splits

same small MLP architecture

same optimizer/training budget

This gives the same 18 paired cells and requires zero additional teacher queries.

Compare only:

Raw: current raw-score regression baseline.

TKCD: identical raw-score regression plus the top-k contrastive term above.

Do not include new samplers, rank targets, additional teachers, or a large hyperparameter sweep in the smoke test.

Primary metric

Use the exact same alarm-set regret at a 5% alarm budget already used in cycle 2.

For each paired cell define:

ΔR=R
raw
	​

−R
TKCD
	​


so positive values mean TKCD is better.

Pre-registered positive-lead threshold

Call the pivot a positive lead only if both conditions hold:

median ΔR ≥ +0.05 across the 18 paired cells; and

TKCD wins at least 14/18 cells (≥77.8%).

Anything weaker is a kill for the strong paper claim.

Why set the bar this high: the project has already shown that 1–2 point gains can be statistically real but scientifically negligible. A pairwise/top-k auxiliary loss is methodologically simple, so it must deliver a clearly operational effect — roughly a five-point absolute reduction in alarm-set regret with broad consistency — to justify a new paper direction.

Interpretation

Pass: There is a clean mechanistic paper: query selection was not the lever; loss allocation was. The teacher's useful information was already present in the existing space-filling queries, but conventional global objectives spent student capacity on regions irrelevant to the deployed alarm decision.

Fail: Stop trying to rescue this project through alternative target shaping. If even a cutoff-aware objective cannot beat raw-score regression by a practically meaningful margin, the cycle-2 “tail-heavy raw score” observation is descriptive rather than exploitable.

Bottom line

This is the pivot I would run next. It directly follows the only new positive mechanism produced by cycle 2, requires no new data and no new teacher queries, has a falsifiable operational metric already implemented, and avoids recycling every sampling-based dead end in the state brief. But the novelty is conditional on a large decision-level gain; a 1–2 point improvement would not be enough.