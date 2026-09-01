# ResearchPivot state brief

## Problem + current claim
Original project: distill a query-only, undeployable unsupervised anomaly-detection pipeline (trained on NORMAL data only) into a small deployable student, by synthesizing off-manifold queries, labelling them with the teacher, and matching the teacher's anomaly-score behavior (student-teacher fidelity, not standalone AUROC). The intuitive claim — "place synthetic queries in a low-density shell just off the normal manifold" — was refuted. The project is currently a clean negative in search of a novel, impactful, publishable reframing.

## Evidence so far (validated, committed)
- 21-24 ADBench datasets, 5 teacher families (kNN, KDE, one-class SVM, IsolationForest, autoencoder), 10 seeds, two-sided paired tests.
- Near-manifold "shell" sampling is NOT special: uniform_shell does not beat normals-only (p=0.56) and is beaten by space-filling uniform_box (mean -0.125, p<1e-4). Per-cell winners: box 48/91, plain shell 3/91.
- Multi-output residual pipeline: off-manifold sampling helps the aggregate SCORE (+0.19, p=3e-4) and the alarm DECISION (+0.11, p=1e-3), but NOT root-cause ATTRIBUTION (ns).
- An attribution-aware "disagreement" sampler fails everywhere (0/21 attribution wins); multi-head ensemble distillation: sampling does not help per-head fidelity (normals-only ties/wins). Coverage predictor weak (r~=0.20, p=0.06).

## Key insight(s)
- Off-manifold query synthesis helps distill the score/decision for single growing teachers, but the specific near-boundary shell is not the lever; space-filling is as good or better in the cheap, low-dimensional, fillable regime.
- CORRECTION from external review: geometry + budget cause any sample-efficiency crossover; teacher COST only converts query savings into money/time — the two are separate axes.

## Tried and rejected (dead-ends — do NOT recycle)
1. Near-manifold shell sampling as the informative region — refuted (box wins).
2. Score/variation importance weighting within the shell — inside noise (p=0.08-0.46).
3. Attribution-aware / inter-output disagreement sampler for multi-output pipelines — fails (0/21).
4. Multi-head ensemble distillation benefiting from sampling — null (normals-only ties/wins).
5. Root-cause attribution as the valuable output sampling unlocks — not supported (attribution resists distillation; score benefits far more).
6. **Regime-crossover mechanism test v1** (controlled limit-state teacher g(x)=min_j ||A^T x - c_j||^2 - r^2; adaptive margin×disagreement + k-center vs Sobol vs random; D in {2,8,32}, m=2, k in {1,3}, B in {64..512}, 5 seeds): VALIDATED NULL for adaptive-beats-Sobol — adaptive ~= Sobol everywhere (diff within +-0.02). D=2 fillable (all ~0.97). D=8 only starts learning at B=512 (band acc 0.72); D=32 stays at chance at all tested budgets. Confound: budgets too small to reach the learnable-but-unsaturated frontier at D>=8.
7. **Regime-crossover frontier re-test** (D=6,8; m=2; k=1; B in {256,512,1024,2048}; 8 seeds) — VALIDATED NULL, confound resolved (the learnable frontier WAS reached: band accuracy climbs 0.55->0.97 across budgets). Adaptive (margin x ensemble-disagreement + k-center) gives at most a ~1-2 point edge over Sobol and NO query-efficiency win: to reach ~0.94 band accuracy both need the same budget (curves overlaid). Adaptive significantly beats Sobol only at D=6/B=2048 (+0.013, 8/8, p=0.008) — near saturation, negligible magnitude, and at HIGH budget (opposite of the "active wins at small budget" prediction). Adaptive >= random throughout (machinery works, not dead code). CONCLUSION: in controlled continuous limit-state distillation, a reasonable active learner does not beat Sobol space-filling in any impactful way — consistent with the DoE literature that space-filling is near-minimax. The "adaptive >> space-filling" positive regime the regime-map paper needs is NOT appearing. Caveat: tested one reasonable active learner (MLP-ensemble margin x disagreement); a GP level-set / AK-MCS learner was not tried, but those are mature/known methods.

8. **Order-Canonical Distillation** (cycle-2 pivot: distill rank/quantile target instead of raw score; claim = lower alarm-set regret + invariance to monotone score rewrites). Smoke: 3 datasets x 2 teachers x 3 seeds, alarm-set regret at 5% budget, worst-case over phi in {identity, asinh, z^3, sigmoid}. VALIDATED NEGATIVE. Invariance holds exactly (rank spread over phi = 0.0000, code correct), but median DeltaR = -0.025, rank wins only 7/18; rank is WORSE than raw even on the IDENTITY transform (e.g. thyroid raw 0.020 vs rank 0.294), failing the pivot's own keep-criterion. MECHANISM (validated): the uniform rank target loses top-k TAIL resolution -- the alarm region is a thin slice [0.95,1] the MLP smooths over -- while the raw (tail-heavy) score gives more signal to separate extreme anomalies. rank_worst (0.497) ~= raw_worst (0.509), so even the worst-case-robustness claim fails. Invariance is real but operationally empty. EMERGENT INVERSE HYPOTHESIS: for deployed top-k alarm decisions, the distillation target should EMPHASIZE the tail (top-k / extreme-value weighted), not canonicalize to uniform rank.

9. **Top-K Contrastive Distillation (TKCD)** (cycle-3 pivot: raw-score MSE + tail-contrastive loss pushing teacher-top-5% query points above 85-95th-pct points; distill the alarm set, not the whole score). Smoke: same 18 cells, alarm-set regret at 5%. VALIDATED NEGATIVE. Even with oracle per-cell lambda selection (leaked, favors TKCD): median ΔR=+0.000, wins 7/18, mean regret 0.283 (raw) vs 0.284 (TKCD); fixed lambda=0.3: median ΔR=-0.010, wins 5/18. Threshold (median>=+0.05 AND >=14/18) failed by a wide margin. Torch raw baseline matches cycle-2 sklearn raw (faithful reimpl); contrastive term runs (not dead code); helps only cardio+AE, null/negative elsewhere. CONCLUSION: the cycle-2 tail-resolution observation is DESCRIPTIVE, not exploitable — a cutoff-aware loss does not beat plain raw-score regression.

## META-FINDING across all 3 cycles + prior work
Raw-score regression on space-filling (box) queries is a STUBBORNLY strong baseline for black-box anomaly-scorer distillation. Everything tried to beat it FAILS at any impactful magnitude: shell placement, adaptive boundary-seeking sampling, rank/quantile canonicalization, tail-contrastive top-k objective, attribution/multi-head extensions. This convergent null says the problem AS POSED (distill a cheap tabular anomaly scorer from off-manifold queries) has no easy lever and is essentially solved by the trivial baseline. The genuinely defensible result is this NEGATIVE/benchmark itself. Any impactful pivot likely requires changing the PROBLEM (an expensive real teacher where query-efficiency has economic value, or a setting where the trivial baseline is not already near-optimal), not another loss/target/sampler on tabular ADBench.

## Assets available
- ADBench tabular datasets (35) on disk; cheap analytic limit-state teachers (arbitrary D, m, k, boundary thickness) — CPU-only, seconds/cell.
- Local 8-core box + RunPod (144-vCPU pods) for parallel sweeps. Compute-light: a full controlled sweep is minutes-to-an-hour.
- Student: small MLP ensembles; sklearn; scipy qmc (Sobol/LHS).
- No expensive real simulator wired yet (F16 GCAS / BenchBase-PostgreSQL are candidate real domains but not set up).

## Target
Novel AND impactful AND valuable AND highly publishable (workshop-to-conference tier, e.g. a methods/benchmark venue). Prefer a compute-light controlled result with a clean mechanistic story, optionally validated on one real black-box teacher.
