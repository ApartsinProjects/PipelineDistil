# NEXT_PROJECT: The Refresh-Churn Floor as an Acceptance Rule for Distilled Anomaly Detectors

Status: proposal after the paper-discovery break attempt (2026-10-15). The load-bearing assumption was tested against primary sources AND against a live 15-cell smoke (`experiment_noisefloor.py`, 3 datasets x 5 teachers x 1 seed, results in the session scratchpad `nf_smoke/results.csv`). Half of the assumption holds, half breaks; the break forces a reframing that is stronger than the original claim. Everything below is written for the reframed paper.

---

## 1. Framing (tied to the reused asset)

The reused asset is the multi-teacher, decision-level distillation harness: 35 ADBench tabular datasets on disk, five black-box teacher families (kNN, KDE, one-class SVM, Isolation Forest, autoencoder), small MLP students trained label-free from normal data plus space-filling box queries, and the operational metric **top-k alarm-set regret** at a fixed alarm budget k (fraction of the teacher's top-k alarm cases the student fails to also flag).

The new object is the teacher's own **refresh churn**: retrain the teacher on a refreshed normal set (bootstrap resample; subsample fractions 50/90/99% as a sensitivity curve; plus seed re-initialisation for Isolation Forest and the autoencoder) and measure the alarm-set regret between the two teacher versions at the same budget k. This is the regret an operator already tolerates every time the detector is refreshed on new normal data. It is therefore the natural bar for a surrogate: a compressed or distilled student whose alarm-set regret against the deployed teacher is at or below the refresh-churn floor is operationally indistinguishable from a routine teacher refresh and can be shipped; a student above the floor changes the alarm list more than a refresh would and needs justification.

The paper delivers three things: (i) the floor as a measurable, tight, reproducible quantity across teacher families and datasets (with two invariants that certify the measurement); (ii) the acceptance rule and its pass/fail map for label-free distillation; (iii) the headroom map: how far above the floor current students sit, and what query budget and student capacity are needed to pass, which is the practitioner's stopping rule.

## 2. Why this and not the others

- C3 and C4 (adversarial cycle): killed upstream for lacking a pragmatic path to publication (already-done or no decisive dataset); not reopened.
- Shell / boundary / adaptive query placement: box and Sobol dominate or tie (p<1e-4; <=2 pt edge at the learnable frontier). Gravestone.
- Rank/quantile canonical targets: invariant but destroy tail resolution (median regret delta -0.025). Gravestone.
- Tail-contrastive top-k loss: median delta +0.00 even with oracle lambda. Gravestone.
- Attribution / multi-head distillation: 0/21 attribution wins. Gravestone.
- "Another sampler or loss on tabular ADBench": the convergent null says the lever is not there; this proposal changes the QUESTION (what bar must a student clear, and does it) rather than adding a lever.

## 3. Load-bearing assumption, break verdict, forced scope

**Assumption as stated:** "The teacher's refresh-churn floor is tight enough to be a meaningful bar (not trivially huge), yet the trivial uniform-box + raw-regression student sits under it across teacher families (not just the stochastic ones)."

**Primary-source attack (floor size).**
- Liang, Soloff, Barber, Willett, "Assumption-free stability for ranking problems" (arXiv 2506.02257, 2025): top-k set instability under leave-one-out is driven entirely by near-ties at the k-th cutoff; it is small for small k and grows with k (Netflix, n=1000: avg delta 0.02 at k=5, 0.12 at k=20, 0.39 at k=100). No separation condition can be assumed on real scores. Implication: the floor is dataset-driven (density of scores at the cutoff), not a universal constant, and must be measured per cell.
- Klüttermann et al., "Why Ranking AD Algorithms Isn't as Reliable as You May Think" (arXiv 2608.04613, 2026): random seeds contribute little to benchmark-level rank instability relative to dataset and hyper-parameters. Implication: seed-only floors for stochastic teachers will be small; the training-set resample is the relevant perturbation.
- Perini, Vercruyssen, Davis, ECML-PKDD 2020 (ExCeeD; verified from the PDF): a prediction's confidence is the probability that it flips when the training set of size n is redrawn, computed in closed form (Beta-Binomial) for a FIXED score function whose THRESHOLD moves with the redraw. This is a threshold-only churn; it lower-bounds a full-retrain churn because it ignores score changes. Implication: an analytic lower bound on the floor exists and should be reported as a pre-registered invariant (full-retrain floor >= ExCeeD threshold-only floor).
- Zimek et al., KDD 2013 (subsampling ensembles) and Aggarwal & Sathe (Outlier Ensembles, 2017): kNN-type outlier scores have a large variance component under training subsampling, which is exactly what subsample ensembles exploit. Implication: the floor for kNN/KDE could be loose at small n and small k; must be scoped by n and k.
- Azzari & Bicego, S+SSPR 2024: Isolation Forest run-to-run variability is real and reduced by ensembling. Implication: the iForest seed floor must shrink with tree count (invariant I2).

**Live smoke (15 cells, k=5% and 1%, 90% subsample floor, 4 resamples).**
- Floor tightness HOLDS: mean floor 0.094 at k=5% (0.153 at k=1%); deterministic teachers 0.000-0.18, stochastic 0.05-0.34. The floor is small but not zero and it is dataset-specific, as 2506.02257 predicts.
- Invariant I1 HOLDS: an independent second resample copy sits at the floor (mean |indep - floor| = 0.033).
- Invariant I2 HOLDS: iForest seed floor 0.208 (100 trees) -> 0.102 (1000 trees).
- Student-under-floor BREAKS: 1/15 cells at k=5% (iForest on vowels only), 2/15 at k=1%. Mean student regret 0.343 (k=5%) and 0.505 (k=1%) against a floor of 0.094 / 0.153. The pre-registered kill condition in the smoke ("student under floor only for iforest/ae") is met; the original claim (b), that the levers nulled because the trivial student already sat under the floor, is refuted.

**Verdict: MUST BE SCOPED (floor half HOLDS; student half BREAKS and is inverted).**
The floor is tight, certifiable, and lower-bounded analytically. The trivial student does not pass it; it sits 3-5x above it. The honest and stronger paper is therefore: the refresh-churn acceptance rule exists and is cheap to compute; label-free distillation of black-box anomaly scorers at current budgets FAILS it on most cells; the gap between student regret and the floor is the true headroom; none of the clever levers moved that gap; the paper maps what does (query budget, student capacity, training-normal reuse) and gives the practitioner a stopping rule.

Forced scope:
- Datasets: all 35 on disk, but headline tables restricted to cells with evaluation-set size >= 600 so that k at 1% is >= 6 and at 5% is >= 30 (small-k cells are reported in an appendix with their quantisation noted; 14 of the 35 datasets have fewer than ~320 validation normals and are affected).
- Budgets: k in {1%, 5%} as deployed budgets; k=0.1% only for the three datasets with eval >= 10k (shuttle, campaign, mammography).
- Floor definition: bootstrap (canonical refresh) as the headline; 50/90/99% subsample as the sensitivity curve; seed-reinit floor reported separately for iForest and AE.
- Teachers: all five families; results reported per family, never pooled across deterministic and stochastic teachers.

## 4. Novelty claim, strongest prior-art threat, why it still stands

**Novelty claim.** First decision-level acceptance rule for surrogates of anomaly detectors that uses the deployed detector's own refresh churn as the reference, measured as top-k alarm-set regret under training-set resampling; first measurement of that floor across five teacher families and 35 public datasets with certifying invariants; first demonstration that label-free score distillation fails the rule at common budgets and a map of the budget/capacity needed to pass it.

**Strongest threat: Stanton et al., "Does Knowledge Distillation Really Work?", NeurIPS 2021.** They define student-teacher fidelity (test agreement, predictive KL) for classifiers and use an independent student as a reference. Why the claim still stands: their reference is a student trained without the teacher, not a retrained teacher; their metric is per-example label agreement, not a top-k alarm set under a fixed budget; their setting is supervised classification, not anomaly detectors trained on normal data only; and they draw no acceptance rule from the reference. The present paper's reference is the teacher's own refresh, the unit is the alarm set, the domain is unsupervised AD, and the output is a pass/fail rule with a measured headroom.

**Second threat: Perini, Vercruyssen, Davis, ECML-PKDD 2020 (ExCeeD).** Closest AD-specific prior: per-example confidence as the probability the prediction flips under a redrawn training set. Distinction: ExCeeD holds the score function fixed and only moves the threshold (closed-form); the refresh floor retrains the scorer and measures the set-level change empirically; ExCeeD is per-example, the floor is per-alarm-set; ExCeeD is not used to judge a surrogate. ExCeeD becomes a baseline and an analytic lower bound inside this paper, which is the right way to absorb it.

**Third threat: Zhou, Xu, Hooker, "A Generic Approach for Reproducible Model Distillation", Machine Learning journal 2024 (arXiv 2211.12631).** Studies student variability over pseudo-sample draws with the teacher fixed and gives a CLT-based corpus-size test. Distinction: variability source is the query corpus, not the teacher's training data; output is a corpus-size rule for interpretable students, not an acceptance bar relative to teacher refresh. Cited as the complementary "how many queries" test; their corpus-size criterion can be run alongside the budget curve in Experiment E3.

## 4b. Venue fit

Siblings (verified venue/year/URL):
1. Stanton et al., NeurIPS 2021, https://arxiv.org/abs/2106.05945 (KD fidelity, independent baseline).
2. Perini, Vercruyssen, Davis, ECML-PKDD 2020, DOI 10.1007/978-3-030-67664-3_14 (ExCeeD; example-wise confidence under training-set redraw).
3. Perini, Galvin, Vercruyssen, ECML-PKDD 2020 Workshops (EDML), DOI 10.1007/978-3-030-65965-3_27 (a ranking stability measure for AD under perturbations; 9 citations; full text is paywalled and its exact perturbation/weighting was NOT verified from the primary text; must be read before submission).
4. Zhou, Xu, Hooker, Machine Learning journal 2024, https://arxiv.org/abs/2211.12631 (reproducible model distillation).
5. Liang, Soloff, Barber, Willett, arXiv 2506.02257 (2025) (assumption-free top-k stability; inflated top-k).
6. Prastalo & Jablonka, arXiv 2605.13826 (2026) (cross-sample prediction churn under bootstrap retraining; chemistry; remedies, no surrogate use).
7. Klüttermann et al., arXiv 2608.04613 (2026) (ranking AD algorithms is unreliable; seeds minor).
8. Hong & Kang, Knowledge-Based Systems 295:111842 (2024) (score distillation for AD; AUC-level fidelity, no floor).
9. Abson et al., arXiv 2609.15295 (2026) (KD of a normalising flow for the LHC L1 trigger; student-teacher fidelity reported as AUC within 0.1 pt, no decision-level or floor comparison). Illustrates the gap the rule fills.
10. Cormier, Milani Fard, Gupta, NIPS 2016 and Watson-Daniels et al., arXiv 2402.07745 (predictive churn; supervised, no top-k, no AD).

Target venues (tier and norms):
- **ECML-PKDD 2027 research track** (Eindhoven, 30 Aug - 3 Sep 2027; deadline not yet posted, the 2026 cycle closed 12 March 2026, so expect early-mid March 2027). Norms verified from the 2026 CFP: 16 pages LNCS including references, double-blind, three reviewers judging novelty / technical quality / impact / clarity, anonymous code strongly encouraged, 10 MB supplement. Expected baselines for this paper: ExCeeD-style threshold-only floor, Hong & Kang style score distillation, Spearman/MSE fidelity as the metric being replaced; expected data: ADBench-scale public tabular; expected reproducibility: seeds, paired two-sided tests, code release. Primary target.
- **TMLR** (rolling; no page limit; acceptance criteria are "claims supported by clear and convincing evidence" and "some of the audience would be interested"). Natural fit for a measurement-plus-rule paper with wins-only claims; reviewers will demand the invariants and the headroom curve. Secondary target, submit here if the ECML-PKDD timing slips.
- **DMKD journal** (Springer, rolling, Q1): accepts evaluation-methodology papers for AD; longer review cycle. Tertiary.
- PAKDD 2027 (Wellington, 29 Jun - 2 Jul 2027; deadline TBA, 2026 cycle was 15 Nov 2025) is the fallback if the full study is ready by November 2026.

Honest acceptance read. The strongest sibling to out-contribute is ExCeeD (same venue family, same perturbation idea, AD-specific). A reviewer accepts if: the floor is shown to be tight and certified by I1/I2 on 20+ datasets x 5 families with paired tests; the ExCeeD lower bound is shown to hold and to be loose enough that the empirical floor is needed; the rule changes a decision (students that look fine on Spearman fail the rule, Experiment E5); and the budget/capacity curve gives a usable stopping rule. A reviewer desk-rejects if the paper reads as "Stanton's independent baseline applied to anomaly detection" with no rule and no headroom map, or if the floor is only shown for stochastic teachers. Estimated chance at ECML-PKDD with all six experiments positive: moderate (a methodology paper with a crisp rule, a negative headline for current practice, and public data); at TMLR: good, given the evidence standard is met.

## 5. Headline results as falsifiable predictions

Units: alarm-set regret in [0,1]; "cell" = (dataset, teacher, seed); headline cells have eval >= 600.

- **P1 (floor tightness).** Median bootstrap refresh floor at k=5% is below 0.15 for kNN/KDE/OCSVM and below 0.30 for iForest(150 trees)/AE across headline cells; I1 holds (|indep - floor| median < 0.05). Smoke: 0.094 mean at 90% subsample.
- **P2 (rule fails current practice).** The trivial box + raw-regression student (1k queries, MLP 32-16) passes the bootstrap floor in fewer than 35% of headline cells at k=5%, and in fewer than 25% at k=1%, in every teacher family. Smoke: 1/15 and 2/15 at the tighter 90% floor.
- **P3 (either way: does budget close the gap?).** Student regret decreases monotonically with query budget 1k -> 50k and with capacity (32-16 -> 256-128-64). Whether the pass rate at 50k queries and the largest student exceeds 50% of headline cells is left open; both outcomes are reported. If it exceeds 50%, the deliverable is a budget stopping rule; if it does not, the deliverable is a certified impossibility for MLP students at 50k label-free queries.
- **P4 (floor ordering).** floor(99%) <= floor(90%) <= floor(bootstrap) <= floor(50%) in >= 90% of cells, and the iForest seed floor at 1000 trees is below the 100-tree floor in >= 90% of cells. Smoke: 0.208 -> 0.102.
- **P5 (analytic lower bound).** The ExCeeD threshold-only churn at the same budget is <= the empirical full-retrain floor in >= 90% of cells, and the ratio empirical/analytic has median >= 2 (the empirical floor is needed).
- **P6 (decision-level metric earns its keep).** Among cells with student-teacher Spearman >= 0.95, at least 30% still fail the floor rule at k=1%; Spearman and MSE predict pass/fail with AUROC < 0.8, so the rule is not a relabelling of score fidelity.
- **P7 (mechanism, cheap floor predictor).** Across cells, the floor correlates with the normalised score-gap density at the k-th cutoff (Spearman rho <= -0.5), so the floor can be predicted from one teacher fit without retraining; this is the 2506.02257 near-tie mechanism seen on AD data.

## 6. Experiments

| # | Experiment | Measured | Pre-registered invariant (must reproduce a known value) | Kill condition |
|---|---|---|---|---|
| E1 | Floor map: 35 datasets x 5 teachers x 5 seeds; bootstrap + 50/90/99% subsample floors; seed floor for iForest/AE; k in {1%, 5%} | Floor per cell, I1 gap, floor vs k | Degenerate: resample fraction 100% (no refresh) must give floor exactly 0.000 for deterministic teachers; I1: independent copy within 0.05 of floor | Median bootstrap floor > 0.5 for deterministic teachers on headline cells (bar uselessly loose) or I1 fails (median gap > 0.1) |
| E2 | Rule applied to the existing students: box + raw regression (headline), plus the archived shell/rank/TKCD students re-scored from saved CSVs where available | Pass rate per family and k; student regret minus floor (headroom) | Degenerate: a "student" that is the teacher itself must give regret 0 and pass every cell; a student that is an independent teacher refresh must pass about half the cells (regret ~ floor) | Pass rate > 80% everywhere (claim (b) resurrects; paper collapses to a one-liner) |
| E3 | Budget x capacity curve: queries {1k, 5k, 20k, 50k} x students {32-16, 128-64, 256-128-64}, 3 seeds, headline cells; Zhou-Xu-Hooker corpus-size test run alongside | Regret vs budget, pass rate vs budget; budget-to-pass per cell | Monotonicity: regret at 50k <= regret at 1k in >= 85% of cells; the 1k/32-16 corner must reproduce E2 within seed noise | Regret non-monotone in budget in > 30% of cells (student training is the bug, not the problem) |
| E4 | ExCeeD threshold-only floor vs empirical floor | Analytic floor, ratio, ordering | ExCeeD with gamma=0 and the true training scores must reproduce the published formula (p_hat^n) on a synthetic 1-D case; analytic <= empirical in >= 90% cells | Analytic floor >= empirical floor in > 30% of cells (then the empirical floor is redundant or mis-measured) |
| E5 | Score-fidelity metrics vs the rule: Spearman, MSE, AUC parity against pass/fail | Confusion of metric thresholds vs rule | Spearman = 1 implies regret 0 and pass (identity check) | Spearman predicts pass/fail with AUROC > 0.9 (the rule is a relabelling) |
| E6 | Mechanism: cutoff gap density vs floor; floor vs n_train (subsample n in {250, 500, 1000, 3000} on the 10 largest datasets) | rho(gap density, floor); floor vs n slope | Larger n must not increase the floor (median slope <= 0) | rho > -0.2 (no mechanism; floor stays a black box, paper still stands on E1-E3 but loses section 6) |

All experiments write one CSV row per cell as soon as it is ready, are resumable (skip rows that exist), and run with OMP/MKL threads = 1 under a process pool.

## 7. Deliverables

1. `experiment_noisefloor.py` promoted to the E1/E2 driver (bootstrap floor added; subsample fractions as a loop; incremental CSV; resume).
2. `experiment_budget_capacity.py` (E3), `experiment_exceed_floor.py` (E4), `experiment_rule_vs_fidelity.py` (E5), `experiment_floor_mechanism.py` (E6).
3. `refresh_floor.py`: a 60-line library function `refresh_floor(fit_fn, X_normal, X_eval, k, scheme, n_rep)` plus `accept(student_scores, teacher_scores, floor, k)`; released with the paper.
4. Figures: F1 floor map (35 x 5 heatmap with I1 whiskers); F2 pass/fail map for current students; F3 budget x capacity curves with the floor as a horizontal band; F4 ExCeeD analytic vs empirical scatter; F5 Spearman-vs-pass scatter; F6 gap-density vs floor.
5. Paper (16-page LNCS for ECML-PKDD, TMLR long form as the fallback), built with `paper-build`; citations checked with `bibtest`; prose pass with `paper-reviewer`.
6. Experiment registry entries for every cell and every killed path (negative paths stay out of the paper).

## 8. Phased plan (agent wall-clock and compute dollars; CPU only)

Measured cost basis: the 15-cell smoke took ~12 min single-threaded (~50 s per cell, dominated by the AE teacher and the four resample refits).

- **Phase 0, done (this session):** smoke, 15 cells, 12 min, $0. Outcome recorded in section 3.
- **Phase 1, E1 + E2 full map:** 875 cells x (1 bootstrap + 3 subsample fractions) x 5 resamples, about 3 CPU-min per cell -> ~45 CPU-hours. Local 8-core: ~6 h wall. RunPod 144-vCPU: ~25 min wall, ~$1-2. Agent time: ~30 min to extend the script and launch, ~30 min to analyse and check invariants.
- **Phase 2, E3 budget x capacity:** 12 configurations x ~300 headline cells x 3 seeds = ~11k student fits, 1-5 CPU-min each -> ~400 CPU-hours. RunPod 144-vCPU: ~3 h wall, ~$6-10. Agent time: ~45 min build + launch, ~45 min analysis.
- **Phase 3, E4 + E5 + E6:** reuse saved scores from Phase 1; E6 adds 4 subsample sizes x 10 datasets x 5 teachers x 5 seeds = 1000 cells, ~1 CPU-h total on the pod. Agent time: ~1.5 h in total, compute < $1.
- **Phase 4, write-up:** paper-build + bibtest + paper-reviewer loop. Agent time: ~3-4 h. Compute $0.
- **Total:** roughly 8-10 h of agent wall-clock across sessions, ~$10-15 cloud compute, or $0 if Phase 2 runs overnight on the local 8-core box (~2 days wall).

## 9. Top-3 risks and the cheapest smoke that de-risks each

1. **The floor is quantised and noisy at small k (small datasets, k=1%).** De-risk: on the three largest datasets (shuttle, campaign, mammography; eval >= 10k) compute the floor at k in {0.1%, 1%, 5%} with 20 resamples and check that the standard error of the floor is below 0.02; if it is not, raise n_rep or drop k=1% from the headline. Cost: 3 x 5 cells x 20 refits, ~20 CPU-min.
2. **The student's failure is an artefact of an under-trained small MLP, not of the problem.** De-risk: one cell (thyroid, kNN, where the student is already near the floor at 0.039 vs 0.015) and one hard cell (cardio, KDE, 0.588 vs 0.176) at 50k queries and a 256-128-64 student with 3 seeds. If the hard cell drops under the floor, P3 resolves toward "budget closes the gap" and the paper keeps the stopping-rule story; if not, P3 resolves the other way and the paper keeps the certified-gap story. Either outcome is publishable; the smoke only decides the emphasis. Cost: ~15 CPU-min.
3. **Reviewer reads it as Stanton's independent baseline plus ExCeeD.** De-risk: write the related-work paragraph (Stanton, ExCeeD, Perini ranking stability, Zhou-Xu-Hooker, Liang et al.) and have `paper-reviewer` attack it before any Phase 2 compute; obtain and read the Perini ranking-stability workshop paper (DOI 10.1007/978-3-030-65965-3_27), whose method was not verifiable from open text in this session. Cost: ~30 min agent time, $0.
