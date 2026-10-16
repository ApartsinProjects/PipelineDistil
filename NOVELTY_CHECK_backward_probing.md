# Deep novelty check: "backward probing" of anomaly-detection pipelines

Date: 2026-10-16. Three independent web-grounded checkers (decision-based attacks; counterfactuals and model inversion; autoencoder geometry and synthetic anomalies), primary sources verified.

## Idea under test
Distill a multi-stage anomaly-detection pipeline (autoencoder -> per-channel residuals -> per-channel calibration CDF -> threshold -> fusion) into one student without training data, by generating queries backwards from the decision: invert the monotone late stages exactly to a required residual, then reach it by a 1-D search along directions the autoencoder cannot reconstruct, with the real pipeline as the stopping oracle.

## Verdict: PARTIAL. Novel as a composition and as a problem setting; every primitive is already published.

### Primitives already owned by prior work (must be credited, cannot be the headline)
| Primitive | Prior work |
|---|---|
| 1-D (binary) search along a ray until the decision flips | Lowd & Meek KDD 2005; OPT / Sign-OPT (ICLR 2019/2020); HopSkipJump 2020 |
| Data-free distillation from a hard-label teacher via boundary line search | Wang, ZSDB3KD, ICML 2021 (https://arxiv.org/abs/2106.03310) |
| Boundary points as the student's transfer set | Heo et al. AAAI 2019; CAKE AISTATS 2024; BAM AISec 2024; counterfactual extraction (Aivodji 2020, DualCF 2022) |
| Gradient-free, hard-label, data-free extraction | DFMS-HL, CVPR 2022 |
| Stepping from a normal sample along the left null space of the decoder Jacobian | Vernekar et al. 2019 (https://arxiv.org/abs/1910.04241) |
| Off-manifold steps sized by calibration quantiles | GCOS, ICML 2026 (https://arxiv.org/abs/2603.08413) |
| Exact reconstruction of axis-parallel rules from counterfactual queries | Khouna, Ferry, Vidal, NeurIPS 2025 |
| Ingredients of the "latent-decode is circular" argument | Bauer 2026 (projection theory, arXiv 2606.15280); Kina 2026 (arXiv 2607.15919); Radhakrishnan et al. PNAS 2020 |

### Standing counter-claim to rebut
Kina 2026 (arXiv 2607.15919): boundary-seeking distillation fails for bottlenecked autoencoders (MNIST). Answer: we seek the boundary of the fused DETECTOR decision along unreconstructable directions, not per-feature reconstruction targets.

### What no checked source states (the surviving novelty)
1. **The problem setting is empty.** No extraction or data-free distillation of anomaly detectors or one-class models was found, and nothing for multi-stage, calibrated, multi-output pipelines.
2. **Exact inversion of the pipeline's own stored calibration/threshold stages** to turn a target decision into a per-channel residual target. Not found anywhere.
3. **The composition**: calibration-inverted residual target + decoder-normal direction + fused-decision oracle, as a transfer-set generator.
4. **The proposition**: the alarm boundary of a calibrated residual pipeline is reachable by 1-D searches along the autoencoder's normal bundle, and latent-decode probing provably cannot reach it.

## Consequences for the paper
- Lead with the PROBLEM (data-free unification of multi-output anomaly pipelines) and the evaluation rule (refresh-churn floor, per-anomaly-type), which are the open parts. Present the method as a principled composition; present "specification of abnormality" as motivation.
- Mandatory baselines / ablations on the same pipeline teacher: (a) Wang 2021 random-direction binary search; (b) Vernekar 2019 random-beta null-space steps; (c) GCOS quantile shells; (d) DFMS-HL and BAM adapted to the pipeline's hard-label output; (e) latent-sample-and-decode (Kina "noise pass"); (f) full-data replay+jitter (Bucila 2006 MUNGE).
- The win must be empirical: fewer queries and higher per-output / per-anomaly-type fidelity than (a)-(e). Novelty alone does not carry it.
- Risks: off-manifold residuals need not grow monotonically along a ray (Smidl et al. 2019), so the search must tolerate non-monotone segments; verify every flip on the real pipeline (Berkeley EECS-2021-126).
