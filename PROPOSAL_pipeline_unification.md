# PROPOSAL: Data-free unification of multi-output anomaly pipelines by probing the pipeline's own stages

Status: survivor of the paper-discovery adversarial cycle (v2 brief), hardened by a source-grounded break attempt on its two load-bearing assumptions (Section 3). Author: Alexander Apartsin. Date: 2026-10-16.

Working title: *The pipeline is its own data generator: stage-native probing for data-free unification of multi-output anomaly pipelines, judged by the refresh-churn floor.*

---

## 1. Framing: objective and assets

**Objective.** A deployed anomaly pipeline on multivariate sensor windows is a graph of stages: per-channel scaler -> autoencoder (AE or VAE) -> per-channel residuals -> per-channel calibration (empirical residual CDF or parametric tail) -> per-channel threshold -> fusion (k-of-n, weighted sum, max) -> alarm; side outputs: channel attribution (which residuals fired), change-point flag (window-to-window score jump), regression quantiles of the fused score. Operators want ONE compact student that reproduces ALL of these outputs for edge deployment, and the training data is gone (regulatory deletion, retention limits, vendor hand-over) or only 1-5% of windows survive.

**The question the paper answers.** When the data is gone, the pipeline's stored stages are the only generator left. Each stage carries sufficient statistics of what it saw: the scaler holds per-channel moments, the AE decoder holds the normal manifold, the calibration stage holds the residual marginals, the threshold and fusion stages hold the decision geometry. Can a transfer set generated *from the stages themselves* train a student that is acceptable by a decision-level criterion, and which stage must the probes target for the decision heads to be learnable at all?

**Assets reused (the moat).**
1. *Refresh-churn floor* (validated on 21 ADBench datasets x 5 detector families x 5 seeds; `results_floormap/`): a replacement is acceptable if its top-k alarm-set change versus the source is below the change the source itself shows when retrained on fresh normal data (median floor 0.05-0.25 at a 5% alarm budget; passing predicts kept detection quality 89% vs 51%; separates good surrogates at 75-98% pass from a weak student at ~12%). For stochastic stages the *seed-churn bound* (re-seed, no new data) equals the fresh-data floor, which gives a data-free proxy usable at deployment time.
2. *Multi-teacher black-box distillation harness* (`experiment_v2.py`, `experiment_pipelines.py`, `runpod_phase*_driver.py`): kNN, KDE, OCSVM, iForest, AE, composite and multi-output residual pipelines, analytic teachers of controllable geometry, RunPod drivers with incremental CSV persistence.
3. *Validated negatives* that shape the design (Section 2): uniform space-filling is near-unbeatable in low dimensions; smart placement pays only where space-filling cannot cover the informative region at budget (high-dimensional windows, thin manifolds), which is exactly the sensor-window regime; attribution resists distillation.
4. *Patent US20240013095A1* (SafeRide, user co-inventor): pipeline-level distillation for edge ML. The paper supplies the method's missing acceptance criterion and the data-free generator; the patent supplies the deployment framing.

**Behaviour change for practitioners.** The paper ends with a one-page prescription: what to store alongside a pipeline at training time (a *distillation certificate*: latent second moments, residual CDFs, stage order and rules) so that it can later be unified without data, and the floor test that certifies the result.

## 2. Why this and not the others (gravestones)

- *Stable-student / self-consistency criterion*: DEAD, tautology (a student that agrees with itself says nothing about the teacher).
- *Query-coverage law* ("fidelity is a function of informative-region coverage"): DEAD, not falsifiable beyond the already-published shell-sampling result and refuted on low-dimensional tabular data by the uniform baseline.
- *Standalone per-family audit* (injected anomaly families as the sole evaluation): demoted to FALLBACK; alone it is a benchmark paper without a method.
- *White-box compile of the pipeline into tensors*: this is Hummingbird (Nakandala et al., OSDI 2020) and is the exact-compile BASELINE, not a contribution; it covers scalers, PCA, iForest and trees, not OCSVM/KDE/LOF, and cannot shrink an AE.
- *Boundary-seeking data-free synthesis* for the AE stage: ill-posed for bottlenecked generators (Kina, arXiv 2607.15919); not pursued.
- *Pure latent-sample-and-decode as the transfer set*: killed by the break attempt in Section 3A; retained only as the degenerate control arm.

## 3. Load-bearing assumptions, break attempts, forced scope

### 3A. "Decoded latent probes reach the region where calibration, threshold and fusion are non-constant"

**Break attempt.** Let the AE be D∘E and the residual stage r(x) = x − D(E(x)). A pure latent probe is x′ = D(z). Its residual is r(x′) = D(z) − D(E(D(z))). Three bodies of evidence say this is close to zero for every z the pipeline will realistically produce:

- Overparameterised AEs store training points as attractors of the iterated map, and points on the decoder's range are near fixed points (Radhakrishnan, Belkin, Uhler, PNAS 2020, https://arxiv.org/abs/1909.12362). The idempotency literature frames the same fact as D∘E∘D ≈ D on the range of D, with exact identity unattainable only because the bottleneck discards information off the range (Fixed points of nonnegative networks, https://arxiv.org/abs/2106.16239; Enforcing Idempotency, ICML 2025).
- Plain AEs reconstruct inputs they never saw, which is precisely why reconstruction error fails as an OOD score (Yoon, Noh, Park, ICML 2021, https://proceedings.mlr.press/v139/yoon21c.html). Decoded points, which lie on the range by construction, are the easiest possible inputs to reconstruct.
- Sampling the latent beyond the aggregate posterior does not rescue this. VAE "prior holes" decode to corrupted outputs with uncontrolled residuals (Dai and Wipf, ICLR 2019, https://arxiv.org/abs/1903.05789; Aneja et al., NCP-VAE, https://arxiv.org/abs/2010.02917), and deep generators distort the latent geometry so latent radius does not map to a controlled input-space displacement (Arvanitidis, Hansen, Hauberg, ICLR 2018, https://arxiv.org/abs/1710.11379). So a latent shell gives a few probes with random, mostly small residuals, not a sweep of the calibration's tails.

Consequence: under pure latent-decode, nearly every probe lands below the lowest calibration quantile the threshold cares about; the threshold outputs a constant, the fusion outputs a constant, the decision heads of the student learn a constant and fail the floor on every series. **Verdict for the assumption as stated: BREAKS.** This is a fatal circularity, and it is exactly the failure that the prior art (2607.15919) reports as "latent-sample-and-decode is the effective baseline" only because its targets were the AE's own reconstructions, not a downstream decision.

**The fix, and why it is the actual method.** The non-constant region of the decision stages is defined in *residual space*, not in latent space, and the stages that define it are white-box and monotone: calibration F_c maps |r_c| to a quantile, the threshold compares the quantile to τ_c, fusion applies a known rule to the exceedance vector. So probes are constructed *backwards through the pipeline*:

1. Pick a target point in decision space: a fusion input pattern (which channels exceed, by how much) drawn to cover every decision region of the fusion rule, including both sides of each threshold, with the calibrated quantile u_c ~ Uniform over the calibration's support (and a stratum above the top stored quantile for the saturated tail).
2. Invert the calibration: |r_c|* = F_c^{-1}(u_c).
3. Pull back to input space: x = D(z) + δ with z from the latent summary (Section 3B) and δ chosen along directions the AE *cannot* absorb. To first order r(D(z)+δ) ≈ (I − J_D J_E) δ, so δ along the normal space of the decoder's tangent plane (equivalently the dominant residual principal directions, which the calibration stage has already summarised per channel, or the orthogonal complement of the decoder Jacobian at z) survives the AE at a controlled magnitude.
4. Query the real pipeline for the true outputs. Approximation error in step 3 only moves where the probe lands, never what it is labelled, so it costs coverage, not correctness. Probes that miss their target stratum are re-stratified from their measured outputs.
5. Add structured probes for family coverage: time-segment mixing of two decoded windows (collective/contextual residual structure), channel swaps between decoded windows (dependency structure), and single-step spikes (point structure). These are generated from decoded windows only, never from stored data.

This is what "stage-native probing" means from here on. The degenerate control (pure latent-decode) stays in the experiments as the arm that must fail, which converts the break into a pre-registered invariant.

### 3B. "A plain AE has no latent distribution to sample from when data is gone"

**Break attempt.** True as stated: a deterministic AE stores no latent density. What the pipeline *does* store by construction: scaler moments (input marginals), residual CDFs (residual marginals), thresholds and rules. None of these is a latent distribution. Candidates to close the gap:

- *Retained latent second moments* (mean and covariance of E(x) over training windows, d_z(d_z+3)/2 numbers; a small GMM if the normal regime is multimodal): the smallest honest addition and trivially storable at training time.
- *VAE prior*: available for free when the AE stage is variational, but prior holes mean prior samples are a biased generator; usable with the Section 3A residual targeting because labels come from the real pipeline.
- *Decoder-only inversion from scaler moments*: encode iid windows drawn from the scaler's N(mean, diag std); gives a latent cloud with zero stored statistics but biased toward the encoder's image of off-manifold noise. Kept as the "zero retained statistics" arm to measure what moments buy.
- *1-5% retained windows*: the budget-curve comparison; latent moments are then estimated from them.

**Verdict: MUST BE SCOPED.** The honest claim is "data-free given the pipeline plus a latent second-moment summary (the distillation certificate)", with the zero-statistics arm and the 1% arm bracketing it. The brief's "data-free" wording in the DAFL/ZSKT sense is not honest for a plain AE and is not claimed. The scoping is a feature: the paper's practitioner deliverable is exactly the list of what to store.

### 3C. Forced scope after the break attempts

- Method = residual-targeted, backwards-constructed probes (stage-native), not latent-decode.
- Data regime = pipeline + latent moments (primary), bracketed by zero statistics and 1% windows.
- Evaluation = refresh-churn floor in the lab (benchmark data used for evaluation only, as every data-free KD paper uses its test set) plus the seed-churn bound as the deployment-time proxy.
- Attribution head reported as a side output with its own floor; the prior negative (attribution resists distillation) is respected and not reproposed as a win.

## 4. Novelty claim and strongest threats

**Claim.** For a stored anomaly pipeline whose training data is gone, the transfer set that makes a single multi-head student pass the refresh-churn floor must be constructed backwards from the pipeline's decision stages into input space; probes that only sample the generative stage reproduce the AE and nothing downstream. The pipeline is its own data generator, and the floor decides whether that generator suffices.

**Threats and why the claim stands.**
- *Bucila, Caruana, Niculescu-Mizil, KDD 2006* (https://cs.cornell.edu/~caruana/compression.kdd06.pdf): MUNGE (replay plus jitter of real unlabelled data) nearly matches real data; RANDOM gets worse. Needs data; it is our full-data arm (i), and the paper's result is relative to it. The stage-native probes are not jitter of data because there is no data; they are jitter of decoded manifold points in directions chosen by the downstream stages.
- *Kina, arXiv 2607.15919 (July 2026)*: boundary-seeking data-free synthesis fails for bottlenecked AEs; latent-sample-and-decode is the effective baseline; MNIST only, AE-output targets only. We agree on the failure and go further: for *downstream decision* targets latent-decode also fails (Section 3A), and the fix is residual targeting. No time series, no multi-output, no acceptance criterion in that paper.
- *Li et al., NeurIPS 2023* (https://proceedings.neurips.cc/paper_files/paper/2023/file/c43b987f23fd5ea840df2b2be426315c-Paper-Conference.pdf): global rule surrogate of black-box AE/VAE/OCSVM/iForest with finite-difference explorers beyond the data. Single detector, single score output, rule surrogate, requires data to start exploration. We unify a multi-stage multi-output pipeline without data and judge by churn, not rule fidelity.
- *Nayak et al., ICML 2019* (https://proceedings.mlr.press/v97/nayak19a.html): data impressions synthesised from the teacher's parameters (classification softmax, Dirichlet over class similarities). Closest in spirit ("the model is its own generator"); classification only, needs class structure, no pipelines.
- *Zhou and Chiam, ESWA 2023* (https://arxiv.org/abs/2301.04338): data-free KD for regression networks via adversarial generator; single scalar regression, no decision stages, no time series.
- *Hong and Kang, KBS 2024, doi 10.1016/j.knosys.2024.111842; Pol et al., arXiv 2310.06047*: AD score distillation with training data. Our full-data arm.
- *Knowledge amalgamation* (Luo et al., IJCAI 2019, https://www.ijcai.org/proceedings/2019/428; Shen et al., ICCV 2019; Ye et al., IJCAI 2019, doi 10.24963/ijcai.2019/573): heterogeneous teachers into one student using unlabelled data; image classification; no data-free setting, no pipelines.
- *ASTER, ICPR 2026 (arXiv 2604.13924); GenIAS (arXiv 2502.08262)*: latent-space pseudo-anomaly generation to *train* a detector from real normal data; the generator is a learned detector component, not a transfer set for distilling a stored pipeline. They confirm that latent perturbations decode to usable anomalous windows, which supports the family-coverage probes in 3A step 5.

## 4b. Venue fit

| Sibling | Venue, year | URL |
|---|---|---|
| Bucila et al., Model compression (MUNGE) | KDD 2006 | https://cs.cornell.edu/~caruana/compression.kdd06.pdf |
| Nayak et al., Zero-shot KD (data impressions) | ICML 2019 | https://proceedings.mlr.press/v97/nayak19a.html |
| Li et al., global rule surrogate of black-box AD | NeurIPS 2023 | link above |
| Hong and Kang, Score distillation for AD | Knowledge-Based Systems 295, 2024 | doi 10.1016/j.knosys.2024.111842 |
| Zhou and Chiam, data-free KD for regression | Expert Systems with Applications 227, 2023 | https://arxiv.org/abs/2301.04338 |
| Luo et al., Knowledge amalgamation from heterogeneous networks | IJCAI 2019 | https://www.ijcai.org/proceedings/2019/428 |
| Kina, failure of boundary-seeking distillation in bottlenecked AEs | arXiv 2607.15919, 2026 | https://arxiv.org/abs/2607.15919 |
| Liu and Paparrizos, TSB-AD | NeurIPS 2024 D&B | https://proceedings.neurips.cc/paper_files/paper/2024/file/c3f3c690b7a99fba16d0efd35cb83b2c-Paper-Datasets_and_Benchmarks_Track.pdf |
| Goswami et al., unsupervised model selection via injection | ICLR 2023 | https://arxiv.org/abs/2210.01078 |
| Stanton et al., Does KD really work | NeurIPS 2021 | https://arxiv.org/abs/2106.05945 |
| Nakandala et al., Hummingbird | OSDI 2020 | https://www.usenix.org/conference/osdi20/presentation/nakandala |

**Target 1: TMLR** (rolling, OpenReview, double blind, mandatory style file, no expanded conference papers; https://jmlr.org/tmlr/author-guide.html). Norm: claims must be correct and supported, interest to a sub-community suffices; reviewers reward pre-registered invariants and honest negatives. Fit: the churn-floor paper line is already known to TMLR reviewers (the harness carries "TMLR review Exp 7" annotations). Honest read: desk-reject risk low; accept probability moderate, roughly 45-55% conditional on the smoke in Section 6 passing on TSB-AD-M with at least two real pipeline configurations. The two reviewer attacks to pre-empt: "the 1% replay arm does as well, so the data-free machinery is unnecessary" (answered by P3, which is allowed to go either way and reshapes the paper if it does) and "synthetic pipelines" (answered by building the pipelines from TSB-AD-M's own strong baselines: AE/VAE plus iForest/PCA fusion).

**Target 2: ECML-PKDD 2027 research track** (Eindhoven, 20-24 Sept 2027; 2026 deadline was 12 March, acceptance 280/1150 ≈ 24%; 2027 CFP not yet posted). Fit: applied ML on time series with a crisp method; page limit forces dropping the attribution side-study. Accept read: 25-35%.

**Not targeted:** KDD ADS (requires a deployed system with real performance data), NeurIPS main (the contribution is a scoped engineering law, not a general learning result).

## 5. Falsifiable predictions

Notation: a *cell* is (series, pipeline config, seed). Pass = student's top-5% alarm-set churn versus the source below the source's own fresh-data refresh churn. Heads: residual (per channel), calibrated score, threshold flags, fusion alarm, attribution, change-point flag, score quantiles.

- **P1 (degenerate invariant, sign certain).** Pure latent-decode probes: ≥ 95% of probes have fused calibrated score below the median of the fusion input's training distribution; the fusion-head output is constant on ≥ 95% of probes; the student's fusion head fails the floor on ≥ 95% of cells. If this fails, the circularity analysis in 3A is wrong and the whole framing must be re-derived.
- **P2 (headline).** Stage-native probes with latent moments only pass the fusion-head floor on at least 0.7 of the cells where full-data replay+jitter passes (ratio ≥ 0.7). Kill: ratio < 0.5.
- **P3 (can go either way).** Moments-only versus 1% retained windows plus stage-native probes: fusion-head pass rates within 10 points. If 1% windows beat moments-only by more than 20 points, a Gaussian latent summary is insufficient (multimodal normal regimes), the GMM certificate is tested, and the paper's prescription changes from "store moments" to "store a k-component latent mixture".
- **P4 (invariant from the validated negative, inverted regime).** Uniform window-space probes at the same budget fail the fusion-head floor on ≥ 90% of cells (window dimension ≥ 64). Kill for the framing: if uniform passes ≥ 50%, the sensor-window regime is not high-dimensional enough to justify placement and the prior tabular negative applies.
- **P5 (per-family, held out from probing).** Point and global families pass at a rate within 10 points of the aggregate; collective and dependency families pass ≥ 15 points lower without mixing probes; adding window-mixing and channel-swap probes lifts collective/dependency pass by ≥ 10 points. The lift sign can fail; the gap sign is predicted.
- **P6 (seed-churn bound).** On stochastic pipeline stages the seed-churn bound equals the fresh-data floor within 0.03 median absolute difference (replicating the ADBench finding on TSB-AD-M), and a student's own reseed churn is below the floor on ≥ 90% of passing cells. The self-copy control (student = the pipeline itself) gives churn 0 on 100% of cells.
- **P7 (side output).** The attribution head passes its floor at a rate at least 20 points below the fusion head under every arm, including full data. Reported as scope, not as a win.

## 6. Experiments

Each row: experiment | measured | ONE pre-registered invariant (a degenerate setting that reproduces a known value) | kill condition.

| # | Experiment | Measured | Pre-registered invariant | Kill condition |
|---|---|---|---|---|
| E0 SMOKE | 5 TSB-AD-M series + 2 analytic multivariate teachers; pipeline scaler -> AE -> residuals -> empirical-CDF calibration -> per-channel threshold -> 2-of-n fusion (+ attribution, change-point, quantiles). Arms: (i) replay+jitter with full data; (ii) stage-native probes, zero data, retained latent moments; (ii-0) pure latent-decode control; (iii) uniform window-space; (iv) 1% windows + stage-native. Budget 20k probes, 3 seeds. | Per-head floor pass rate; probe coverage of fusion decision regions (entropy of fusion output over probe set) | (iii) fails the floor on the fusion head on every series; self-copy student passes with churn 0; (ii-0) fusion-output entropy ≈ 0 (P1) | (ii) fails on fusion or threshold head on every series while (i) passes on them: method dead, fall back to the partial-data budget curve |
| E1 | Residual-landing audit: for each arm, histogram of per-channel calibrated quantiles of probes; fraction in each fusion decision region | Coverage per region, per arm | Latent-decode control puts ≥ 95% of probes below the training median (P1) | Stage-native covers any decision region with < 5% of probes after re-stratification: fix the inversion before any student is trained |
| E2 HEADLINE | All TSB-AD-M multivariate series (200), 3 pipeline configs (AE k-of-n; VAE weighted-sum; AE+iForest max-fusion as the known-hard case), 5 seeds, arms (i)-(iv) + (ii-0) | Per-head pass rate; pass ratio (ii)/(i); P2, P4 | Full-data arm (i) reproduces the ADBench floor-pass range (75-98%) on smooth fusion; max-fusion pipeline is undistillable by the small student under every arm (known negative reproduced) | P2 ratio < 0.5 on smooth-fusion configs |
| E3 | Budget curve: 0 stats, moments-only, GMM-3, 1%, 5%, 100% windows, each + stage-native | Pass rate vs retained information (bytes) | 100% windows + stage-native ≥ replay+jitter (more information cannot hurt; if it does, a code bug) | 1% beats moments-only by > 20 points AND GMM-3 does not close the gap: prescription becomes "retain 1% windows", paper narrows |
| E4 | Per-family floors with injected families held out from probing (point, contextual, collective; local, global, dependency, cluster), TSB-AD-M subset of 60 series | Pass rate by family; mixing-probe ablation (P5) | Injected family with zero amplitude reproduces the aggregate floor exactly | Collective/dependency pass < 30% under every arm including full data: scope the claim to point/global families |
| E5 | Seed-churn bound as deployment proxy (P6) | Median |seed churn − fresh-data floor|; student reseed churn | Deterministic pipeline (PCA residual, fixed thresholds) has seed churn exactly 0 | Proxy error > 0.10 median: drop the deployment-time proxy, keep the lab floor |
| E6 | Baselines: Hummingbird exact compile where applicable (scaler, PCA, iForest) for latency and size; Li et al. style rule surrogate on the fused score; single-head students per output vs one multi-head student | Size, latency, pass rate | Exact compile passes the floor with churn 0 wherever it applies | Multi-head student loses > 10 points to per-head students on fusion: report per-head ensemble as the deliverable |

All runs append one CSV row per cell on completion and skip existing rows on resume.

## 7. Deliverables

1. Paper (TMLR format) with the headline law, the budget curve, the per-family audit, and the practitioner prescription (distillation certificate: latent moments or k-GMM, residual CDFs, stage order and rules).
2. Open harness extension: `experiment_unify.py` (stage-native probe constructor with calibration inversion and Jacobian-normal pushes; multi-head student; per-head floor), `runpod_unify_driver.py`, TSB-AD-M loader.
3. Registry of invariants and kill results (`results_unify/invariants.txt`), including the latent-decode gravestone with its measured numbers.
4. Figures: residual-landing histograms per arm (the circularity made visible), pass-ratio vs budget, per-family pass matrix.

## 8. Phased plan (agent wall-clock and compute cost)

Agent time is my execution time on tool calls and babysitting; compute is RunPod only.

| Phase | Work | Agent wall-clock | Compute |
|---|---|---|---|
| 0 Smoke (E0, E1) | Build probe constructor and multi-head student on top of `experiment_v2.py`; 5 series + 2 analytic teachers; local CPU | 3-4 h | $0 |
| 1 Data | TSB-AD-M download (TheDatumOrg/TSB-AD, released Nov 2024) into an isolated directory; loader with length/channel filters | 30-45 min agent; external download 10-30 min | $0 |
| 2 Headline (E2, E3) | 200 series x 3 configs x 5 seeds x 6 arms; small AEs train in seconds; RunPod 1x RTX A5000 or a 32-vCPU pod, chunked drivers with incremental CSVs | 4-6 h agent over 2 sessions | 8-12 h pod, $15-30 |
| 3 Families and proxy (E4, E5) | 60-series subset, 7 families, mixing ablation; seed-churn bound | 3 h agent | 3-4 h pod, $8-12 |
| 4 Baselines (E6) | Hummingbird compile, rule surrogate, per-head students | 2 h agent | $0-5 |
| 5 Paper | Draft, paper-reviewer pass, bibtest on all citations, paper-build HTML/DOCX, TMLR bundle via html2tex | 5-6 h agent | $0 |
| Total | | 18-22 h agent | $25-50 |

Phase 0 gates everything: its kill condition ends the direction and switches to the budget-curve fallback (E3 only, still a TMLR-shaped paper on what to retain).

## 9. Top-3 risks and the cheapest smoke for each

1. **Circularity survives the fix** (residual pushes get absorbed by the AE because the decoder Jacobian is near full rank in window space, so δ has no normal direction to live in). Smoke: on one series, measure ||r(D(z)+δ)|| / ||δ|| for δ along the top-10 residual principal directions versus random directions; expect ≥ 0.5 for principal directions and the ratio ordering principal > random. 15 minutes local CPU.
2. **Moments-only is not enough** (multimodal normal regimes in TSB-AD-M; Gaussian latent summary decodes to regime blends). Smoke: on 5 series, compare the latent-moment Gaussian's decoded windows against 1% real windows by the pipeline's own fused score distribution (two-sample KS on the score); if KS > 0.3 on ≥ 3 series, switch the certificate to GMM-3 before Phase 2. 20 minutes local CPU.
3. **The floor is too loose or too tight on TSB-AD-M** (series too short for a meaningful fresh-data retrain, or stochastic stages so noisy that any student passes). Smoke: compute the fresh-data floor and the seed-churn bound on 10 series for the AE pipeline; require median floor in [0.03, 0.40] and a weak MLP student (the known ~12% passer) to pass < 30%; otherwise filter series by length and raise the alarm budget to 10%. 30 minutes local CPU.

---

Gravestones (one line each, kept out of the paper body): stable-student criterion (tautology); coverage law (unfalsifiable, refuted in low dimensions); boundary-seeking latent synthesis (ill-posed, 2607.15919); pure latent-sample-and-decode transfer set (circular, Section 3A, measured in E1); attribution as a win (resists distillation, reported as scope).
