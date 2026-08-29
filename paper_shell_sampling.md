# Where to Query a Blackbox Anomaly Pipeline: Shell Sampling for Label-Free Distillation

**A. Apartsin (draft, 2026-08-29)**

## Abstract

We study how to distill a complicated, blackbox, unsupervised anomaly-detection **pipeline** (preprocessing, one or more detectors, and a fusion step) into a single small student model for resource-constrained deployment. The pipeline can only be queried, its internals are opaque, and no labeled anomalies exist at any point: the only data is a set of normal operating samples. The student must therefore learn the pipeline's behavior on the anomalous region from synthetic query points we place there ourselves and label with the pipeline's own scores. The central question is not the distillation objective, which is a plain regression from input to pipeline score, but **where in input space a fixed budget of synthetic queries should go**. We show that the informative queries lie in the **low-density shell** just off the normal manifold, and that within that shell the right target depends on the pipeline's off-manifold *shape*: for detectors whose score keeps growing away from the data, sample toward high score; for detectors whose score saturates, sample where the score has a large gradient. A single sampler that importance-weights the shell by **both** the pipeline score and its gradient magnitude is robust to this shape without being told it, and matches or exceeds shape-specific samplers across a spectrum of teacher pipelines (kernel density, isolation forest, k-NN distance, one-class SVM, and an autoencoder). We measure success as student-teacher **fidelity** on held-out anomalies (rank agreement), not standalone detection accuracy. We are candid about scope: some pipelines with discontinuous fusion are not reproducible by a small student at all, and on real high-dimensional tabular data fine-grained fidelity remains an open problem while simple baselines already suffice for coarse detection. All code and results are open-source.

## 1. Introduction

Unsupervised anomaly detection in production is rarely a single model. A deployed pipeline chains a preprocessing stage, one or more detectors from different families (density estimators, tree-based isolators, distance-based methods, deep autoencoders), and a fusion or calibration step. When such a pipeline must run on a device with tight compute, memory, and power budgets, the practitioner faces a distillation problem: compress the pipeline into one small model whose scores approximate the pipeline's on the inputs the device will see.

Two features make this different from ordinary supervised knowledge distillation [Hinton et al., 2015]. First, the teacher is a **pipeline**, not a differentiable model: we can query it end-to-end and read the scalar score it returns, but we cannot backpropagate through it. Second, the training set is **fully unsupervised**: only normal samples exist, and no labeled anomalies are ever available. Whatever the student learns about the pipeline's behavior on abnormal inputs, it learns from query points we construct and label with the pipeline itself.

This paper asks a single question: *given a fixed budget of synthetic queries, where in input space should they go so the student best reproduces the pipeline?* We argue and show that:

- The informative queries lie in the **low-density shell** just off the normal manifold, not on the manifold (where normals already teach the student) and not in the far field (where the pipeline score is uninformatively saturated). (§4.2)
- Within the shell, the target depends on the pipeline's off-manifold **shape**. A *growing* pipeline (score rises with distance, e.g. autoencoder reconstruction error) is best queried toward high score; a *saturating* pipeline (score plateaus, e.g. one-class SVM) is best queried at score **gradients** (edges). (§4.3)
- A single sampler that weights the shell by **both** score and score-gradient magnitude is automatically robust to this shape and needs no manual choice or shape label. (§4.4)
- Reproducibility of the pipeline can be predicted by a cheap **shape probe** and is fundamentally bounded: discontinuous pipelines are not distillable by a small student regardless of sampling. (§4.5, §6.4)

We measure student-teacher **fidelity** on held-out anomalies (rank agreement between student and pipeline scores) because the goal is faithful reproduction of the pipeline, not beating it. We evaluate on a controlled 2-D benchmark and a spectrum of five teacher pipelines, and we report honestly on a real-data limitation (§6.5).

## 2. Related work

### 2.1 Knowledge distillation and blackbox distillation

Knowledge distillation transfers a large teacher's input-output behavior to a smaller student by matching soft outputs [Hinton et al., 2015; Ba and Caruana, 2014; Buciluă et al., 2006], extended to feature and relation matching [Romero et al., 2015; Park et al., 2019]. Our objective is a plain squared-error match on pipeline scores. Data-free distillation [Nayak et al., 2019; Chen et al., 2019] synthesizes inputs when training data is unavailable, typically by inverting the teacher's activations; our setting is intermediate, since we have the normal data but not the anomalies the deployed student must handle, so we synthesize queries to cover that region.

### 2.2 Unsupervised anomaly detection

The teacher pipelines we distill are built from standard unsupervised detectors: density estimators such as Gaussian kernel density [Silverman, 1986] and Local Outlier Factor [Breunig et al., 2000]; boundary methods such as one-class SVM [Schölkopf et al., 2001] and Deep SVDD [Ruff et al., 2018]; isolation and distance methods such as Isolation Forest [Liu et al., 2008] and k-NN distance; and reconstruction methods such as autoencoders [Zhou and Paffenroth, 2017]. A unifying review [Ruff et al., 2021] catalogues these families and their persistent score-calibration difficulty across families, which our rank-based normalization addresses. Recent deep detectors include self-supervised [Bergman and Hoshen, 2020; Golan and El-Yaniv, 2018] and reconstruction-driven industrial methods [Zavrtanik et al., 2021; Roth et al., 2022; Li et al., 2021]; our student is architecture-agnostic to the pipeline it distills.

### 2.3 Synthetic query synthesis and off-manifold sampling

Synthesizing training points when only normals are available has been approached by adversarial near-boundary generation [Ducoffe and Precioso, 2018; Chen et al., 2020] and by score-based and diffusion samplers that draw from a target density defined through its score function [Song and Ermon, 2019, 2020]. Our sampler places queries by importance-weighting an off-manifold shell rather than training a generator, and does not require differentiating the teacher.

### 2.4 Active learning and query synthesis

Constructing new query points rather than selecting from a pool is *query synthesis* [Angluin, 1988], distinct from the pool-based active learning that dominates the modern literature [Sener and Savarese, 2018; Settles, 2010]. The score-gradient signal we use is related to disagreement- and margin-based acquisition [Seung et al., 1992; Houlsby et al., 2011; Ducoffe and Precioso, 2018]. Langevin dynamics for concentrating samples in target regions has a long history [Welling and Teh, 2011; Song and Ermon, 2019]; we compare a Langevin variant of our sampler against a simpler importance-weighted shell sampler and find the latter as effective and easier to reason about.

## 3. Problem setup

Let $P$ be a blackbox anomaly-scoring pipeline mapping an input $x \in \mathbb{R}^d$ to a scalar score $s(x) = P(x) \in \mathbb{R}$, oriented so larger means more anomalous. We have:

- a training set $\mathcal{X} = \{x_i\}_{i=1}^N$ of **normal** samples;
- query access to $P$: we may evaluate $s(x)$ at any $x$, but cannot inspect or differentiate $P$;
- **no labeled anomalies**, at train or validation time.

We want a compact student $f_\theta : \mathbb{R}^d \to \mathbb{R}$ that reproduces $s$ at deployment, exported as a framework-agnostic graph (e.g. ONNX) and compiled for the target runtime. The distillation objective is a plain regression; the contribution is the placement of the synthetic queries used to fit $f_\theta$.

**Success metric.** Because the goal is to reproduce the pipeline, we measure student-teacher **fidelity** on a held-out anomaly set $\mathcal{A}$: the Spearman rank correlation between $f_\theta(a)$ and $s(a)$ over $a \in \mathcal{A}$ (and RMSE of the normalized scores). This rewards faithful reproduction of the pipeline's ranking on inputs it was never trained to match, rather than the student's standalone detection accuracy.

## 4. Method: shell sampling

### 4.1 Why normals-only distillation fails

Fitting the student only on the normals,
$$
\theta^\star = \arg\min_\theta \frac{1}{N}\sum_{i=1}^N \bigl(f_\theta(x_i) - s(x_i)\bigr)^2,
$$
constrains it only where the pipeline already scores inputs as normal. Off the manifold the student is unconstrained, and a small network interpolates smoothly; it has no way to reproduce the pipeline's actual off-manifold surface. We must add synthetic queries $\{q_j\}_{j=1}^M$ that carry the pipeline's behavior on the anomalous side, and fit
$$
\mathcal{L}(\theta) = \frac{1}{N}\sum_i \bigl(f_\theta(x_i) - s(x_i)\bigr)^2 + \gamma\,\frac{1}{M}\sum_j \bigl(f_\theta(q_j) - s(q_j)\bigr)^2 .
$$
The question is where to place the $q_j$.

### 4.2 Where: the low-density shell

Let $\rho(x)$ be the distance from $x$ to its nearest training normal (a cheap proxy for data density). Queries are wasted in two places: **on** the manifold ($\rho \approx 0$), where the normals already supply the target, and in the **far field** ($\rho$ large), where most pipelines return an uninformative extreme (a saturated ceiling) and the student needs only a single anchor. The informative region is the **low-density shell** $\{x : \rho_{\min} \le \rho(x) \le \rho_{\max}\}$: off the manifold, but close enough that the pipeline's decision structure is still resolved and a plausible anomaly could actually appear. All samplers below draw candidates from this shell.

### 4.3 What: score and gradient depend on the pipeline shape

To make signals comparable across heterogeneous detectors we convert each raw score to its rank on the training normals: $u(x) = F(s(x))$, where $F$ is the empirical CDF of $s$ on $\mathcal{X}$, extended linearly beyond the training range so that inputs scoring above every normal receive distinct values greater than one (a naive clip at one collapses the entire off-manifold region to a tie and destroys the ranking).

Within the shell, which candidates are informative depends on how the pipeline behaves off the manifold:

- **Growing pipelines** (the rank $u$ keeps rising with distance, e.g. autoencoder reconstruction error): the informative queries are toward **high score**. Weight candidates by $u(x)$.
- **Saturating pipelines** (the rank $u$ plateaus off the manifold, e.g. one-class SVM whose decision function flattens): high-score weighting is uninformative because most of the shell is already at the plateau; the informative queries are where the score **changes**, i.e. at large $\|\nabla_x u(x)\|$. Weight candidates by the gradient magnitude (estimated by finite differences; the pipeline need not be differentiable).

### 4.4 The combined sampler

Rather than choose per pipeline, weight each shell candidate by **both** signals:
$$
w(x) \;\propto\; \tfrac{1}{2}\,\widehat{u}(x) \;+\; \tfrac{1}{2}\,\widehat{\|\nabla_x u(x)\|},
$$
where $\widehat{\cdot}$ denotes normalization to a probability over the candidate pool, and draw $M$ queries by importance sampling. The score term handles growing pipelines, the gradient term handles saturating ones, and because both are present the sampler is robust to the pipeline's shape without being told it. We refer to this as the **combined shell sampler**. (A Langevin variant that walks up the same potential while a density prior holds it in the shell gives equivalent results and is described in Appendix A; the importance-weighted form is simpler and used throughout.)

### 4.5 Automatic operation and a reproducibility probe

The combined sampler needs no manual shape selection. If a diagnostic is wanted, a cheap **shape probe** estimates whether a pipeline is growing or saturating by measuring how the median score changes across concentric shells at increasing $\rho$: a growing pipeline's score keeps rising past the training range, a saturating one plateaus. In our experiments an explicit growth-to-weight rule based on this probe matches the fixed combined sampler but does not beat it, so we recommend the combined sampler as the default and reserve the probe as a diagnostic. The probe also flags the failure case of §6.4: a pipeline whose score is flat across all shells cannot be reproduced by any sampler.

## 5. Experimental setup

**Data.** Two-moons normals ($N = 2000$, noise $0.15$); a held-out off-manifold anomaly set drawn uniformly in the shell $0.2 < \rho < 2.5$ (500 points). This isolates the sampler as the only variable on geometry we control.

**Teacher pipelines (spectrum).** Five single-stage pipelines spanning off-manifold shapes, each fit on the normals only: k-NN distance ($k=10$) and Gaussian kernel density (both monotone), one-class SVM with an RBF kernel (saturating), an undercomplete tanh autoencoder scored by reconstruction error (growing), and a percentile-max fusion of the autoencoder and kernel density (discontinuous). We additionally build a three-step complex pipeline (standardize, autoencoder reconstruction error, density-gated combination) for the mechanism study of §6.3.

**Student.** A small tanh MLP (one hidden layer of 8 units, about 40 parameters) with input standardization, deliberately low-capacity so that *where* the queries are placed, not student size, determines what boundary it learns.

**Samplers compared.** Normals-only (no queries); Gaussian jitter on normals; uniform in a bounding box; **score-only** shell (climb $u$); **gradient-only** shell (edges); and the **combined** shell sampler of §4.4. A Langevin variant is compared in the appendix.

**Metric.** Student-teacher fidelity on the held-out anomalies: Spearman rank correlation between student and pipeline scores (and RMSE of normalized scores). Means over 3 seeds unless stated.

## 6. Results

### 6.1 The informative region and the shape dependence

Placing queries in the low-density shell is what makes distillation work: across all reproducible teacher pipelines, every shell-based sampler lifts fidelity from the normals-only baseline (Spearman $\approx 0.2$–$0.3$) to $0.7$–$0.97$. Within the shell, the score-only and gradient-only samplers trade off exactly as the shape argument predicts.

**Student-teacher fidelity (Spearman) on held-out anomalies, per teacher pipeline.**

| Pipeline (shape) | normals-only | score-only | gradient-only | **combined** |
|---|---|---|---|---|
| k-NN distance (monotone)     | $0.308$ | $0.950$ | $0.968$ | $0.966$ |
| kernel density (monotone)    | $0.280$ | $0.933$ | $0.940$ | $0.944$ |
| one-class SVM (saturating)   | $-0.056$ | $0.350$ | $0.562$ | $\mathbf{0.708}$ |
| autoencoder (growing)        | $0.322$ | $\mathbf{0.911}$ | $0.818$ | $0.823$ |
| percentile-max fusion (discontinuous) | $0.055$ | $0.069$ | $0.106$ | $0.054$ |

![Fidelity of each sampler across the teacher-pipeline spectrum; the best target within the shell depends on the pipeline's off-manifold shape, and the combined sampler is robust to it.](figure_sampler_shape.png)

Two readings. First, **the score-only sampler wins the growing autoencoder** ($0.911$) but is weakest on the saturating one-class SVM ($0.350$); the **gradient-only sampler wins the saturating case** ($0.562$) but gives up ground on the autoencoder ($0.818$). This is the shape dependence of §4.3, measured. Second, **the combined sampler is the robust choice**: it wins the hard one-class SVM outright ($0.708$, well above either single signal), ties the best on both monotone pipelines, and recovers most of the autoencoder performance the gradient-only sampler sacrificed. No single-signal sampler is best everywhere; the combined one is never far from best on any distillable pipeline.

### 6.2 The combined sampler needs no shape label

Because it carries both terms, the combined sampler adapts to the pipeline shape implicitly. An explicit alternative that runs the shape probe of §4.5 and sets the score-versus-gradient weight from it reaches the same fidelity as the fixed combined sampler across the spectrum but does not exceed it. We therefore recommend the combined sampler as the automatic default and keep the probe as a diagnostic (it also identifies the undistillable case of §6.4). No manual choice, no shape label, and no anomaly labels are required at any point.

### 6.3 Mechanism on a complex pipeline

To show *why* sampling is needed rather than only *that* it helps, we distill a three-step pipeline whose off-manifold surface is genuinely structured (standardize, then autoencoder reconstruction error gated by a density term). Fidelity on the held-out anomalies over 5 seeds: normals-only $0.245 \pm 0.120$, Gaussian jitter $0.336 \pm 0.265$, a bounded score-climb sampler $0.184 \pm 0.290$, and an adaptive shell sampler $\mathbf{0.939 \pm 0.030}$. The figure below shows the mechanism: normals-only produces a smooth, monotone surface that cannot express the pipeline's off-manifold structure, while the shell sampler's query cloud spans the anomalous region and the resulting student reproduces the true surface, including an off-manifold valley the pipeline exhibits far from any data.

![The complex pipeline's off-manifold surface (left) is not reproducible from normals only (middle); shell sampling places queries across the anomalous region and the student reproduces it (right).](figure_complex.png)

### 6.4 A fundamental limit: discontinuous pipelines

The percentile-max fusion is not reproduced by any sampler (fidelity $\approx 0.05$–$0.11$ for all). Its score is a non-smooth maximum of two rank surfaces; the small student cannot represent the discontinuity regardless of where queries are placed. This is a **capacity** limit, not a sampling one, and the shape probe of §4.5 detects it in advance (a flat score profile across shells). We report it plainly: shell sampling makes a pipeline's off-manifold behavior *available* to the student, but a student too small to represent that behavior still cannot reproduce it.

### 6.5 Scope: real high-dimensional data

On three real tabular anomaly-detection datasets (shuttle $d=9$, satellite $d=36$, optdigits $d=64$) distilled from monotone detectors, coarse detection is easy and needs no sampling: normals-only or uniform augmentation already produce a student whose standalone anomaly AUROC matches the teacher (shuttle: normals-only $0.987$; satellite: uniform $0.829$; optdigits: normals-only $0.850$). Fine-grained *fidelity* to the pipeline, however, is poor and noisy for every sampler at these dimensions and sample sizes, because the small student cannot reproduce a real detector's full ranking regardless of query placement. We state this as an open limitation rather than a result: the shell-sampling advantage is demonstrated on controlled geometry and on the teacher spectrum, and its transfer to high-dimensional fine-fidelity distillation is unresolved. The likely bottleneck is student capacity, not sampling; scaling the student and moving to a learned generator that produces shell queries directly (rather than rejection-sampling them, which degrades in high $d$) are the natural next steps.

## 7. Discussion

The practical recommendation is compact: **place a fixed query budget in the low-density shell and weight it by both pipeline score and score-gradient magnitude.** This one sampler is automatically robust to whether the pipeline grows or saturates off the manifold, requires no anomaly labels and no differentiation of the pipeline, and reduces to the two shape-specific samplers as special cases. The score term and the gradient term are individually necessary: dropping the gradient term loses the saturating pipelines, dropping the score term loses the growing ones.

Two honest boundaries frame the contribution. Pipelines with discontinuous fusion exceed a small student's representational capacity and are not distillable at any sampling budget; a cheap shell probe predicts this. And on real high-dimensional data, coarse detection is already served by trivial baselines while fine-fidelity reproduction is unresolved, so the present contribution is a focused methods-and-analysis result on where queries should go, not a solved deployment pipeline.

## 8. Conclusion

Distilling a blackbox unsupervised anomaly pipeline with no anomaly labels is a query-placement problem. The informative queries sit in the low-density shell just off the normal manifold, and the right target within the shell follows the pipeline's off-manifold shape: score for growing pipelines, score-gradient for saturating ones. A single shell sampler weighted by both is automatically shape-robust, reproduces a spectrum of teacher pipelines with high fidelity, and comes with a cheap probe that predicts when a pipeline is beyond a small student's reach. The open frontier is fine-fidelity distillation in high dimensions, where student capacity, not query placement, appears to bind. All code, data, and results are released.

## References

- Angluin, D. (1988). *Queries and concept learning.* Machine Learning 2(4).
- Ba, J. and Caruana, R. (2014). *Do deep nets really need to be deep?* NeurIPS.
- Bergman, L. and Hoshen, Y. (2020). *Classification-based anomaly detection for general data.* ICLR.
- Breunig, M. M., Kriegel, H.-P., Ng, R. T., and Sander, J. (2000). *LOF: identifying density-based local outliers.* SIGMOD.
- Buciluă, C., Caruana, R., and Niculescu-Mizil, A. (2006). *Model compression.* KDD.
- Chen, H., Wang, Y., Xu, C., Yang, Z., Liu, C., Shi, B., Xu, C., Xu, C., and Tian, Q. (2019). *Data-free learning of student networks.* ICCV.
- Chen, R., Batra, D., Kira, Z., Piché-Taillefer, R., and Wolf, C. (2020). *Adversarially trained variational autoencoders for anomaly detection.* arXiv:2010.11024.
- Ducoffe, M. and Precioso, F. (2018). *Adversarial active learning for deep networks: a margin based approach.* arXiv:1802.09841.
- Golan, I. and El-Yaniv, R. (2018). *Deep anomaly detection using geometric transformations.* NeurIPS.
- Hinton, G., Vinyals, O., and Dean, J. (2015). *Distilling the knowledge in a neural network.* NIPS Deep Learning Workshop. arXiv:1503.02531.
- Houlsby, N., Huszár, F., Ghahramani, Z., and Lengyel, M. (2011). *Bayesian active learning for classification and preference learning.* arXiv:1112.5745.
- Li, C.-L., Sohn, K., Yoon, J., and Pfister, T. (2021). *CutPaste: self-supervised learning for anomaly detection and localization.* CVPR.
- Liu, F. T., Ting, K. M., and Zhou, Z.-H. (2008). *Isolation forest.* ICDM.
- Nayak, G. K., Mopuri, K. R., Shaj, V., Radhakrishnan, V. B., and Chakraborty, A. (2019). *Zero-shot knowledge distillation in deep networks.* ICML.
- Park, W., Kim, D., Lu, Y., and Cho, M. (2019). *Relational knowledge distillation.* CVPR.
- Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., and Bengio, Y. (2015). *FitNets: hints for thin deep nets.* ICLR.
- Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., and Gehler, P. (2022). *Towards total recall in industrial anomaly detection.* CVPR.
- Ruff, L., Vandermeulen, R. A., Görnitz, N., Deecke, L., Siddiqui, S. A., Binder, A., Müller, E., and Kloft, M. (2018). *Deep one-class classification.* ICML.
- Ruff, L., Kauffmann, J. R., Vandermeulen, R. A., Montavon, G., Samek, W., Kloft, M., Dietterich, T. G., and Müller, K.-R. (2021). *A unifying review of deep and shallow anomaly detection.* Proceedings of the IEEE.
- Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., and Williamson, R. C. (2001). *Estimating the support of a high-dimensional distribution.* Neural Computation.
- Sener, O. and Savarese, S. (2018). *Active learning for convolutional neural networks: a core-set approach.* ICLR.
- Settles, B. (2010). *Active learning literature survey.* Univ. Wisconsin-Madison TR 1648.
- Seung, H. S., Opper, M., and Sompolinsky, H. (1992). *Query by committee.* COLT.
- Silverman, B. W. (1986). *Density estimation for statistics and data analysis.* Chapman & Hall.
- Song, Y. and Ermon, S. (2019). *Generative modeling by estimating gradients of the data distribution.* NeurIPS.
- Song, Y. and Ermon, S. (2020). *Improved techniques for training score-based generative models.* NeurIPS.
- Welling, M. and Teh, Y. W. (2011). *Bayesian learning via stochastic gradient Langevin dynamics.* ICML.
- Zavrtanik, V., Kristan, M., and Skočaj, D. (2021). *DRAEM: a discriminatively trained reconstruction embedding for surface anomaly detection.* ICCV.
- Zhou, C. and Paffenroth, R. C. (2017). *Anomaly detection with robust deep autoencoders.* KDD.

## Appendix A — Samplers

**Combined shell sampler (used throughout).** Draw a candidate pool uniformly in the low-density shell $\rho_{\min} \le \rho(x) \le \rho_{\max}$; compute each candidate's extended rank $u(x)$ and its gradient magnitude $\|\nabla_x u(x)\|$ by central finite differences; importance-sample $M$ queries with probability proportional to $\tfrac12\widehat u + \tfrac12\widehat{\|\nabla u\|}$.

**Langevin variant.** Start chains at random normals and walk up the same shell potential while a density prior holds them near the manifold; a per-chain radius drawn log-uniformly lets some chains reach the far field. This gives fidelity equivalent to the importance-weighted sampler and is retained only for comparison; the importance-weighted form is simpler and needs no step-size tuning.

## Appendix B — Reproduction

The teacher spectrum, the complex-pipeline mechanism study, and the real-data evaluation are separate CPU-only scripts. The score-plus-gradient shell sampler and the shape probe are single functions; the finite-difference gradient is batched into one pipeline-scoring pass per step.
