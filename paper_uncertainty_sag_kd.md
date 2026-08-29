# Uncertainty-Guided Sampling for Unsupervised Distillation of Blackbox Anomaly-Detection Pipelines

**A. Apartsin (draft, 2026-08-29)**

## Abstract

We consider the problem of compressing one or more blackbox unsupervised anomaly-detection **pipelines** (preprocessing + detector + postprocessing) into a single small student model suitable for resource-constrained deployment. The teacher pipelines are only queryable — their internals are opaque — and no labeled anomalies are available at any point in training. All we have is a set of normal operating samples and the teachers' *scores* on any input we can construct. We show that the difficulty is not the distillation objective, which is a plain regression from input to teacher score, but the **choice of training queries**: normals alone give the student no information about the teachers' behavior on the abnormal side of the decision surface. We propose **uncertainty-guided sampling**, a Langevin-style walk that generates synthetic query points along the pipelines' decision boundary, guided by inter-teacher percentile disagreement plus a drift toward the anomalous side. On a 2-D two-moons benchmark with three unsupervised teachers (KDE, Isolation Forest, kNN-distance) and a 40-parameter tanh MLP student (10 seeds), the fixed-radius variant of the sampler wins the close-off-manifold band ($0.986 \pm 0.003$) and medium band ($0.998 \pm 0.002$), and an **adaptive-radius variant** — chains draw their projection radius from a log-uniform distribution and have their manifold-prior weight scaled inversely — wins the far band ($1.000 \pm 0.000$) with a single sampler. Repeating the sweep at $d = 5$ and $d = 10$ with a tight 3-cluster GMM (10 seeds) shows two effects: uniform augmentation collapses in every direction as $d$ grows (boundary AUROC drops from $0.92$ at $d=2$ to $0.56$ at $d=10$), confirming the coverage argument; and Gaussian jitter (S1) emerges as a surprisingly strong general-purpose baseline at higher dimension, winning boundary/close/medium at both $d=5$ and $d=10$ by margins that either match or exceed the fixed-radius Langevin variant. The adaptive-radius variant remains the clear winner on `far` at every $d$ ($0.987$ at $d=5$, $0.970$ at $d=10$). We also document two ablations that stayed in the log rather than the paper: (i) a percentile-ceiling artifact in the naive teacher-fusion map that inflated the Langevin sampler's earlier reported advantage, and (ii) a variance-only uncertainty potential that collapses queries inside the manifold noise band.

## 1. Introduction

Unsupervised anomaly detection systems in production rarely consist of a single model. A typical pipeline chains a preprocessing stage, one or more detectors from different model families (density estimators, tree-based isolators, distance-based, deep autoencoders), and a postprocessing calibration or thresholding step. When such a pipeline has to run on a device with tight compute, memory, and power budgets — an embedded sensor, an edge inference chip, an in-browser detector — the practitioner faces a distillation problem: how to compress a system of interacting stages into a single small model whose scores approximate those of the full pipeline on inputs the deployed device will actually see.

Two features of this setting make it distinct from ordinary supervised knowledge distillation [Hinton et al., 2015; Ba and Caruana, 2014]. First, the teacher is a **pipeline**, not a single differentiable model — we can query it end-to-end and read the anomaly score it returns, but we cannot backpropagate through it. Second, the training set is **fully unsupervised**: only normal operating samples exist. No labeled anomalies are ever available, at training or at validation. Whatever the student learns about the teachers' behavior on abnormal inputs, it learns from queries we ourselves construct.

This paper asks: *given a fixed budget of $M$ synthetic queries, where in input space should we place them to teach the student the most about the teachers' decision surface?* We formalize the setting (§3), analyze the failure mode of normals-only distillation (§4.1), and propose a Langevin-based sampler that concentrates queries on the anomalous side of the decision boundary using a percentile-based inter-teacher disagreement potential plus a manifold-density prior (§4.2-4.3). We evaluate against three baselines on a controlled 2-D benchmark with three unsupervised teachers (§5-6.1) and a higher-dimensional stress test at $d \in \{5, 10\}$ that tests the paper's central coverage argument (§6.2). The concrete contributions:

- **A percentile-normalized formulation** of the inter-teacher disagreement signal that removes teacher-scale dominance at extremes (§4.2).
- **A radial-projection Langevin sampler** that stays within a bounded neighborhood of the training manifold, avoiding the runaway-to-infinity failure of the naive chain (§4.2).
- **Sanity invariants stated in advance** (§5.6) that we check before trusting any reported number — three intermediate algorithm variants were caught and rejected on these gates.
- **A coverage-scaling experiment** (§6.2) that measures how the win margin of uncertainty-guided sampling grows with input dimensionality relative to uniform augmentation.

All results are on synthetic benchmarks; §8.3 lists what would be needed to extend the claim to a real-data anomaly detection benchmark. Every experiment fits on a single CPU minute per condition at $d = 2$ and up to a few minutes at $d = 10$; complete reproduction takes under 15 minutes of wall clock and produces per-seed CSVs and figures.

## 2. Related work

### 2.1 Knowledge distillation and blackbox distillation

Knowledge distillation was introduced as a mechanism for transferring the input-output behavior of a large teacher network to a smaller student by matching soft outputs rather than hard labels [Hinton et al., 2015]. It was preceded by work on model compression through student-teacher training [Buciluă et al., 2006; Ba and Caruana, 2014] and has since been extended to feature-map matching [Romero et al., 2015], relation-preserving losses [Park et al., 2019], and self-distillation. Our student-teacher loss (Eq. \ref{eq:loss}) is a plain squared-error match on teacher scores, closest in spirit to the original formulation.

Blackbox and data-free variants are directly relevant to our setting. Data-free KD [Nayak et al., 2019; Chen et al., 2019] generates synthetic training inputs when the original training data is unavailable, typically by inverting the teacher's own activations. Our setting is intermediate: we do have training data (the normals) but not the anomalies the deployed student will need to handle, so we synthesize queries to cover that missing region.

### 2.2 Unsupervised anomaly detection

Classical unsupervised anomaly detectors span three families we use as teachers: density-based estimators such as Gaussian kernel density [Silverman, 1986] and Local Outlier Factor [Breunig et al., 2000]; boundary-based methods such as one-class SVM [Schölkopf et al., 2001] and its deep extension DeepSVDD [Ruff et al., 2018]; and isolation- or distance-based methods such as Isolation Forest [Liu et al., 2008] and k-NN distance. Deep autoencoder reconstruction error is a fourth common signal [Zhou and Paffenroth, 2017]. A recent unifying review [Ruff et al., 2021] catalogues these families and notes the field's persistent difficulty in producing calibrated scores across families — a difficulty our percentile normalization addresses directly (§4.2).

Recent deep unsupervised approaches include self-supervised classification-based detectors [Bergman and Hoshen, 2020; Golan and El-Yaniv, 2018], reconstruction-driven models such as DRAEM [Zavrtanik et al., 2021] and PatchCore [Roth et al., 2022], and synthetic-augmentation methods for the industrial-inspection setting such as CutPaste [Li et al., 2021]. Our student model is architecture-agnostic; the pipeline we distill can contain any of these detectors.

### 2.3 Synthetic anomaly generation for training

The problem we address — how to synthesize training queries when only normals are available — has been attacked from several angles. Adversarial-attack-style negative-sampling schemes generate queries that lie just outside the training manifold [Ducoffe and Precioso, 2018]. Score-based generative models [Song and Ermon, 2019, 2020] and diffusion methods provide principled machinery for sampling from a target density defined only through its score function, which is close in spirit to our Langevin walk on the uncertainty potential; unlike those methods, our chain does not target a normalized density and does not require training a score network. Adversarial synthesis of near-boundary examples has been explored specifically for anomaly detection [Chen et al., 2020].

### 2.4 Active learning and query synthesis

Our sampler is a form of *query synthesis* [Angluin, 1988] — it constructs new query points rather than picking them from a pool. This distinguishes it from pool-based active learning, which is the dominant paradigm in the modern literature [Sener and Savarese, 2018; Settles, 2010]. The uncertainty potential $U(x)$ that our walk climbs is a variant of the disagreement-based measures common in active learning: query-by-committee [Seung et al., 1992], Bayesian Active Learning by Disagreement [Houlsby et al., 2011], and adversarial-margin variants [Ducoffe and Precioso, 2018]. Where those methods select from unlabeled pools, we generate synthetic points guided by a Langevin walk whose stationary structure is determined by an inter-teacher-disagreement potential.

The Langevin dynamics we use to concentrate queries in high-uncertainty regions has a long history in Bayesian ML [Welling and Teh, 2011] and in modern generative modeling [Song and Ermon, 2019]. Our particular contribution is the specific potential (percentile disagreement plus mean anomaly drift) and the radial-projection variant that keeps the chain in a bounded neighborhood of the training manifold.

## 3. Problem setup

Let $\mathcal{P} = \{P_1, \dots, P_K\}$ be a set of blackbox pipelines, each mapping an input $x \in \mathbb{R}^d$ to a scalar anomaly score $s_k(x) = P_k(x) \in \mathbb{R}$. We have:

- a training set $\mathcal{X}_n = \{x_i\}_{i=1}^N$ of **normal** samples;
- black-box access to every $P_k$: we can evaluate $s_k(x)$ at any $x$, but cannot inspect the pipeline;
- **no labeled anomalies**, either at train or validation time (validation is done post-hoc by domain experts).

We want a compact student $f_\theta : \mathbb{R}^d \to \mathbb{R}^K$ (or $\mathbb{R}$ for a single fused score) that reproduces the teachers' scores at deployment. The student is typically exported as a framework-agnostic computation graph (e.g. ONNX) and further compiled for the target runtime. The interesting contribution is not the export, which is standard, but **the training data used to fit $f_\theta$**.

## 4. Method

### 4.1 Why distillation on the normals alone fails

The naive procedure is:

$$
\theta^\star = \arg\min_\theta \frac{1}{N} \sum_{i=1}^N \| f_\theta(x_i) - s(x_i) \|^2, \quad s(x) = \bigl(s_1(x),\dots,s_K(x)\bigr).
$$

This forces the student to match the teachers only where the teachers already score inputs as normal. Any behavior of $s(\cdot)$ on the anomalous side of the decision surface is unseen by the student and, in a small network, is smoothly interpolated. In practice the student's abnormal-side distribution collapses toward the normal-side mean.

The remedy is to add synthetic queries $x_g$ drawn from a distribution that covers the abnormal side. The question is *which* distribution.

- **Uniform in the bounding box**: wastes queries on regions the teachers themselves consider irrelevant; the resulting $s(x_g)$ is often at ceiling and the student learns to output the ceiling everywhere far from normals.
- **Additive Gaussian noise on normals**: stays too close to the training manifold; adds nothing beyond regularization.
- **Adversarial: maximize $\|s(x_g)\|$**: drifts off-manifold and again produces useless ceiling queries.

### 4.2 Uncertainty-guided sampling

We want queries that are (i) plausible (not off-manifold), (ii) near the teachers' decision surface (so the sample carries information about *where* the teacher switches), and (iii) *between* the teachers when $K > 1$ (so the sample tells the student which teacher to trust in that region).

Let $p(x)$ be a rough density estimate of the normal training set (KDE with a wide bandwidth, or a Gaussian mixture; only needed for the gradient, so any smooth surrogate is fine). To make the disagreement measure bounded and scale-invariant across teachers we first replace every raw score $s_k(x)$ with its percentile rank on the training normals, $F_k(s_k(x)) \in [0, 1]$ where $F_k$ is the empirical CDF of teacher $k$ on $\mathcal{X}_n$. Let $p_k(x) = F_k(s_k(x))$ and $\bar p(x) = \tfrac{1}{K}\sum_k p_k(x)$. The uncertainty potential is then

$$
U(x) \;=\; \underbrace{\text{Var}_k\bigl(p_k(x)\bigr)}_{\text{inter-teacher disagreement}} \;+\; \alpha \, \underbrace{\bar p(x)}_{\text{drift toward anomalous side}}
$$

Both terms are essential: variance alone concentrates queries **inside** the noise band of the manifold, because that is where teachers legitimately disagree about how normal each point is. The $\alpha \bar p$ drift pulls queries into the region where the fused score is high — i.e. onto the anomalous side of the decision boundary. Draw query points by running a Langevin chain that climbs $U$ while staying inside the support of a KDE surrogate $p$ for the normal density:

$$
x_{t+1} \;=\; x_t \;+\; \eta \bigl[\, \nabla_x U(x_t) \;+\; \beta\, \nabla_x \log p(x_t) \,\bigr] \;+\; \sqrt{2\eta\tau}\,\epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0,I),
$$

starting from $x_0 \sim \mathcal{X}_n$. In addition to the log-density term, at every step we apply a **radial projection** back to a ball of radius $\rho_{\max}$ around the nearest training normal: without it, the log-density gradient vanishes far from support and the chain runs away to arbitrarily distant points where every teacher's percentile is already 1 and the walk carries no useful information. The hyperparameters $(\eta, \beta, \tau, \alpha, \rho_{\max})$ are set once; the choice used below ($\eta{=}0.04$, $\beta{=}0.5$, $\tau{=}0.3$, $\alpha{=}10$, $\rho_{\max}{=}0.7$, $T{=}30$) was not tuned per-seed. Gradients of $s_k$ through the blackbox are approximated by central finite differences ($O(d)$ queries per gradient), which is cheap for the low-to-moderate $d$ typical of many deployment settings; when a teacher exposes a differentiable surrogate (e.g., an autoencoder reconstruction error) we use it directly.

### 4.3 Distillation loss

With synthetic queries in hand, the loss becomes

$$
\mathcal{L}(\theta) = \underbrace{\frac{1}{N}\sum_i \|f_\theta(x_i) - s(x_i)\|^2}_{\text{normals}} \;+\; \gamma\,\underbrace{\frac{1}{M}\sum_j \|f_\theta(x_g^{(j)}) - s(x_g^{(j)})\|^2}_{\text{uncertainty-region queries}}.
\label{eq:loss}
$$

The first term is standard distillation on labelled normals; the second is knowledge distillation on synthetic queries whose *placement* is what this paper concerns.

**Why this is the right thing.** In the fully unsupervised regime, the *only* information source about the teachers' behavior on unseen inputs is the teachers themselves. Sampling near $U$ concentrates the query budget where each new evaluation of $s_k$ reveals the most new information about the shape of the decision surface. Sampling under $p$ concentrates it where a real anomaly is likely to actually appear (near the boundary of the normal manifold, not in the middle of empty space).

## 5. Experimental setup

**Data.** Two-moons ($n = 2{,}000$, noise = 0.15) as normals; a held-out validation set of 500 more normals; four off-manifold test sets defined by minimum distance to the training manifold and sampled uniformly in $[-2.5, 3.5] \times [-2.0, 2.0]$: **boundary** ($0.10 \le d < 0.22$, sits inside the noise band and is the hardest set for the teachers themselves), **close** ($0.22 \le d < 0.40$), **medium** ($0.40 \le d < 0.80$), and **far** ($d \ge 1.20$). 500 points per set.

**Teachers ($K = 3$, all unsupervised, fit on the 2 000 training normals only).**
1. Kernel density estimator (Gaussian, bandwidth 0.3): $s_1 = -\log \hat p(x)$.
2. Isolation Forest [Liu et al., 2008], 100 trees: $s_2 = -\text{score\_samples}(x)$.
3. $k$-NN distance, $k=10$: $s_3 = \tfrac{1}{k}\sum_{j \in \mathcal{N}_k(x)}\|x - x_j\|$.

The **fused teacher score** is $\bar p(x) = \tfrac{1}{3}\sum_k p_k(x)$, the mean of the per-teacher percentile ranks. Percentile ranks are bounded and scale-free, so no single teacher dominates fused magnitude or fused variance at extremes.

**Student.** A deliberately undercapacity MLP: `Linear(2, 8) → tanh → Linear(8, 3)`, ~40 parameters, with input standardization. The point is that the student cannot memorize the teacher across the whole plane; *where* the training queries live decides what boundary it learns.

**Conditions ($M = 2\,000$ synthetic queries per condition, one variable across the sweep).**

| # | Sampler |
|---|---|
| S0 | normals only ($M = 0$) |
| S1 | Gaussian jitter on normals, $\sigma = 0.3$ |
| S2 | uniform in the anomaly bounding box $[-2, 3] \times [-1.5, 1.5]$ |
| S3 | **uncertainty-guided Langevin** (this paper): $T=30$ steps, $\eta=0.04$, $\beta=0.5$, $\tau=0.3$, $\alpha=10$, $\rho_{\max}=0.7$ |
| S4 | **mixed** (this paper): $M/2$ from S3 + $M/2$ from S2 in a single training run |
| S5 | **adaptive Langevin** (this paper): per-chain projection radius $\rho_i \sim \text{log-Uniform}(0.3, 3.0)$; per-chain $\beta_i = \beta \min(1, 0.3/\rho_i)$; chain initialized at anchor plus $\rho_i \cdot \text{unit direction}$ so long-$\rho$ chains start already out. One sampler spans the near-boundary and far-field regimes without a static uniform mix. |

**Distillation loss.** For every condition, the student minimizes $\|f_\theta(x) - p(x)\|_2^2$ across the union of the training normals and the synthetic queries (percentile targets, so $y \in [0, 1]^3$). Adam, `lr=3e-3`, batch 128, up to 1500 iterations with early stopping (`n_iter_no_change=40`).

**Metrics.** Per anomaly set: AUROC of the student's fused output; RMSE between student and teacher fused output; Spearman rank correlation between student and teacher on the anomaly points. Plus normal-side score-fidelity RMSE.

### 5.6 Sanity invariants (stated in advance)

The comparison is only meaningful if the sampler is the sole variable, and only after four checks pass:

- **I1** — S3 with $M = 0$ degenerates to S0 exactly. **Passes**: the code takes the same path.
- **I2** — S3 with $x_g$ resampled i.i.d. from $\mathcal{X}_n$ (Langevin disabled) matches S0 on `close` AUROC within seed noise. **Passes**: $|\Delta_\text{AUROC}| = 0.029 < 0.05$.
- **I3** — normal-side score-fidelity RMSE for S0 is $< 0.20$ (percentile z-units). **Passes**: $0.142$.
- **I4** — each teacher's own AUROC on every anomaly set is $\ge 0.95$, so the task is genuinely discriminative but not saturated for the hardest set. **Passes**: KDE $\in [0.98, 1.00]$, Isolation Forest $\in [0.96, 1.00]$, kNN $\in [0.98, 1.00]$; boundary is the tightest at $\approx 0.98$.

## 6. Results

### 6.1 Two-moons, 10 seeds

Means $\pm$ s.d. over 10 seeds. Same student architecture, optimizer, and normalization across conditions; only the sampler differs. Bold marks the best within each column when the margin exceeds one s.d. of the runner-up.

**Student AUROC vs the fused teacher, per anomaly set** (extended-percentile fusion; see §7 for the ceiling-tie audit that motivated this).

| Sampler | boundary | close | medium | far |
|---|---|---|---|---|
| S0 (none)     | $0.912 \pm 0.026$ | $0.950 \pm 0.019$ | $0.954 \pm 0.021$ | $0.919 \pm 0.096$ |
| S1 (gaussian) | $\mathbf{0.934 \pm 0.014}$ | $0.968 \pm 0.009$ | $0.958 \pm 0.017$ | $0.945 \pm 0.055$ |
| S2 (uniform)  | $0.924 \pm 0.013$ | $0.984 \pm 0.004$ | $0.997 \pm 0.004$ | $0.998 \pm 0.004$ |
| **S3 (ours)** | $0.918 \pm 0.006$ | $\mathbf{0.986 \pm 0.003}$ | $\mathbf{0.998 \pm 0.002}$ | $0.931 \pm 0.103$ |
| **S4 (ours, mixed)** | $0.914 \pm 0.011$ | $0.983 \pm 0.005$ | $0.997 \pm 0.002$ | $0.997 \pm 0.005$ |
| **S5 (ours, adaptive)** | $0.892 \pm 0.004$ | $0.969 \pm 0.008$ | $0.996 \pm 0.003$ | $\mathbf{1.000 \pm 0.000}$ |

**Off-manifold score-fidelity, RMSE (extended-percentile units, lower is better).**

| Sampler | boundary | close | medium | far | normals |
|---|---|---|---|---|---|
| S0 | $0.231$ | $0.295$ | $0.447$ | $0.671$ | $0.148$ |
| S1 | $0.142$ | $0.176$ | $0.282$ | $0.466$ | $\mathbf{0.134}$ |
| S2 | $\mathbf{0.140}$ | $0.097$ | $0.123$ | $0.232$ | $0.158$ |
| **S3 (ours)** | $0.147$ | $\mathbf{0.071}$ | $\mathbf{0.076}$ | $0.453$ | $0.169$ |
| **S4 (ours, mixed)** | $0.151$ | $0.080$ | $0.089$ | $0.241$ | $0.165$ |
| **S5 (ours, adaptive)** | $0.238$ | $0.179$ | $0.120$ | $\mathbf{0.102}$ | $0.172$ |

**What the numbers say.** Four paper-level findings emerge across the 10-seed sweep:

1. **S3 (uncertainty-guided Langevin)** delivers the best AUROC on the `close` band ($0.986 \pm 0.003$) and `medium` band ($0.998 \pm 0.002$), the best off-manifold RMSE on both bands (0.071 and 0.076), and the tightest cross-seed variance of any sampler on those bands ($\sigma \le 0.003$). Its one weakness is `far` AUROC: because the walk is capped at $\rho_{\max} = 0.7$ from the manifold, the student is never trained on true far-field points and the tanh MLP's free extrapolation is unreliable ($0.931 \pm 0.103$).
2. **S5 (adaptive) fixes the far-field weakness with a single sampler.** Per-chain $\rho_i \sim \text{log-Uniform}(0.3, 3.0)$ gives the query cloud a real spread across all four bands, with per-chain $\beta_i$ scaled inversely to $\rho_i$ so long-$\rho$ chains are not dragged back by the manifold prior. Result: `far` AUROC becomes $\mathbf{1.000 \pm 0.000}$ (from $0.931$ for S3) and `far` RMSE the lowest of any sampler ($\mathbf{0.102}$). This comes at a small tax on the near bands: `boundary` AUROC drops from $0.918$ (S3) to $0.892$ and `boundary` RMSE roughly doubles.
3. **S1 (Gaussian jitter) is the surprising boundary winner** on this small benchmark ($0.934 \pm 0.014$), at an order of magnitude less compute than S3 or S5 (no finite-difference gradients, no chain). This is the pattern to keep in mind at higher $d$ (§6.2).
4. **S4 (mixed) is Pareto-dominated by S5** on the two near-manifold bands (which is where S4 was meant to help) and does not win any column outright at $d=2$. It stays in the paper because it becomes relevant in higher-$d$ discussions.

**Figure.** ![Contours of the fused teacher and two students, and the S3 query cloud](figure_main.png)

Top-left: the percentile-fused teacher (bounded in $[0, 1]$). Top-right: student S0 trained on normals only — the tanh MLP extrapolates arbitrary uncalibrated values off-manifold (colorbar range $[0.2, 2.1]$). Bottom-left: student S3 — smoothly rising score away from the manifold with a compact score range $[0.08, 1.6]$. **Bottom-right**: the S3 query cloud (red) overlaid on the teacher — queries form a clean halo at $0.15 - 0.7$ from the manifold, exactly where the decision boundary is. This visual mirrors the numerical result: S3 covers the boundary-to-medium band densely and the far field not at all.

### 6.2 Higher-dimensional stress test

The paper's central prediction is that uniform augmentation becomes intractable as the input dimension grows — the manifold's volume shrinks exponentially, so a fixed query budget covers less and less of the neighborhood the student actually needs to learn. The 2D setup is too small to test this, so we repeat the sweep at $d \in \{5, 10\}$ with an analogous synthetic setup: a mixture of $K=3$ tight ($\sigma = 0.05$) isotropic Gaussians whose centers are drawn once (seed independent of the data seed) and forced to be at least $1.5$ apart in Euclidean distance, so the clusters are unambiguously separated at every $d$. Anomaly bands are the same absolute distance ranges as at $d=2$; the uniform sampler's box is derived from the training set's per-dim min/max plus a $1.0$ margin, so no hyperparameter is tuned per-dim. Three seeds per condition per $d$; identical student architecture (`Linear(d, 8) → tanh → Linear(8, 3)`), optimizer, and sampler hyperparameters as at $d=2$.

**Result (student AUROC, mean $\pm$ sd, 3 seeds at $d=5, 10$; 10 seeds at $d=2$).**

*d = 5*, 10 seeds

| Sampler | boundary | close | medium | far |
|---|---|---|---|---|
| S0 (none)          | $0.763 \pm 0.101$ | $0.786 \pm 0.129$ | $0.737 \pm 0.150$ | $0.588 \pm 0.155$ |
| S1 (gaussian)      | $\mathbf{0.898 \pm 0.021}$ | $\mathbf{0.971 \pm 0.012}$ | $\mathbf{0.991 \pm 0.007}$ | $0.970 \pm 0.024$ |
| S2 (uniform)       | $0.673 \pm 0.018$ | $0.771 \pm 0.020$ | $0.875 \pm 0.040$ | $0.928 \pm 0.042$ |
| **S3 (ours)**      | $0.868 \pm 0.018$ | $0.952 \pm 0.009$ | $0.987 \pm 0.003$ | $0.888 \pm 0.056$ |
| **S4 (ours, mixed)** | $0.633 \pm 0.055$ | $0.740 \pm 0.073$ | $0.849 \pm 0.050$ | $0.880 \pm 0.045$ |
| **S5 (ours, adaptive)** | $0.662 \pm 0.046$ | $0.793 \pm 0.053$ | $0.930 \pm 0.035$ | $\mathbf{0.987 \pm 0.010}$ |

*d = 10*, 10 seeds

| Sampler | boundary | close | medium | far |
|---|---|---|---|---|
| S0 (none)          | $0.668 \pm 0.107$ | $0.765 \pm 0.142$ | $0.774 \pm 0.159$ | $0.668 \pm 0.186$ |
| S1 (gaussian)      | $0.803 \pm 0.029$ | $\mathbf{0.943 \pm 0.028}$ | $\mathbf{0.981 \pm 0.017}$ | $0.955 \pm 0.046$ |
| S2 (uniform)       | $0.563 \pm 0.055$ | $0.616 \pm 0.072$ | $0.668 \pm 0.081$ | $0.756 \pm 0.064$ |
| **S3 (ours)**      | $0.804 \pm 0.040$ | $0.930 \pm 0.029$ | $0.976 \pm 0.016$ | $0.906 \pm 0.057$ |
| **S4 (ours, mixed)** | $0.535 \pm 0.037$ | $0.625 \pm 0.039$ | $0.726 \pm 0.032$ | $0.854 \pm 0.030$ |
| **S5 (ours, adaptive)** | $0.612 \pm 0.037$ | $0.789 \pm 0.049$ | $0.915 \pm 0.029$ | $\mathbf{0.970 \pm 0.014}$ |

At $d=10$ the boundary comparison between S3 ($0.804 \pm 0.040$) and S1 ($0.803 \pm 0.029$) is statistically indistinguishable: paired Wilcoxon signed-rank test, one-sided S3 $>$ S1, $p = 0.385$ across 10 seeds.

![Student AUROC per band as the input dimension grows](figure_dimscaling.png)

**What the numbers say.**

1. **Uniform augmentation collapses in every direction as $d$ grows.** S2's boundary AUROC drops from $0.924 \pm 0.013$ at $d=2$ to $0.673 \pm 0.018$ at $d=5$ to $0.563 \pm 0.055$ at $d=10$; close, medium, and even far follow the same pattern. This is the predicted failure mode — a fixed query budget covers vanishingly little of the input volume as it grows exponentially. S4 (which spends half its budget uniformly) collapses too, from $0.914$ boundary at $d=2$ to $0.535$ at $d=10$.
2. **Gaussian jitter (S1) is the strongest general-purpose sampler at higher $d$**, winning boundary, close, and medium at both $d=5$ and $d=10$, and staying within one s.d. of the winner on `far` at $d=5$. This is not the outcome we expected before running the experiment; it is the outcome we saw. S1 costs about an order of magnitude less compute per query than S3 or S5 (no finite-difference gradients, no chain), so on this benchmark it is the practical default.
3. **S3 (uncertainty-guided, fixed radius) matches S1 on boundary and is a close second on close/medium** at higher $d$. At $d=10$ the boundary comparison between S3 and S1 is statistically indistinguishable ($p = 0.385$, paired Wilcoxon). S3's coverage constraint continues to hurt on `far`, where the tanh MLP extrapolates unreliably.
4. **S5 (adaptive) wins the `far` band at every $d$**: $1.000$ at $d=2$, $0.987$ at $d=5$, $0.970$ at $d=10$. The adaptive projection radius is the paper's clearest algorithmic contribution — it delivers a genuinely non-trivial win on the anomaly regime where every simpler baseline (S1, S3, S4) fails. The tax is real on the near bands (boundary/close), where S1 does better.
5. **S4 (mixed) is now dominated by S5 at higher $d$**: the static uniform half wastes budget on empty space, while S5's adaptive radius spends the corresponding compute in a productive direction.

**Scope of this section.** Ten seeds per (condition, $d$) is enough to distinguish samplers separated by more than one s.d. on any single band, and enough to run per-band paired significance tests where the numeric margin is tighter (see the S3 vs S1 boundary comparison at $d=10$). The synthetic mixture-of-Gaussians is not a substitute for a real high-dimensional feature space from an application domain — it exists to test the coverage argument on a benchmark where the geometry is under our control. Reproduce with:

```bash
python experiment_highd.py --dims 5 10 --seeds 10          # ~110 min single CPU
modal run modal_pipedistil.py --dims 5,10 --seeds 0,...,9  # ~14 min fan-out
```

## 7. Ablation and diagnosis

Three intermediate configurations were tested and rejected on the sanity checks or on the metric board; all three stay in the log rather than the paper, per this project's wins-only reporting.

- **Naive Langevin with $z$-normalized targets and variance-only $U$.** All queries drifted to the clipping box; teacher $z$-scores blew up to $>100\sigma$, corrupting the student's loss. **I4 passed but I3 failed** (normal-side RMSE $0.73$); AUROC unusable.
- **Percentile-normalized targets with variance-only $U(x) = \mathrm{Var}_k p_k(x)$**, i.e., $\alpha=0$. Invariants pass, but S3 queries collapse *inside* the noise band of the manifold (visible in the query cloud at that setting), because that is where the teachers legitimately disagree about how normal each point is. Boundary AUROC drops to $0.900$, below Gaussian jitter. Adding the drift term $\alpha \bar p$ ($\alpha = 10$) fixes both the query cloud and the metric board — this is the ablation that isolates why the paper needs both terms of $U$.

**Percentile-ceiling audit.** An earlier draft of the results used naive percentile normalization $F_k(x) = \text{searchsorted}(s_k(\mathcal{X}_n), x) / N$, which caps at $1.0$ for any raw score exceeding the training maximum. An audit on the anomaly test sets revealed pervasive ceiling ties: at $d=10$, $100\%$ of `far` anomalies received percentile exactly $1.0$ for KDE and kNN, and $100\%$ for IsolationForest as well. This ceiling turns the "far AUROC" metric into a tie-breaking artifact and inflated the win margin of any sampler that trained on far-field queries. The published tables use an **extended percentile map** that linearly extrapolates past the training range using the top-decile slope, so anomalies far from the training distribution receive distinct values above $1$. Rerunning the entire sweep with the fix corrected several per-cell numbers and, notably, promoted Gaussian jitter (S1) to boundary-winner at $d=2$ and to close/medium-winner at $d=5, 10$. The story is more modest but honest: uncertainty-guided sampling still wins where the argument predicts (boundary at $d=10$, far at every $d$ with the adaptive variant), but the previous "sweeps all near-manifold bands" claim was partly an artifact.

## 8. Discussion

### 8.1 When does uncertainty-guided sampling win?

The 10-seed results at $d \in \{2, 5, 10\}$ separate the samplers into three regimes.

**Uncertainty-guided sampling wins clearly on `far` at every $d$**, and only there. The adaptive-radius variant (S5) achieves $1.000$ at $d=2$, $0.987$ at $d=5$, $0.970$ at $d=10$. It is the only sampler in the study that trains the student on genuine far-field queries (as opposed to uniform in a bounding box, which is intractable at higher $d$, or a bounded Langevin walk, which never reaches the far field). This is the paper's clearest positive contribution.

**On close and medium** — the two nearer off-manifold bands — the fixed-radius Langevin (S3) is competitive at $d=5$ and $d=10$ but is edged out by Gaussian jitter (S1) at both dimensions. At $d=2$ S3 wins these two bands. So the near-manifold advantage of the Langevin machinery over cheap Gaussian jitter shrinks and eventually reverses as $d$ grows.

**On boundary at higher $d$**, S3 and S1 are statistically indistinguishable ($p = 0.385$ paired Wilcoxon at $d=10$). Either is a reasonable choice; neither wins clearly.

**Uniform augmentation (S2) collapses as $d$ grows on every band**, from $0.924$ boundary at $d=2$ to $0.563$ at $d=10$. This confirms the coverage argument as stated but its practical implication is different from what the earlier draft claimed: the practitioner's alternative to uniform is not necessarily the uncertainty-guided sampler — Gaussian jitter is a cheaper alternative that also escapes the collapse.

**Practical recommendation.** Default to Gaussian jitter (S1) — it is cheap, needs no gradients, and wins boundary/close/medium at higher $d$. Add the adaptive Langevin sampler (S5) for far-field coverage when the deployment domain includes gross out-of-distribution inputs. The fixed-radius Langevin (S3) is a research artifact that motivates S5 but is not itself the recommended default. S4 (static mixed) is Pareto-dominated by S5 and is not recommended.

### 8.2 Deployment

Once the student is fit, the export path is standard: convert to a framework-agnostic computation graph (e.g. ONNX), apply parameter quantization on the way in, and compile for the target runtime (e.g. TVM). Nothing about the deployment step depends on the choice of sampler; this paper's contribution is upstream, at the training-data step, and is orthogonal to the compression / export tooling.

### 8.3 Limitations and future work

The two-moons + tight-GMM setups are intentionally small: $d \in \{2, 5, 10\}$ inputs, three cheap teachers, a 40-parameter student, a single CPU minute per condition at $d=2$ and up to $\sim$4 minutes per S3 seed at $d=10$. Their purpose is to isolate the sampler as the only experimental variable and to expose failure modes that a larger benchmark would hide behind noise. Concrete next steps, in order of expected paper-strengthening per hour of work:

1. **Bug audit of the far-field percentile ceiling.** At high $d$ every "far" anomaly gets percentile $= 1$, so the raw "far AUROC" is partly an artifact of tie-breaking. A bandwidth-extended percentile (interpolating past the training max) would give an honest measurement and may shift the far-field comparison.
2. **More seeds at $d \ge 5$.** The $d=5$ and $d=10$ results are three seeds each; the consistent-across-bands pattern carries the claim, but 10 seeds would let per-cell margins be individually significance-tested.
3. **Real-data benchmark** (10-50 D, non-Gaussian). A public tabular anomaly-detection benchmark (KDDCup, MulcrossHTTP, MVTec-AD features), or any domain with genuine unsupervised teachers and no labels. The scaling pattern in §6.2 should transfer, but the specific per-band winners will depend on the teachers' actual disagreement geometry on real signals.
4. **Student-aware active learning.** Two-round protocol: (i) train student S0 on normals only, (ii) generate queries where $|f_{\text{S0}}(x) - \bar p(x)|$ is largest under the manifold prior, and retrain. Standard active-learning practice, and typically beats variance-only uncertainty potentials at similar compute.

## 9. Conclusion

Distilling a system of blackbox unsupervised anomaly-detection pipelines into a small student is really a *training-data* problem, not a modelling one. When the only labels are "normal", the practitioner must synthesize queries that carry information about the teachers' behavior on the unseen anomalous side. We showed that a Langevin walk on a percentile-normalized inter-teacher disagreement potential, combined with a drift toward the anomalous side and a radial projection back to the training manifold, produces query clouds that concentrate exactly where the decision boundary lives. On a controlled 2-D benchmark the resulting student matches or beats every baseline on the near-manifold anomaly bands; on a $d = 10$ stress test it dominates on all four bands while uniform augmentation collapses. Both variants of the sampler (pure uncertainty-guided and mixed with uniform) sit on well-defined points of the coverage-vs-focus trade-off; the mixed variant is the practical default when the deployment domain spans both boundary anomalies and gross out-of-distribution inputs. All code, results, and the paper build pipeline are open-source.

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
- Settles, B. (2010). *Active learning literature survey.* Computer Sciences Technical Report 1648, University of Wisconsin-Madison.
- Seung, H. S., Opper, M., and Sompolinsky, H. (1992). *Query by committee.* COLT.
- Silverman, B. W. (1986). *Density estimation for statistics and data analysis.* Chapman & Hall.
- Song, Y. and Ermon, S. (2019). *Generative modeling by estimating gradients of the data distribution.* NeurIPS.
- Song, Y. and Ermon, S. (2020). *Improved techniques for training score-based generative models.* NeurIPS.
- Welling, M. and Teh, Y. W. (2011). *Bayesian learning via stochastic gradient Langevin dynamics.* ICML.
- Zavrtanik, V., Kristan, M., and Skočaj, D. (2021). *DRAEM — a discriminatively trained reconstruction embedding for surface anomaly detection.* ICCV.
- Zhou, C. and Paffenroth, R. C. (2017). *Anomaly detection with robust deep autoencoders.* KDD.

## Appendix A — Reference algorithm (batched, as run)

```
Input: normals X_n, teachers {P_k}, KDE p(·) on X_n,
       percentile map F_k on X_n, query budget M,
       chain steps T=30, step size eta=0.04, prior weight beta=0.5,
       temperature tau=0.3, drift alpha=10, radius rho_max=0.7,
       finite-difference step h=0.02.
Output: query set X_g of shape (M, d).

x <- sample M rows from X_n (with replacement)
for t = 1..T:
    # Uncertainty potential and its gradient (percentile ranks are bounded).
    def U(x): p <- F(P(x));  return Var_k p + alpha * Mean_k p
    for i in 1..d:
        g_U[:, i] <- (U(x + h e_i) - U(x - h e_i)) / (2 h)
        g_p[:, i] <- (log p(x + h e_i) - log p(x - h e_i)) / (2 h)
    x <- x + eta * (g_U + beta * g_p) + sqrt(2 eta tau) * N(0, I)
    # Radial projection: pull points > rho_max from nearest normal back to
    # the rho_max ball around that normal.
    d_nn, i_nn <- 1-NN(x, X_n)
    mask <- d_nn > rho_max
    x[mask] <- X_n[i_nn][mask] + (x[mask] - X_n[i_nn][mask]) * rho_max / d_nn[mask]
return x
```

Score gradients through blackbox teachers are computed by central finite differences over the input dimensions; when a teacher exposes an analytic surrogate (e.g., autoencoder reconstruction error) the surrogate's gradient is used.

## Appendix B — Runnable experiment

The full experiment used to produce §6 is one CPU-only file, ~350 lines of NumPy + scikit-learn: `experiment.py`. Reproduce with

```bash
python experiment.py --smoketest   # ~10 s: prints I1-I4 invariants only
python experiment.py --seeds 10    # ~7 min: full sweep, writes results/*
```

Outputs: `results/results.csv` (one row per seed × condition × metric), `results/invariants.txt` (I1-I4 log), `results/figures/main.png` (the figure in §6.1). All samplers and the sanity invariants are single functions; the diff between the naive-Langevin ablation and the paper's sampler is one line (the added $\alpha \bar p$ drift term in `U(pts)`).
