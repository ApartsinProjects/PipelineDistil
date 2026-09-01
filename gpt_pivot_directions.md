Library
/
anomaly_distill_pivot_response.md
Research Pivot: When Does Adaptive Sampling Actually Pay for Black-Box Distillation?

Project confirmation. This project is about distilling a query-only, undeployable unsupervised anomaly-detection pipeline trained on normal data into a small deployable student, using synthetic off-manifold queries to reproduce the teacher's anomaly-score behavior; the decisive experiments showed that the proposed near-manifold shell sampling is not special, space-filling wins in the cheap low-dimensional regime, and several attribution/multi-head extensions also fail.

Executive verdict

The original anomaly-distillation thesis should not be rescued by inventing another synthetic anomaly sampler. Your negative result is too clean, and the surrounding literature is now too close. In particular, Hong and Kang's 2024 Score Distillation for Anomaly Detection already formulates a small student that mimics a larger anomaly detector's scoring function, while a large teacher-student anomaly-detection literature already uses synthetic anomalies or teacher-student discrepancies. Your distinctive contribution is no longer "anomaly-score distillation."

The F-16 GCAS pivot is directionally right but too close to existing work if presented as active safety-boundary learning. The killer prior is Qin et al., ICCPS 2022, Statistical Verification of Cyber-Physical Systems using Surrogate Models and Conformal Inference: they explicitly learn surrogate models of black-box CPS robustness, perform GP-based refinement, and evaluate F-16 GCAS. S-TaLiRo, VerifAI, Bayesian/optimization-based falsification, GP level-set estimation, adaptive stress testing, and the structural-reliability literature all establish the general idea of spending expensive simulator calls near a failure boundary.

The strongest paper is instead:

A regime map for black-box surrogate distillation: when adaptive boundary-seeking beats passive space-filling, and when it does not.

That framing converts the anomaly result into the first, necessary negative-control regime rather than an embarrassment. The scientific claim is not that a new sampler wins everywhere. It is that the benefit of adaptive sampling has a measurable crossover governed primarily by coverage difficulty of the informative region relative to query budget, while query cost determines the practical value of that sample-efficiency gain.

One correction is essential before writing the paper: per-query expense and geometric difficulty are separate axes. High dimension, low intrinsic dimension, rare/thin decision surfaces, or multimodality determine whether an adaptive strategy can beat space-filling in number of queries. Per-query expense does not mathematically create that sample-efficiency advantage; it determines whether saving queries is economically meaningful. Do not claim that both are necessary for a statistical win. Claim instead:

geometry/budget determine the sample-efficiency crossover;

teacher cost converts query savings into time/money/compute savings.

Overall ranking
Rank	Pivot	Novelty	2–3 month feasibility	Clean causal story	Verdict
1	Cross-regime study: when adaptive sampling beats space-filling for black-box map distillation	8.5/10	9/10	10/10	Best bet
2	F-16/ARCH-style safety-margin distillation, but only as a real-system component of #1	5.5/10 alone; 8/10 inside #1	9/10	9/10	Use, but do not make it the sole novelty
3	Database/SLO violation-map distillation with BenchBase + PostgreSQL	7/10	8/10	8/10	Strong second real domain
4	OpenFAST ultimate-load / operating-envelope map distillation	6/10	6/10	8/10	Good engineering validation, more setup
5	Power-grid transient-stability map or robot-policy failure map	4/10	5/10	8/10	Already crowded; use only as validation, not thesis
1. Critique of the F-16 GCAS "safety-map distillation" hero pick
Bottom line

As currently stated, it is not genuinely novel enough. The application is excellent, the geometry is excellent, and the deployment story is excellent, but the core methodological claim is already covered by several mature literatures.

I would score the current hero concept:

Problem importance: 9/10

Benchmark credibility: 9/10

Methodological novelty as "actively learn the thin safety boundary": 3/10

Novelty as "query-budget crossover for faithful deployable safety-map distillation": 7/10

Novelty as one domain in a broader regime-characterization paper: 8/10

The closest prior work
1. Qin et al. 2022 is the closest and most dangerous prior

Xin Qin, Yuan Xian, Aditya Zutshi, Chuchu Fan, Jyotirmoy V. Deshmukh. "Statistical Verification of Cyber-Physical Systems using Surrogate Models and Conformal Inference." ICCPS 2022. DOI: 10.1109/ICCPS54341.2022.00017.

This paper is unusually close to your proposed F-16 story. It:

treats the CPS as a black-box simulator;

learns a surrogate from simulations;

predicts quantitative temporal-logic robustness, not merely a hard pass/fail label;

uses GP-based refinement to obtain finer partitions where needed;

explicitly evaluates F-16 GCAS.

That means a paper whose claim is "we query F-16 rollouts intelligently to learn a safety-margin map" will immediately be compared to Qin et al.

2. S-TaLiRo / Breach / CPS falsification

Annpureddy et al., 2011, S-TaLiRo, TACAS. DOI: 10.1007/978-3-642-19835-9_21.

Donzé, 2010, Breach, CAV. DOI: 10.1007/978-3-642-14295-6_17.

These systems use quantitative robustness of temporal-logic specifications to guide simulation-based search. Their objective is usually falsification rather than globally faithful surrogate distillation, but the core notion that robustness gives a useful graded signal around a safety boundary is old.

3. VerifAI

Dreossi et al., 2019, "VerifAI: A Toolkit for the Formal Design and Analysis of Artificial Intelligence-Based Systems," CAV.

VerifAI uses active samplers driven by specification robustness to discover failures in simulation. It supports falsification, parameter synthesis, systematic testing, and data augmentation. Again, it is not primarily trying to compress the simulator into a deployable student, but it occupies much of the conceptual territory around adaptive safety testing.

4. GP level-set estimation

Gotovos, Casati, Hitz, Krause, 2013, "Active Learning for Level Set Estimation," IJCAI.

This is the generic statistical version of your geometry: learn the set where an unknown function crosses a threshold using sequential measurements and GP confidence bounds. It has theoretical sample-complexity guarantees. Calling boundary-focused sampling itself new is therefore impossible.

5. Structural reliability / limit-state learning

Echard, Gayton, Lemaire, 2011, "AK-MCS: An active learning reliability method combining Kriging and Monte Carlo Simulation," Structural Safety 33(2):145–154. DOI: 10.1016/j.strusafe.2011.01.002.

This literature is even more structurally similar. It explicitly assumes:

an expensive performance function (G(x));

safe and failure regions separated by (G(x)=0);

an expensive simulator/finite-element model;

sequential sampling concentrated near the limit-state surface;

the objective of reducing expensive simulator calls.

Bichon's EGRA and many later AK-MCS variants make this a mature line of work.

6. Adaptive stress testing

Lee et al., "Adaptive Stress Testing: Finding Likely Failure Events with Reinforcement Learning," 2018/2019.

AST targets likely failure trajectories in black-box stochastic simulators, including safety-critical autonomy applications. It is more trajectory-search than surrogate-map learning, but it is another strong prior against "smart safety simulation queries" as a novelty claim.

7. F-16 benchmark itself

Heidlauf, Collins, Bolender, Bak, 2018, "Verification Challenges in F-16 Ground Collision Avoidance and Other Automated Maneuvers," ARCH.

AeroBenchVV/AeroBenchVVPython deliberately exposes roughly 10–20-dimensional nonlinear hybrid aircraft dynamics as a verification benchmark. It is an excellent public benchmark, but many verification papers know it.

What is still plausibly novel

The F-16 direction survives if you make the target different from falsification or reliability estimation.

Your target should be:

Given a strict rollout budget, learn a small deployable student that reproduces the teacher's continuous safety-margin function over an operational envelope, and characterize when adaptive querying reduces the number of expensive rollouts needed to reach a specified map-fidelity target.

That is distinguishable along four axes:

Global/function fidelity rather than counterexample discovery.
Falsification asks whether a violation can be found. You ask whether the entire relevant safety-margin ranking/function can be copied.

Deployable compressed student rather than a GP analysis object.
The student should have an explicit inference-latency/size constraint and be evaluated as a substitute for online screening.

Crossover characterization rather than "our active sampler wins."
Explicitly show the low-dimensional/big-budget regime where Sobol/LHS catches or beats active sampling and the high-coverage-difficulty regime where active sampling wins.

Controlled dimension / thickness sweep.
The F-16 is one real point on the phase diagram; synthetic or semi-synthetic tasks establish the mechanism.

What not to claim

Do not claim:

"boundary-seeking is provably the right tool" unless you actually prove a theorem under stated smoothness/noise assumptions;

"F-16 safety-map surrogate learning is new";

"adaptive sampling for an expensive simulator is new";

"the failure set being thin makes our method novel";

"query cost causes active learning to outperform space filling."

A defensible claim is:

Boundary-focused active sampling has long been known to help expensive limit-state estimation. What has not been cleanly characterized in this setting is the empirical crossover between adaptive and space-filling query designs for faithful black-box map distillation, including regimes where the adaptive machinery is unnecessary.

2. Ranked alternative directions you are missing

These are intentionally directions not already central in the briefing.

Rank 1 — Cross-regime benchmark: a "phase diagram" of adaptive vs. space-filling distillation
Research question

Under what combinations of effective dimension, informative-region thickness, multimodality, query budget, and teacher cost does adaptive sampling provide a real advantage over Sobol/LHS/random space-filling for black-box function distillation?

This is the direction I recommend most strongly.

Expensive teacher

Use a family rather than a single teacher:

your existing ADBench black-box anomaly pipelines as the cheap/low-dimensional negative-control regime;

F-16 GCAS rollouts from AeroBenchVVPython as a real nonlinear safety function;

BenchBase/PostgreSQL benchmark runs as a wall-clock-expensive software-system teacher;

controlled analytic/simulation functions where dimension and boundary thickness can be swept exactly.

The real contribution is the cross-regime law, not any individual simulator.

Why the informative region is thin/high-dimensional

For a continuous teacher score (g(x)) with a deployment decision (g(x)\leq 0), the operationally important set is a band

[
\mathcal{I}_\tau={x: |g(x)|\leq \tau}
]

around the decision/limit-state surface.

You can control:

ambient dimension (D);

intrinsic dimension (m);

number of disconnected modes/components (k);

band thickness (\tau);

failure volume;

smoothness/nonlinearity.

This lets you test the briefing's mechanism directly rather than infer it post hoc from one dataset family.

Smart sampling method

Do not invent a baroque sampler. Use an intentionally simple, defensible acquisition function:

[
a(x)=
\underbrace{\exp(-|\hat g(x)|/T)}{\text{near predicted boundary}}
\times
\underbrace{\mathrm{Var}{j}[\hat g_j(x)]}_{\text{committee disagreement}}
]

with batch diversity enforced by k-center/farthest-first selection in feature/input space.

Ablate:

margin only;

disagreement only;

margin × disagreement;

margin × disagreement + diversity.

The point is not to win an acquisition-function contest. The point is to establish the regime where any informed acquisition is worth using.

Query-budget crossover experiment

For each regime, train at budgets such as:

[
B \in {16,32,64,128,256,512,1024}
]

scaled when dimension makes these obviously too small.

Compare:

iid random;

Sobol;

Latin hypercube/maximin;

your original shell;

uncertainty/disagreement;

boundary margin;

boundary × disagreement;

GP-LSE or AK-MCS-style baseline where computationally feasible.

Plot fidelity vs. number of teacher queries and fidelity vs. measured teacher wall-clock cost separately.

Estimate:

(B_{win}): first budget where active significantly beats the best passive baseline;

(B_{close}): budget where passive closes to within 5% of active;

query reduction required to hit a fixed fidelity target.

Public benchmarks

ADBench tabular datasets.

AeroBenchVVPython F-16 GCAS.

BenchBase + PostgreSQL (TPC-C/YCSB/SmallBank).

Public standard reliability/level-set test functions; Surjanovic-Bingham can be used for controlled analytic functions, but include canonical structural-reliability limit-state functions too.

Closest prior work

Pronzato & Müller, Design of computer experiments: space filling and beyond, 2012.

Crombecq et al., Efficient space-filling and non-collapsing sequential design strategies for simulation-based modeling, EJOR 2011.

Gotovos et al., IJCAI 2013 level-set estimation.

Echard et al., AK-MCS, 2011.

Qin et al., ICCPS 2022.

Dakota adaptive sampling.

Why this is still publishable

The individual ingredients are old. The publishable contribution is a carefully controlled regime characterization with strong negative controls:

"Adaptive sampling is not intrinsically superior. Its advantage emerges only when the passive design cannot cover the task-relevant set at the available query budget."

That is a much stronger scientific story than another acquisition function.

Single failure mode

If the crossover cannot be made systematic after controlling dimension/thickness/budget, then the proposed unifying law is too weak and the paper becomes an empirical benchmark paper rather than a mechanistic result.

Rank 2 — Database SLO-map distillation: query-efficient cloning of latency/throughput failure surfaces
Research question

Can a small student learn the SLO violation map of a real database/workload system with fewer expensive benchmark runs by querying configurations near the performance threshold rather than space-filling the whole configuration space?

Expensive teacher

A real DBMS deployment:

PostgreSQL;

BenchBase workload generator;

TPC-C, YCSB, SmallBank, or a mixture.

One teacher query is not a model inference. It is an actual warmup + benchmark run under a specific:

DB configuration;

workload rate;

concurrency;

workload mix;

cache/memory setting.

A query can naturally cost tens of seconds or minutes, which gives the query-efficiency story real operational meaning.

Why the informative region is thin/high-dimensional

Let the teacher output:

p95 or p99 latency;

throughput;

timeout/error rate;

optionally memory/CPU saturation.

Define a safety/SLO margin such as

[
g(x)=L_{\max}-\mathrm{p95Latency}(x).
]

The surface (g(x)=0) separates configurations that meet the SLO from those that do not.

Input variables can include:

10–30 PostgreSQL knobs;

offered load;

terminal/concurrency count;

read/write mixture;

scale factor;

buffer/cache controls.

The SLO boundary is a thin subset of a combinatorial mixed discrete/continuous space.

Smart sampling method

Two-stage pool-based active learning:

Sobol/LHS seed of (2D) to (4D) configurations.

Train a 5-member MLP or gradient-boosted ensemble.

Generate a large candidate pool of valid configurations.

Rank candidates by:

small predicted absolute SLO margin;

ensemble disagreement;

diversity from already measured configurations.

Run the real benchmark for the selected batch.

Repeat.

Query-budget crossover experiment

Budgets:

40, 80, 160, 320, 640 benchmark runs.

At each budget compare:

random;

Sobol;

LHS;

active boundary;

active uncertainty only;

FLASH-like sequential model-based baseline.

Primary target:

student-teacher fidelity on a large held-out measured set;

SLO classification error;

error within a narrow boundary band;

query count required for <2% SLO sign error or a fixed continuous-margin RMSE.

Public benchmark

BenchBase, the CMU multi-DBMS benchmarking framework, supports PostgreSQL and workloads including TPC-C, YCSB, SmallBank, Twitter, Wikipedia, SEATS, AuctionMark, and others.

Closest prior work

Nair et al., FLASH, IEEE TSE 2020, DOI: 10.1109/TSE.2018.2870895.

Ha & Zhang, DeepPerf, ICSE 2019, DOI: 10.1109/ICSE.2019.00113.

SPLConqueror/configurable-systems performance modeling.

CM-CASL, JSS 2023, active/semi-supervised performance modeling.

What is actually new enough

Do not sell this as "active learning for software configuration"; that is old.

Sell it as:

query-efficient fidelity learning of an entire SLO boundary under a fixed black-box benchmark budget, with a pre-registered active-vs-space-filling crossover analysis.

Optimization papers like FLASH are usually judged by finding good configurations, not by faithful reconstruction of the threshold/margin function over the operational envelope.

Single failure mode

Real benchmark noise can dominate the geometric effect. You need repeated measurements for a subset of configurations and either model heteroscedastic noise or define a tolerance band around the SLO.

Rank 3 — OpenFAST operating-envelope / ultimate-load map distillation
Research question

Can adaptive sampling reduce the number of expensive aero-hydro-servo-elastic simulations needed to learn a deployable surrogate of a wind turbine's limit-state margin over environmental and control conditions?

Expensive teacher

OpenFAST, NREL's open-source coupled wind-turbine simulation stack.

A teacher query runs a nonlinear time-domain simulation and produces responses such as:

blade-root bending moment;

tower-base load;

platform pitch;

rotor speed;

generator torque;

maximum structural response.

Define a margin to an engineering limit:

[
g(x)=L_{\mathrm{allow}}-\max_t L(t;x).
]

Why the informative region is thin/high-dimensional

Input space can include:

mean wind speed;

turbulence intensity;

yaw misalignment;

gust parameters;

wave height;

wave period;

wind-wave direction;

controller set points;

selected structural/control parameters.

The exceedance boundary (g(x)=0) is a lower-dimensional surface inside an 8–15D operating envelope.

Smart sampling

Use the same fixed method as Rank 1:

small Sobol seed;

ensemble surrogate;

acquisition = near-limit × disagreement;

diversity filter.

The same sampler across domains strengthens the regime paper.

Query-budget crossover

Budgets of roughly 100–2,000 simulations depending on runtime and turbine model.

Compare active to:

Sobol;

LHS;

iid random;

GP/AK-MCS-style active reliability baseline.

Measure:

continuous margin fidelity;

sign accuracy;

boundary-weighted error;

limit-state distance on low-dimensional slices;

query count to a fixed error.

Public benchmark/simulator

OpenFAST.

NREL 5-MW reference turbine and/or IEA 15-MW reference turbine.

Public OpenFAST regression/reference cases.

Closest prior work

This is not empty territory:

structural-reliability active learning and AK-MCS;

active-learning reliability methods applied to wind-turbine structures;

OpenFAST surrogate modeling for loads;

recent multi-fidelity OpenFAST surrogate work.

Novelty verdict

Moderate, not high. The value is a strong real engineering point on your crossover diagram, not a standalone claim that active learning can save simulations.

Single failure mode

The setup burden can consume the entire 2–3 month window, and reviewers from wind engineering may demand domain-specific validation well beyond the methodological question.

Rank 4 — Power-grid transient-stability margin distillation
Research question

Learn a fast surrogate of transient-stability margin over uncertain operating conditions and contingencies using fewer full time-domain simulations.

Expensive teacher

Dynamic simulation of an IEEE test system using a public simulator such as ANDES or another reproducible dynamic power-system tool.

Outputs:

stable/unstable;

critical clearing time;

rotor-angle stability margin;

frequency nadir or recovery margin.

Why thin/high-dimensional

The transient-stability boundary is a classic limit-state surface in a high-dimensional space of:

generator dispatch;

load levels;

renewable injections;

fault location/type;

clearing time;

topology.

Smart sampler

uncertainty + predicted-margin proximity;

batch diversity;

optionally GP/Kriging for small-dimensional variants.

Public benchmark

IEEE 39-bus New England system;

larger IEEE/WECC-style public cases where reproducible data are available.

Closest prior work

This direction is already heavily occupied:

active learning for transient stability assessment has existed for years;

Zhang et al. 2021 explicitly use active learning to reduce time-domain simulations;

2026 work uses Kriging-based active learning for rare transient-instability events on IEEE 59-bus and WECC 240-bus systems;

additional 2026 work uses multi-metric adaptive active learning on IEEE 39-bus.

Verdict

Do not make this the main pivot. It is almost a textbook instance of your mechanism, which is useful as validation but poor for novelty.

Single failure mode

A reviewer can plausibly say: "This is another active-learning transient-stability paper with a different student."

Rank 5 — Robot-policy operating-envelope distillation
Research question

Given an expensive closed-loop robot simulator and fixed controller/policy, learn a compact map from scenario parameters to probability/margin of task failure.

Expensive teacher

Possible public systems:

Isaac Sim;

MuJoCo;

CommonRoad/vehicle simulation;

Scenic + CARLA/Webots.

A query is a complete episode/rollout.

Why thin/high-dimensional

The success/failure surface lives in a scenario space containing:

initial poses;

velocities;

friction;

object locations;

masses;

disturbances;

perception noise;

policy/environment parameters.

Smart sampler

Boundary-focused disagreement sampling with k-center diversity.

Query crossover

Vary:

scenario dimension;

episode budget;

disturbance variance;

failure rarity.

Compare to random, Sobol/LHS, cross-entropy, and falsification methods.

Closest prior work

This is crowded from several sides:

adaptive stress testing;

VerifAI/Scenic;

rare-event simulation for AV validation;

CPS falsification;

recent boundary-focused robot failure discovery such as ROBOGATE (2026).

Verdict

Conceptually clean, but not a good 2–3 month novelty bet.

Single failure mode

You spend substantial compute and engineering effort only to rediscover an active-testing result the robotics-safety community already expects.

3. SINGLE best bet and concrete experimental plan
Pick

Build the paper around the crossover law, not around F-16.

Working research question:

When does adaptive query selection reduce the number of black-box evaluations required to faithfully distill a teacher function, relative to strong passive space-filling designs?

The original anomaly study becomes the empirical anchor for the "adaptive does not help" regime. F-16 and one second real expensive domain become positive regimes.

The paper's central hypothesis

Let:

(B): teacher-query budget;

(D): ambient dimension;

(m): intrinsic dimension of the relevant support;

(\tau): thickness of the task-relevant boundary band;

(k): number of disconnected relevant regions;

(C_q): measured cost per teacher query.

Define a coverage-difficulty variable conceptually as

[
\kappa = \frac{N_{\text{passive}}(\mathcal I_\tau)}{B},
]

where (N_{\text{passive}}(\mathcal I_\tau)) is the number of passive points required to cover the informative set at a fixed resolution.

You do not need to estimate this exact quantity perfectly. The experimental claim can use controlled proxies (D), (D/m), (\tau), and (k).

Sharpened hypothesis

At low coverage difficulty, Sobol/LHS is competitive or better.

As coverage difficulty rises relative to (B), adaptive boundary sampling gains query efficiency.

As (B) becomes large, passive methods eventually catch up.

Higher (C_q) does not cause the statistical crossover; it increases the practical value of a given query reduction.

That is much more defensible than "smart sampling pays iff queries are expensive and the region is thin."

Experimental suite
Family A — Original anomaly-distillation regime: mandatory negative control

Use exactly the already careful setup:

21–24 ADBench tabular datasets;

kNN, KDE, one-class SVM, Isolation Forest, autoencoder teachers;

normals-only;

uniform shell;

uniform box;

current best active strategies if desired;

10 or preferably 20 paired seeds.

Primary purpose:

Demonstrate that in a cheap, low-dimensional/fillable space, sophisticated sampling does not beat strong space filling.

Do not hide the failure. Put it in Figure 1.

Family B — Controlled geometry sweep: mechanism test

Create teacher functions where ground truth can be evaluated cheaply for exhaustive analysis, but count calls as if they were black-box queries.

Sweep:

(D \in {2,4,8,16,32,64});

intrinsic dimension (m \in {2,4,8});

(D/m);

boundary thickness (\tau);

number of disconnected modes (k \in {1,2,4,8});

smooth vs non-smooth limit-state surfaces;

balanced vs rare failure regions.

Important: do not use artificial sleep time to claim sample efficiency. These functions establish geometry. Actual simulator runtimes establish economic cost.

Construct functions such as:

hypersphere/ellipsoid boundaries;

rotated anisotropic boundaries;

multiple disconnected spheres/components;

nonlinear sinusoidal/ridge limit states;

low-dimensional latent functions randomly embedded in a higher-dimensional ambient space;

canonical structural-reliability limit-state test functions.

Family C — F-16 GCAS: real nonlinear dynamic safety map

Use AeroBenchVVPython.

Teacher output:

[
g(x)=\min_t h(t;x)
]

or a quantitative safety robustness margin with zero at ground collision.

Input envelope:

Start with the standard benchmark dimensions, then build nested parameter sets:

2D;

4D;

8D;

as high as physically meaningful without introducing nonsense parameters.

This creates a real dimension sweep rather than comparing unrelated tasks.

Family D — one second real expensive domain

Preferred: BenchBase/PostgreSQL SLO map.

Why I prefer it over OpenFAST for a 2–3 month paper:

setup is easier;

queries have obvious wall-clock cost;

mixed discrete/continuous dimension is naturally large;

the SLO threshold creates a genuine thin boundary;

it demonstrates the rule outside physics/safety verification.

If you already have OpenFAST expertise, OpenFAST can replace BenchBase.

Fixed student

Avoid turning the paper into architecture search.

Use one primary student family:

2–3 hidden-layer MLP;

fixed parameter budget, e.g. 50k–200k parameters;

5 independent ensemble members for acquisition;

same architecture family across methods.

Add one non-neural check:

gradient-boosted trees or random forest.

If conclusions change completely with the student, that is itself important and should be reported.

Sampling baselines
Passive baselines

iid uniform/random.

Sobol sequence.

Latin hypercube.

maximin LHS if affordable.

original uniform shell where applicable.

normals-only for the original anomaly case.

Sobol should be treated as the primary passive opponent, not random sampling. Your own results already show why weak random baselines are misleading.

Adaptive baselines

uncertainty/disagreement only;

predicted-margin proximity only;

margin × disagreement;

margin × disagreement + diversity;

GP-LSE on low/moderate-dimensional tasks;

AK-MCS-style learning function on reliability-style tasks when feasible.

Do not add 15 acquisition functions. Five strong baselines are enough.

Proposed acquisition rule

Maintain ensemble ({\hat g_j}_{j=1}^M).

For candidate (x),

[
u(x)=\mathrm{Var}_j[\hat g_j(x)]
]

and

[
b(x)=\exp\left(-\frac{|\bar g(x)|}{T}\right).
]

Score:

[
a(x)=u(x),b(x).
]

For a batch, take a large top-scoring pool, then select by farthest-first/k-center to avoid querying duplicates from the same local patch.

This method is deliberately simple. If a simple standard method exhibits the predicted crossover, the regime claim is stronger.

Axes that must be pre-registered
Axis 1 — dimension

Controlled (D) and (D/m).

Axis 2 — boundary thickness / rarity

Use either:

(\tau), width of the evaluation band around the limit state;

failure volume;

or both.

Axis 3 — query budget

Log-spaced budgets.

Axis 4 — student capacity

At least:

small;

medium.

You need to ensure that "active fails" is not merely student under-capacity.

Axis 5 — teacher cost

Use measured wall-clock or monetary cost for real teachers.

Report cost separately from query count.

Primary metrics

Do not rely only on global RMSE. A huge safe interior can make a bad safety map look good.

1. Global score fidelity

Spearman rank correlation;

normalized RMSE or MAE.

Spearman keeps continuity with the anomaly project.

2. Boundary-band fidelity

Evaluate separately on:

[
|g(x)| \le \tau_{\text{eval}}.
]

Metrics:

Spearman;

MAE;

sign error.

3. Decision fidelity

balanced accuracy;

AUROC only if threshold varies;

false-safe rate and false-unsafe rate at the deployed threshold.

For safety applications, false-safe error deserves explicit reporting.

4. Geometric error

For low-dimensional controlled/F-16 slices:

symmetric Chamfer distance or Hausdorff-style distance between true and predicted (g(x)=0) contours.

5. Query efficiency

For target fidelity (F^*):

[
Q(F^*)=\min{B(B)\ge F^*}.
]

Report query reduction:

[
1-\frac{Q_{\text{active}}}{Q_{\text{Sobol}}}.
]

6. Cost efficiency

[
\text{cost to target}=Q(F^*)\times \mathrm{median\ measured\ query\ cost}.
]

For noisy DB benchmarks, use actual cumulative measured cost rather than multiplying by a median.

7. Deployment value

student inference latency;

teacher/student speedup;

student memory/parameter count.

This preserves the original distillation motivation.

The pre-registered crossover invariant

This is the key part of the paper.

I would pre-register three invariants.

Invariant A — negative-control regime

For low-dimensional/fillable settings at moderate-to-large budget:

Adaptive sampling must not be claimed superior unless its 95% paired confidence interval excludes zero against Sobol.

Expected result: no meaningful advantage, and possibly a Sobol win.

This formally protects the original negative result.

Invariant B — high-coverage-difficulty regime

For pre-specified high-(D/m), thin-boundary settings:

To count as a successful positive regime, adaptive sampling must reduce the median teacher-query count required to reach the fixed target boundary-fidelity level by at least 30% relative to Sobol, with a positive paired effect in at least 80% of seeds/task instances.

Do not lower this threshold after seeing results.

Invariant C — crossover shift

Define (B_{\text{close}}) as the smallest budget after the active advantage appears at which Sobol comes back to within 5% of active on the primary boundary-fidelity metric.

Pre-register:

Across the controlled geometry sweep, (B_{\text{close}}) should increase as coverage difficulty increases, operationalized separately by increasing (D/m), decreasing (\tau), or increasing disconnected-mode count (k).

This is a much more interesting result than "active > random."

If it fails, report that the simple coverage hypothesis is incomplete.

Statistics

Use:

20 seeds for synthetic/cheap tasks;

at least 10 seeds for expensive teachers;

paired comparisons at identical budgets;

bootstrap 95% CIs for query-to-target metrics;

Holm correction within each experiment family;

effect sizes, not only p-values;

area under the fidelity-vs-log-query-budget curve as a secondary aggregate measure.

Avoid testing every budget independently and then highlighting the most favorable point.

The figure set I would build
Figure 1 — The failed original thesis

ADBench: shell vs normals vs uniform box, with the strong negative result visible.

Caption message:

Sophisticated off-manifold proximity is not automatically informative.

Figure 2 — Controlled phase diagram

Heat map:

x-axis: log query budget / dimension;

y-axis: (D/m) or boundary thinness;

color: active minus Sobol fidelity.

This is potentially the paper's signature figure.

Figure 3 — Crossover curves

Fidelity vs. teacher queries at:

low coverage difficulty;

medium;

high.

Show the actual crossing/catching-up behavior.

Figure 4 — F-16 boundary map

2D slice with:

true simulator margin;

Sobol-trained student;

active-trained student;

queried points.

Figure 5 — Real expensive domain

BenchBase SLO margin vs. query budget and cumulative benchmark time.

Figure 6 — Query savings vs. measured query cost

Keep this explicitly economic, not causal.

Win condition for the whole paper

I would call the paper successful only if all of these hold:

Original anomaly negative result remains robust.

Controlled sweep shows a reproducible geometry/budget crossover.

At least one real expensive high-coverage-difficulty teacher shows ≥30% query reduction at fixed fidelity.

At least one real or controlled low-coverage-difficulty case shows no adaptive advantage.

The effect survives Sobol/LHS, not merely iid random.

The conclusion survives at least two student families.

Query cost is reported separately from sample-efficiency geometry.

If only item 3 holds, you have an application paper.
If only item 2 holds, you have a methodological simulation paper.
If 1–7 hold, you have the strong general paper.

4. Directions that are traps
Trap 1 — Keep anomaly detection and invent sampler #4

Avoid.

Your own experiments already show:

shell not special;

uniform box wins;

attribution-aware disagreement sampler fails;

multi-head sampling does not help;

coverage predictor is weak.

Adding another synthetic sampler after this looks like fishing for a positive result.

The external literature makes this worse:

Hong & Kang 2024 already do anomaly-score distillation;

anomaly teacher-student distillation is a large field;

synthetic anomaly generation/distillation is already common.

The honest contribution from the anomaly study is the negative regime, not another anomaly-specific method.

Trap 2 — F-16 paper whose novelty is "actively sample near the safety boundary"

Avoid as a standalone thesis.

Qin et al. 2022 already use surrogate refinement on F-16 GCAS. S-TaLiRo, VerifAI, GP level-set estimation, and reliability analysis make the generic idea old.

Use F-16 as a strong benchmark in the regime paper.

Trap 3 — Foundation interatomic potential distillation via active frame selection

Avoid unless you have a genuinely different question.

Active learning/committee uncertainty for molecular potentials is already standard practice. Querying expensive DFT labels selectively is one of the canonical active-learning success stories.

You would be entering a mature domain with substantial compute and domain-expertise expectations simply to demonstrate a mechanism everyone there already accepts.

Trap 4 — Generic CFD/FEA surrogate + adaptive sampling

Avoid as the primary novelty.

"Expensive simulator + adaptive design of experiments + surrogate" is an enormous literature.

AirfRANS is useful as data, but if you only query a precomputed dataset then "teacher query expense" becomes artificial. If you rerun OpenFOAM, you gain authentic cost but inherit substantial simulation engineering.

Use CFD only if the paper's contribution is the cross-regime law.

Trap 5 — Generic black-box model extraction

Avoid.

Closest work includes:

Orekondy et al., Knockoff Nets, CVPR 2019;

Pal et al., ActiveThief, AAAI 2020;

Truong et al., Data-Free Model Extraction, 2020/2021;

numerous query-efficient extraction attacks and synthetic-query methods.

This literature already asks how to choose limited black-box queries to maximize substitute-model fidelity.

Your anomaly setting had a special unsupervised/off-manifold story, but after the shell hypothesis failed, moving to generic extraction sacrifices most of the distinctive motivation.

Trap 6 — LLM API distillation under token cost

Avoid for this paper.

It is timely but exceptionally crowded, changes quickly, and makes controlled "ground truth" fidelity difficult because:

outputs are stochastic;

prompts are discrete/semantic rather than a clean continuous geometry;

API/model versions change;

model extraction and active data selection for LLMs are already fast-moving.

It would sever the clean mechanistic connection to your negative result.

Trap 7 — Power-grid transient stability as the hero

Avoid.

Recent work is especially close:

active-learning TSA already existed before 2021;

2026 papers explicitly target rare transient instability with Kriging active learning;

2026 work uses multi-metric adaptive active learning and IEEE 39-bus.

It is a validation domain, not a novelty domain.

Trap 8 — Root-cause attribution as the rescue

Avoid unless the target representation changes completely.

Your own result is telling you something substantive: reproducing the aggregate anomaly score/decision can be much easier than reproducing discrete attribution/ranking of residual channels.

Attribution is unstable because small score perturbations can change argmax/top-k identities even when aggregate score fidelity is good.

A new sampler is unlikely to fix a target-instability problem.

Trap 9 — Multi-head ensemble distillation as the rescue

Avoid.

You already have the decisive result: sampling does not improve per-head fidelity. Unless the new paper is explicitly a negative study of when black-box ensemble decomposition is not identifiable, there is no reason to keep pushing this branch.

Trap 10 — Claim "expensive queries are necessary for active learning to work"

Conceptual trap.

Expense is not a statistical requirement.

Example:

A 100-dimensional cheap analytic teacher can still show an enormous query-count advantage for active boundary learning.

A one-hour 2D smooth simulator may still be best handled by a 30-point space-filling design.

The correct decomposition is:

f(\text{geometry},\text{target loss},B,\text{noise},\text{student})
]

while

[
\text{practical value}
\approx
\text{queries saved}\times\text{cost per query}.
]

That distinction will make the final paper stronger.

5. Compelling and honest paper framing/title
Recommended framing

Do not frame the anomaly result as a failed preliminary experiment that motivated a different application.

Frame it as the counterexample that exposed the real scientific question:

We began from a seemingly intuitive premise: to distill the off-manifold behavior of an unsupervised anomaly detector, synthetic queries should be concentrated just outside the normal manifold. Across a large controlled study this premise failed; simple space-filling queries were substantially better. This result suggested a more general question: when should adaptive query selection beat space filling at all? We show that the answer is not "whenever the teacher is black-box" or "whenever the boundary matters." The advantage appears when the query budget is small relative to the coverage complexity of the task-relevant region, and disappears when that region can be adequately space-filled. We validate the crossover in controlled geometries and real black-box systems, while separating statistical query efficiency from the economic cost of each teacher call.

That is scientifically stronger than trying to hide the refutation.

Best title
When Active Queries Pay: Regime Crossovers in Black-Box Surrogate Distillation

This is my first choice.

It is:

broad enough for anomaly + safety + systems;

honest;

not tied to a method that may later lose;

centered on the actual contribution.

Other strong titles
Beyond Space Filling: When Adaptive Sampling Helps Black-Box Function Distillation

Clear and conservative.

The Limits of Smart Sampling: Query-Budget Crossovers in Black-Box Distillation

Best if the negative result is visually central.

From Failed Shell Sampling to Query-Efficient Surrogates: A Regime Map for Black-Box Distillation

Memorable, but I would use this for a talk rather than the final paper title.

When Does Active Sampling Beat Space Filling? A Controlled Study of Black-Box Map Distillation

Very explicit and reviewer-friendly.

Geometry, Budget, and Cost in Black-Box Surrogate Distillation

More journal-like and less catchy.

Recommended claim hierarchy

The paper should make claims in this order.

Claim 1 — Negative empirical fact

In the original unsupervised anomaly-distillation regime, near-manifold shell sampling is not privileged; strong passive space filling is better.

This is already supported by your 21–24 dataset, 5-teacher, multi-seed study.

Claim 2 — Mechanistic hypothesis

Adaptive sampling becomes useful when passive coverage of the task-relevant set is poor relative to the available query budget.

This is broader and cleaner than "off-manifold points are useful."

Claim 3 — Controlled crossover

As (D/m), boundary thinness, or disconnectedness increase, the budget at which space filling catches active sampling shifts upward.

This is the paper's scientific centerpiece.

Claim 4 — Real-system validation

The predicted high-coverage-difficulty regime occurs in at least one real expensive black-box system such as F-16 safety-margin mapping or database SLO mapping.

Claim 5 — Economic consequence

When teacher calls are expensive, the query-count reduction yields meaningful wall-clock/compute savings.

This should be the final consequence, not the mechanism.

Closest-prior-work map
Your component	Closest prior	What they already establish	What you must add
Small student mimics anomaly score	Hong & Kang, Score Distillation for Anomaly Detection, KBS 2024	anomaly-score distillation itself	black-box query-design regime, not score KD
Teacher-student anomaly models	STPM/MKD/DTSNE and many later works	normal-only teacher/student anomaly learning	black-box score fidelity under query budget
Synthetic queries for substitute models	Papernot-style substitute training; Data-Free Model Extraction	generated queries can train a black-box substitute	regime crossover with space filling
Query-efficient extraction	Knockoff Nets; ActiveThief	active/public-data query selection can improve extraction	continuous map/limit-state fidelity, negative regimes
Expensive simulator surrogate	classical computer experiments	LHS/Sobol, sequential surrogate design	explicit active/passive crossover law
Thin failure boundary	AK-MCS / EGRA	sample near limit state	map-distillation fidelity and regime map
Threshold-set learning	Gotovos et al. 2013	GP level-set active learning with theory	compression + empirical high-D crossover
CPS robustness/surrogate	Qin et al. 2022	surrogate quantitative robustness; F-16 GCAS	deployable student + crossover, not verification guarantee
CPS active testing	S-TaLiRo, VerifAI, AST	intelligent simulation finds failures	globally faithful student map
Software configuration	FLASH, DeepPerf, CM-CASL	few expensive measurements can model/optimize configs	SLO-boundary fidelity and cross-regime test
Power-grid stability	multiple AL-TSA papers	active simulation labeling saves TDS calls	little novelty left
Molecular potentials	active-learning potentials	committee uncertainty for expensive labels	little novelty unless claim changes
Realistic paper shape
Introduction

Distillation of black-box pipelines is attractive when deployment cost prevents direct inference.

In anomaly detection, anomalies are absent at training time, suggesting synthetic off-manifold queries.

Intuitive hypothesis: query near the normal manifold.

Large study falsifies it: space filling wins.

This exposes the real question: when does smart/adaptive sampling earn its complexity?

Contributions:

negative anomaly result;

controlled regime sweep;

crossover metric/invariants;

real safety/system validation;

separation of sample efficiency from teacher economics.

Related work

Four explicit subsections:

anomaly-detection distillation;

model extraction/data-free KD;

active learning/level-set/reliability analysis;

design of computer experiments/surrogate modeling and CPS falsification.

Do not bury AK-MCS or Qin et al.; reviewers will know them.

Methods

No need for a highly novel network.

Define:

teacher;

student;

passive design;

adaptive design;

task-relevant band;

query budget;

cost;

crossover metrics.

Experiments

negative anomaly regime;

controlled geometry;

F-16;

one orthogonal expensive real system.

Discussion

The most important section.

State explicitly:

why shell failed;

why adaptive works elsewhere;

why teacher cost and geometric difficulty are distinct;

where passive designs remain preferable;

where the adaptive overhead is not justified.

Final recommendation

Do not publish the F-16 GCAS idea as "a novel active safety-map sampler." The closest prior work is too close, especially Qin et al. 2022 plus the mature level-set/reliability/falsification literature.

Do use F-16 as the strongest positive real-system benchmark in a broader paper.

The single best pivot is:

Turn the anomaly refutation into a controlled study of the active-vs-space-filling crossover in black-box surrogate distillation.

The anomaly experiments give you something most active-learning papers do not have: a large, careful regime in which the sophisticated method fails. That negative control is valuable if the next experiments produce the complementary positive regime under pre-registered geometry/budget conditions.

The paper becomes substantially stronger if it can say:

We do not propose another sampler that wins on selected benchmarks. We identify when adaptive sampling should be expected to help, demonstrate a regime where it does not, and show the crossover as coverage difficulty increases.

That is the paper I would pursue.

Selected references

Hong, J.; Kang, S. Score distillation for anomaly detection. Knowledge-Based Systems (2024), 111842. DOI: 10.1016/j.knosys.2024.111842.

Orekondy, T.; Schiele, B.; Fritz, M. Knockoff Nets: Stealing Functionality of Black-Box Models. CVPR 2019. DOI: 10.1109/CVPR.2019.00509.

Pal, S. et al. ActiveThief: Model Extraction Using Active Learning and Unannotated Public Data. AAAI 2020. DOI: 10.1609/aaai.v34i01.5432.

Truong, J.-B.; Maini, P.; Walls, R. J.; Papernot, N. Data-Free Model Extraction. arXiv:2011.14779.

Gotovos, A.; Casati, N.; Hitz, G.; Krause, A. Active Learning for Level Set Estimation. IJCAI 2013, 1344–1350.

Echard, B.; Gayton, N.; Lemaire, M. AK-MCS: An active learning reliability method combining Kriging and Monte Carlo Simulation. Structural Safety 33(2), 2011, 145–154. DOI: 10.1016/j.strusafe.2011.01.002.

Bichon, B. et al. Efficient Global Reliability Analysis for Nonlinear Implicit Performance Functions. AIAA Journal 2008. DOI: 10.2514/1.34321.

Annpureddy, Y.; Liu, C.; Fainekos, G.; Sankaranarayanan, S. S-TaLiRo: A Tool for Temporal Logic Falsification for Hybrid Systems. TACAS 2011. DOI: 10.1007/978-3-642-19835-9_21.

Donzé, A. Breach, A Toolbox for Verification and Parameter Synthesis of Hybrid Systems. CAV 2010. DOI: 10.1007/978-3-642-14295-6_17.

Dreossi, T. et al. VerifAI: A Toolkit for the Formal Design and Analysis of Artificial Intelligence-Based Systems. CAV 2019.

Qin, X.; Xian, Y.; Zutshi, A.; Fan, C.; Deshmukh, J. V. Statistical Verification of Cyber-Physical Systems using Surrogate Models and Conformal Inference. ICCPS 2022. DOI: 10.1109/ICCPS54341.2022.00017.

Heidlauf, P.; Collins, A.; Bolender, M.; Bak, S. Verification Challenges in F-16 Ground Collision Avoidance and Other Automated Maneuvers. ARCH 2018.

Lee, R. et al. Adaptive Stress Testing: Finding Likely Failure Events with Reinforcement Learning. arXiv:1811.02188.

Pronzato, L.; Müller, W. G. Design of computer experiments: space filling and beyond. Statistics and Computing 22, 2012, 681–701. DOI: 10.1007/s11222-011-9242-3.

Crombecq, K.; Laermans, E.; Dhaene, T. Efficient space-filling and non-collapsing sequential design strategies for simulation-based modeling. European Journal of Operational Research 214(3), 2011, 683–696. DOI: 10.1016/j.ejor.2011.05.032.

Ha, H.; Zhang, H. DeepPerf: Performance Prediction for Configurable Software with Deep Sparse Neural Network. ICSE 2019. DOI: 10.1109/ICSE.2019.00113.

Nair, V.; Yu, Z.; Menzies, T.; Siegmund, N.; Apel, S. Finding Faster Configurations Using FLASH. IEEE Transactions on Software Engineering 46(7), 2020, 794–811. DOI: 10.1109/TSE.2018.2870895.

AirfRANS: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier–Stokes Solutions. NeurIPS 2022 Datasets and Benchmarks.

OpenFAST, NREL open-source aero-hydro-servo-elastic wind-turbine simulation framework.

Khandait et al. ARCH-COMP 2025 Category Report: Falsification. EPiC Series in Computing 108, 2025, 169–189. DOI: 10.29007/dgnn.

Liu, J.; Wang, X.; Wang, X. Probabilistic Assessment of Rare Transient Instability Events via Kriging-based Active Learning Framework. International Journal of Electrical Power & Energy Systems 178 (2026), 111915. DOI: 10.1016/j.ijepes.2026.111915.

Zhang, Y.; Zhao, Q.; Tan, B.; Yang, J. A power system transient stability assessment method based on active learning. The Journal of Engineering 2021. DOI: 10.1049/tje2.12068.

Tang, W.; Dang, C.; Xu, J. Parallel active learning XGBoost for structural reliability analysis with application to an onshore wind turbine tower. Reliability Engineering & System Safety 265 (2026), 111390. DOI: 10.1016/j.ress.2025.111390.