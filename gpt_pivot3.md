Library
/
blackbox_distillation_new_pivot.md

The project distills a query-only, undeployable unsupervised anomaly-detection teacher into a small deployable student, and the current evidence says passive space-filling queries are essentially as effective as a reasonable adaptive boundary-seeking learner, so the next contribution must come from changing what behavior is distilled, not from another query policy.

One new pivot: Order-Canonical Distillation — anomaly scores modulo monotone transformations
(a) New hypothesis / reframing

An anomaly score is generally not a cardinal quantity: if a teacher score s(x) and its alarm threshold tau are transformed by any strictly increasing function phi, then (phi(s(x)), phi(tau)) implements exactly the same ranking and exactly the same alarm decisions. The project should therefore pivot from query placement to behavioral equivalence: distill the teacher's order/quantile structure rather than its arbitrary raw score scale, using an order-canonical target such as the empirical score quantile. The hypothesis is that, at the same Sobol queries and same query budget, order-canonical distillation gives materially lower alarm-decision regret and is invariant to behavior-preserving monotone rewrites of the teacher score, while ordinary raw-score MSE distillation is not.

(b) Why this is novel, impactful, and who would cite it

The important claim is not that ranking losses or quantile transforms are new. The publishable claim is that black-box anomaly-detector distillation is currently posed on the wrong mathematical object: a numerical anomaly score should be treated as an equivalence class under strictly monotone transformations, because those transformations leave the teacher's operational behavior unchanged. A distillation method that produces different deployed students for two behaviorally identical parameterizations of the same teacher fails a basic invariance criterion.

That gives the paper a much stronger center than another sampler comparison:

Principle: define behavior-preserving distillation for anomaly detectors modulo monotone score reparameterization.

Failure test for incumbents: take one teacher and rewrite its score by several strictly monotone nonlinear functions; a correct distillation procedure should not materially change its deployed alarm behavior.

Simple canonicalization: replace raw teacher values by their empirical CDF / normalized rank on the query set, then train the same small MLP student on that bounded canonical target. This requires zero extra teacher queries.

Mechanistic story: raw-score regression spends capacity reproducing arbitrary calibration, tails, and curvature that do not affect the teacher's decisions; order-canonical regression spends capacity only on the behavior that survives all monotone reparameterizations.

If the effect is real, the work is relevant to three communities: anomaly detection (heterogeneous detectors whose scores have arbitrary scales), black-box model extraction / knowledge distillation (what constitutes behavioral fidelity), and surrogate modeling / deployed monitoring (small surrogates that must preserve alarm decisions rather than numerical artifacts). A sufficiently broad benchmark could plausibly target data-mining / anomaly-detection methods venues such as KDD, ICDM, or ECML-PKDD, with workshop-level publication possible from the controlled invariance result alone.

The reason this pivot survives the null you just obtained is that it makes no claim that adaptive querying should beat Sobol. In fact, your negative result becomes supporting evidence: once space-filling is already near-minimax for acquiring locations, the remaining avoidable error may be in the representation of the black-box response rather than in where queries are placed.

(c) Single cheapest confirm-or-kill experiment
Experiment: monotone-equivalence stress test with no new teacher queries

Reuse the existing ADBench teacher/query outputs for the current strongest passive baseline: Sobol + small MLP raw-score regression. For each existing dataset × teacher-family cell, keep the exact same query points, query budget, train/test split, architecture, optimizer, and seed.

For each teacher score vector s, first robust-standardize it on the query set to z, then create four behaviorally equivalent teacher parameterizations:

phi_1(z) = z

phi_2(z) = asinh(z)

phi_3(z) = z^3

phi_4(z) = sigmoid(z)

All four are strictly increasing, so they induce exactly the same teacher ordering. Standardize each transformed target before fitting the raw-score baseline, so the test is not merely detecting trivial scale differences.

Train only two students on the same Sobol queries:

Incumbent: the current small MLP minimizing MSE to the transformed raw teacher score.

Pivot: the identical MLP minimizing MSE to the teacher's normalized empirical rank

u_i = (rank(s_i) - 0.5) / B.

Because ranks are unchanged by every phi_j, the pivot receives exactly the same canonical labels under all four behaviorally equivalent teacher parameterizations.

Exact primary metric

On the held-out ADBench test points, let the deployed system have a fixed 5% alarm budget. Define

A_T = teacher top-5% points

and

A_S = student top-5% points.

Measure alarm-set regret

R = 1 - |A_T intersect A_S| / |A_T|.

R = 0 means the distilled student sends exactly the same cases to the alarm queue as the teacher; R = 0.20 means 20% of the teacher's alarm slots are replaced by different cases. This is a deployed-decision metric, not score correlation or R-squared.

For each dataset × teacher cell, take the worst-case R across the four monotone-equivalent score parameterizations. The single comparison is then

DeltaR = R_worst(raw-score MSE) - R_worst(order-canonical).

Pre-registered positive lead

Call the pivot alive only if all of the following hold across the existing ADBench × teacher cells:

median DeltaR >= 0.05 absolute alarm-regret reduction;

order-canonical distillation wins on at least 70% of cells;

the paired 95% bootstrap CI for the mean DeltaR excludes zero;

on the identity parameterization phi_1, order-canonical distillation is not worse than raw-score MSE by more than 0.01 median regret.

A result such as raw-score worst-case regret 0.18 versus order-canonical 0.11 would be a strong positive lead: the same teacher behavior, same samples, same architecture, and same query budget, but a 7-point reduction in deployed alarm disagreement purely from choosing the correct invariant target.

Kill criterion

Kill this pivot if the median worst-case improvement is < 0.02, or if order-canonical distillation wins on < 60% of cells. In that case, arbitrary score parameterization is not a material bottleneck, and turning the project into a monotone-invariance paper would be mostly formalism without operational payoff.

Blunt verdict

Stop spending effort on query acquisition. Your two controlled tests already say that a competent adaptive policy does not buy meaningful query efficiency over Sobol in the regime you can study cleanly. The strongest remaining pivot on the assets already on disk is to ask whether you have been distilling an arbitrary numerical representation of the teacher instead of its invariant operational behavior; the monotone-equivalence stress test above can confirm or kill that idea with essentially only a student retraining sweep.