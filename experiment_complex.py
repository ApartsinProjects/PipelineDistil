"""Complex-pipeline teacher: does sampling let a student reproduce a
NON-MONOTONIC anomaly pipeline that normals-only distillation cannot?

Motivation. The other experiments use monotone distance-like teachers (KDE,
IsolationForest, kNN): their anomaly score rises smoothly away from the normal
manifold, so a student fit on normals ALONE already extrapolates them, and
sampling adds nothing (confirmed on real data). The paper's actual claim is
about a COMPLICATED multi-step pipeline. The canonical hard case is an
autoencoder: a nonlinear encode -> decode -> reconstruction-error chain whose
off-manifold score is NON-MONOTONIC (it reconstructs some anomalies well and
others poorly), so its behavior on anomalous inputs genuinely cannot be guessed
from normals. This is exactly the regime where copying the pipeline's behavior
onto the student REQUIRES sampling the anomalous region.

Teacher pipeline P(x) (3 steps):
  1. standardize (fit on normals)
  2. autoencoder reconstruction error: e(x) = || x - AE(x) ||^2, AE trained on
     normals with a bottleneck (nonlinear, so e is non-monotonic off-manifold)
  3. density gate: P(x) = e(x) * (1 + kde_nll(x)), a non-linear combination of
     reconstruction error and negative log density

We distill P into a small student with each sampler and measure how faithfully
the student reproduces P ON HELD-OUT ANOMALIES (Spearman rank + RMSE), which is
the metric that matches the "copy the pipeline behavior on anomalies" claim.

Usage:
  python experiment_complex.py --seeds 5
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import make_moons
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiment import (
    make_percentile_maps, train_student, auroc,
    sampler_none, sampler_gaussian, sampler_langevin, sampler_langevin_adaptive,
)

N_TRAIN = 2000
N_ANOM = 500


# ---------------------------------------------------------------------------
# Complex teacher pipeline
# ---------------------------------------------------------------------------
class ComplexPipeline:
    """A 3-step blackbox anomaly pipeline: standardize -> AE recon error ->
    density-gated combination. Only queryable as P(x) -> scalar score."""

    def __init__(self, X_normals, seed=0):
        self.scaler = StandardScaler().fit(X_normals)
        Xs = self.scaler.transform(X_normals)
        d = Xs.shape[1]
        # Autoencoder: reconstruct x through a bottleneck. Undercomplete +
        # nonlinear => reconstructs the normal manifold, unpredictable off it.
        bottleneck = max(1, d // 2)
        self.ae = MLPRegressor(
            hidden_layer_sizes=(16, bottleneck, 16),
            activation="tanh", solver="adam", learning_rate_init=3e-3,
            max_iter=800, batch_size=64, random_state=seed, tol=1e-5,
            n_iter_no_change=30,
        ).fit(Xs, Xs)
        self.kde = KernelDensity(kernel="gaussian", bandwidth=0.3).fit(Xs)

    def __call__(self, X):
        Xs = self.scaler.transform(X)
        recon = self.ae.predict(Xs)
        e = ((Xs - recon) ** 2).mean(axis=1)          # reconstruction error
        nll = -self.kde.score_samples(Xs)             # negative log density
        nll = np.clip(nll, 0, None)
        return e * (1.0 + nll)                         # non-linear gate


# ---------------------------------------------------------------------------
# Wrap the single pipeline as a 1-teacher "committee" for the shared samplers.
# The samplers expect score_batch(teachers, pts) -> (n, K); we give K=1.
# ---------------------------------------------------------------------------
def make_teacher_fns(pipeline):
    return [lambda pts: pipeline(pts)]


def score_batch_1(teacher_fns, X):
    return np.stack([f(X) for f in teacher_fns], axis=-1)


def make_pct_map_1(S_train):
    # reuse the module's percentile map (handles K=1 fine)
    return make_percentile_maps(S_train)


# ---------------------------------------------------------------------------
# Data: two-moons normals + ring anomalies (same geometry as experiment.py)
# ---------------------------------------------------------------------------
def gen_normals(n, seed):
    X, _ = make_moons(n_samples=n, noise=0.15, random_state=seed)
    return X


def gen_anomalies(X_train, seed):
    rng = np.random.default_rng(seed + 777)
    nn = NearestNeighbors(n_neighbors=1).fit(X_train)
    lo, hi = np.array([-2.5, -2.0]), np.array([3.5, 2.0])
    pts = []
    while len(pts) < N_ANOM:
        cand = rng.uniform(lo, hi, size=(5000, 2))
        dist = nn.kneighbors(cand)[0][:, 0]
        pts.extend(cand[(dist > 0.25) & (dist < 1.5)].tolist())
    return np.array(pts[:N_ANOM])


# ---------------------------------------------------------------------------
# One (seed) run: distill the pipeline with each sampler, measure fidelity
# ---------------------------------------------------------------------------
SAMPLER_FNS = {
    "S0_none": sampler_none,
    "S1_gaussian": sampler_gaussian,
    "S3_langevin": sampler_langevin,
    "S5_adaptive": sampler_langevin_adaptive,
}


def run_seed(seed, M=2000):
    X = gen_normals(N_TRAIN, seed)
    A = gen_anomalies(X, seed)
    pipe = ComplexPipeline(X, seed=seed)
    teachers = make_teacher_fns(pipe)
    S_train = score_batch_1(teachers, X)
    pct = make_pct_map_1(S_train)
    kde = pipe.kde  # reuse the pipeline's KDE for the log-density prior

    # teacher target on anomalies (percentile-normalized)
    t_anom = pct(score_batch_1(teachers, A))[:, 0]

    rows = []
    for cond, fn in SAMPLER_FNS.items():
        rng = np.random.default_rng(seed * 131 + hash(cond) % 9973)
        kw = dict(teachers=teachers, mu=None, sigma=None, kde=kde, pct_map=pct)
        Xg = fn(X, M, rng, **kw) if cond != "S0_none" else sampler_none(X, M, rng)
        X_all = np.concatenate([X, Xg]) if len(Xg) else X
        Y_all = pct(score_batch_1(teachers, X_all))
        student = train_student(X_all, Y_all, seed=seed)
        pred = student.predict(A)
        s_anom = pred if pred.ndim == 1 else pred[:, 0]

        rho = spearmanr(s_anom, t_anom).statistic
        rmse = float(np.sqrt(np.mean((s_anom - t_anom) ** 2)))
        rows.append({"seed": seed, "cond": cond, "spearman_anom": rho, "rmse_anom": rmse})
        print(f"[complex seed={seed}] {cond:<12} spearman={rho:.3f} rmse={rmse:.3f}", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--outdir", default="results_complex")
    args = p.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True)

    all_rows = []
    for s in range(args.seeds):
        all_rows.extend(run_seed(s))

    csv_path = outdir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "cond", "spearman_anom", "rmse_anom"])
        w.writeheader(); w.writerows(all_rows)
    print(f"\nWrote {csv_path}")

    # Summary
    import collections
    agg = collections.defaultdict(lambda: {"sp": [], "rm": []})
    for r in all_rows:
        agg[r["cond"]]["sp"].append(r["spearman_anom"])
        agg[r["cond"]]["rm"].append(r["rmse_anom"])
    print("\n=== fidelity to the complex pipeline ON ANOMALIES (mean +/- sd) ===")
    for cond, v in agg.items():
        sp, rm = np.array(v["sp"]), np.array(v["rm"])
        print(f"  {cond:<12} spearman={sp.mean():.3f}+/-{sp.std():.3f}   rmse={rm.mean():.3f}+/-{rm.std():.3f}")


if __name__ == "__main__":
    main()
