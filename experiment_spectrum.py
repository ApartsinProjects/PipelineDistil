"""Experiment A+B: sampling benefit as a function of pipeline non-monotonicity.

Builds a SPECTRUM of teacher pipelines from monotone to strongly non-monotone,
measures each teacher's non-monotonicity index, and shows that the benefit of
uncertainty-guided sampling (S5 fidelity - S0 fidelity, on held-out anomalies)
grows with that index. This turns the qualitative claim ("sampling helps for
complex pipelines") into a PREDICTOR: measure your pipeline's non-monotonicity;
if high, sampling is required.

Teachers (increasing off-manifold complexity):
  knn         mean distance to k nearest normals            (monotone)
  kde         negative log kernel density                   (monotone)
  ocsvm       negative one-class SVM decision function      (saturating)
  ae          autoencoder reconstruction error              (non-monotone)
  ae_kde_max  percentile-max fusion of AE and KDE           (strongly non-monotone)

Non-monotonicity index of a teacher T:
  sample off-manifold points at a range of distances d(x) to the manifold;
  index = 1 - |Spearman(T(x), d(x))|.
  A purely monotone (distance-like) teacher has |rho|=1 -> index 0.
  A teacher whose score is unrelated to distance -> index near 1.

Usage:
  python experiment_spectrum.py --seeds 5
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import make_moons
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from experiment import (
    make_percentile_maps, train_student,
    sampler_none, sampler_langevin_adaptive,
)

N_TRAIN = 2000
N_ANOM = 500


def gen_normals(n, seed):
    X, _ = make_moons(n_samples=n, noise=0.15, random_state=seed)
    return X


def gen_offmanifold(X_train, n, seed, dmin=0.2, dmax=2.5):
    """Points spread across the off-manifold region, with their distance."""
    rng = np.random.default_rng(seed + 555)
    nn = NearestNeighbors(n_neighbors=1).fit(X_train)
    lo, hi = np.array([-2.5, -2.0]), np.array([3.5, 2.0])
    pts = []
    while len(pts) < n:
        cand = rng.uniform(lo, hi, size=(6000, 2))
        dist = nn.kneighbors(cand)[0][:, 0]
        keep = (dist > dmin) & (dist < dmax)
        pts.extend(cand[keep].tolist())
    pts = np.array(pts[:n])
    dist = nn.kneighbors(pts)[0][:, 0]
    return pts, dist


# ---------------------------------------------------------------------------
# Teacher factory: each returns a callable X -> raw score (larger = anomalous)
# ---------------------------------------------------------------------------
def build_teachers(X_train, seed):
    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    d = Xs.shape[1]

    knn = NearestNeighbors(n_neighbors=10).fit(Xs)
    kde = KernelDensity(kernel="gaussian", bandwidth=0.3).fit(Xs)
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1).fit(Xs)
    ae = MLPRegressor(hidden_layer_sizes=(16, max(1, d // 2), 16),
                      activation="tanh", solver="adam", learning_rate_init=3e-3,
                      max_iter=800, batch_size=64, random_state=seed, tol=1e-5,
                      n_iter_no_change=30).fit(Xs, Xs)

    def s_knn(X):  return knn.kneighbors(scaler.transform(X))[0].mean(1)
    def s_kde(X):  return -kde.score_samples(scaler.transform(X))
    def s_ocsvm(X): return -ocsvm.decision_function(scaler.transform(X))
    def s_ae(X):
        Xt = scaler.transform(X); return ((Xt - ae.predict(Xt)) ** 2).mean(1)

    # percentile-max fusion of AE and KDE (strongly non-monotone)
    ae_tr = np.sort(s_ae(X_train)); kde_tr = np.sort(s_kde(X_train))
    N = len(X_train)
    def _pct(sorted_ref, v): return np.searchsorted(sorted_ref, v) / N
    def s_ae_kde_max(X):
        return np.maximum(_pct(ae_tr, s_ae(X)), _pct(kde_tr, s_kde(X)))

    return {"knn": s_knn, "kde": s_kde, "ocsvm": s_ocsvm,
            "ae": s_ae, "ae_kde_max": s_ae_kde_max}


def nonmonotonicity_index(teacher_fn, X_train, seed):
    pts, dist = gen_offmanifold(X_train, 3000, seed)
    scores = teacher_fn(pts)
    rho = spearmanr(scores, dist).statistic
    return 1.0 - abs(rho)


def distill_fidelity(teacher_fn, X_train, X_anom, kde_for_prior, seed, cond, M=2000):
    """Distill teacher into a student with sampler `cond`; return Spearman
    fidelity of student vs teacher ON the anomalies."""
    teachers = [teacher_fn]
    sb = lambda X: np.stack([f(X) for f in teachers], axis=-1)
    S = sb(X_train); pct = make_percentile_maps(S)
    t_anom = pct(sb(X_anom))[:, 0]

    rng = np.random.default_rng(seed * 149 + hash(cond) % 9973)
    kw = dict(teachers=teachers, mu=None, sigma=None, kde=kde_for_prior, pct_map=pct)
    fn = sampler_none if cond == "S0_none" else sampler_langevin_adaptive
    Xg = fn(X_train, M, rng, **kw) if cond != "S0_none" else sampler_none(X_train, M, rng)
    Xall = np.concatenate([X_train, Xg]) if len(Xg) else X_train
    Yall = pct(sb(Xall))
    st = train_student(Xall, Yall, seed)
    p = st.predict(X_anom); s_anom = p if p.ndim == 1 else p[:, 0]
    return spearmanr(s_anom, t_anom).statistic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--outdir", default="results_spectrum")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True)

    order = ["knn", "kde", "ocsvm", "ae", "ae_kde_max"]
    rows = []
    for seed in range(args.seeds):
        X = gen_normals(N_TRAIN, seed)
        Xan, _ = gen_offmanifold(X, N_ANOM, seed + 1)  # held-out off-manifold anomalies
        teachers = build_teachers(X, seed)
        kde_prior = KernelDensity(kernel="gaussian", bandwidth=0.3).fit(
            StandardScaler().fit(X).transform(X))
        # the sampler's log-density prior expects a KDE on raw X:
        kde_prior = KernelDensity(kernel="gaussian", bandwidth=0.3).fit(X)
        for name in order:
            tfn = teachers[name]
            nmi = nonmonotonicity_index(tfn, X, seed)
            f0 = distill_fidelity(tfn, X, Xan, kde_prior, seed, "S0_none")
            f5 = distill_fidelity(tfn, X, Xan, kde_prior, seed, "S5_adaptive")
            rows.append({"seed": seed, "teacher": name, "nmi": nmi,
                         "fid_S0": f0, "fid_S5": f5, "gain": f5 - f0})
            print(f"[spectrum seed={seed}] {name:<11} nmi={nmi:.3f} "
                  f"S0={f0:.3f} S5={f5:.3f} gain={f5-f0:+.3f}", flush=True)

    csv_path = outdir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "teacher", "nmi", "fid_S0", "fid_S5", "gain"])
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {csv_path}")

    import collections
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for k in ("nmi", "fid_S0", "fid_S5", "gain"):
            agg[r["teacher"]][k].append(r[k])
    print("\n=== teacher spectrum (mean over seeds) ===")
    print(f"  {'teacher':<11} {'non-mono':>9} {'S0 fid':>8} {'S5 fid':>8} {'gain':>8}")
    for name in order:
        a = agg[name]
        print(f"  {name:<11} {np.mean(a['nmi']):>9.3f} {np.mean(a['fid_S0']):>8.3f} "
              f"{np.mean(a['fid_S5']):>8.3f} {np.mean(a['gain']):>+8.3f}")


if __name__ == "__main__":
    main()
