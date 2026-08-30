"""High-dimensional controlled-manifold study (TMLR review Exp 4 / W12).

Tests whether the shell-placement effect and the score/variation weighting
survive as ambient dimension grows, and whether the O(K) random-direction
variation estimator stays usable where coordinate finite differences (O(d))
become expensive.

Design:
  - normal data: intrinsic-m GMM latent mapped into ambient d by a fixed random
    orthonormal embedding, plus small ambient noise (an m-dim manifold in R^d);
  - shell defined from NORMAL DATA ONLY (leave-one-out kNN distance scale), so
    shell construction never sees the test anomalies (addresses the review's
    shell/eval-overlap concern);
  - anomalies: held-out points at scale-relative off-manifold distances;
  - teachers: kNN distance (monotone), one-class SVM (saturating), autoencoder
    (growing), each fit on normals only;
  - samplers: normals-only, uniform-shell, combined (coordinate FD, O(d)),
    combined (K random directions, O(K));
  - metric: Spearman student-teacher fidelity on held-out anomalies.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse, csv, collections
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from experiment import make_percentile_maps, train_student, sampler_none

N_TRAIN, N_ANOM, M, OVERSAMPLE = 2000, 500, 1000, 8


# ---------------------------------------------------------------------------
# high-dimensional manifold data
# ---------------------------------------------------------------------------
def _embedding(d, m, seed=12345):
    rng = np.random.default_rng(seed * 1000 + d * 10 + m)
    A = rng.normal(size=(d, m))
    Q, _ = np.linalg.qr(A)          # d x m orthonormal columns
    means = rng.uniform(-1.5, 1.5, size=(3, m))
    return Q, means


def gen_normals(n, d, m, seed):
    Q, means = _embedding(d, m)
    rng = np.random.default_rng(seed)
    asg = rng.integers(0, 3, size=n)
    latent = means[asg] + rng.normal(scale=0.15, size=(n, m))
    X = latent @ Q.T                # embed into R^d
    X += rng.normal(scale=0.02, size=(n, d))   # small ambient noise
    return X


def gen_anomalies(X_train, d, seed, lo_mult=1.0, hi_mult=6.0):
    """Off-manifold points at scale-relative distance bands, verified by a
    global 1-NN distance so they genuinely sit off the training manifold."""
    rng = np.random.default_rng(seed + 10_000)
    nn = NearestNeighbors(n_neighbors=1).fit(X_train)
    # normal-data distance scale (median NN distance among a sample of normals)
    idx = rng.choice(len(X_train), size=min(500, len(X_train)), replace=False)
    scale = np.median(nn.kneighbors(X_train[idx], n_neighbors=2)[0][:, 1])
    lo, hi = lo_mult * scale, hi_mult * scale
    pts = []
    tries = 0
    while len(pts) < N_ANOM:
        anchors = X_train[rng.integers(0, len(X_train), size=4000)]
        dirs = rng.normal(size=(4000, d)); dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
        radii = rng.uniform(lo, hi, size=(4000, 1))
        cand = anchors + dirs * radii
        dd = nn.kneighbors(cand)[0][:, 0]
        pts.extend(cand[(dd > lo) & (dd <= hi)].tolist())
        tries += 1
        if tries > 60:
            raise RuntimeError(f"anomaly gen d={d}: {len(pts)}/{N_ANOM}")
    return np.array(pts[:N_ANOM]), scale


# ---------------------------------------------------------------------------
# teachers (fit on normals only)
# ---------------------------------------------------------------------------
def build_teachers(X, seed):
    sc = StandardScaler().fit(X); Xs = sc.transform(X); d = Xs.shape[1]
    knn = NearestNeighbors(n_neighbors=10).fit(Xs)
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1).fit(Xs)
    ae = MLPRegressor(hidden_layer_sizes=(32, max(2, d // 2), 32), activation="tanh",
                      solver="adam", learning_rate_init=3e-3, max_iter=600,
                      batch_size=64, random_state=seed, tol=1e-5, n_iter_no_change=25).fit(Xs, Xs)
    return {
        "knn":   lambda P: knn.kneighbors(sc.transform(P))[0].mean(1),
        "ocsvm": lambda P: -ocsvm.decision_function(sc.transform(P)),
        "ae":    lambda P: ((sc.transform(P) - ae.predict(sc.transform(P))) ** 2).mean(1),
    }


# ---------------------------------------------------------------------------
# normal-data-derived shell + samplers
# ---------------------------------------------------------------------------
def shell_bounds(X, seed):
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(500, len(X)), replace=False)
    scale = np.median(nn.kneighbors(X[idx])[0][:, 1])
    return 1.0 * scale, 6.0 * scale   # rho_min, rho_max from normals only


def shell_pool(X, need, rng, rmin, rmax):
    d = X.shape[1]; nn = NearestNeighbors(n_neighbors=1).fit(X)
    pts = []; tries = 0
    while len(pts) < need:
        anchors = X[rng.integers(0, len(X), size=6000)]
        dirs = rng.normal(size=(6000, d)); dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
        radii = rng.uniform(rmin, rmax, size=(6000, 1))
        cand = anchors + dirs * radii
        dd = nn.kneighbors(cand)[0][:, 0]
        pts.extend(cand[(dd > rmin) & (dd <= rmax)].tolist())
        tries += 1
        if tries > 80:
            break
    return np.array(pts[:need])


def signals(pool, teachers, pct, h, K):
    fused = lambda P: pct(np.stack([f(P) for f in teachers], axis=-1)).mean(1)
    d = pool.shape[1]; u = fused(pool)
    if K is None:
        g2 = np.zeros(len(pool))
        for i in range(d):
            e = np.zeros(d); e[i] = h
            g2 += ((fused(pool + e) - fused(pool - e)) / (2 * h)) ** 2
        v = np.sqrt(g2)
    else:
        rng = np.random.default_rng(0); acc = np.zeros(len(pool))
        for _ in range(K):
            z = rng.normal(size=(len(pool), d)); z /= (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
            acc += np.abs(fused(pool + h * z) - fused(pool - h * z)) / (2 * h)
        v = acc / K
    return u, v


def make_queries(kind, X, teachers, pct, rng, rmin, rmax, h=0.05, K=None):
    if kind == "uniform_shell":
        return shell_pool(X, M, rng, rmin, rmax)
    pool = shell_pool(X, M * OVERSAMPLE, rng, rmin, rmax)
    if len(pool) < 10:
        return sampler_none(X, M, rng)
    u, v = signals(pool, teachers, pct, h, K)
    un = np.clip(u, 0, None); un = un / (un.sum() + 1e-12); vn = v / (v.sum() + 1e-12)
    w = 0.5 * un + 0.5 * vn; w = w / w.sum()
    return pool[rng.choice(len(pool), size=M, replace=True, p=w)]


def fidelity(tf, X, Xan, seed, kind, rmin, rmax, K=None):
    teachers = [tf]; sb = lambda A: np.stack([f(A) for f in teachers], axis=-1)
    pct = make_percentile_maps(sb(X)); t = pct(sb(Xan))[:, 0]
    rng = np.random.default_rng(seed * 233 + hash((kind, K)) % 9973)
    Xg = sampler_none(X, M, rng) if kind == "normals_only" else make_queries(kind, X, teachers, pct, rng, rmin, rmax, K=K)
    Xall = np.concatenate([X, Xg]) if len(Xg) else X
    st = train_student(Xall, pct(sb(Xall)), seed)
    p = st.predict(Xan); s = p if p.ndim == 1 else p[:, 0]
    return spearmanr(s, t).statistic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[8, 32, 64])
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--outdir", default="results_highd_v2")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(exist_ok=True)
    teach_order = ["knn", "ocsvm", "ae"]
    rows = []
    for d in a.dims:
        for seed in range(a.seeds):
            X = gen_normals(N_TRAIN, d, a.m, seed)
            Xan, scale = gen_anomalies(X, d, seed)
            rmin, rmax = shell_bounds(X, seed)
            T = build_teachers(X, seed)
            for tn in teach_order:
                tf = T[tn]; r = {"d": d, "m": a.m, "seed": seed, "teacher": tn}
                r["normals_only"] = fidelity(tf, X, Xan, seed, "normals_only", rmin, rmax)
                r["uniform_shell"] = fidelity(tf, X, Xan, seed, "uniform_shell", rmin, rmax)
                r["combined_FD"] = fidelity(tf, X, Xan, seed, "mix", rmin, rmax, K=None)
                r["combined_K4"] = fidelity(tf, X, Xan, seed, "mix", rmin, rmax, K=4)
                rows.append(r)
                print(f"[train] d={d} seed={seed} {tn:<6} "
                      f"none={r['normals_only']:.3f} unif={r['uniform_shell']:.3f} "
                      f"FD={r['combined_FD']:.3f} K4={r['combined_K4']:.3f}", flush=True)
    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        agg[(r["d"], r["teacher"])]
        for k in ("normals_only", "uniform_shell", "combined_FD", "combined_K4"):
            agg[(r["d"], r["teacher"])][k].append(r[k])
    print("\n=== fidelity by (d, teacher) ===")
    for d in a.dims:
        for tn in teach_order:
            a_ = agg[(d, tn)]
            print(f"  d={d:>3} {tn:<6} none={np.mean(a_['normals_only']):.3f} "
                  f"unif={np.mean(a_['uniform_shell']):.3f} FD={np.mean(a_['combined_FD']):.3f} "
                  f"K4={np.mean(a_['combined_K4']):.3f}")
    print(f"\nWrote {out/'results.csv'}")


if __name__ == "__main__":
    main()
