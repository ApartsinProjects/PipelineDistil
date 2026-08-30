"""Real composite-pipeline teachers (TMLR review Exp 7 / W13).

The spectrum teachers are mostly single detectors. A reader drawn by the word
"pipeline" expects preprocessing, multiple detectors, calibration, and fusion.
We build three genuine composite pipelines and distill each as a blackbox:

  P1 smooth ensemble:   standardize -> PCA -> {kNN dist, OCSVM, AE recon}
                        -> per-detector percentile calibration -> weighted mean
  P2 gated ensemble:    standardize -> {AE recon, KDE nll}
                        -> density-dependent soft gate between them
  P3 non-smooth prod:   standardize -> {Isolation Forest, kNN dist, AE recon}
                        -> percentile calibration -> max (top-1) fusion

Each is queryable only as x -> scalar score. We distill with normals-only,
uniform-shell, and the combined score/variation shell sampler, and report
student-teacher fidelity (Spearman) on held-out off-manifold anomalies.
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
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from experiment import make_percentile_maps, train_student, sampler_none
from experiment_spectrum import gen_normals, gen_offmanifold
import experiment_v2 as v2


def _pctmap(vals):
    s = np.sort(vals); N = len(s)
    return lambda x: np.searchsorted(s, x) / (N + 1)


class P1SmoothEnsemble:
    """standardize -> PCA -> kNN+OCSVM+AE -> percentile -> weighted mean."""
    def __init__(self, X, seed):
        self.sc = StandardScaler().fit(X); Xs = self.sc.transform(X)
        self.pca = PCA(n_components=min(X.shape[1], 8)).fit(Xs); Z = self.pca.transform(Xs)
        d = Z.shape[1]
        self.knn = NearestNeighbors(n_neighbors=10).fit(Z)
        self.ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1).fit(Z)
        self.ae = MLPRegressor(hidden_layer_sizes=(16, max(2, d // 2), 16), activation="tanh",
                               max_iter=500, random_state=seed, tol=1e-5, n_iter_no_change=25).fit(Z, Z)
        s1 = self.knn.kneighbors(Z)[0].mean(1); s2 = -self.ocsvm.decision_function(Z)
        s3 = ((Z - self.ae.predict(Z)) ** 2).mean(1)
        self.p1, self.p2, self.p3 = _pctmap(s1), _pctmap(s2), _pctmap(s3)
        self.w = np.array([0.4, 0.3, 0.3])
    def __call__(self, X):
        Z = self.pca.transform(self.sc.transform(X))
        s1 = self.knn.kneighbors(Z)[0].mean(1); s2 = -self.ocsvm.decision_function(Z)
        s3 = ((Z - self.ae.predict(Z)) ** 2).mean(1)
        return self.w[0] * self.p1(s1) + self.w[1] * self.p2(s2) + self.w[2] * self.p3(s3)


class P2GatedEnsemble:
    """standardize -> AE recon and KDE nll -> density-dependent soft gate."""
    def __init__(self, X, seed):
        self.sc = StandardScaler().fit(X); Xs = self.sc.transform(X); d = Xs.shape[1]
        self.ae = MLPRegressor(hidden_layer_sizes=(16, max(2, d // 2), 16), activation="tanh",
                               max_iter=500, random_state=seed, tol=1e-5, n_iter_no_change=25).fit(Xs, Xs)
        self.kde = KernelDensity(bandwidth=0.3).fit(Xs)
        ae_s = ((Xs - self.ae.predict(Xs)) ** 2).mean(1); nll = -self.kde.score_samples(Xs)
        self.pae, self.pnll = _pctmap(ae_s), _pctmap(nll)
        self.nll_ref = np.median(nll); self.nll_scale = np.std(nll) + 1e-9
    def __call__(self, X):
        Xs = self.sc.transform(X)
        ae_s = ((Xs - self.ae.predict(Xs)) ** 2).mean(1); nll = -self.kde.score_samples(Xs)
        g = 1.0 / (1.0 + np.exp(-(nll - self.nll_ref) / self.nll_scale))   # low density -> g~1
        return g * self.pae(ae_s) + (1 - g) * self.pnll(nll)               # gate: far -> trust AE


class P3NonSmoothProd:
    """standardize -> IsolationForest + kNN + AE -> percentile -> max fusion."""
    def __init__(self, X, seed):
        self.sc = StandardScaler().fit(X); Xs = self.sc.transform(X); d = Xs.shape[1]
        self.iforest = IsolationForest(n_estimators=100, random_state=seed).fit(Xs)
        self.knn = NearestNeighbors(n_neighbors=10).fit(Xs)
        self.ae = MLPRegressor(hidden_layer_sizes=(16, max(2, d // 2), 16), activation="tanh",
                               max_iter=500, random_state=seed, tol=1e-5, n_iter_no_change=25).fit(Xs, Xs)
        s1 = -self.iforest.score_samples(Xs); s2 = self.knn.kneighbors(Xs)[0].mean(1)
        s3 = ((Xs - self.ae.predict(Xs)) ** 2).mean(1)
        self.p1, self.p2, self.p3 = _pctmap(s1), _pctmap(s2), _pctmap(s3)
    def __call__(self, X):
        Xs = self.sc.transform(X)
        s1 = -self.iforest.score_samples(Xs); s2 = self.knn.kneighbors(Xs)[0].mean(1)
        s3 = ((Xs - self.ae.predict(Xs)) ** 2).mean(1)
        return np.maximum.reduce([self.p1(s1), self.p2(s2), self.p3(s3)])


PIPELINES = {"P1_smooth": P1SmoothEnsemble, "P2_gated": P2GatedEnsemble, "P3_nonsmooth": P3NonSmoothProd}


def fidelity(pipe, X, Xan, seed, kind, K=None):
    teachers = [pipe]; sb = lambda A: np.stack([f(A) for f in teachers], axis=-1)
    pct = make_percentile_maps(sb(X)); t = pct(sb(Xan))[:, 0]
    rng = np.random.default_rng(seed * 271 + hash((kind, K)) % 9973)
    Xg = sampler_none(X, v2.M, rng) if kind == "normals_only" else v2.make_queries(kind, X, teachers, pct, rng, lam=0.5, K=K)
    Xall = np.concatenate([X, Xg]) if len(Xg) else X
    st = train_student(Xall, pct(sb(Xall)), seed)
    p = st.predict(Xan); s = p if p.ndim == 1 else p[:, 0]
    return spearmanr(s, t).statistic


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--outdir", default="results_pipelines"); a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(exist_ok=True)
    rows = []
    for seed in range(a.seeds):
        X = gen_normals(v2.N_TRAIN, seed); Xan, _ = gen_offmanifold(X, v2.N_ANOM, seed + 1)
        for pn, cls in PIPELINES.items():
            pipe = cls(X, seed); r = {"seed": seed, "pipeline": pn}
            r["normals_only"] = fidelity(pipe, X, Xan, seed, "normals_only")
            r["uniform_shell"] = fidelity(pipe, X, Xan, seed, "uniform_shell")
            r["score"] = fidelity(pipe, X, Xan, seed, "score")
            r["variation"] = fidelity(pipe, X, Xan, seed, "variation", K=None)
            r["combined"] = fidelity(pipe, X, Xan, seed, "mix", K=None)
            rows.append(r)
            print(f"[pipe seed={seed}] {pn:<12} none={r['normals_only']:.3f} unif={r['uniform_shell']:.3f} "
                  f"score={r['score']:.3f} var={r['variation']:.3f} comb={r['combined']:.3f}", flush=True)
    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for k in ("normals_only", "uniform_shell", "score", "variation", "combined"):
            agg[r["pipeline"]][k].append(r[k])
    print("\n=== composite-pipeline fidelity (mean over seeds) ===")
    print(f"  {'pipeline':<12}{'none':>8}{'unif':>8}{'score':>8}{'var':>8}{'comb':>8}")
    for pn in PIPELINES:
        a_ = agg[pn]
        print(f"  {pn:<12}" + "".join(f"{np.mean(a_[k]):>8.3f}" for k in ('normals_only','uniform_shell','score','variation','combined')))
    print(f"\nWrote {out/'results.csv'}")


if __name__ == "__main__":
    main()
