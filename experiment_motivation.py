"""Two motivation experiments (reviewer questions).

Q1  Is normal-data-only sampling sufficient? Directly measure, per real dataset,
    the fraction of real anomalies that fall inside the normal-data-derived shell,
    and correlate it with the shell-sampling fidelity gain. "Normal-data sampling
    suffices" == "anomalies fall in the normal-data shell": this makes that
    equivalence measurable and shows when it holds and when it fails.

Q2  Why distill at all? The single-detector teachers are non-parametric: their
    deployed size grows with the training set (kNN/KDE store all points, OCSVM
    stores support vectors, IsolationForest stores trees), while the student is a
    fixed 33-parameter net. Measure the teacher footprint vs the student.
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).parent / "data" / "adbench"
DATASETS = ["23_mammography", "38_thyroid", "6_cardio", "30_satellite", "28_pendigits",
            "40_vowels", "32_shuttle", "26_optdigits", "41_Waveform", "20_letter"]
# fidelity gains (uniform-shell - normals-only) from results_realbench (5 seeds)
FID_GAIN = {"23_mammography": -0.166, "38_thyroid": 0.338, "6_cardio": 0.122,
            "30_satellite": 0.315, "28_pendigits": 0.081, "40_vowels": 0.344,
            "32_shuttle": -0.147, "26_optdigits": 0.072, "41_Waveform": 0.121,
            "20_letter": 0.322}


def load(name, seed=0, max_train=3000):
    a = np.load(DATA / f"{name}.npz")
    X, y = a["X"].astype(np.float64), a["y"].astype(int)
    normals, anoms = X[y == 0], X[y == 1]
    rng = np.random.default_rng(seed); rng.shuffle(normals)
    n_tr = min(max_train, len(normals) // 2)
    Xtr = normals[:n_tr]
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(anoms)


def q1_coverage():
    print("=== Q1: fraction of real anomalies inside the normal-data shell ===")
    print(f"  {'dataset':<15}{'d':>4}{'in-shell %':>12}{'fid-gain':>10}")
    cov, gain = [], []
    for name in DATASETS:
        Xtr, Xan = load(name)
        nn = NearestNeighbors(n_neighbors=2).fit(Xtr)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(Xtr), size=min(500, len(Xtr)), replace=False)
        scale = np.median(nn.kneighbors(Xtr[idx])[0][:, 1])
        rmin, rmax = 1.0 * scale, 6.0 * scale
        da = NearestNeighbors(n_neighbors=1).fit(Xtr).kneighbors(Xan)[0][:, 0]
        frac = float(((da > rmin) & (da <= rmax)).mean())
        cov.append(frac); gain.append(FID_GAIN[name])
        print(f"  {name:<15}{Xtr.shape[1]:>4}{100*frac:>11.1f}%{FID_GAIN[name]:>+10.3f}")
    r = np.corrcoef(cov, gain)[0, 1]
    print(f"\n  correlation(in-shell fraction, fidelity gain) = {r:+.3f}")
    print(f"  (positive => normal-data sampling helps exactly when anomalies land in the shell)")


def _fit_size(model):
    return len(pickle.dumps(model)) / 1024.0   # KB


def q2_footprint():
    print("\n=== Q2: non-parametric teacher footprint vs fixed student ===")
    print(f"  {'dataset':<15}{'N_train':>8}{'d':>4}"
          f"{'kNN KB':>9}{'KDE KB':>9}{'OCSVM KB':>10}{'iForest KB':>12}{'student KB':>12}")
    for name in DATASETS:
        Xtr, _ = load(name)
        knn = NearestNeighbors(n_neighbors=10).fit(Xtr)
        kde = KernelDensity(bandwidth=0.3).fit(Xtr)
        ocs = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1).fit(Xtr)
        ifo = IsolationForest(n_estimators=100, random_state=0).fit(Xtr)
        # student is a fixed width-8 tanh MLP: 8*d + 8 + 8 + 1 params, ~a few KB
        d = Xtr.shape[1]; npar = 8 * d + 8 + 8 + 1
        student_kb = (npar * 8) / 1024.0        # float64 weights
        print(f"  {name:<15}{len(Xtr):>8}{d:>4}"
              f"{_fit_size(knn):>9.0f}{_fit_size(kde):>9.0f}{_fit_size(ocs):>10.0f}"
              f"{_fit_size(ifo):>12.0f}{student_kb:>12.2f}")
    print("\n  kNN/KDE store the whole training set; OCSVM stores support vectors;")
    print("  IsolationForest stores 100 trees. All grow with N. The student is a")
    print("  fixed ~a-few-KB net regardless of N -- which is why distillation is worthwhile.")


if __name__ == "__main__":
    q1_coverage()
    q2_footprint()
