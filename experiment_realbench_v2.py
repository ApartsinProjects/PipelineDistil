"""Decisive real-data study (hardening round).

Fixes every protocol hole the audit exposed and tests a method improvement:

  * TEACHER FAMILIES (5): kNN, KDE, one-class SVM, Isolation Forest, autoencoder.
    All are non-parametric / undeployable (grow with N). This turns 10 dataset
    cells into up to 50 (dataset, teacher) cells and lets us report significance
    per teacher family instead of one fragile aggregate.
  * OFF-MANIFOLD BASELINES that are NOT the shell (kills the "shell beats no
    queries" straw man): gaussian_jitter and uniform_box are off-manifold
    placements too. The real question is whether the SHELL placement beats these.
  * METHOD IMPROVEMENT: wide_shell extends the outer radius (rmax 6->12 sigma) to
    raise anomaly coverage; multiscale_shell samples radius log-uniformly over a
    wide band. Tests whether covering more of the off-manifold region recovers the
    audit's failure cases (mammography, shuttle).
  * TEACHER-QUALITY GATING: we log teacher AUROC per cell so the analysis can
    restrict to teachers worth reproducing (AUROC >= floor); reproducing a
    near-random teacher (e.g. waveform AE = 0.56) is meaningless.
  * OPERATIONAL CO-METRIC: top-k alarm agreement (fraction of the teacher's top-k
    scored eval points that are also in the student's top-k), k = #anomalies.
  * COVERAGE COVARIATE: fraction of real anomalies inside the normal-data shell,
    for the applicability diagnostic.

Raw per-(dataset, teacher, seed, sampler) rows are written; all statistics
(two-sided, gated, coverage-conditioned) are computed in analyze_realbench_v2.py.
No anomaly labels are used for training, acquisition, shell definition, or model
selection; labels enter only the final AUROC / top-k / coverage evaluation.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse, csv, zlib
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr


def _seed_of(*parts):
    """Deterministic across processes (Python's hash() is PYTHONHASHSEED-salted)."""
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from experiment import make_percentile_maps, train_student, sampler_none

DATA_DIR = Path(__file__).parent / "data" / "adbench"
M, OVERSAMPLE = 1000, 8


import warnings
warnings.filterwarnings("ignore")


def all_datasets(min_anom=20, min_normal=400, max_d=200):
    """Keep datasets with enough anomalies for a stable Spearman, enough normals
    to train+val, and moderate dimension (the shell is meaningless at huge d)."""
    keep = []
    for p in sorted(DATA_DIR.glob("*.npz")):
        try:
            a = np.load(p); y = a["y"].astype(int); X = a["X"]
            if int((y == 1).sum()) >= min_anom and int((y == 0).sum()) >= min_normal \
                    and X.shape[1] <= max_d:
                keep.append(p.stem)
        except Exception:
            pass
    return keep


def load(name, seed, max_train=3000):
    a = np.load(DATA_DIR / f"{name}.npz")
    X, y = a["X"].astype(np.float64), a["y"].astype(int)
    normals, anoms = X[y == 0], X[y == 1]
    rng = np.random.default_rng(seed); rng.shuffle(normals)
    n_tr = min(max_train, len(normals) // 2)
    Xtr = normals[:n_tr]
    Xval = normals[n_tr:n_tr + max(500, len(normals) // 4)]
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xval), sc.transform(anoms)


# ---- teacher families (all fit on train normals only; return P -> anomaly score)
def teacher_knn(X, seed):
    nn = NearestNeighbors(n_neighbors=10).fit(X)
    return lambda P: nn.kneighbors(P)[0].mean(1)

def teacher_kde(X, seed):
    kde = KernelDensity(bandwidth=0.5).fit(X)
    return lambda P: -kde.score_samples(P)

def teacher_ocsvm(X, seed):
    m = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1).fit(X)
    return lambda P: -m.decision_function(P)

def teacher_iforest(X, seed):
    m = IsolationForest(n_estimators=150, random_state=seed).fit(X)
    return lambda P: -m.score_samples(P)

def teacher_ae(X, seed):
    d = X.shape[1]
    ae = MLPRegressor(hidden_layer_sizes=(64, max(2, d // 2), 64), activation="tanh",
                      solver="adam", learning_rate_init=3e-3, max_iter=600, batch_size=64,
                      random_state=seed, tol=1e-5, n_iter_no_change=25).fit(X, X)
    return lambda P: ((P - ae.predict(P)) ** 2).mean(1)

TEACHERS = {"knn": teacher_knn, "kde": teacher_kde, "ocsvm": teacher_ocsvm,
            "iforest": teacher_iforest, "ae": teacher_ae}


def shell_scale(X, seed):
    nn = NearestNeighbors(n_neighbors=2).fit(X); rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(500, len(X)), replace=False)
    return float(np.median(nn.kneighbors(X[idx])[0][:, 1]))


def shell_pool(X, need, rng, rmin, rmax, logr=False):
    """Anchored directional shell sampler: land points at dist in (rmin, rmax]."""
    d = X.shape[1]; nn = NearestNeighbors(n_neighbors=1).fit(X); pts = []; tries = 0
    while len(pts) < need:
        anchors = X[rng.integers(0, len(X), size=6000)]
        dirs = rng.normal(size=(6000, d)); dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
        if logr:
            rad = np.exp(rng.uniform(np.log(rmin), np.log(rmax), size=(6000, 1)))
        else:
            rad = rng.uniform(rmin, rmax, size=(6000, 1))
        cand = anchors + dirs * rad
        dd = nn.kneighbors(cand)[0][:, 0]
        pts.extend(cand[(dd > rmin) & (dd <= rmax)].tolist())
        tries += 1
        if tries > 100:
            break
    return np.array(pts[:need]) if pts else np.zeros((0, d))


def make_queries(kind, X, rng, scale):
    """Off-manifold query placements. All spend M teacher labels; the shell
    variants spend no extra selection queries (score weighting is separate)."""
    d = X.shape[1]
    if kind == "normals_only":
        return sampler_none(X, M, rng)
    if kind == "gaussian_jitter":
        base = X[rng.integers(0, len(X), size=M)]
        return base + rng.normal(scale=1.0 * scale, size=(M, d))
    if kind == "uniform_box":
        lo, hi = X.min(0) - 1.0 * scale, X.max(0) + 1.0 * scale
        return rng.uniform(lo, hi, size=(M, d))
    if kind == "uniform_shell":
        return shell_pool(X, M, rng, 1.0 * scale, 6.0 * scale)
    if kind == "wide_shell":
        return shell_pool(X, M, rng, 1.0 * scale, 12.0 * scale)
    if kind == "multiscale_shell":
        return shell_pool(X, M, rng, 0.5 * scale, 12.0 * scale, logr=True)
    if kind == "shell_box_mix":
        # Improved method: hedge across teacher geometry with ONE label-free
        # sampler. Half the queries land in the near-boundary shell (structure
        # for non-radial teachers), half fill the ambient box (far-field
        # coverage for radial teachers whose anomalies sit beyond the shell).
        half = M // 2
        lo, hi = X.min(0) - 1.0 * scale, X.max(0) + 1.0 * scale
        box = rng.uniform(lo, hi, size=(M - half, d))
        sh = shell_pool(X, half, rng, 1.0 * scale, 6.0 * scale)
        return np.concatenate([sh, box]) if len(sh) else box
    raise ValueError(kind)


def topk_agree(t_scores, s_scores, k):
    if k <= 0 or k >= len(t_scores):
        k = max(1, len(t_scores) // 5)
    tt = set(np.argsort(t_scores)[-k:]); ss = set(np.argsort(s_scores)[-k:])
    return len(tt & ss) / k


SAMPLERS = ["normals_only", "gaussian_jitter", "uniform_box",
            "uniform_shell", "wide_shell", "multiscale_shell", "shell_box_mix"]


def run_cell(name, tname, seed):
    Xtr, Xval, Xan = load(name, seed)
    teacher = TEACHERS[tname](Xtr, seed)
    sb = lambda A: np.stack([teacher(A)], axis=-1)
    pct = make_percentile_maps(sb(Xtr))
    t_an = pct(sb(Xan))[:, 0]; t_val = pct(sb(Xval))[:, 0]
    scale = shell_scale(Xtr, seed)
    y_true = np.r_[np.zeros(len(Xval)), np.ones(len(Xan))]
    t_eval = np.r_[t_val, t_an]
    teach_auroc = roc_auc_score(y_true, t_eval)
    # coverage: fraction of anomalies inside the normal-data shell (1..6 sigma)
    da = NearestNeighbors(n_neighbors=1).fit(Xtr).kneighbors(Xan)[0][:, 0]
    cov = float(((da > 1.0 * scale) & (da <= 6.0 * scale)).mean())
    cov_wide = float(((da > 1.0 * scale) & (da <= 12.0 * scale)).mean())
    # radial-overlap coverage: fraction of anomalies whose distance falls in the
    # [q10,q90] of the query radii (in-shell in the sense that matters).
    qr = da[(da > 1.0 * scale) & (da <= 6.0 * scale)]
    # teacher radiality on anomalies: does the teacher just rank by distance?
    # High radiality => a normals-only student already reproduces it (no headroom).
    radiality = float(spearmanr(t_an, da).statistic) if len(da) > 2 else np.nan
    rows = []
    for kind in SAMPLERS:
        rng = np.random.default_rng(seed * 1_000_003 + _seed_of(tname, kind))
        Xg = make_queries(kind, Xtr, rng, scale)
        Xall = np.concatenate([Xtr, Xg]) if len(Xg) else Xtr
        st = train_student(Xall, pct(sb(Xall)), seed)
        s_an = st.predict(Xan); s_an = s_an if s_an.ndim == 1 else s_an[:, 0]
        s_val = st.predict(Xval); s_val = s_val if s_val.ndim == 1 else s_val[:, 0]
        s_eval = np.r_[s_val, s_an]
        rows.append({"dataset": name, "teacher": tname, "d": Xtr.shape[1], "seed": seed,
                     "sampler": kind, "teacher_auroc": teach_auroc, "coverage": cov,
                     "coverage_wide": cov_wide, "radiality": radiality,
                     "fid": spearmanr(s_an, t_an).statistic,
                     "student_auroc": roc_auc_score(y_true, s_eval),
                     "topk": topk_agree(t_eval, s_eval, int(len(Xan)))})
    return rows


def _work(task):
    name, tname, seed = task
    try:
        return run_cell(name, tname, seed)
    except Exception as e:
        return [{"__fail__": f"{name}/{tname}/s{seed}: {e}"}]


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--teachers", default="knn,kde,ocsvm,iforest,ae")
    ap.add_argument("--outdir", default="results_realbench_v2")
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(exist_ok=True)
    tlist = a.teachers.split(","); datasets = all_datasets()
    tasks = [(n, t, s) for n in datasets for t in tlist for s in range(a.seeds)]
    print(f"datasets={len(datasets)} teachers={tlist} seeds={a.seeds} "
          f"cells={len(tasks)} workers={a.workers}", flush=True)
    print("  datasets:", ", ".join(datasets), flush=True)
    rows, done, fails = [], 0, 0
    with mp.Pool(a.workers) as pool:
        for res in pool.imap_unordered(_work, tasks, chunksize=1):
            if res and res[0].get("__fail__"):
                fails += 1; print(f"[FAIL] {res[0]['__fail__']}", flush=True); continue
            rows.extend(res); done += 1
            if done % 25 == 0:
                print(f"  ... {done}/{len(tasks)} cells done ({fails} fails)", flush=True)
    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out/'results.csv'} ({len(rows)} rows, {done} cells, {fails} fails)", flush=True)


if __name__ == "__main__":
    main()
