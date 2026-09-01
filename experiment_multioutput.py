"""Multi-OUTPUT pipeline distillation (the core thesis).

A realistic anomaly pipeline is not one scalar: it runs several stages trained on
normal data (an autoencoder, a density model, a distance model, an isolation
model), each emitting an output with its OWN response surface and its own hidden
inductive bias. On the normals they roughly agree ("all normal"); OFF the manifold
-- where the anomalies we never saw actually live -- their surfaces diverge, and
that joint off-manifold behavior is exactly what a deployable student must copy.

This experiment distills the WHOLE OUTPUT VECTOR (not a fused scalar) into one
small multi-output student, and tests whether an ADVANCED sampler that targets
where the pipeline's heads DISAGREE beats plain shell sampling.

Teacher heads (all fit on train normals only), per-head percentile-calibrated:
  o1 = autoencoder reconstruction error      (grows off-manifold)
  o2 = KDE negative log-density              (grows off-manifold, different shape)
  o3 = kNN mean distance                     (roughly radial)
  o4 = IsolationForest anomaly score         (piecewise, tree-structured)

Student: one MLP, K outputs, trained to reproduce the calibrated head vector on
(real normals + synthetic queries). Metric: mean over heads of Spearman(student
head, teacher head) on held-out real anomalies; plus fidelity of the fused
decision (mean / max of calibrated heads) and its ranking.

Samplers (label-free; all add M synthetic queries to the real normals):
  normals_only      : no synthetic queries (control)
  uniform_shell     : uniform in the normal-data shell (placement only)
  disagreement_shell: NEW -- importance-sample the shell by INTER-HEAD variance of
                      the calibrated outputs (query where the heads disagree; needs
                      no student proxy and no anomaly labels; defined only because
                      the teacher is multi-output)
  shell_box_mix     : half shell + half ambient box (geometry hedge)

Pre-stated tests:
  T1  sampling helps multi-output distillation (shell mean-fid >> normals_only).
  T2  KEY: disagreement_shell > uniform_shell for multi-output teachers.
  T3  multi-output distillation benefits at least as much from sampling as the
      single fused-scalar student does (run a scalar control on the fused score).
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, csv, zlib, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from experiment import make_percentile_maps

DATA_DIR = Path(__file__).parent / "data" / "adbench"
M = 1000


def _seed_of(*p):
    return zlib.crc32("|".join(map(str, p)).encode()) & 0x7FFFFFFF


def all_datasets(min_anom=20, min_normal=400, max_d=120):
    keep = []
    for p in sorted(DATA_DIR.glob("*.npz")):
        try:
            a = np.load(p); y = a["y"].astype(int); X = a["X"]
            if (y == 1).sum() >= min_anom and (y == 0).sum() >= min_normal and X.shape[1] <= max_d:
                keep.append(p.stem)
        except Exception:
            pass
    return keep


def load(name, seed, max_train=2500):
    a = np.load(DATA_DIR / f"{name}.npz")
    X, y = a["X"].astype(np.float64), a["y"].astype(int)
    normals, anoms = X[y == 0], X[y == 1]
    rng = np.random.default_rng(seed); rng.shuffle(normals)
    n_tr = min(max_train, len(normals) // 2)
    Xtr = normals[:n_tr]; Xval = normals[n_tr:n_tr + max(500, len(normals) // 4)]
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xval), sc.transform(anoms)


class MultiHeadTeacher:
    """x -> (ae_err, kde_nll, knn_dist, iforest_score); each fit on normals only."""
    HEADS = ["ae", "kde", "knn", "iforest"]

    def __init__(self, X, seed):
        d = X.shape[1]
        self.ae = MLPRegressor(hidden_layer_sizes=(64, max(2, d // 2), 64), activation="tanh",
                               solver="adam", learning_rate_init=3e-3, max_iter=500, batch_size=64,
                               random_state=seed, tol=1e-5, n_iter_no_change=25).fit(X, X)
        self.kde = KernelDensity(bandwidth=0.5).fit(X)
        self.knn = NearestNeighbors(n_neighbors=10).fit(X)
        self.iforest = IsolationForest(n_estimators=150, random_state=seed).fit(X)

    def __call__(self, P):
        o1 = ((P - self.ae.predict(P)) ** 2).mean(1)
        o2 = -self.kde.score_samples(P)
        o3 = self.knn.kneighbors(P)[0].mean(1)
        o4 = -self.iforest.score_samples(P)
        return np.stack([o1, o2, o3, o4], axis=-1)


def shell_scale(X, seed):
    nn = NearestNeighbors(n_neighbors=2).fit(X); rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(500, len(X)), replace=False)
    return float(np.median(nn.kneighbors(X[idx])[0][:, 1]))


def shell_pool(X, need, rng, rmin, rmax):
    d = X.shape[1]; nn = NearestNeighbors(n_neighbors=1).fit(X); pts = []; tries = 0
    while len(pts) < need:
        anchors = X[rng.integers(0, len(X), size=6000)]
        dirs = rng.normal(size=(6000, d)); dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
        cand = anchors + dirs * rng.uniform(rmin, rmax, size=(6000, 1))
        dd = nn.kneighbors(cand)[0][:, 0]
        pts.extend(cand[(dd > rmin) & (dd <= rmax)].tolist()); tries += 1
        if tries > 100:
            break
    return np.array(pts[:need]) if pts else np.zeros((0, d))


def make_queries(kind, X, teacher, pct, rng, scale):
    d = X.shape[1]
    if kind == "normals_only":
        return np.zeros((0, d))
    if kind == "uniform_shell":
        return shell_pool(X, M, rng, 1.0 * scale, 6.0 * scale)
    if kind == "shell_box_mix":
        half = M // 2; lo, hi = X.min(0) - scale, X.max(0) + scale
        box = rng.uniform(lo, hi, size=(M - half, d)); sh = shell_pool(X, half, rng, scale, 6 * scale)
        return np.concatenate([sh, box]) if len(sh) else box
    if kind == "disagreement_shell":
        # oversample the shell, keep the M points where the calibrated heads
        # DISAGREE most (variance across heads). No proxy, no labels.
        pool = shell_pool(X, M * 8, rng, 1.0 * scale, 6.0 * scale)
        if len(pool) < 10:
            return shell_pool(X, M, rng, scale, 6 * scale)
        C = pct(teacher(pool))                 # (n, K) calibrated
        disagree = C.var(axis=1)               # inter-head variance
        w = disagree / (disagree.sum() + 1e-12)
        return pool[rng.choice(len(pool), size=M, replace=True, p=w)]
    raise ValueError(kind)


def train_multi_student(X, Y, seed):
    return MLPRegressor(hidden_layer_sizes=(16,), activation="tanh", solver="adam",
                        learning_rate_init=5e-3, max_iter=800, random_state=seed,
                        tol=1e-6, n_iter_no_change=30).fit(X, Y)


SAMPLERS = ["normals_only", "uniform_shell", "disagreement_shell", "shell_box_mix"]


def run_cell(name, seed):
    Xtr, Xval, Xan = load(name, seed)
    teacher = MultiHeadTeacher(Xtr, seed)
    pct = make_percentile_maps(teacher(Xtr))          # per-head calibration on normals
    T_an = pct(teacher(Xan))                           # (n_an, K) target heads
    K = T_an.shape[1]
    # fused decision the pipeline would emit (mean of calibrated heads) + its AUROC
    fused_an = T_an.mean(1); fused_val = pct(teacher(Xval)).mean(1)
    y_true = np.r_[np.zeros(len(Xval)), np.ones(len(Xan))]
    teach_auroc = roc_auc_score(y_true, np.r_[fused_val, fused_an])
    scale = shell_scale(Xtr, seed)
    rows = []
    for kind in SAMPLERS:
        rng = np.random.default_rng(seed * 1_000_003 + _seed_of(kind))
        Xg = make_queries(kind, Xtr, teacher, pct, rng, scale)
        Xall = np.concatenate([Xtr, Xg]) if len(Xg) else Xtr
        Yall = pct(teacher(Xall))
        st = train_multi_student(Xall, Yall, seed)
        S_an = st.predict(Xan); S_an = S_an.reshape(len(Xan), -1)
        # per-head fidelity
        head_fid = [spearmanr(S_an[:, k], T_an[:, k]).statistic for k in range(K)]
        # fused-decision fidelity (student's fused vs teacher's fused) + student AUROC
        s_fused_an = S_an.mean(1); s_fused_val = st.predict(Xval).reshape(len(Xval), -1).mean(1)
        rows.append({"dataset": name, "d": Xtr.shape[1], "seed": seed, "sampler": kind,
                     "teacher_auroc": teach_auroc,
                     "mean_head_fid": float(np.nanmean(head_fid)),
                     "min_head_fid": float(np.nanmin(head_fid)),
                     **{f"fid_{h}": head_fid[i] for i, h in enumerate(MultiHeadTeacher.HEADS)},
                     "fused_fid": spearmanr(s_fused_an, fused_an).statistic,
                     "student_auroc": roc_auc_score(y_true, np.r_[s_fused_val, s_fused_an])})
    return rows


def _work(task):
    name, seed = task
    try:
        return run_cell(name, seed)
    except Exception as e:
        return [{"__fail__": f"{name}/s{seed}: {e}"}]


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--outdir", default="results_multioutput")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(exist_ok=True)
    datasets = all_datasets()
    tasks = [(n, s) for n in datasets for s in range(a.seeds)]
    print(f"datasets={len(datasets)} seeds={a.seeds} cells={len(tasks)} workers={a.workers}", flush=True)
    print("  " + ", ".join(datasets), flush=True)
    rows, done, fails = [], 0, 0
    with mp.Pool(a.workers) as pool:
        for res in pool.imap_unordered(_work, tasks, chunksize=1):
            if res and res[0].get("__fail__"):
                fails += 1; print(f"[FAIL] {res[0]['__fail__']}", flush=True); continue
            rows.extend(res); done += 1
            if done % 20 == 0:
                print(f"  ... {done}/{len(tasks)} cells ({fails} fails)", flush=True)
    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out/'results.csv'} ({len(rows)} rows, {done} cells, {fails} fails)", flush=True)


if __name__ == "__main__":
    main()
