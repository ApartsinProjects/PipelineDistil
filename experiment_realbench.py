"""Real tabular anomaly benchmark (TMLR review Exp 6 / W12).

Tests the central practical claim on real data: does placing distillation queries
in a normal-data-defined shell give a reliable student-teacher FIDELITY lift over
normals-only, on real datasets with a real (growing) autoencoder teacher?

Protocol (label-free, no anomaly labels for training/acquisition/selection):
  - split each dataset's normals into train / val;
  - fit an autoencoder teacher on the train normals only (growing off-manifold);
  - define the shell from the train normals only (leave-one-out kNN scale);
  - distill with normals-only, uniform-shell, and score-shell samplers;
  - primary metric: Spearman student-teacher fidelity on the held-out real
    anomalies; secondary: student standalone AUROC vs ground-truth labels
    (used only for evaluation).
Reports per dataset and a paired aggregate (Wilcoxon) of uniform-shell vs
normals-only fidelity.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse, csv, collections
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from experiment import make_percentile_maps, train_student, sampler_none

DATA_DIR = Path(__file__).parent / "data" / "adbench"
DATASETS = ["23_mammography", "38_thyroid", "6_cardio", "30_satellite", "28_pendigits",
            "40_vowels", "32_shuttle", "26_optdigits", "41_Waveform", "20_letter"]
M, OVERSAMPLE = 1000, 8


def _find(name):
    p = DATA_DIR / f"{name}.npz"
    if p.exists():
        return p
    here = Path(__file__).parent
    for c in here.rglob(f"{name}.npz"):
        return c
    raise FileNotFoundError(f"{name}.npz not found under {here}")


def load(name, seed, max_train=3000):
    a = np.load(_find(name))
    X, y = a["X"].astype(np.float64), a["y"].astype(int)
    normals, anoms = X[y == 0], X[y == 1]
    rng = np.random.default_rng(seed); rng.shuffle(normals)
    n_tr = min(max_train, len(normals) // 2)
    Xtr, Xval = normals[:n_tr], normals[n_tr:n_tr + max(500, len(normals) // 4)]
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xval), sc.transform(anoms)


def ae_teacher(X, seed):
    d = X.shape[1]
    ae = MLPRegressor(hidden_layer_sizes=(64, max(2, d // 2), 64), activation="tanh",
                      solver="adam", learning_rate_init=3e-3, max_iter=600, batch_size=64,
                      random_state=seed, tol=1e-5, n_iter_no_change=25).fit(X, X)
    return lambda P: ((P - ae.predict(P)) ** 2).mean(1)


def shell_bounds(X, seed):
    nn = NearestNeighbors(n_neighbors=2).fit(X); rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(500, len(X)), replace=False)
    scale = np.median(nn.kneighbors(X[idx])[0][:, 1])
    return 1.0 * scale, 6.0 * scale


def shell_pool(X, need, rng, rmin, rmax):
    d = X.shape[1]; nn = NearestNeighbors(n_neighbors=1).fit(X); pts = []; tries = 0
    while len(pts) < need:
        anchors = X[rng.integers(0, len(X), size=6000)]
        dirs = rng.normal(size=(6000, d)); dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
        cand = anchors + dirs * rng.uniform(rmin, rmax, size=(6000, 1))
        dd = nn.kneighbors(cand)[0][:, 0]
        pts.extend(cand[(dd > rmin) & (dd <= rmax)].tolist())
        tries += 1
        if tries > 80:
            break
    return np.array(pts[:need]) if pts else np.zeros((0, d))


def make_queries(kind, X, teacher, pct, rng, rmin, rmax):
    if kind == "uniform_shell":
        return shell_pool(X, M, rng, rmin, rmax)
    pool = shell_pool(X, M * OVERSAMPLE, rng, rmin, rmax)
    if len(pool) < 10:
        return sampler_none(X, M, rng)
    u = pct(np.stack([teacher(pool)], axis=-1))[:, 0]
    w = np.clip(u, 0, None); w = w / (w.sum() + 1e-12)
    return pool[rng.choice(len(pool), size=M, replace=True, p=w)]


def run_one(name, seed):
    Xtr, Xval, Xan = load(name, seed)
    teacher = ae_teacher(Xtr, seed); sb = lambda A: np.stack([teacher(A)], axis=-1)
    pct = make_percentile_maps(sb(Xtr))
    t_an = pct(sb(Xan))[:, 0]; t_val = pct(sb(Xval))[:, 0]
    rmin, rmax = shell_bounds(Xtr, seed)
    y_true = np.r_[np.zeros(len(Xval)), np.ones(len(Xan))]
    teach_auroc = roc_auc_score(y_true, np.r_[t_val, t_an])
    out = {"dataset": name, "d": Xtr.shape[1], "seed": seed, "teacher_auroc": teach_auroc}
    for kind in ("normals_only", "uniform_shell", "score_shell"):
        rng = np.random.default_rng(seed * 313 + hash(kind) % 9973)
        Xg = sampler_none(Xtr, M, rng) if kind == "normals_only" else make_queries(
            "score" if kind == "score_shell" else "uniform_shell", Xtr, teacher, pct, rng, rmin, rmax)
        Xall = np.concatenate([Xtr, Xg]) if len(Xg) else Xtr
        st = train_student(Xall, pct(sb(Xall)), seed)
        s_an = st.predict(Xan); s_an = s_an if s_an.ndim == 1 else s_an[:, 0]
        s_val = st.predict(Xval); s_val = s_val if s_val.ndim == 1 else s_val[:, 0]
        out[f"fid_{kind}"] = spearmanr(s_an, t_an).statistic
        out[f"auroc_{kind}"] = roc_auc_score(y_true, np.r_[s_val, s_an])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--outdir", default="results_realbench"); a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(exist_ok=True)
    rows = []
    for name in DATASETS:
        for seed in range(a.seeds):
            try:
                r = run_one(name, seed); rows.append(r)
                print(f"[train] {name:<15} d={r['d']:>3} seed={seed} "
                      f"fid: none={r['fid_normals_only']:.3f} unif={r['fid_uniform_shell']:.3f} "
                      f"score={r['fid_score_shell']:.3f} (teacher AUROC {r['teacher_auroc']:.3f})", flush=True)
            except Exception as e:
                print(f"[train] {name} seed={seed} FAILED: {e}", flush=True)
    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # aggregate: per-dataset mean, paired uniform-shell vs normals-only
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for k in ("fid_normals_only", "fid_uniform_shell", "fid_score_shell",
                  "auroc_normals_only", "auroc_uniform_shell", "auroc_score_shell"):
            agg[r["dataset"]][k].append(r[k])
    print("\n=== per-dataset fidelity (mean) ===")
    print(f"  {'dataset':<15}{'none':>8}{'unif':>8}{'score':>8}{'lift':>8}")
    lifts = []
    for name in DATASETS:
        a_ = agg[name]
        if not a_: continue
        none, unif = np.mean(a_["fid_normals_only"]), np.mean(a_["fid_uniform_shell"])
        lifts.append(unif - none)
        print(f"  {name:<15}{none:>8.3f}{unif:>8.3f}{np.mean(a_['fid_score_shell']):>8.3f}{unif-none:>+8.3f}")
    lifts = np.array(lifts)
    print(f"\n  median shell lift (uniform-shell - normals-only): {np.median(lifts):+.3f}")
    print(f"  datasets improved: {(lifts > 0).sum()}/{len(lifts)}")
    # paired wilcoxon on per-dataset means
    per_none = [np.mean(agg[n]["fid_normals_only"]) for n in DATASETS if agg[n]]
    per_unif = [np.mean(agg[n]["fid_uniform_shell"]) for n in DATASETS if agg[n]]
    try:
        stat, p = wilcoxon(per_unif, per_none, alternative="greater")
        print(f"  paired Wilcoxon (uniform-shell > normals-only): p={p:.4f}")
    except Exception as e:
        print(f"  wilcoxon failed: {e}")
    print(f"\nWrote {out/'results.csv'}")


if __name__ == "__main__":
    main()
