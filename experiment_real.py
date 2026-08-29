"""Real-data anomaly-detection sweep on ADBench tabular datasets.

Reuses everything from experiment.py (samplers, student, invariants, eval) and
only changes the data generator: real X_normal from an ADBench .npz, real
X_anom for the test set, no synthetic anomaly bands.

Datasets used (spanning the paper's d range):
  32_shuttle.npz    d=9,  n=49k, anom_frac=0.0715
  30_satellite.npz  d=36, n=6.4k, anom_frac=0.3164
  26_optdigits.npz  d=64, n=5.2k, anom_frac=0.0288

Usage:
  python experiment_real.py --datasets shuttle satellite --seeds 5
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from experiment import (
    SAMPLERS, fit_teachers, score_batch, normalize_stats,
    make_percentile_maps, train_student, auroc,
    sampler_none, sampler_langevin, sampler_gaussian, sampler_mixed,
)

DATA_DIR = Path(__file__).parent / "data" / "adbench"

DATASETS = {
    "shuttle":    "32_shuttle.npz",
    "satellite":  "30_satellite.npz",
    "optdigits":  "26_optdigits.npz",
}


def load_dataset(name: str, seed: int, max_train: int = 4000):
    """Load an ADBench .npz. Returns (X_train_normal, X_val_normal, X_anom).

    Splits: half of normals -> train, quarter -> val, quarter unused. All
    anomalies used for test. Standardized to zero-mean / unit-variance on
    training normals only (scaler fit on X_train_normal, applied to all).
    Optional max_train cap (uniform subsample) so d=64 dataset stays fast.
    """
    d = np.load(DATA_DIR / DATASETS[name])
    X, y = d["X"].astype(np.float64), d["y"].astype(int)
    normals = X[y == 0]
    anoms   = X[y == 1]

    rng = np.random.default_rng(seed)
    rng.shuffle(normals)
    n_norm = len(normals)
    n_train = min(max_train, n_norm // 2)
    n_val   = max(500, n_norm // 4)
    X_train = normals[:n_train]
    X_val   = normals[n_train : n_train + n_val]

    # Standardize using training normals only.
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    X_anom  = scaler.transform(anoms)

    return X_train, X_val, X_anom


def make_d_dim_uniform_sampler(X_train):
    lo = X_train.min(axis=0) - 1.0
    hi = X_train.max(axis=0) + 1.0
    def _u(X_n, M, rng, **kw):
        return rng.uniform(lo, hi, size=(M, X_n.shape[1]))
    return _u


def make_d_dim_mixed_sampler(uniform_fn):
    def _m(X_n, M, rng, *, teachers, mu, sigma, kde, pct_map, **kw):
        m_l = M // 2
        m_u = M - m_l
        Xg_l = sampler_langevin(X_n, m_l, rng,
                                teachers=teachers, mu=mu, sigma=sigma,
                                kde=kde, pct_map=pct_map)
        Xg_u = uniform_fn(X_n, m_u, rng)
        return np.concatenate([Xg_l, Xg_u], axis=0)
    return _m


def eval_condition_real(student, teachers, mu, sigma, X_val, X_anom, pct_map):
    """Fused AUROC of student on real held-out normals vs real anomalies."""
    def fused_teacher(pts):
        return pct_map(score_batch(teachers, pts)).mean(axis=1)
    def fused_student(pts):
        return student.predict(pts).mean(axis=1)

    val_s = fused_student(X_val)
    anom_s = fused_student(X_anom)
    val_t = fused_teacher(X_val)
    anom_t = fused_teacher(X_anom)
    return {
        "auroc_student":        auroc(val_s, anom_s),
        "auroc_teacher":        auroc(val_t, anom_t),
        "rmse_student_normals": float(np.sqrt(np.mean((val_s - val_t) ** 2))),
        "rmse_student_anom":    float(np.sqrt(np.mean((anom_s - anom_t) ** 2))),
    }


def run_shard(dataset: str, seed: int, M: int = 2000):
    """Run all six SAMPLERS at one (dataset, seed). Returns per-metric rows."""
    X_train, X_val, X_anom = load_dataset(dataset, seed=seed)
    d = X_train.shape[1]

    teachers, kde = fit_teachers(X_train, seed=0)
    S_train = score_batch(teachers, X_train)
    mu, sigma = normalize_stats(S_train)
    pct_map = make_percentile_maps(S_train)

    u_fn = make_d_dim_uniform_sampler(X_train)
    m_fn = make_d_dim_mixed_sampler(u_fn)
    local_samplers = dict(SAMPLERS)
    local_samplers["S2_uniform"] = u_fn
    local_samplers["S4_mixed"]   = m_fn

    rows = []
    for cond in local_samplers:
        t0 = time.time()
        rng = np.random.default_rng(seed * 97 + hash(cond) % 10_000)
        kw = dict(teachers=teachers, mu=mu, sigma=sigma, kde=kde, pct_map=pct_map)
        sampler = local_samplers[cond]
        Xg = sampler(X_train, M, rng, **kw) if cond != "S0_none" else sampler_none(X_train, M, rng)
        X_all = np.concatenate([X_train, Xg]) if len(Xg) else X_train
        Y_all = pct_map(score_batch(teachers, X_all))
        student = train_student(X_all, Y_all, seed=seed)
        metrics = eval_condition_real(student, teachers, mu, sigma, X_val, X_anom, pct_map)
        wall = time.time() - t0
        for metric, value in metrics.items():
            rows.append({"dataset": dataset, "d": d, "seed": seed, "cond": cond,
                         "metric": metric, "value": float(value)})
        rows.append({"dataset": dataset, "d": d, "seed": seed, "cond": cond,
                     "metric": "wall_s", "value": float(wall)})
        print(f"[{dataset} d={d} seed={seed}] {cond}: auroc={metrics['auroc_student']:.3f} "
              f"(teacher {metrics['auroc_teacher']:.3f})  ({wall:.1f}s)", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--outdir", default="results_real")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    all_rows = []
    for name in args.datasets:
        for s in range(args.seeds):
            all_rows.extend(run_shard(name, s))

    csv_path = outdir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "d", "seed", "cond", "metric", "value"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {csv_path} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
