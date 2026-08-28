"""Higher-dimensional stress test of the sampler comparison.

The 2D two-moons experiment (experiment.py) is a smoke test. The paper's
central claim is that uniform augmentation stops being feasible in higher
dimensions (curse of dimensionality) while uncertainty-guided sampling
degrades gracefully. This file tests that claim at d in {5, 10} with a
d-dim Gaussian-mixture normals setup.

Reuses experiment.py's samplers, student, evaluation, and invariants.
Only the data generator is new.

Usage:
    python experiment_highd.py --dims 5 --seeds 3
    python experiment_highd.py --dims 10 --seeds 3
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

# Reuse everything from the 2D experiment.
from experiment import (
    SAMPLERS, fit_teachers, score_batch, normalize_stats,
    make_percentile_maps, train_student, eval_condition, auroc,
    sampler_none, sampler_langevin, sampler_gaussian, sampler_mixed,
)


def make_d_dim_uniform_sampler(X_train):
    """Rebuild sampler_uniform with a d-dim box derived from the training set
    (plus a margin), so the same code works at any dimensionality."""
    lo = X_train.min(axis=0) - 1.0
    hi = X_train.max(axis=0) + 1.0

    def _u(X_n, M, rng, **kw):
        return rng.uniform(lo, hi, size=(M, X_n.shape[1]))
    return _u


def make_d_dim_mixed_sampler(uniform_fn):
    """S4 mixed with the d-dim uniform half."""
    def _m(X_n, M, rng, *, teachers, mu, sigma, kde, pct_map, **kw):
        m_l = M // 2
        m_u = M - m_l
        Xg_l = sampler_langevin(X_n, m_l, rng,
                                teachers=teachers, mu=mu, sigma=sigma,
                                kde=kde, pct_map=pct_map)
        Xg_u = uniform_fn(X_n, m_u, rng)
        return np.concatenate([Xg_l, Xg_u], axis=0)
    return _m


N_TRAIN = 2000
N_VAL = 500
N_ANOM = 500
K_CLUSTERS = 3


CLUSTER_STD = 0.05  # tight so "boundary" band (0.15-0.30 min-dist) is truly off-manifold at any d
MIN_MEAN_SEP = 1.5  # cluster means kept at least this far apart


def _gmm_means(d: int, seed_means: int = 12345) -> np.ndarray:
    """Fixed cluster means for a given d, seeded independently so that
    train/val/anom all agree on WHERE the clusters live."""
    rng = np.random.default_rng(seed_means * 1000 + d)
    for _ in range(200):
        means = rng.uniform(-1.5, 1.5, size=(K_CLUSTERS, d))
        diffs = means[:, None] - means[None, :]
        dm = np.linalg.norm(diffs, axis=-1) + np.eye(K_CLUSTERS) * 1e6
        if dm.min() >= MIN_MEAN_SEP:
            return means
    raise RuntimeError(f"Could not sample well-separated means at d={d}")


def gen_normals_gmm(n: int, d: int, seed: int) -> np.ndarray:
    """Draw n normals from the fixed d-dim GMM. Data seed varies; means don't."""
    means = _gmm_means(d)
    rng = np.random.default_rng(seed)
    assignments = rng.integers(0, K_CLUSTERS, size=n)
    return means[assignments] + rng.normal(scale=CLUSTER_STD, size=(n, d))


def gen_anomalies_highd(X_train: np.ndarray, d: int, seed: int) -> dict[str, np.ndarray]:
    """d-dim anomaly buckets by distance to nearest normal.

    At higher d the manifold occupies a vanishing fraction of any box, so
    rejection from a uniform box fails. We instead sample each anomaly by
    picking a random training normal, a random unit-sphere direction, and a
    distance drawn uniformly in the band. Every accepted point is verified
    against a global 1-NN distance to guarantee band membership (starting
    from an anchor's shell does not guarantee that anchor is the NEAREST
    normal). Rejection is cheap because we already start at the target band.
    """
    rng = np.random.default_rng(seed + 10_000)
    nn = NearestNeighbors(n_neighbors=1).fit(X_train)

    # Absolute distance bands. Tight clusters (std=0.05) keep the within-
    # cluster spread below 0.15 at any d, so "boundary" (0.15-0.30) is
    # unambiguously off-manifold at every dimensionality.
    bands = {"boundary": (0.15, 0.30),
             "close":    (0.30, 0.60),
             "medium":   (0.60, 1.20),
             "far":      (2.00, 3.00)}
    out = {}
    for name, (lo, hi) in bands.items():
        pts = []
        tries = 0
        while len(pts) < N_ANOM:
            batch = 4000
            # random anchors
            anchors = X_train[rng.integers(0, len(X_train), size=batch)]
            # random unit directions
            directions = rng.normal(size=(batch, d))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
            # target distances in the band
            radii = rng.uniform(lo, hi, size=(batch, 1))
            cand = anchors + directions * radii
            # global 1-NN distance check: keep only cand whose ACTUAL nearest
            # normal (which may not be the anchor) sits in the band.
            true_d, _ = nn.kneighbors(cand)
            true_d = true_d[:, 0]
            mask = (true_d > lo) & (true_d <= hi)
            pts.extend(cand[mask].tolist())
            tries += 1
            if tries > 40:
                raise RuntimeError(f"gen_anomalies_highd d={d} band={name}: "
                                   f"only {len(pts)}/{N_ANOM} after {tries} rounds")
        out[name] = np.array(pts[:N_ANOM])
    return out


def run_one_dsamp(seed, cond, X_train, X_val, anoms, teachers, mu, sigma, kde,
                  pct_map, local_samplers, M):
    rng = np.random.default_rng(seed * 97 + hash(cond) % 10_000)
    kw = dict(teachers=teachers, mu=mu, sigma=sigma, kde=kde, pct_map=pct_map)
    sampler = local_samplers[cond]
    Xg = sampler(X_train, M, rng, **kw) if cond != "S0_none" else sampler_none(X_train, M, rng)
    X_all = np.concatenate([X_train, Xg]) if len(Xg) else X_train
    Y_all = pct_map(score_batch(teachers, X_all))
    student = train_student(X_all, Y_all, seed=seed)
    metrics = eval_condition(student, teachers, mu, sigma, X_val, anoms, pct_map)
    return student, Xg, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--M", type=int, default=2000)
    parser.add_argument("--outdir", default="results_highd")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    rows = []
    for d in args.dims:
        print(f"\n=== d = {d} ===", flush=True)
        X_train = gen_normals_gmm(N_TRAIN, d, seed=0)
        X_val = gen_normals_gmm(N_VAL, d, seed=1)
        try:
            anoms = gen_anomalies_highd(X_train, d, seed=0)
        except RuntimeError as e:
            print(f"[skip d={d}] {e}", flush=True)
            continue

        teachers, kde = fit_teachers(X_train, seed=0)
        S_train = score_batch(teachers, X_train)
        mu, sigma = normalize_stats(S_train)
        pct_map = make_percentile_maps(S_train)

        # Rebuild d-dim uniform + mixed samplers for this dimensionality.
        u_fn = make_d_dim_uniform_sampler(X_train)
        m_fn = make_d_dim_mixed_sampler(u_fn)
        local_samplers = dict(SAMPLERS)
        local_samplers["S2_uniform"] = u_fn
        local_samplers["S4_mixed"] = m_fn

        # I4 quick check.
        for name in anoms:
            for k, tname in enumerate(["kde", "iforest", "knn"]):
                def _pct_scalar(pts, k=k):
                    return pct_map(score_batch(teachers, pts))[:, k]
                au = auroc(_pct_scalar(X_val), _pct_scalar(anoms[name]))
                if au < 0.8:
                    print(f"[warn d={d}] teacher {tname} on {name} AUROC={au:.3f} (< 0.8, task may be too hard)")

        for seed in range(args.seeds):
            for cond in local_samplers:
                t0 = time.time()
                _, Xg, m = run_one_dsamp(seed, cond, X_train, X_val, anoms,
                                          teachers, mu, sigma, kde, pct_map,
                                          local_samplers, M=args.M)
                wall = time.time() - t0
                for k, v in m.items():
                    rows.append({"d": d, "seed": seed, "cond": cond,
                                 "metric": k, "value": v})
                print(f"d={d} seed={seed} cond={cond:<12} "
                      f"bdy={m['auroc_student_boundary']:.3f} "
                      f"close={m['auroc_student_close']:.3f} "
                      f"med={m['auroc_student_medium']:.3f} "
                      f"far={m['auroc_student_far']:.3f} "
                      f"({wall:.1f}s)", flush=True)

    csv_path = outdir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["d", "seed", "cond", "metric", "value"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
