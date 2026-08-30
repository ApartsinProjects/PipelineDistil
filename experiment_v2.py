"""Revision experiments (TMLR review response):
  - uniform-shell baseline (does importance allocation inside the shell matter,
    or is any shell coverage enough?)
  - K random-direction local-variation estimator vs coordinate finite differences
    (query cost O(K) instead of O(d))
  - lambda mixture sweep (is 0.5 a robust hedge?)
  on the 5-teacher spectrum, 10 seeds.

Fidelity = Spearman(student, teacher) on held-out off-manifold anomalies.
"""
from __future__ import annotations
import argparse, csv, collections
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from experiment import make_percentile_maps, train_student, sampler_none
from experiment_spectrum import gen_normals, gen_offmanifold, build_teachers

N_TRAIN, N_ANOM, M = 2000, 500, 2000
SHELL = (0.15, 2.5)


def shell_pool(X, need, rng):
    d = X.shape[1]; nn = NearestNeighbors(n_neighbors=1).fit(X)
    lo, hi = X.min(0) - 1.0, X.max(0) + 1.0; pool = []
    while len(pool) < need:
        c = rng.uniform(lo, hi, size=(6000, d))
        dist = nn.kneighbors(c)[0][:, 0]
        pool.extend(c[(dist > SHELL[0]) & (dist < SHELL[1])].tolist())
    return np.array(pool[:need])


def signals(pool, teachers, pct, h=0.05, K=None, rng=None):
    """Return (u, var) where u is fused rank score and var is local variation.
    K=None -> coordinate finite differences (O(d)); K=int -> K random directions (O(K))."""
    d = pool.shape[1]
    fused = lambda P: pct(np.stack([f(P) for f in teachers], axis=-1)).mean(1)
    u = fused(pool)
    if K is None:
        g2 = np.zeros(len(pool))
        for i in range(d):
            e = np.zeros(d); e[i] = h
            g2 += ((fused(pool + e) - fused(pool - e)) / (2 * h)) ** 2
        var = np.sqrt(g2)
    else:
        acc = np.zeros(len(pool))
        for _ in range(K):
            z = rng.normal(size=(len(pool), d)); z /= (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
            acc += np.abs(fused(pool + h * z) - fused(pool - h * z)) / (2 * h)
        var = acc / K
    return u, var


def make_queries(kind, X, teachers, pct, rng, lam=0.5, K=None):
    if kind == "uniform_shell":
        return shell_pool(X, M, rng)
    pool = shell_pool(X, M * 12, rng)
    u, var = signals(pool, teachers, pct, K=K, rng=rng)
    un = np.clip(u, 0, None); un = un / (un.sum() + 1e-12)
    vn = var / (var.sum() + 1e-12)
    if kind == "score":      w = un
    elif kind == "variation": w = vn
    else:                    w = lam * un + (1 - lam) * vn   # mix
    w = w / w.sum()
    return pool[rng.choice(len(pool), size=M, replace=True, p=w)]


def fidelity(teacher_fn, X, Xan, seed, kind, lam=0.5, K=None):
    teachers = [teacher_fn]; sb = lambda A: np.stack([f(A) for f in teachers], axis=-1)
    pct = make_percentile_maps(sb(X)); t = pct(sb(Xan))[:, 0]
    rng = np.random.default_rng(seed * 191 + hash((kind, lam, K)) % 9973)
    Xg = sampler_none(X, M, rng) if kind == "normals_only" else make_queries(kind, X, teachers, pct, rng, lam, K)
    Xall = np.concatenate([X, Xg]) if len(Xg) else X
    st = train_student(Xall, pct(sb(Xall)), seed)
    p = st.predict(Xan); s = p if p.ndim == 1 else p[:, 0]
    return spearmanr(s, t).statistic


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--outdir", default="results_v2"); a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(exist_ok=True)
    order = ["knn", "kde", "ocsvm", "ae", "ae_kde_max"]
    rows = []
    for seed in range(a.seeds):
        X = gen_normals(N_TRAIN, seed); Xan, _ = gen_offmanifold(X, N_ANOM, seed + 1)
        T = build_teachers(X, seed)
        for tn in order:
            tf = T[tn]
            # baselines + main samplers (coordinate FD variation, K=None)
            r = {"seed": seed, "teacher": tn}
            r["normals_only"] = fidelity(tf, X, Xan, seed, "normals_only")
            r["uniform_shell"] = fidelity(tf, X, Xan, seed, "uniform_shell")
            r["score"] = fidelity(tf, X, Xan, seed, "score")
            r["variation_FD"] = fidelity(tf, X, Xan, seed, "variation", K=None)
            r["combined_FD"] = fidelity(tf, X, Xan, seed, "mix", lam=0.5, K=None)
            # K random-direction estimator (query-efficient)
            r["variation_K2"] = fidelity(tf, X, Xan, seed, "variation", K=2)
            r["combined_K2"] = fidelity(tf, X, Xan, seed, "mix", lam=0.5, K=2)
            # lambda sweep (combined, coordinate FD)
            for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
                r[f"lam{lam}"] = fidelity(tf, X, Xan, seed, "mix", lam=lam, K=None)
            rows.append(r)
            print(f"[v2 seed={seed}] {tn:<11} unif={r['uniform_shell']:.3f} "
                  f"score={r['score']:.3f} varFD={r['variation_FD']:.3f} "
                  f"combFD={r['combined_FD']:.3f} combK2={r['combined_K2']:.3f}", flush=True)
    keys = list(rows[0].keys())
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    # summary
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for k in keys:
            if k not in ("seed", "teacher"): agg[r["teacher"]][k].append(r[k])
    print("\n=== fidelity means (10 seeds) ===")
    cols = ["normals_only", "uniform_shell", "score", "variation_FD", "combined_FD", "variation_K2", "combined_K2"]
    print("  teacher      " + " ".join(f"{c[:11]:>12}" for c in cols))
    for tn in order:
        print(f"  {tn:<11} " + " ".join(f"{np.mean(agg[tn][c]):>12.3f}" for c in cols))
    print("\n=== lambda sweep (combined FD) ===")
    lams = ["lam0.0", "lam0.25", "lam0.5", "lam0.75", "lam1.0"]
    print("  teacher      " + " ".join(f"{l:>9}" for l in lams))
    for tn in order:
        print(f"  {tn:<11} " + " ".join(f"{np.mean(agg[tn][l]):>9.3f}" for l in lams))
    print(f"\nWrote {out/'results.csv'}")


if __name__ == "__main__":
    main()
