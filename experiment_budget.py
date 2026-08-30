"""Matched-cost budget curves (TMLR review W1/W5).

Fixes the number of student-training queries (M_train) and varies the
ACQUISITION budget = total teacher evaluations spent selecting them, so methods
are compared at equal blackbox cost. Uniform-shell spends only M_train
evaluations (no selection). Score-shell scoring E candidates spends E. The
variation samplers spend more per candidate (coordinate FD: 1+2d; K directions:
1+2K), so at a matched eval budget they can score fewer candidates. The question:
does the smarter (variation) selection ever beat cheap (score / uniform)
selection once its selection cost is charged?

Teachers: 2D one-class SVM (saturating, where variation helped at fixed M) and
autoencoder (growing). Fidelity = Spearman on held-out anomalies, 5 seeds.
"""
from __future__ import annotations
import argparse, csv, collections
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from experiment import make_percentile_maps, train_student, sampler_none
from experiment_spectrum import gen_normals, gen_offmanifold, build_teachers

N_TRAIN, N_ANOM, M_TRAIN, SHELL = 2000, 500, 500, (0.15, 2.5)


def shell_pool(X, need, rng):
    d = X.shape[1]; nn = NearestNeighbors(n_neighbors=1).fit(X); pts = []
    while len(pts) < need:
        c = rng.uniform(X.min(0) - 1, X.max(0) + 1, size=(6000, d))
        dd = nn.kneighbors(c)[0][:, 0]
        pts.extend(c[(dd > SHELL[0]) & (dd < SHELL[1])].tolist())
    return np.array(pts[:need])


def select(kind, X, teacher, pct, rng, n_cand, h=0.05, K=None):
    """Score n_cand shell candidates and importance-sample M_train of them.
    Returns (queries, teacher_evals_spent)."""
    d = X.shape[1]
    if kind == "uniform_shell":
        q = shell_pool(X, M_TRAIN, rng)
        return q, M_TRAIN                       # only the training labels
    pool = shell_pool(X, n_cand, rng)
    fused = lambda P: pct(np.stack([teacher(P)], axis=-1))[:, 0]
    u = fused(pool); evals = len(pool)
    if kind == "score":
        w = np.clip(u, 0, None)
    else:                                        # combined: score + variation
        if K is None:
            g2 = np.zeros(len(pool))
            for i in range(d):
                e = np.zeros(d); e[i] = h
                g2 += ((fused(pool + e) - fused(pool - e)) / (2 * h)) ** 2
            v = np.sqrt(g2); evals += len(pool) * 2 * d
        else:
            acc = np.zeros(len(pool))
            for _ in range(K):
                z = rng.normal(size=(len(pool), d)); z /= (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
                acc += np.abs(fused(pool + h * z) - fused(pool - h * z)) / (2 * h)
            v = acc / K; evals += len(pool) * 2 * K
        un = np.clip(u, 0, None); un /= (un.sum() + 1e-12); vn = v / (v.sum() + 1e-12)
        w = 0.5 * un + 0.5 * vn
    w = w / (w.sum() + 1e-12)
    q = pool[rng.choice(len(pool), size=M_TRAIN, replace=True, p=w)]
    return q, evals


def fidelity(teacher, X, Xan, seed, kind, n_cand, K=None):
    teachers = [teacher]; sb = lambda A: np.stack([teacher(A)], axis=-1)
    pct = make_percentile_maps(sb(X)); t = pct(sb(Xan))[:, 0]
    rng = np.random.default_rng(seed * 337 + hash((kind, n_cand, K)) % 9973)
    if kind == "normals_only":
        Xg, evals = sampler_none(X, M_TRAIN, rng), M_TRAIN
    else:
        Xg, evals = select(kind, X, teacher, pct, rng, n_cand, K=K)
    Xall = np.concatenate([X, Xg]) if len(Xg) else X
    st = train_student(Xall, pct(sb(Xall)), seed)
    p = st.predict(Xan); s = p if p.ndim == 1 else p[:, 0]
    return spearmanr(s, t).statistic, evals


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--outdir", default="results_budget"); a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(exist_ok=True)
    cands = [500, 1000, 2000, 5000, 12000, 24000]     # candidate-pool sizes for the score sampler
    rows = []
    for seed in range(a.seeds):
        X = gen_normals(N_TRAIN, seed); Xan, _ = gen_offmanifold(X, N_ANOM, seed + 1)
        T = build_teachers(X, seed)
        for tn in ["ocsvm", "ae"]:
            tf = T[tn]
            f0, e0 = fidelity(tf, X, Xan, seed, "normals_only", 0)
            fu, eu = fidelity(tf, X, Xan, seed, "uniform_shell", 0)
            rows.append({"seed": seed, "teacher": tn, "method": "normals_only", "evals": e0, "fid": f0})
            rows.append({"seed": seed, "teacher": tn, "method": "uniform_shell", "evals": eu, "fid": fu})
            for c in cands:
                fs, es = fidelity(tf, X, Xan, seed, "score", c)
                fc, ec = fidelity(tf, X, Xan, seed, "combined", c, K=None)
                fk, ek = fidelity(tf, X, Xan, seed, "combined", c, K=4)
                rows.append({"seed": seed, "teacher": tn, "method": "score", "evals": es, "fid": fs})
                rows.append({"seed": seed, "teacher": tn, "method": "combined_FD", "evals": ec, "fid": fc})
                rows.append({"seed": seed, "teacher": tn, "method": "combined_K4", "evals": ek, "fid": fk})
            print(f"[budget] seed={seed} {tn} done", flush=True)
    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "teacher", "method", "evals", "fid"]); w.writeheader(); w.writerows(rows)
    print(f"Wrote {out/'results.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
