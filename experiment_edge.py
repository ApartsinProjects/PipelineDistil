"""S6 edge-of-low-density sampler vs S0/S5 across the teacher spectrum.

Idea (from a design discussion): for a non-monotonic pipeline the informative
queries are where the teacher score CHANGES (large gradient) in regions of LOW
data density (off-manifold, no training signal) -- the 'edge of low density'.
S6 draws candidates in the off-manifold shell and importance-samples them by
teacher score-gradient magnitude, instead of climbing the score like S5.

Memory-efficient: small candidate pool, chunked gradient.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors, KernelDensity
from experiment import make_percentile_maps, train_student, sampler_none, sampler_langevin_adaptive
from experiment_spectrum import gen_normals, gen_offmanifold, build_teachers

N_TRAIN, N_ANOM = 2000, 500


def _shell_pool(X_n, need, rng, dmin, dmax):
    d = X_n.shape[1]
    nn = NearestNeighbors(n_neighbors=1).fit(X_n)
    lo, hi = X_n.min(0) - 1.0, X_n.max(0) + 1.0
    pool = []
    while len(pool) < need:
        cand = rng.uniform(lo, hi, size=(6000, d))
        dist = nn.kneighbors(cand)[0][:, 0]
        pool.extend(cand[(dist > dmin) & (dist < dmax)].tolist())
    return np.array(pool[:need])


def _fused_and_grad(pool, teachers, pct_map, h):
    d = pool.shape[1]
    def fused(P): return pct_map(np.stack([f(P) for f in teachers], axis=-1)).mean(1)
    s = fused(pool)
    g2 = np.zeros(len(pool))
    for i in range(d):
        e = np.zeros(d); e[i] = h
        g2 += ((fused(pool + e) - fused(pool - e)) / (2 * h)) ** 2
    return s, np.sqrt(g2)


def sampler_edge(X_n, M, rng, *, teachers, pct_map, h=0.05, dmin=0.15, dmax=2.5, oversample=12):
    pool = _shell_pool(X_n, M * oversample, rng, dmin, dmax)
    _, g = _fused_and_grad(pool, teachers, pct_map, h)
    w = g / (g.sum() + 1e-12)
    return pool[rng.choice(len(pool), size=M, replace=True, p=w)]


def sampler_combined(X_n, M, rng, *, teachers, pct_map, h=0.05, dmin=0.15, dmax=2.5, oversample=12):
    """S7: importance-sample the low-density shell by BOTH the teacher score
    (climb toward anomalous side, like S5) AND the score-gradient magnitude
    (edges, like S6). Combines the two preferences by averaging their
    normalized weights, so it adapts to whichever signal the teacher provides:
    growing detectors (AE) via the score term, saturating detectors (OC-SVM)
    via the gradient term."""
    pool = _shell_pool(X_n, M * oversample, rng, dmin, dmax)
    s, g = _fused_and_grad(pool, teachers, pct_map, h)
    sn = np.clip(s, 0, None); sn = sn / (sn.sum() + 1e-12)
    gn = g / (g.sum() + 1e-12)
    w = 0.5 * sn + 0.5 * gn
    w = w / w.sum()
    return pool[rng.choice(len(pool), size=M, replace=True, p=w)]


def growth_signature(X_n, teachers, pct_map, rng, radii=(0.3, 0.6, 1.0, 1.5, 2.2), npershell=400):
    """Probe how the teacher's fused score behaves as we move OFF the manifold.
    Sample points in concentric shells at increasing distance and measure the
    median fused percentile at each. Returns g in [0,1]:
      g ~ 1  => GROWING teacher (score keeps rising with distance; use score-climb)
      g ~ 0  => SATURATING teacher (score plateaus off-manifold; use edge/gradient)
    g is the normalized slope of median-score vs shell-distance (Spearman-like),
    computed purely from blackbox queries -- no anomaly labels needed."""
    d = X_n.shape[1]
    nn = NearestNeighbors(n_neighbors=1).fit(X_n)
    lo, hi = X_n.min(0) - 1.0, X_n.max(0) + 1.0
    raw = []   # mean RAW fused teacher score per shell
    ext = []   # mean EXTENDED percentile per shell (how far past training range)
    for r in radii:
        pts = []
        while len(pts) < npershell:
            cand = rng.uniform(lo, hi, size=(4000, d))
            dist = nn.kneighbors(cand)[0][:, 0]
            band = cand[(dist > r * 0.8) & (dist < r * 1.2)]
            pts.extend(band.tolist())
        pts = np.array(pts[:npershell])
        Sraw = np.stack([f(pts) for f in teachers], axis=-1).mean(1)
        raw.append(np.mean(Sraw))
        ext.append(np.mean(pct_map(np.stack([f(pts) for f in teachers], axis=-1)).mean(1)))
    raw, ext = np.array(raw), np.array(ext)
    # A GROWING teacher keeps rising past the training range: its raw score at
    # the far shell greatly exceeds the near shell, and its extended percentile
    # climbs well above 1. A SATURATING teacher plateaus: raw ratio ~1 and
    # extended percentile stays ~1. Combine both cues.
    raw_growth = (raw[-1] - raw[0]) / (abs(raw[0]) + abs(raw[-1]) + 1e-9)   # in [~0, 1]
    ext_growth = np.clip(ext[-1] - 1.0, 0.0, None)                         # >0 if past range
    g = 0.5 * np.clip(raw_growth, 0, 1) + 0.5 * np.clip(ext_growth / 0.3, 0, 1)
    return float(np.clip(g, 0.0, 1.0)), raw


def sampler_auto(X_n, M, rng, *, teachers, pct_map, h=0.05, dmin=0.15, dmax=2.5, oversample=12):
    """AUTOMATIC sampler: inspect the teacher's growth signature and set the
    score-vs-gradient weight accordingly, then importance-sample the low-density
    shell. Growing teachers -> weight the score term; saturating -> the gradient
    term. One sampler, no manual choice, no teacher labels."""
    g, _ = growth_signature(X_n, teachers, pct_map, rng)
    pool = _shell_pool(X_n, M * oversample, rng, dmin, dmax)
    s, grad = _fused_and_grad(pool, teachers, pct_map, h)
    sn = np.clip(s, 0, None); sn = sn / (sn.sum() + 1e-12)
    gn = grad / (grad.sum() + 1e-12)
    # alpha in [0.2, 0.8] from the growth signature: more score weight when
    # growing, more gradient weight when saturating. Never fully drop either.
    alpha = 0.2 + 0.6 * g
    w = alpha * sn + (1 - alpha) * gn
    w = w / w.sum()
    return pool[rng.choice(len(pool), size=M, replace=True, p=w)], g, alpha


def sampler_active(X_n, M, rng, *, teachers, pct_map, proxy_student, h=0.05,
                   dmin=0.15, dmax=2.5, oversample=12):
    """S8: S7's score+gradient shell sampler PLUS the teacher-disagreement term
    from the original SAG loss ((y_g - T(x_g))). A first-pass normals-only
    student is the proxy; candidates are weighted by |proxy(x) - teacher(x)|
    (where the normals-only student is currently WRONG about the pipeline) on
    top of the score and gradient terms. This is active learning: sample where
    correcting the student would help most."""
    pool = _shell_pool(X_n, M * oversample, rng, dmin, dmax)
    s, g = _fused_and_grad(pool, teachers, pct_map, h)
    t = pct_map(np.stack([f(pool) for f in teachers], axis=-1)).mean(1)   # teacher fused
    pp = proxy_student.predict(pool); pp = pp if pp.ndim == 1 else pp.mean(1)
    disagree = np.abs(pp - t)
    sn = np.clip(s, 0, None); sn = sn / (sn.sum() + 1e-12)
    gn = g / (g.sum() + 1e-12)
    dn = disagree / (disagree.sum() + 1e-12)
    w = (sn + gn + dn) / 3.0
    w = w / w.sum()
    return pool[rng.choice(len(pool), size=M, replace=True, p=w)]


def fidelity(teacher_fn, X, Xan, kde, seed, which, M=2000):
    teachers = [teacher_fn]; sb = lambda A: np.stack([f(A) for f in teachers], axis=-1)
    pct = make_percentile_maps(sb(X)); t = pct(sb(Xan))[:, 0]
    rng = np.random.default_rng(seed * 151 + hash(which) % 9973)
    if which == "S0":
        Xg = sampler_none(X, M, rng)
    elif which == "S5":
        Xg = sampler_langevin_adaptive(X, M, rng, teachers=teachers, mu=None, sigma=None, kde=kde, pct_map=pct)
    elif which == "S6":
        Xg = sampler_edge(X, M, rng, teachers=teachers, pct_map=pct)
    elif which == "S7":
        Xg = sampler_combined(X, M, rng, teachers=teachers, pct_map=pct)
    elif which == "S8":  # active learning w/ normals-only proxy (naive)
        proxy = train_student(X, pct(sb(X)), seed)
        Xg = sampler_active(X, M, rng, teachers=teachers, pct_map=pct, proxy_student=proxy)
    else:  # S9: 2-round. Round1 = S7 (M/2). Proxy trained on round1 (informative
           # off-manifold). Round2 (M/2) = where that proxy STILL disagrees.
        Xg1 = sampler_combined(X, M // 2, rng, teachers=teachers, pct_map=pct)
        proxy = train_student(np.concatenate([X, Xg1]), pct(sb(np.concatenate([X, Xg1]))), seed)
        Xg2 = sampler_active(X, M - M // 2, rng, teachers=teachers, pct_map=pct, proxy_student=proxy)
        Xg = np.concatenate([Xg1, Xg2])
    Xall = np.concatenate([X, Xg]) if len(Xg) else X
    st = train_student(Xall, pct(sb(Xall)), seed)
    p = st.predict(Xan); s = p if p.ndim == 1 else p[:, 0]
    return spearmanr(s, t).statistic


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--outdir", default="results_edge"); args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True)
    order = ["knn", "kde", "ocsvm", "ae", "ae_kde_max"]
    rows = []
    for seed in range(args.seeds):
        X = gen_normals(N_TRAIN, seed); Xan, _ = gen_offmanifold(X, N_ANOM, seed + 1)
        T = build_teachers(X, seed); kde = KernelDensity(bandwidth=0.3).fit(X)
        for tn in order:
            r = {"seed": seed, "teacher": tn}
            for w in ["S0", "S7", "S8", "S9"]:
                r[w] = fidelity(T[tn], X, Xan, kde, seed, w)
            rows.append(r)
            print(f"[edge seed={seed}] {tn:<11} S7={r['S7']:.3f} S8={r['S8']:.3f} S9_2round={r['S9']:.3f}", flush=True)
    with (outdir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "teacher", "S0", "S7", "S8", "S9"]); w.writeheader(); w.writerows(rows)
    import collections
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for k in ["S0", "S7", "S8", "S9"]: agg[r["teacher"]][k].append(r[k])
    print("\n=== fidelity (mean over seeds) ===")
    print(f"  {'teacher':<11} {'S0':>7} {'S7comb':>8} {'S8active':>9} {'S9_2round':>10}")
    for tn in order:
        a = agg[tn]
        print(f"  {tn:<11} {np.mean(a['S0']):>7.3f} {np.mean(a['S7']):>8.3f} "
              f"{np.mean(a['S8']):>9.3f} {np.mean(a['S9']):>10.3f}")


if __name__ == "__main__":
    main()
