"""Small-scale experiment for uncertainty-guided distillation of blackbox
anomaly-detection pipelines.

Runs 4 conditions x N seeds on two-moons data with 3 unsupervised teachers
(KDE, IsolationForest, kNN-distance) and a tiny MLP student. The single
variable across conditions is the synthetic-query sampler:

    S0: none            (normals only)
    S1: Gaussian jitter (near-manifold noise)
    S2: uniform         (space-filling)
    S3: Langevin        (uncertainty-guided, ours)

Outputs:
    results.csv        one row per (seed, condition, metric)
    invariants.txt     sanity-check log
    figures/main.png   contour plots + S3 query cloud

Usage:
    python experiment.py --seeds 10
    python experiment.py --seeds 1 --smoketest
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import make_moons
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.neural_network import MLPRegressor

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

MOON_NOISE = 0.15
N_TRAIN = 2000
N_VAL = 500
N_ANOM = 500


def gen_normals(n: int, seed: int) -> np.ndarray:
    X, _ = make_moons(n_samples=n, noise=MOON_NOISE, random_state=seed)
    return X


def _min_dist(pts: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Min Euclidean distance from each pts to any ref point."""
    nn = NearestNeighbors(n_neighbors=1).fit(ref)
    d, _ = nn.kneighbors(pts)
    return d[:, 0]


def gen_anomalies(X_train: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Three anomaly test sets, defined by distance to the normal manifold:

    near:   0.30 < min-dist < 0.70    (close-off-manifold, HARDEST)
    medium: 0.70 < min-dist < 1.20    (moderately off-manifold)
    far:    min-dist > 1.50           (far field, easy)

    All three are sampled from a wide bounding box, so they cover multiple
    directions of departure from the manifold (not just one annulus).
    """
    rng = np.random.default_rng(seed + 10_000)
    box_lo = np.array([-2.5, -2.0])
    box_hi = np.array([3.5, 2.0])
    nn = NearestNeighbors(n_neighbors=1).fit(X_train)

    # boundary sits INSIDE the noise band -> genuinely hard (teachers < 1.0).
    # close and medium are progressively further off-manifold.
    buckets = {"boundary": ([], 0.10, 0.22),
               "close":    ([], 0.22, 0.40),
               "medium":   ([], 0.40, 0.80),
               "far":      ([], 1.20, np.inf)}
    tries = 0
    while any(len(b[0]) < N_ANOM for b in buckets.values()):
        cand = rng.uniform(box_lo, box_hi, size=(5000, 2))
        d, _ = nn.kneighbors(cand)
        d = d[:, 0]
        for name, (bucket, lo, hi) in buckets.items():
            if len(bucket) < N_ANOM:
                mask = (d > lo) & (d <= hi)
                bucket.extend(cand[mask].tolist())
        tries += 1
        if tries > 40:
            raise RuntimeError("gen_anomalies could not fill buckets; check box / distance thresholds")

    return {name: np.array(b[:N_ANOM]) for name, (b, _, _) in buckets.items()}


# -----------------------------------------------------------------------------
# Teachers
# -----------------------------------------------------------------------------

def fit_teachers(X: np.ndarray, seed: int):
    """Return three callables that map (n, d) -> (n,) anomaly scores.

    Scores are oriented so LARGER = MORE ANOMALOUS.
    """
    kde = KernelDensity(kernel="gaussian", bandwidth=0.3).fit(X)
    iforest = IsolationForest(n_estimators=100, random_state=seed).fit(X)
    knn = NearestNeighbors(n_neighbors=10).fit(X)

    def s_kde(pts):
        return -kde.score_samples(pts)  # -log p; larger = anomalous

    def s_if(pts):
        # score_samples returns high for normal; negate.
        return -iforest.score_samples(pts)

    def s_knn(pts):
        d, _ = knn.kneighbors(pts)
        return d.mean(axis=1)

    return [s_kde, s_if, s_knn], kde


def score_batch(fns, X):
    """Stack teacher scores into (n, K)."""
    return np.stack([f(X) for f in fns], axis=-1)


def normalize_stats(S: np.ndarray):
    mu = S.mean(axis=0)
    sd = S.std(axis=0) + 1e-9
    return mu, sd


def znorm_clip(S, mu, sigma, clip=3.0):
    """z-normalize then clip to +-clip: bounded and scale-invariant across
    teachers, so no single detector dominates variance/fused mean at extremes.
    """
    Z = (S - mu) / sigma
    return np.clip(Z, -clip, clip)


def make_percentile_maps(S_train):
    """Return one callable per teacher that maps raw score -> percentile in
    [0, 1] via linear interpolation of the training normals' sorted scores.
    Percentile is a scale-free, bounded per-teacher signal: variance across
    teachers then measures genuine inter-detector disagreement, not scale."""
    K = S_train.shape[1]
    sorted_scores = [np.sort(S_train[:, k]) for k in range(K)]
    def pct(S):
        # S: (n, K) -> (n, K) in [0, 1]
        out = np.zeros_like(S)
        for k in range(S.shape[1]):
            out[:, k] = np.searchsorted(sorted_scores[k], S[:, k]) / len(sorted_scores[k])
        return out
    return pct


# -----------------------------------------------------------------------------
# Samplers
# -----------------------------------------------------------------------------

def sampler_none(X_n, M, rng, **kw):
    return np.zeros((0, X_n.shape[1]))


def sampler_gaussian(X_n, M, rng, noise_scale=0.3, **kw):
    idx = rng.integers(0, len(X_n), size=M)
    return X_n[idx] + rng.normal(scale=noise_scale, size=(M, X_n.shape[1]))


def sampler_uniform(X_n, M, rng, box=((-2.0, 3.0), (-1.5, 1.5)), **kw):
    lo = np.array([box[0][0], box[1][0]])
    hi = np.array([box[0][1], box[1][1]])
    return rng.uniform(lo, hi, size=(M, X_n.shape[1]))


def sampler_langevin(
    X_n, M, rng, *, teachers, mu, sigma, kde, pct_map,
    T=30, eta=0.04, beta=0.5, tau=0.3, h=0.02, max_dist=0.7, alpha_mean=10.0,
):
    """Batched Langevin walk climbing inter-teacher variance,
    with a KDE log-density prior to keep queries near-manifold.

    A hard projection at every step keeps the walk within `max_dist` of the
    normal training set: without it, the log-density gradient vanishes far
    from support and the chain runs away to the clip box.
    """
    if M == 0:
        return np.zeros((0, X_n.shape[1]))
    idx = rng.integers(0, len(X_n), size=M)
    x = X_n[idx].copy().astype(float)
    d = x.shape[1]
    nn = NearestNeighbors(n_neighbors=1).fit(X_n)

    def U(pts):
        # Percentile-normalized teacher scores in [0, 1] per teacher.
        # Uncertainty potential = inter-teacher disagreement PLUS a term that
        # pulls the walk toward higher fused-score regions (the decision
        # boundary), so the queries do not collapse onto the normal manifold
        # where every teacher agrees "this is normal but slightly different".
        Z = pct_map(score_batch(teachers, pts))
        return Z.var(axis=1) + alpha_mean * Z.mean(axis=1)

    def log_p(pts):
        return kde.score_samples(pts)      # (n,)

    def project(pts):
        """Radial projection back to a max_dist ball around nearest normal."""
        dist, idx_nn = nn.kneighbors(pts)
        dist = dist[:, 0]
        anchor = X_n[idx_nn[:, 0]]
        too_far = dist > max_dist
        if not too_far.any():
            return pts
        v = pts[too_far] - anchor[too_far]
        v = v * (max_dist / dist[too_far, None])
        pts[too_far] = anchor[too_far] + v
        return pts

    e_basis = np.eye(d) * h

    for _ in range(T):
        gU = np.zeros_like(x)
        gp = np.zeros_like(x)
        for i in range(d):
            xp = x + e_basis[i]
            xm = x - e_basis[i]
            gU[:, i] = (U(xp) - U(xm)) / (2 * h)
            gp[:, i] = (log_p(xp) - log_p(xm)) / (2 * h)
        noise = rng.normal(size=x.shape)
        x = x + eta * (gU + beta * gp) + np.sqrt(2 * eta * tau) * noise
        x = project(x)
    return x


def sampler_mixed(X_n, M, rng, *, teachers, mu, sigma, kde, pct_map, **kw):
    """S4: half Langevin (uncertainty-guided, ~boundary region), half uniform
    (far-field coverage). Same total budget M as any single sampler."""
    m_l = M // 2
    m_u = M - m_l
    Xg_l = sampler_langevin(X_n, m_l, rng,
                            teachers=teachers, mu=mu, sigma=sigma,
                            kde=kde, pct_map=pct_map)
    Xg_u = sampler_uniform(X_n, m_u, rng)
    return np.concatenate([Xg_l, Xg_u], axis=0)


SAMPLERS = {
    "S0_none": sampler_none,
    "S1_gaussian": sampler_gaussian,
    "S2_uniform": sampler_uniform,
    "S3_langevin": sampler_langevin,
    "S4_mixed": sampler_mixed,
}


# -----------------------------------------------------------------------------
# Student + evaluation
# -----------------------------------------------------------------------------

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_student(X: np.ndarray, Y: np.ndarray, seed: int):
    """Tiny MLP student: 8 tanh units, ~40 params. Deliberately undercapacity
    so that WHERE the training queries live matters, not just the loss."""
    pipe = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(8,),
            activation="tanh",
            solver="adam",
            learning_rate_init=3e-3,
            max_iter=1500,
            batch_size=128,
            random_state=seed,
            tol=1e-5,
            n_iter_no_change=40,
        ),
    )
    pipe.fit(X, Y)
    return pipe


def auroc(y_normal_scores, y_anom_scores):
    y_true = np.concatenate([np.zeros(len(y_normal_scores)),
                             np.ones(len(y_anom_scores))])
    y_score = np.concatenate([y_normal_scores, y_anom_scores])
    return roc_auc_score(y_true, y_score)


def eval_condition(student, teachers, mu, sigma, X_val, anoms, pct_map):
    """AUROC per anomaly set, score-fidelity RMSE, Spearman rank calibration."""
    out = {}

    def fused(pts):
        Sn = pct_map(score_batch(teachers, pts))
        return Sn.mean(axis=1)

    def fused_student(pts):
        pred = student.predict(pts)  # (n, K) (already normalized target)
        return pred.mean(axis=1)

    val_teacher = fused(X_val)
    val_student = fused_student(X_val)

    for name, A in anoms.items():
        a_teacher = fused(A)
        a_student = fused_student(A)
        out[f"auroc_teacher_{name}"] = auroc(val_teacher, a_teacher)
        out[f"auroc_student_{name}"] = auroc(val_student, a_student)
        out[f"rmse_score_{name}"] = float(np.sqrt(np.mean(
            (a_student - a_teacher) ** 2)))
        # Spearman rank calibration between student and teacher on anom pts.
        rho, _ = spearmanr(a_student, a_teacher)
        out[f"spearman_{name}"] = float(rho)

    # Normal-side fit RMSE (invariant I3).
    out["rmse_score_normals"] = float(np.sqrt(np.mean(
        (val_student - val_teacher) ** 2)))
    return out


# -----------------------------------------------------------------------------
# Run one (seed, condition)
# -----------------------------------------------------------------------------

def run_one(seed: int, cond: str, X_train, X_val, anoms, teachers, mu, sigma, kde,
            pct_map, M=2000):
    rng = np.random.default_rng(seed * 97 + hash(cond) % 10_000)
    kw = dict(teachers=teachers, mu=mu, sigma=sigma, kde=kde, pct_map=pct_map)
    sampler = SAMPLERS[cond]
    Xg = sampler(X_train, M, rng, **kw) if cond != "S0_none" else sampler_none(X_train, M, rng)

    X_all = np.concatenate([X_train, Xg]) if len(Xg) else X_train
    # Percentile targets are bounded in [0, 1] per teacher: prevents extreme
    # far-off-manifold values from dominating the tiny student's loss.
    Y_all = pct_map(score_batch(teachers, X_all))

    student = train_student(X_all, Y_all, seed=seed)
    metrics = eval_condition(student, teachers, mu, sigma, X_val, anoms, pct_map)
    return student, Xg, metrics


# -----------------------------------------------------------------------------
# Sanity invariants
# -----------------------------------------------------------------------------

def run_invariants(X_train, X_val, anoms, teachers, mu, sigma, kde, pct_map, log_path: Path):
    lines = ["=== Sanity invariants ==="]

    rng = np.random.default_rng(0)

    # I1: S3 with M=0 collapses to S0.
    _, _, s0 = run_one(0, "S0_none", X_train, X_val, anoms, teachers, mu, sigma, kde, pct_map)
    Xg0 = sampler_langevin(X_train, 0, rng, teachers=teachers, mu=mu, sigma=sigma, kde=kde, pct_map=pct_map)
    assert len(Xg0) == 0
    lines.append("I1 (M=0 => S3==S0): OK, empty query set path taken")

    # I2: S3 with x_g = i.i.d. from X_n (no walk) is same order as S0.
    idx = rng.integers(0, len(X_train), size=2000)
    Xg_iid = X_train[idx]
    Y_all = pct_map(score_batch(teachers, np.concatenate([X_train, Xg_iid])))
    student_iid = train_student(np.concatenate([X_train, Xg_iid]), Y_all, seed=0)
    m_iid = eval_condition(student_iid, teachers, mu, sigma, X_val, anoms, pct_map)
    gap = abs(m_iid["auroc_student_close"] - s0["auroc_student_close"])
    lines.append(f"I2 (iid-resample => S0): |auroc_close gap| = {gap:.4f} (want < 0.05)")

    # I3: normal-side fit RMSE for S0.
    lines.append(f"I3 (normal-side RMSE): {s0['rmse_score_normals']:.4f} (want < 0.20)")

    # I4: each teacher's own AUROC on every anomaly set (percentile-normalized).
    def _pct_scalar(k, pts):
        S = score_batch(teachers, pts)   # (n, K)
        return pct_map(S)[:, k]
    for name in anoms:
        for k, tname in enumerate(["kde", "iforest", "knn"]):
            au = auroc(_pct_scalar(k, X_val), _pct_scalar(k, anoms[name]))
            lines.append(f"I4 teacher={tname:<8} anom={name:<7} AUROC={au:.3f}")

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--M", type=int, default=2000)
    parser.add_argument("--smoketest", action="store_true")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)

    # Fixed data across all seeds; only teacher/student seeds vary.
    X_train = gen_normals(N_TRAIN, seed=0)
    X_val = gen_normals(N_VAL, seed=1)
    anoms = gen_anomalies(X_train, seed=0)

    # Fit teachers once (deterministic-ish; IsolationForest depends on seed).
    teachers, kde = fit_teachers(X_train, seed=0)
    S_train = score_batch(teachers, X_train)
    mu, sigma = normalize_stats(S_train)
    pct_map = make_percentile_maps(S_train)

    # -- Invariants --
    t0 = time.time()
    run_invariants(X_train, X_val, anoms, teachers, mu, sigma, kde, pct_map,
                   log_path=outdir / "invariants.txt")
    print(f"[invariants] {time.time() - t0:.1f}s")

    if args.smoketest:
        print("Smoke test only; exiting after invariants.")
        return

    # -- Full sweep --
    conditions = list(SAMPLERS.keys())
    rows = []
    students_by_cond = {}  # keep last-seed student per condition for the figure

    for seed in range(args.seeds):
        for cond in conditions:
            t0 = time.time()
            student, Xg, m = run_one(seed, cond, X_train, X_val, anoms,
                                     teachers, mu, sigma, kde, pct_map,
                                     M=args.M)
            wall = time.time() - t0
            for k, v in m.items():
                rows.append({"seed": seed, "cond": cond, "metric": k, "value": v})
            rows.append({"seed": seed, "cond": cond, "metric": "wall_s", "value": wall})
            students_by_cond[cond] = (student, Xg)
            print(f"seed={seed} cond={cond:<12} "
                  f"bdy={m['auroc_student_boundary']:.3f} "
                  f"close={m['auroc_student_close']:.3f} "
                  f"med={m['auroc_student_medium']:.3f} "
                  f"far={m['auroc_student_far']:.3f} "
                  f"rmseN={m['rmse_score_normals']:.3f} "
                  f"({wall:.1f}s)")

    # Save CSV.
    csv_path = outdir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "cond", "metric", "value"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    # Save teachers/students state for later plotting via a separate script
    # (kept simple: re-run this file after --skip-sweep is fine; we plot inline).
    plot_main(outdir, X_train, X_val, anoms, teachers, mu, sigma, pct_map, students_by_cond)


def plot_main(outdir, X_train, X_val, anoms, teachers, mu, sigma, pct_map, students_by_cond):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xx, yy = np.meshgrid(np.linspace(-2.5, 3.5, 200),
                         np.linspace(-2.0, 2.0, 200))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    def fused_teacher(pts):
        return pct_map(score_batch(teachers, pts)).mean(axis=1)

    def fused_student(student, pts):
        return student.predict(pts).mean(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    def draw(ax, Z, title, xg=None):
        cs = ax.contourf(xx, yy, Z.reshape(xx.shape), levels=20, cmap="viridis")
        ax.scatter(X_train[:, 0], X_train[:, 1], s=2, c="white",
                   alpha=0.35, label="normals")
        if xg is not None and len(xg):
            ax.scatter(xg[:, 0], xg[:, 1], s=6, c="red", alpha=0.6,
                       label="S3 queries")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-2.5, 3.5)
        ax.set_ylim(-2.0, 2.0)
        fig.colorbar(cs, ax=ax, fraction=0.04)

    draw(axes[0, 0], fused_teacher(grid), "Fused teacher score")
    if "S0_none" in students_by_cond:
        s0, _ = students_by_cond["S0_none"]
        draw(axes[0, 1], fused_student(s0, grid),
             "Student S0 (normals only)")
    if "S3_langevin" in students_by_cond:
        s3, xg3 = students_by_cond["S3_langevin"]
        draw(axes[1, 0], fused_student(s3, grid),
             "Student S3 (uncertainty-guided)")
        draw(axes[1, 1], fused_teacher(grid),
             "Teacher + S3 queries (red)", xg=xg3)

    fig_path = outdir / "figures" / "main.png"
    fig.savefig(fig_path, dpi=140)
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
