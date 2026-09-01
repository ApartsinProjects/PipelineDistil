"""Cycle-2 pivot smoke: Order-Canonical Distillation.

Claim: distilling the teacher's rank (order-canonical target) instead of its raw
score gives lower deployed ALARM-SET regret and is INVARIANT to strictly-monotone
rewrites of the teacher score; raw-score MSE distillation is not.

No new teacher queries: same query points, same student architecture/seed. For each
dataset x teacher, standardize the teacher score to z, form 4 strictly-increasing
parameterizations phi in {z, asinh(z), z^3, sigmoid(z)} (identical ordering), and
train two students on space-filling (box) + normals queries:
  raw : MSE to standardized phi(z)   (incumbent; depends on phi)
  rank: MSE to normalized rank u=(rank-0.5)/N  (pivot; invariant to phi)
Metric: alarm-set regret R = 1 - |A_T ∩ A_S| / |A_T| at a 5% alarm budget on the
held-out test set (val normals + anomalies), A_* = top-5% by teacher/student score.
Report worst-case R over the 4 phi, and DeltaR = R_worst(raw) - R_worst(rank).
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
from sklearn.neural_network import MLPRegressor
from experiment_realbench_v2 import load, TEACHERS, shell_scale

M_BOX = 1000
ALARM = 0.05


def _rng(*p):
    return np.random.default_rng(zlib.crc32("|".join(map(str, p)).encode()) & 0x7FFFFFFF)


PHIS = {
    "identity": lambda z: z,
    "asinh":    lambda z: np.arcsinh(z),
    "cube":     lambda z: z ** 3,
    "sigmoid":  lambda z: 1.0 / (1.0 + np.exp(-z)),
}


def zstd(v):
    med = np.median(v); mad = np.median(np.abs(v - med)) + 1e-9
    return (v - med) / (1.4826 * mad)


def student(X, y, seed):
    return MLPRegressor(hidden_layer_sizes=(32, 16), activation="tanh", solver="adam",
                        learning_rate_init=5e-3, max_iter=800, random_state=seed,
                        tol=1e-6, n_iter_no_change=30).fit(X, y)


def alarm_regret(teacher_score, student_score, frac=ALARM):
    k = max(1, int(round(frac * len(teacher_score))))
    at = set(np.argsort(teacher_score)[-k:]); as_ = set(np.argsort(student_score)[-k:])
    return 1.0 - len(at & as_) / len(at)


def run_cell(name, tname, seed):
    Xtr, Xval, Xan = load(name, seed)
    teacher = TEACHERS[tname](Xtr, seed)
    scale = shell_scale(Xtr, seed)
    rng = _rng("q", name, tname, seed)
    d = Xtr.shape[1]
    lo, hi = Xtr.min(0) - scale, Xtr.max(0) + scale
    box = rng.uniform(lo, hi, size=(M_BOX, d))
    Xq = np.concatenate([Xtr, box])                       # query points (space-filling + normals)
    sq = teacher(Xq)                                       # raw teacher score on queries
    zq = zstd(sq)
    # order-canonical target on queries (invariant to phi): normalized rank
    order = np.argsort(np.argsort(sq)); u = (order + 0.5) / len(sq)
    # held-out eval
    Xe = np.concatenate([Xval, Xan]); te_raw = teacher(Xe)
    rank_st = student(Xq, u, seed)
    r_rank = {ph: alarm_regret(te_raw, rank_st.predict(Xe)) for ph in PHIS}   # invariant
    r_raw = {}
    for ph, f in PHIS.items():
        yq = zstd(f(zq))                                   # standardized transformed target
        raw_st = student(Xq, yq, seed)
        r_raw[ph] = alarm_regret(te_raw, raw_st.predict(Xe))
    rraw_worst = max(r_raw.values()); rrank_worst = max(r_rank.values())
    return {"dataset": name, "teacher": tname, "seed": seed,
            "raw_identity": r_raw["identity"], "rank_identity": r_rank["identity"],
            "raw_sigmoid": r_raw["sigmoid"], "raw_cube": r_raw["cube"], "raw_asinh": r_raw["asinh"],
            "raw_worst": rraw_worst, "rank_worst": rrank_worst,
            "rank_invariance_spread": max(r_rank.values()) - min(r_rank.values()),
            "deltaR": rraw_worst - rrank_worst}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="6_cardio,40_vowels,38_thyroid")
    ap.add_argument("--teachers", default="knn,ae")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--outdir", default="results_ordercanon")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(exist_ok=True)
    rows = []
    for name in a.datasets.split(","):
        for tname in a.teachers.split(","):
            for s in range(a.seeds):
                try:
                    r = run_cell(name, tname, s); rows.append(r)
                    print(f"[oc] {name:<14}{tname:<5}s{s} raw_worst={r['raw_worst']:.3f} "
                          f"rank_worst={r['rank_worst']:.3f} dR={r['deltaR']:+.3f} "
                          f"raw_id={r['raw_identity']:.3f} rank_id={r['rank_identity']:.3f} "
                          f"rank_inv_spread={r['rank_invariance_spread']:.3f} "
                          f"raw_sig={r['raw_sigmoid']:.3f}", flush=True)
                except Exception as e:
                    print(f"[oc] {name}/{tname}/s{s} FAIL {e}", flush=True)
    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    dR = np.array([r["deltaR"] for r in rows])
    print(f"\n=== median DeltaR={np.median(dR):+.3f}  rank wins {int((dR>0).sum())}/{len(dR)} "
          f"| mean raw_worst={np.mean([r['raw_worst'] for r in rows]):.3f} "
          f"rank_worst={np.mean([r['rank_worst'] for r in rows]):.3f} ===", flush=True)
    print(f"invariance check: max rank spread over phi = "
          f"{max(r['rank_invariance_spread'] for r in rows):.4f} (should be ~0)", flush=True)


if __name__ == "__main__":
    main()
