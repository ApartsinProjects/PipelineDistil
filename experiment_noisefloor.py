"""De-risking smoke for C1: the retraining/refresh noise floor.

Question that decides the whole direction: is the teacher's own REFRESH CHURN
(top-k alarm-set regret between the teacher and a retrained copy on a resampled
normal set) a meaningful bar, AND does the trivial uniform-box + raw-regression
student sit UNDER that floor across teacher families (not only the stochastic ones)?

For each (dataset, teacher):
  floor_boot90/99 : alarm-set regret between teacher-on-full-normals and
                    teacher-on-{90%,99%} subsample, averaged over several resamples.
  floor_seed      : (stochastic teachers iforest/ae only) regret between two seeds.
  student_regret  : uniform-box + raw-score MLP student vs teacher.
  indep_copy      : regret of a SECOND independent resample copy (invariant: ~= floor).
Metric: alarm-set regret at k in {1%, 5%} on a fixed eval set (val normals + anomalies).

Pre-registered invariants:
  I1 an independent retrained copy sits AT the floor (|indep - floor| small), not below.
  I2 iForest seed floor shrinks as trees grow (100 -> 1000).
Positive lead (C1 lives): student_regret <= floor_boot90 in a MAJORITY of cells
ACROSS teacher families (incl. deterministic kNN/KDE/OCSVM), and the floor is not
trivially ~1.0 (tightness). Kill: student under floor only for iforest/ae.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, csv, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from experiment_realbench_v2 import load, shell_scale


def teacher_fn(tname, X, seed):
    if tname == "knn":
        nn = NearestNeighbors(n_neighbors=10).fit(X); return lambda P: nn.kneighbors(P)[0].mean(1)
    if tname == "kde":
        kde = KernelDensity(bandwidth=0.5).fit(X); return lambda P: -kde.score_samples(P)
    if tname == "ocsvm":
        m = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1).fit(X); return lambda P: -m.decision_function(P)
    if tname == "iforest":
        m = IsolationForest(n_estimators=150, random_state=seed).fit(X); return lambda P: -m.score_samples(P)
    if tname == "ae":
        d = X.shape[1]
        ae = MLPRegressor(hidden_layer_sizes=(64, max(2, d // 2), 64), activation="tanh", solver="adam",
                          learning_rate_init=3e-3, max_iter=400, batch_size=64, random_state=seed,
                          tol=1e-5, n_iter_no_change=20).fit(X, X)
        return lambda P: ((P - ae.predict(P)) ** 2).mean(1)
    raise ValueError(tname)


def iforest_fn(X, seed, trees):
    m = IsolationForest(n_estimators=trees, random_state=seed).fit(X); return lambda P: -m.score_samples(P)


def regret(a, b, frac):
    k = max(1, int(round(frac * len(a))))
    A = set(np.argsort(a)[-k:]); B = set(np.argsort(b)[-k:])
    return 1.0 - len(A & B) / len(A)


def student(X, y, seed):
    return MLPRegressor(hidden_layer_sizes=(32, 16), activation="tanh", solver="adam",
                        learning_rate_init=5e-3, max_iter=800, random_state=seed,
                        tol=1e-6, n_iter_no_change=30).fit(X, y)


STOCH = {"iforest", "ae"}


def run_cell(name, tname, seed, n_resample=4):
    rng = np.random.default_rng(seed * 101 + 3)
    Xtr, Xval, Xan = load(name, seed)
    Xe = np.concatenate([Xval, Xan]); ntr = len(Xtr)
    base = teacher_fn(tname, Xtr, seed); se = base(Xe)
    out = {"dataset": name, "teacher": tname, "seed": seed, "n_train": ntr, "stochastic": tname in STOCH}
    for frac in (0.01, 0.05):
        tag = f"k{int(frac*100)}"
        # refresh churn via subsample resamples
        for sub, key in ((0.90, "boot90"), (0.99, "boot99")):
            vals = []
            for r in range(n_resample):
                idx = rng.choice(ntr, size=max(10, int(sub * ntr)), replace=False)
                t2 = teacher_fn(tname, Xtr[idx], seed + 100 + r)
                vals.append(regret(se, t2(Xe), frac))
            out[f"floor_{key}_{tag}"] = float(np.mean(vals))
        # independent copy invariant (another 90% resample)
        idx = rng.choice(ntr, size=int(0.90 * ntr), replace=False)
        out[f"indep_{tag}"] = regret(se, teacher_fn(tname, Xtr[idx], seed + 999)(Xe), frac)
        # seed churn for stochastic teachers
        if tname in STOCH:
            out[f"floor_seed_{tag}"] = regret(se, teacher_fn(tname, Xtr, seed + 1)(Xe), frac)
        # student (uniform-box + raw-score regression)
        scale = shell_scale(Xtr, seed); d = Xtr.shape[1]
        lo, hi = Xtr.min(0) - scale, Xtr.max(0) + scale
        box = rng.uniform(lo, hi, size=(1000, d)); Xq = np.concatenate([Xtr, box])
        sq = base(Xq); med = np.median(sq); mad = np.median(np.abs(sq - med)) + 1e-9
        st = student(Xq, (sq - med) / (1.4826 * mad), seed)
        out[f"student_{tag}"] = regret(se, st.predict(Xe), frac)
    # I2: iforest seed floor should shrink with trees
    if tname == "iforest":
        s100 = iforest_fn(Xtr, seed, 100)(Xe); s100b = iforest_fn(Xtr, seed + 1, 100)(Xe)
        s1000 = iforest_fn(Xtr, seed, 1000)(Xe); s1000b = iforest_fn(Xtr, seed + 1, 1000)(Xe)
        out["ifo_seedfloor_100_k5"] = regret(s100, s100b, 0.05)
        out["ifo_seedfloor_1000_k5"] = regret(s1000, s1000b, 0.05)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="6_cardio,40_vowels,38_thyroid,30_satellite,20_letter,2_annthyroid,44_Wilt,7_Cardiotocography")
    ap.add_argument("--teachers", default="knn,kde,ocsvm,iforest,ae")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--outdir", default="results_noisefloor")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(exist_ok=True); rows = []
    for name in a.datasets.split(","):
        for tname in a.teachers.split(","):
            for s in range(a.seeds):
                try:
                    r = run_cell(name, tname, s); rows.append(r)
                    print(f"[nf] {name:<16}{tname:<8}s{s} k5: floor90={r['floor_boot90_k5']:.3f} "
                          f"indep={r['indep_k5']:.3f} student={r['student_k5']:.3f} "
                          f"under_floor={r['student_k5']<=r['floor_boot90_k5']}", flush=True)
                except Exception as e:
                    print(f"[nf] {name}/{tname}/s{s} FAIL {e}", flush=True)
    keys = sorted({k for r in rows for k in r})
    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out/'results.csv'} ({len(rows)} rows)", flush=True)
    # headline summary
    import collections
    for frac in ("k1", "k5"):
        by = collections.defaultdict(lambda: [0, 0])
        floors = []; studs = []
        for r in rows:
            fl = r.get(f"floor_boot90_{frac}"); stu = r.get(f"student_{frac}")
            if fl is None or stu is None: continue
            floors.append(fl); studs = studs if False else studs  # noqa
            under = stu <= fl + 1e-9
            by[r["teacher"]][0] += int(under); by[r["teacher"]][1] += 1
        print(f"\n=== {frac}: student <= refresh-floor(90%) by teacher family ===", flush=True)
        for t in ("knn", "kde", "ocsvm", "iforest", "ae"):
            u, n = by[t]
            if n: print(f"  {t:<8} {u}/{n} cells under floor", flush=True)
        print(f"  mean floor_boot90_{frac} = {np.mean([r[f'floor_boot90_{frac}'] for r in rows]):.3f} "
              f"(tightness: closer to 0 = tighter/harder bar)", flush=True)
        print(f"  mean student_{frac}      = {np.mean([r[f'student_{frac}'] for r in rows]):.3f}", flush=True)
    # invariants
    idep = [abs(r[f"indep_k5"] - r[f"floor_boot90_k5"]) for r in rows]
    print(f"\nI1 |indep - floor| mean={np.mean(idep):.3f} (should be small)", flush=True)
    ifo = [r for r in rows if r["teacher"] == "iforest" and "ifo_seedfloor_100_k5" in r]
    if ifo:
        print(f"I2 iforest seed floor: 100 trees={np.mean([r['ifo_seedfloor_100_k5'] for r in ifo]):.3f} "
              f"-> 1000 trees={np.mean([r['ifo_seedfloor_1000_k5'] for r in ifo]):.3f} (should shrink)", flush=True)


if __name__ == "__main__":
    main()
