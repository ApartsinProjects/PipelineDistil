"""De-risk #2: does query budget + student capacity close the student-to-floor gap?

Resolves P3. Two cells: an EASY one (38_thyroid/knn, student already near the floor)
and a HARD one (6_cardio/kde, student ~0.59 vs floor ~0.18). For each: bootstrap
refresh floor, then student alarm-set regret at query budgets {1k, 50k} x student
capacities {(32,16),(256,128,64)}, 3 seeds, k in {1%,5%}.
Outcome: if the HARD cell drops UNDER the floor at 50k + big student -> P3 = "budget
closes the gap" (paper ships a stopping rule). If it stays above -> P3 = "certified
gap" (paper ships an impossibility-at-budget result). Either is publishable.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import csv, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.neural_network import MLPRegressor
from experiment_realbench_v2 import load, shell_scale
from experiment_noisefloor import teacher_fn, regret

CELLS = [("38_thyroid", "knn", "easy"), ("6_cardio", "kde", "hard")]
BUDGETS = [1000, 50000]
CAPS = {"small": (32, 16), "big": (256, 128, 64)}
OUT = Path("results_budget_capacity"); OUT.mkdir(exist_ok=True)


def floor_boot(tname, Xtr, Xe, se, frac, seed, n=4):
    rng = np.random.default_rng(seed * 17 + 1); ntr = len(Xtr); vals = []
    for r in range(n):
        idx = rng.integers(0, ntr, size=ntr)   # bootstrap
        vals.append(regret(se, teacher_fn(tname, Xtr[idx], seed + 50 + r)(Xe), frac))
    return float(np.mean(vals))


def run():
    rows = []
    for name, tname, tag in CELLS:
        for seed in range(3):
            Xtr, Xval, Xan = load(name, seed); Xe = np.concatenate([Xval, Xan])
            base = teacher_fn(tname, Xtr, seed); se = base(Xe)
            scale = shell_scale(Xtr, seed); d = Xtr.shape[1]
            rng = np.random.default_rng(seed * 7 + 2)
            for M in BUDGETS:
                lo, hi = Xtr.min(0) - scale, Xtr.max(0) + scale
                box = rng.uniform(lo, hi, size=(M, d)); Xq = np.concatenate([Xtr, box])
                sq = base(Xq); med = np.median(sq); mad = np.median(np.abs(sq - med)) + 1e-9
                y = (sq - med) / (1.4826 * mad)
                for cap, hid in CAPS.items():
                    st = MLPRegressor(hidden_layer_sizes=hid, activation="tanh", solver="adam",
                                      learning_rate_init=5e-3, max_iter=1000, random_state=seed,
                                      tol=1e-6, n_iter_no_change=30, batch_size=min(256, len(Xq))).fit(Xq, y)
                    pe = st.predict(Xe)
                    row = {"cell": tag, "dataset": name, "teacher": tname, "seed": seed,
                           "budget": M, "cap": cap}
                    for frac, fk in ((0.01, "k1"), (0.05, "k5")):
                        row[f"student_{fk}"] = regret(se, pe, frac)
                        row[f"floor_{fk}"] = floor_boot(tname, Xtr, Xe, se, frac, seed)
                        row[f"under_{fk}"] = row[f"student_{fk}"] <= row[f"floor_{fk}"] + 1e-9
                    rows.append(row)
                    print(f"[bc] {tag:<4}{name:<12}{tname:<4}s{seed} M={M:<6}{cap:<6} "
                          f"k5 student={row['student_k5']:.3f} floor={row['floor_k5']:.3f} "
                          f"under={row['under_k5']}", flush=True)
    with (OUT / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\n=== HARD cell (6_cardio/kde) student_k5 vs floor by budget x cap ===", flush=True)
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        if r["cell"] == "hard":
            agg[(r["budget"], r["cap"])].append((r["student_k5"], r["floor_k5"]))
    for (M, cap), v in sorted(agg.items()):
        s = np.mean([a for a, _ in v]); fl = np.mean([b for _, b in v])
        print(f"  M={M:<6}{cap:<6} student={s:.3f} floor={fl:.3f} gap={s-fl:+.3f} "
              f"{'UNDER' if s<=fl else 'above'}", flush=True)
    print("P3 verdict: if big+50k goes UNDER on the hard cell -> stopping-rule; else -> certified-gap", flush=True)


if __name__ == "__main__":
    run()
