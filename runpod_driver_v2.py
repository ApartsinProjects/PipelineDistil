"""RunPod driver for the revision experiments (CPU work on a rented pod).

Runs two things and writes CSVs to results/:
  1) experiment_v2 spectrum: uniform-shell baseline, coordinate-FD vs
     K-random-direction variation, and a lambda mixture sweep, 10 seeds.
  2) capacity sweep: fidelity vs student width for every teacher, to separate
     query-limited from capacity-limited failure (esp. the max-fusion teacher).

Uses the [train] print convention so the runpod monitor shows progress. The
work is CPU-only; the pod's GPU is unused (the runner just needs CUDA present
to pass its bootstrap assert).
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv, sys, time, collections
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)

from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from experiment import make_percentile_maps, sampler_none
from experiment_spectrum import gen_normals, gen_offmanifold, build_teachers
import experiment_v2 as v2

ORDER = ["knn", "kde", "ocsvm", "ae", "ae_kde_max"]


def run_spectrum(seeds=10):
    print("[train] spectrum: uniform-shell + variation(FD/K) + lambda sweep", flush=True)
    rows = []
    for seed in range(seeds):
        X = gen_normals(v2.N_TRAIN, seed); Xan, _ = gen_offmanifold(X, v2.N_ANOM, seed + 1)
        T = build_teachers(X, seed)
        for tn in ORDER:
            tf = T[tn]; r = {"seed": seed, "teacher": tn}
            r["normals_only"] = v2.fidelity(tf, X, Xan, seed, "normals_only")
            r["uniform_shell"] = v2.fidelity(tf, X, Xan, seed, "uniform_shell")
            r["score"] = v2.fidelity(tf, X, Xan, seed, "score")
            r["variation_FD"] = v2.fidelity(tf, X, Xan, seed, "variation", K=None)
            r["combined_FD"] = v2.fidelity(tf, X, Xan, seed, "mix", lam=0.5, K=None)
            r["variation_K2"] = v2.fidelity(tf, X, Xan, seed, "variation", K=2)
            r["combined_K2"] = v2.fidelity(tf, X, Xan, seed, "mix", lam=0.5, K=2)
            for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
                r[f"lam{lam}"] = v2.fidelity(tf, X, Xan, seed, "mix", lam=lam, K=None)
            rows.append(r)
        print(f"[train]   spectrum seed {seed+1}/{seeds} done", flush=True)
    keys = list(rows[0].keys())
    with (RESULTS / "v2_spectrum.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"[train] wrote v2_spectrum.csv ({len(rows)} rows)", flush=True)


def student_width(width):
    return lambda X, Y, seed: make_pipeline(StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(width,), activation="tanh", solver="adam",
                     learning_rate_init=3e-3, max_iter=1500, batch_size=128,
                     random_state=seed, tol=1e-5, n_iter_no_change=40)).fit(X, Y)


def run_capacity(seeds=5, widths=(4, 8, 16, 32, 64, 128)):
    print("[train] capacity sweep: fidelity vs student width, combined sampler", flush=True)
    rows = []
    for seed in range(seeds):
        X = gen_normals(v2.N_TRAIN, seed); Xan, _ = gen_offmanifold(X, v2.N_ANOM, seed + 1)
        T = build_teachers(X, seed)
        for tn in ORDER:
            tf = T[tn]; teachers = [tf]; sb = lambda A: np.stack([f(A) for f in teachers], axis=-1)
            pct = make_percentile_maps(sb(X)); t = pct(sb(Xan))[:, 0]
            rng = np.random.default_rng(seed * 211 + hash(tn) % 9973)
            Xg = v2.make_queries("mix", X, teachers, pct, rng, lam=0.5, K=None)
            Xall = np.concatenate([X, Xg]); Yall = pct(sb(Xall))
            for wdt in widths:
                st = student_width(wdt)(Xall, Yall, seed)
                p = st.predict(Xan); s = p if p.ndim == 1 else p[:, 0]
                rows.append({"seed": seed, "teacher": tn, "width": wdt,
                             "fidelity": spearmanr(s, t).statistic})
        print(f"[train]   capacity seed {seed+1}/{seeds} done", flush=True)
    with (RESULTS / "v2_capacity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "teacher", "width", "fidelity"]); w.writeheader(); w.writerows(rows)
    print(f"[train] wrote v2_capacity.csv ({len(rows)} rows)", flush=True)


def main():
    print("[train] GPU: (CPU-only revision experiments; GPU unused)", flush=True)
    print("[train] Model loaded: PipeDistil-v2", flush=True)
    print("[train] Starting work...", flush=True)
    t0 = time.time()
    run_spectrum(seeds=10)
    run_capacity(seeds=5)
    print(f"[train] === DONE === ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
