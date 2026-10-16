"""Phase 1 (E1+E2): the full refresh-churn FLOOR MAP + the acceptance rule applied
to a weak student and to positive-control surrogates. Incremental + resumable.

For each (dataset, teacher, seed) with eval >= 600 (headline scope):
  floors: bootstrap (canonical refresh), 90% and 99% subsample; seed floor for
          stochastic teachers (iforest, ae). All as top-k alarm-set regret on a
          fixed eval set at k in {1%, 5%}.
  surrogates judged by the rule (regret vs teacher):
    - weak      : box(1k) + raw-score MLP(32-16)   [expected to FAIL]
    - refresh   : teacher retrained on a 90% subsample [positive anchor, == the floor]
    - iforest1k : 1000-tree iForest standing in for the 150-tree teacher (iforest only)
  invariants: I1 independent-copy ~= floor; I2 iForest seed floor shrinks with trees.

One CSV row per (dataset,teacher,seed) written the moment it is ready; a re-launch
skips rows already in the CSV (resume). OMP/MKL threads pinned to 1 under a pool.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, csv, warnings, glob
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from experiment_realbench_v2 import load, shell_scale
from experiment_noisefloor import teacher_fn, regret, iforest_fn

DATA_DIR = Path(__file__).parent / "data" / "adbench"
STOCH = {"iforest", "ae"}


def eligible_datasets(min_eval=600, max_d=256):
    keep = []
    for p in sorted(DATA_DIR.glob("*.npz")):
        try:
            a = np.load(p); y = a["y"].astype(int)
            n_norm = int((y == 0).sum()); n_an = int((y == 1).sum())
            if n_an < 20 or n_norm < 400 or a["X"].shape[1] > max_d:
                continue  # drop ultra-high-d (e.g. InternetAds d=1555): out of scope + OOM
            # eval = val normals (<=1500, >=400) + anomalies; approximate
            val = min(1500, max(400, n_norm // 4))
            if val + n_an >= min_eval:
                keep.append(p.stem)
        except Exception:
            pass
    return keep


def student_weak(Xtr, base, scale, seed):
    d = Xtr.shape[1]; rng = np.random.default_rng(seed * 7 + 2)
    lo, hi = Xtr.min(0) - scale, Xtr.max(0) + scale
    box = rng.uniform(lo, hi, size=(1000, d)); Xq = np.concatenate([Xtr, box])
    sq = base(Xq); med = np.median(sq); mad = np.median(np.abs(sq - med)) + 1e-9
    return MLPRegressor(hidden_layer_sizes=(32, 16), activation="tanh", solver="adam",
                        learning_rate_init=5e-3, max_iter=800, random_state=seed,
                        tol=1e-6, n_iter_no_change=30).fit(Xq, (sq - med) / (1.4826 * mad))


def run_cell(name, tname, seed, n_rep=5):
    rng = np.random.default_rng(seed * 101 + 3)
    Xtr, Xval, Xan = load(name, seed); Xe = np.concatenate([Xval, Xan]); ntr = len(Xtr)
    base = teacher_fn(tname, Xtr, seed); se = base(Xe); scale = shell_scale(Xtr, seed)
    out = {"dataset": name, "teacher": tname, "seed": seed, "n_train": ntr,
           "n_eval": len(Xe), "stochastic": tname in STOCH}
    for frac, tag in ((0.01, "k1"), (0.05, "k5")):
        # floors
        for sub, key in ((None, "boot"), (0.90, "sub90"), (0.99, "sub99")):
            vals = []
            for r in range(n_rep):
                if sub is None:
                    idx = rng.integers(0, ntr, size=ntr)                 # bootstrap
                else:
                    idx = rng.choice(ntr, size=max(10, int(sub * ntr)), replace=False)
                vals.append(regret(se, teacher_fn(tname, Xtr[idx], seed + 100 + r)(Xe), frac))
            out[f"floor_{key}_{tag}"] = float(np.mean(vals))
        idx = rng.choice(ntr, size=int(0.90 * ntr), replace=False)
        out[f"indep_{tag}"] = regret(se, teacher_fn(tname, Xtr[idx], seed + 999)(Xe), frac)
        if tname in STOCH:
            out[f"floor_seed_{tag}"] = regret(se, teacher_fn(tname, Xtr, seed + 1)(Xe), frac)
        # surrogates
        st = student_weak(Xtr, base, scale, seed)
        out[f"student_weak_{tag}"] = regret(se, st.predict(Xe), frac)
        ridx = rng.choice(ntr, size=int(0.90 * ntr), replace=False)       # positive anchor = a refresh
        out[f"surr_refresh_{tag}"] = regret(se, teacher_fn(tname, Xtr[ridx], seed + 7)(Xe), frac)
        if tname == "iforest":
            out[f"surr_iforest1k_{tag}"] = regret(se, iforest_fn(Xtr, seed, 1000)(Xe), frac)
        # pass flags vs bootstrap floor
        out[f"weak_pass_{tag}"] = out[f"student_weak_{tag}"] <= out[f"floor_boot_{tag}"] + 1e-9
    if tname == "iforest":
        out["ifo_seedfloor_100_k5"] = regret(iforest_fn(Xtr, seed, 100)(Xe), iforest_fn(Xtr, seed + 1, 100)(Xe), 0.05)
        out["ifo_seedfloor_1000_k5"] = regret(iforest_fn(Xtr, seed, 1000)(Xe), iforest_fn(Xtr, seed + 1, 1000)(Xe), 0.05)
    return out


def _work(task):
    try:
        return run_cell(*task)
    except Exception as e:
        return {"__fail__": f"{task}: {e}"}


def done_keys(csv_path):
    seen = set()
    if csv_path.exists():
        import csv as _c
        with csv_path.open() as f:
            for row in _c.DictReader(f):
                seen.add((row["dataset"], row["teacher"], row["seed"]))
    return seen


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--teachers", default="knn,kde,ocsvm,iforest,ae")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--outdir", default="results_floormap")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(exist_ok=True); csv_path = out / "results.csv"
    datasets = eligible_datasets(); tlist = a.teachers.split(",")
    seen = done_keys(csv_path)
    tasks = [(n, t, s) for n in datasets for t in tlist for s in range(a.seeds)
             if (n, t, str(s)) not in seen]
    print(f"datasets={len(datasets)} teachers={tlist} seeds={a.seeds} "
          f"todo={len(tasks)} (skipped {len(seen)} done) workers={a.workers}", flush=True)
    print("  " + ", ".join(datasets), flush=True)
    # fixed header (superset of keys); unknown cells leave blanks
    fields = ["dataset", "teacher", "seed", "n_train", "n_eval", "stochastic"]
    for tag in ("k1", "k5"):
        fields += [f"floor_boot_{tag}", f"floor_sub90_{tag}", f"floor_sub99_{tag}",
                   f"indep_{tag}", f"floor_seed_{tag}", f"student_weak_{tag}",
                   f"surr_refresh_{tag}", f"surr_iforest1k_{tag}", f"weak_pass_{tag}"]
    fields += ["ifo_seedfloor_100_k5", "ifo_seedfloor_1000_k5"]
    new = not csv_path.exists()
    f = csv_path.open("a", newline=""); w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    if new:
        w.writeheader(); f.flush()
    done = fails = 0
    with mp.Pool(a.workers) as pool:
        for res in pool.imap_unordered(_work, tasks, chunksize=1):
            if "__fail__" in res:
                fails += 1; print(f"[FAIL] {res['__fail__']}", flush=True); continue
            w.writerow(res); f.flush(); done += 1
            if done % 15 == 0:
                print(f"  ... {done}/{len(tasks)} ({fails} fails)", flush=True)
    f.close()
    print(f"\nWrote {csv_path} (+{done} rows, {fails} fails)", flush=True)


if __name__ == "__main__":
    main()
