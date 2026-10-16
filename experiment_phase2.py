"""Phase 2: P4 fix (data-vs-seed floor decomposition) + E3 (budget x capacity) +
E7 (ground-truth precision@k validation of the rule). One row per
(dataset, teacher, seed), incremental + resumable.

P4 fix: fraction floors (sub99, sub90, boot) refit with the SAME seed as the
teacher, so they measure DATA churn only; seed churn is measured separately by
refitting on the full data with other seeds (stochastic teachers only).
E3: student alarm-set regret over query budget {1k,10k,50k} x capacity
{small (32,16), big (256,128,64)}.
E7: precision@k (fraction of top-k that are true anomalies) of the teacher, of
each bootstrap refresh (the refresh distribution), and of each student; the rule
is decision-relevant if under-floor students keep p@k inside the refresh spread.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, csv, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.neural_network import MLPRegressor
from experiment_realbench_v2 import load, shell_scale
from experiment_noisefloor import teacher_fn, regret
from experiment_floormap import eligible_datasets

STOCH = {"iforest", "ae"}
FRACS = ((0.01, "k1"), (0.05, "k5"))
BUDGETS = (1000, 10000, 50000)
CAPS = {"small": (32, 16), "big": (256, 128, 64)}


def topk(s, frac):
    k = max(1, int(round(frac * len(s)))); return np.argsort(s)[-k:]


def prec_at_k(s, y, frac):
    return float(y[topk(s, frac)].mean())


def fit_student(Xq, y, hid, seed):
    return MLPRegressor(hidden_layer_sizes=hid, activation="tanh", solver="adam",
                        learning_rate_init=5e-3, max_iter=300, batch_size=256,
                        random_state=seed, tol=1e-6, n_iter_no_change=20).fit(Xq, y)


def run_cell(name, tname, seed, n_rep=5):
    rng = np.random.default_rng(seed * 101 + 3)
    Xtr, Xval, Xan = load(name, seed)
    Xe = np.concatenate([Xval, Xan]); ye = np.r_[np.zeros(len(Xval)), np.ones(len(Xan))]
    ntr = len(Xtr); base = teacher_fn(tname, Xtr, seed); se = base(Xe)
    out = {"dataset": name, "teacher": tname, "seed": seed, "n_eval": len(Xe)}
    # data-churn floors: SAME seed, resampled data
    boot_scores = []
    for key, sub in (("sub99", 0.99), ("sub90", 0.90), ("boot", None)):
        sc = []
        for r in range(n_rep):
            idx = (rng.integers(0, ntr, size=ntr) if sub is None
                   else rng.choice(ntr, size=max(10, int(sub * ntr)), replace=False))
            sc.append(teacher_fn(tname, Xtr[idx], seed)(Xe))
        if key == "boot":
            boot_scores = sc
        for frac, tag in FRACS:
            out[f"dfloor_{key}_{tag}"] = float(np.mean([regret(se, s, frac) for s in sc]))
    # seed churn: full data, other seeds (stochastic only)
    if tname in STOCH:
        sc = [teacher_fn(tname, Xtr, seed + 1 + r)(Xe) for r in range(n_rep)]
        for frac, tag in FRACS:
            out[f"sfloor_{tag}"] = float(np.mean([regret(se, s, frac) for s in sc]))
    # E7: precision@k of teacher and refresh distribution
    for frac, tag in FRACS:
        out[f"p_teacher_{tag}"] = prec_at_k(se, ye, frac)
        ps = [prec_at_k(s, ye, frac) for s in boot_scores]
        out[f"p_refresh_lo_{tag}"] = float(np.min(ps)); out[f"p_refresh_hi_{tag}"] = float(np.max(ps))
        out[f"p_refresh_mean_{tag}"] = float(np.mean(ps))
    # E3: budget x capacity students
    scale = shell_scale(Xtr, seed); d = Xtr.shape[1]
    lo, hi = Xtr.min(0) - scale, Xtr.max(0) + scale
    big_box = rng.uniform(lo, hi, size=(max(BUDGETS), d))
    for M in BUDGETS:
        Xq = np.concatenate([Xtr, big_box[:M]]); sq = base(Xq)
        med = np.median(sq); mad = np.median(np.abs(sq - med)) + 1e-9
        yq = (sq - med) / (1.4826 * mad)
        for cap, hid in CAPS.items():
            pe = fit_student(Xq, yq, hid, seed).predict(Xe)
            for frac, tag in FRACS:
                out[f"reg_{M}_{cap}_{tag}"] = regret(se, pe, frac)
                out[f"p_{M}_{cap}_{tag}"] = prec_at_k(pe, ye, frac)
    return out


def fields():
    f = ["dataset", "teacher", "seed", "n_eval"]
    for tag in ("k1", "k5"):
        f += [f"dfloor_sub99_{tag}", f"dfloor_sub90_{tag}", f"dfloor_boot_{tag}", f"sfloor_{tag}",
              f"p_teacher_{tag}", f"p_refresh_lo_{tag}", f"p_refresh_hi_{tag}", f"p_refresh_mean_{tag}"]
        for M in BUDGETS:
            for cap in CAPS:
                f += [f"reg_{M}_{cap}_{tag}", f"p_{M}_{cap}_{tag}"]
    return f


def _work(task):
    try:
        return run_cell(*task)
    except Exception as e:
        return {"__fail__": f"{task}: {e}"}


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--outdir", default="results_phase2")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True); path = out / "results.csv"
    seen = set()
    if path.exists():
        with path.open() as fh:
            seen = {(r["dataset"], r["teacher"], r["seed"]) for r in csv.DictReader(fh)}
    tasks = [(n, t, s) for n in eligible_datasets() for t in ("knn", "kde", "ocsvm", "iforest", "ae")
             for s in range(a.seeds) if (n, t, str(s)) not in seen]
    print(f"todo={len(tasks)} skipped={len(seen)} workers={a.workers}", flush=True)
    new = not path.exists()
    fh = path.open("a", newline=""); w = csv.DictWriter(fh, fieldnames=fields(), extrasaction="ignore")
    if new:
        w.writeheader(); fh.flush()
    done = fails = 0
    with mp.Pool(a.workers) as pool:
        for res in pool.imap_unordered(_work, tasks, chunksize=1):
            if "__fail__" in res:
                fails += 1; print(f"[FAIL] {res['__fail__']}", flush=True); continue
            w.writerow(res); fh.flush(); done += 1
            if done % 15 == 0:
                print(f"  ... {done}/{len(tasks)} ({fails} fails)", flush=True)
    fh.close(); print(f"Wrote {path} (+{done}, {fails} fails)", flush=True)


if __name__ == "__main__":
    main()
