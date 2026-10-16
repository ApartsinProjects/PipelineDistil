"""Phase 3: E3 redone with properly trained students + strong competing surrogates.

Question: is the student-to-floor gap a real limit of label-free distillation, or
an artefact of under-trained small MLPs? For each (dataset, teacher, seed):
  data floor  : bootstrap refresh floor (same seed), as in Phase 2.
  students on space-filling queries (budgets 10k, 50k), all with standardized
  inputs, early stopping on a held-out 10% of the queries, up to 1000 epochs:
    mlp_big   : MLP (256,128,64)
    mlp_ens   : mean of 3 MLPs (256,128,64), different seeds
    hgb       : HistGradientBoostingRegressor (strong non-neural student)
  compressed teacher (white-box compression control, deterministic teachers):
    coreset10 : teacher refit on a 10% k-center coreset of the training normals.
Recorded per student: alarm-set regret (k=1%,5%), precision@k, and the student's
FIT QUALITY = R^2 of student vs teacher score on held-out queries and on the eval
set. High query R^2 + high regret => a real decision-level gap, not under-training.
Incremental + resumable CSV.
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from experiment_realbench_v2 import load, shell_scale
from experiment_noisefloor import teacher_fn, regret
from experiment_floormap import eligible_datasets

FRACS = ((0.01, "k1"), (0.05, "k5"))
BUDGETS = (10000, 50000)
DETERMINISTIC = {"knn", "kde", "ocsvm"}


def prec_at_k(s, y, frac):
    k = max(1, int(round(frac * len(s)))); return float(y[np.argsort(s)[-k:]].mean())


def mlp(seed):
    return MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation="relu", solver="adam",
                        learning_rate_init=1e-3, max_iter=1000, batch_size=256,
                        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
                        random_state=seed)


def kcenter_idx(X, m, seed):
    rng = np.random.default_rng(seed); idx = [int(rng.integers(len(X)))]
    d = ((X - X[idx[0]]) ** 2).sum(1)
    for _ in range(m - 1):
        i = int(np.argmax(d)); idx.append(i); d = np.minimum(d, ((X - X[i]) ** 2).sum(1))
    return np.array(idx)


def score_block(out, name, pe, se, ye, fit_q, fit_e):
    for frac, tag in FRACS:
        out[f"reg_{name}_{tag}"] = regret(se, pe, frac)
        out[f"p_{name}_{tag}"] = prec_at_k(pe, ye, frac)
    out[f"r2q_{name}"] = fit_q; out[f"r2e_{name}"] = fit_e


def run_cell(name, tname, seed, n_rep=5):
    rng = np.random.default_rng(seed * 101 + 3)
    Xtr, Xval, Xan = load(name, seed)
    Xe = np.concatenate([Xval, Xan]); ye = np.r_[np.zeros(len(Xval)), np.ones(len(Xan))]
    ntr = len(Xtr); base = teacher_fn(tname, Xtr, seed); se = base(Xe)
    out = {"dataset": name, "teacher": tname, "seed": seed, "n_eval": len(Xe)}
    sc = [teacher_fn(tname, Xtr[rng.integers(0, ntr, size=ntr)], seed)(Xe) for _ in range(n_rep)]
    for frac, tag in FRACS:
        out[f"dfloor_boot_{tag}"] = float(np.mean([regret(se, s, frac) for s in sc]))
        out[f"p_teacher_{tag}"] = prec_at_k(se, ye, frac)
        out[f"p_refresh_lo_{tag}"] = float(np.min([prec_at_k(s, ye, frac) for s in sc]))
    scale = shell_scale(Xtr, seed); d = Xtr.shape[1]
    lo, hi = Xtr.min(0) - scale, Xtr.max(0) + scale
    box = rng.uniform(lo, hi, size=(max(BUDGETS) + 5000, d))
    Xhold = box[-5000:]; shold = base(Xhold)            # held-out queries for fit quality
    for M in BUDGETS:
        Xq = np.concatenate([Xtr, box[:M]]); sq = base(Xq)
        med = np.median(sq); mad = np.median(np.abs(sq - med)) + 1e-9
        yq = (sq - med) / (1.4826 * mad)
        ss = StandardScaler().fit(Xq); Zq = ss.transform(Xq); Ze = ss.transform(Xe); Zh = ss.transform(Xhold)
        yh = (shold - med) / (1.4826 * mad); ye_t = (se - med) / (1.4826 * mad)
        m = mlp(seed).fit(Zq, yq)
        score_block(out, f"mlp_{M}", m.predict(Ze), se, ye, r2_score(yh, m.predict(Zh)), r2_score(ye_t, m.predict(Ze)))
        ens = [m] + [mlp(seed + 10 + j).fit(Zq, yq) for j in range(2)]
        pe = np.mean([e.predict(Ze) for e in ens], 0); ph = np.mean([e.predict(Zh) for e in ens], 0)
        score_block(out, f"ens_{M}", pe, se, ye, r2_score(yh, ph), r2_score(ye_t, pe))
        h = HistGradientBoostingRegressor(max_iter=500, early_stopping=True, random_state=seed).fit(Zq, yq)
        score_block(out, f"hgb_{M}", h.predict(Ze), se, ye, r2_score(yh, h.predict(Zh)), r2_score(ye_t, h.predict(Ze)))
    if tname in DETERMINISTIC:
        ci = kcenter_idx(Xtr, max(20, ntr // 10), seed)
        pe = teacher_fn(tname, Xtr[ci], seed)(Xe)
        score_block(out, "coreset10", pe, se, ye, np.nan, r2_score(se, pe))
    return out


def fields():
    f = ["dataset", "teacher", "seed", "n_eval"]
    for tag in ("k1", "k5"):
        f += [f"dfloor_boot_{tag}", f"p_teacher_{tag}", f"p_refresh_lo_{tag}"]
    names = [f"{s}_{M}" for M in BUDGETS for s in ("mlp", "ens", "hgb")] + ["coreset10"]
    for n in names:
        for tag in ("k1", "k5"):
            f += [f"reg_{n}_{tag}", f"p_{n}_{tag}"]
        f += [f"r2q_{n}", f"r2e_{n}"]
    return f


def _work(task):
    try:
        return run_cell(*task)
    except Exception as e:
        return {"__fail__": f"{task}: {e}"}


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--outdir", default="results_phase3")
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
            if done % 10 == 0:
                print(f"  ... {done}/{len(tasks)} ({fails} fails)", flush=True)
    fh.close(); print(f"Wrote {path} (+{done}, {fails} fails)", flush=True)


if __name__ == "__main__":
    main()
