"""Regime crossover (Family B): does ADAPTIVE boundary-seeking beat passive
SPACE-FILLING for distilling a black-box function's decision boundary, and does
the crossover shift with geometry?

Correct framing (per external review): geometry + budget cause the sample-efficiency
crossover; teacher COST only makes query savings economically valuable. So here we
count QUERIES (the teacher is a cheap analytic limit-state function evaluated as if
black-box) and ask when adaptive query design needs FEWER queries than Sobol to
reach a target boundary fidelity.

Teacher (controlled limit-state): project x to an intrinsic m-dim subspace z=A^T x;
g(x) = min_j (||z - c_j||^2 - r^2) over k centers -> a union of k (hyper)spheres,
i.e. k disconnected decision surfaces g=0 embedded in ambient dimension D. The
operationally informative region is the thin band |g| <= tau around g=0.

Samplers (fixed query budget B, all label points with g):
  sobol   : Sobol low-discrepancy space-filling (PRIMARY passive opponent)
  random  : iid uniform (weak passive control)
  adaptive: pool-based active learning. Seed with a small Sobol set, train a small
            MLP ensemble, then iteratively query the pool where the ensemble is
            uncertain AND predicts near the boundary (margin x disagreement), with
            k-center diversity, retraining each batch until the budget is spent.

Metric (on a large held-out uniform set): boundary-band sign accuracy and Spearman
of predicted-vs-true g inside |g|<=tau_eval (the region that matters), plus overall
sign accuracy. Pre-registered expectation: at low D / large B, Sobol ties or beats
adaptive; as D (with fixed m) and #modes k grow and the band is thin, adaptive
reaches the target with fewer queries -> a crossover.
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
from scipy.stats import spearmanr, qmc
from sklearn.neural_network import MLPRegressor

BOX = 2.5                    # domain [-BOX, BOX]^D
R = 1.0                      # sphere radius (limit-state)
N_EVAL = 20000


def _rng(*p):
    return np.random.default_rng(zlib.crc32("|".join(map(str, p)).encode()) & 0x7FFFFFFF)


def make_teacher(D, m, k, seed):
    rng = _rng("teacher", D, m, k, seed)
    A = np.linalg.qr(rng.normal(size=(D, m)))[0]           # D x m orthonormal
    centers = rng.uniform(-1.0, 1.0, size=(k, m))          # k modes in intrinsic space
    def g(X):
        Z = X @ A                                          # (n, m)
        d2 = np.stack([((Z - c) ** 2).sum(1) for c in centers], axis=1)  # (n, k)
        return d2.min(1) - R * R                           # union of k spheres
    return g


def sobol_pts(D, n, rng):
    s = qmc.Sobol(d=D, scramble=True, seed=int(rng.integers(1 << 31)))
    u = s.random(n)
    return (u * 2 - 1) * BOX


def rand_pts(D, n, rng):
    return rng.uniform(-BOX, BOX, size=(n, D))


def ensemble(Xtr, ytr, seed, n_models=3):
    ms = []
    for j in range(n_models):
        m = MLPRegressor(hidden_layer_sizes=(64, 32), activation="tanh", solver="adam",
                         learning_rate_init=5e-3, max_iter=600, random_state=seed * 17 + j,
                         tol=1e-6, n_iter_no_change=25).fit(Xtr, ytr)
        ms.append(m)
    return ms


def ens_pred(ms, X):
    P = np.stack([m.predict(X) for m in ms], axis=1)       # (n, n_models)
    return P.mean(1), P.var(1)


def kcenter(cand, chosen_pts, n_pick):
    """farthest-first selection of n_pick from cand, seeded by chosen_pts."""
    pick = []
    if len(chosen_pts):
        mind = np.min(((cand[:, None, :] - chosen_pts[None, :, :]) ** 2).sum(2), axis=1)
    else:
        mind = np.full(len(cand), np.inf)
    for _ in range(min(n_pick, len(cand))):
        i = int(np.argmax(mind)); pick.append(i)
        d = ((cand - cand[i]) ** 2).sum(1)
        mind = np.minimum(mind, d)
    return pick


def adaptive_query(g, D, B, seed):
    rng = _rng("adapt", D, B, seed)
    n0 = max(16, B // 4)
    X = sobol_pts(D, n0, rng); y = g(X)
    batch = max(8, B // 8)
    while len(X) < B:
        ms = ensemble(X, y, seed)
        pool = rand_pts(D, 4000, rng)
        gm, gv = ens_pred(ms, pool)
        T = np.std(y) + 1e-9
        score = gv * np.exp(-np.abs(gm) / (0.5 * T))       # disagreement x near-boundary
        top = np.argsort(score)[-min(400, len(pool)):]     # candidate shortlist
        n_pick = min(batch, B - len(X))
        sel = kcenter(pool[top], X, n_pick)
        Xnew = pool[top][sel]
        X = np.concatenate([X, Xnew]); y = np.concatenate([y, g(Xnew)])
    return X[:B], y[:B]


def fidelity(g, ms, D, seed, tau_eval):
    rng = _rng("eval", D, seed)
    Xe = rand_pts(D, N_EVAL, rng); ge = g(Xe)
    gm, _ = ens_pred(ms, Xe)
    band = np.abs(ge) <= tau_eval
    out = {"overall_sign": float(np.mean(np.sign(gm) == np.sign(ge)))}
    if band.sum() > 10:
        out["band_sign"] = float(np.mean(np.sign(gm[band]) == np.sign(ge[band])))
        out["band_spearman"] = float(spearmanr(gm[band], ge[band]).statistic)
    else:
        out["band_sign"] = np.nan; out["band_spearman"] = np.nan
    return out


def run_cell(D, m, k, B, seed, tau_eval=0.5):
    g = make_teacher(D, m, k, seed)
    rows = []
    for method in ["sobol", "random", "adaptive"]:
        rng = _rng("run", method, D, B, seed)
        if method == "sobol":
            X = sobol_pts(D, B, rng); y = g(X)
        elif method == "random":
            X = rand_pts(D, B, rng); y = g(X)
        else:
            X, y = adaptive_query(g, D, B, seed)
        ms = ensemble(X, y, seed)
        f = fidelity(g, ms, D, seed, tau_eval)
        rows.append({"D": D, "m": m, "k": k, "B": B, "seed": seed, "method": method, **f})
    return rows


def _work(task):
    try:
        return run_cell(*task)
    except Exception as e:
        return [{"__fail__": f"{task}: {e}"}]


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--dims", default="2,8,32")
    ap.add_argument("--modes", default="1,3")
    ap.add_argument("--budgets", default="64,128,256,512")
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir", default="results_regime_crossover")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(exist_ok=True)
    Ds = [int(x) for x in a.dims.split(",")]; Ks = [int(x) for x in a.modes.split(",")]
    Bs = [int(x) for x in a.budgets.split(",")]
    tasks = [(D, a.m, k, B, s) for D in Ds for k in Ks for B in Bs for s in range(a.seeds)]
    print(f"dims={Ds} m={a.m} modes={Ks} budgets={Bs} seeds={a.seeds} cells={len(tasks)}", flush=True)
    fields = ["D", "m", "k", "B", "seed", "method", "overall_sign", "band_sign", "band_spearman"]
    f = (out / "results.csv").open("w", newline=""); w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    done = fails = 0
    with mp.Pool(a.workers) as pool:
        for res in pool.imap_unordered(_work, tasks, chunksize=1):
            if res and res[0].get("__fail__"):
                fails += 1; print(f"[FAIL] {res[0]['__fail__']}", flush=True); continue
            w.writerows(res); f.flush(); done += 1
            if done % 10 == 0:
                print(f"  ... {done}/{len(tasks)} cells ({fails} fails)", flush=True)
    f.close()
    print(f"\nWrote {out/'results.csv'} ({done} cells, {fails} fails)", flush=True)


if __name__ == "__main__":
    main()
