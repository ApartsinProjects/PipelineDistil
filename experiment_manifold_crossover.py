"""Geometry-governed crossover: when does manifold-aware sampling beat space-filling?

Our real-tabular study found dumb space-filling (uniform box) beats near-manifold
"shell" sampling -- because those teachers were low-dimensional and roughly
single-blob, so the ambient box was still fillable. This experiment tests the
regime the tabular study missed: a teacher whose NORMAL support is a MULTIMODAL,
LOW-DIMENSIONAL MANIFOLD embedded in a HIGH ambient dimension (intrinsic m << D,
k modes). There the box has volume ~exp(D), almost all of it empty space where the
teacher is uninformative, so a fixed query BUDGET spent space-filling is wasted,
while sampling anchored on the manifold concentrates queries where the teacher's
response surface varies.

Setup (fixed small query budget M = expensive teacher):
  normals: union of k modes, each an m-dim Gaussian pancake (low-rank) centered on
           a random point on a sphere in R^D, plus small ambient noise.
  teacher: kNN mean-distance anomaly score fit on the normals (cheap here; the
           geometry, not the per-query cost, drives the crossover).
  anomalies (eval): off-manifold points = a random mode center pushed along a
           random AMBIENT-noise direction by a moderate distance (the near-manifold
           shell region we actually care about).
  samplers (each adds exactly M synthetic queries, labelled by the teacher):
    uniform_box  : uniform in the normals' bounding box (+pad)   [space-filling]
    gaussian_jitter : normal points + isotropic noise            [local, manifold-aware]
    global_shell : anchored directional shell around all normals [manifold-aware]
    per_mode_shell: shell anchored per detected mode (kmeans)     [mode + manifold aware]
  metric: student-teacher fidelity (Spearman) on held-out anomalies.

Sweep D in {8,32,128,512}, k in {1,8}, m fixed. Prediction: uniform_box fidelity
collapses as D grows and k>1, while shell/per_mode hold -> a geometry-driven
crossover where structured sampling wins.
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
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from experiment import make_percentile_maps

M = 500                      # fixed query budget (expensive teacher)
N_NORMAL, N_ANOM = 2000, 500
M_INTRINSIC = 4              # manifold intrinsic dimension


def _rng(*p):
    return np.random.default_rng(zlib.crc32("|".join(map(str, p)).encode()) & 0x7FFFFFFF)


def gen_data(D, k, seed):
    rng = _rng("data", D, k, seed)
    # k mode centers on a sphere of radius R in R^D
    C = rng.normal(size=(k, D)); C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12); C *= 6.0
    # each mode: an m-dim low-rank subspace (pancake) + small ambient noise
    bases = [np.linalg.qr(rng.normal(size=(D, M_INTRINSIC)))[0] for _ in range(k)]
    def sample_manifold(n, rng):
        who = rng.integers(0, k, size=n); X = np.empty((n, D))
        for j in range(k):
            idx = np.where(who == j)[0]
            if len(idx) == 0:
                continue
            z = rng.normal(size=(len(idx), M_INTRINSIC)) * 1.2      # on-manifold coords
            X[idx] = C[j] + z @ bases[j].T + rng.normal(size=(len(idx), D)) * 0.1
        return X, who
    Xn, _ = sample_manifold(N_NORMAL, rng)
    # anomalies: a mode center pushed OFF-manifold (perpendicular) by moderate dist
    ra = _rng("anom", D, k, seed); who = ra.integers(0, k, size=N_ANOM); Xa = np.empty((N_ANOM, D))
    for j in range(k):
        idx = np.where(who == j)[0]
        if len(idx) == 0:
            continue
        z = ra.normal(size=(len(idx), M_INTRINSIC)) * 1.2
        onman = C[j] + z @ bases[j].T
        perp = ra.normal(size=(len(idx), D)); perp -= (perp @ bases[j]) @ bases[j].T   # remove in-manifold comp
        perp /= (np.linalg.norm(perp, axis=1, keepdims=True) + 1e-12)
        Xa[idx] = onman + perp * ra.uniform(1.0, 4.0, size=(len(idx), 1))               # near-manifold shell
    return Xn, Xa


def teacher_of(Xn):
    nn = NearestNeighbors(n_neighbors=10).fit(Xn)
    return lambda P: nn.kneighbors(P)[0].mean(1)


def shell_scale(Xn):
    nn = NearestNeighbors(n_neighbors=2).fit(Xn)
    return float(np.median(nn.kneighbors(Xn[:min(500, len(Xn))])[0][:, 1]))


def shell_pool(anchors_fn, Xn, need, rng, rmin, rmax):
    D = Xn.shape[1]; nn = NearestNeighbors(n_neighbors=1).fit(Xn); pts = []; tries = 0
    while len(pts) < need:
        A = anchors_fn(6000, rng)
        dirs = rng.normal(size=(6000, D)); dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
        cand = A + dirs * rng.uniform(rmin, rmax, size=(6000, 1))
        dd = nn.kneighbors(cand)[0][:, 0]
        pts.extend(cand[(dd > rmin) & (dd <= rmax)].tolist()); tries += 1
        if tries > 60:
            break
    return np.array(pts[:need]) if pts else np.zeros((0, D))


def make_queries(kind, Xn, rng, scale, k):
    D = Xn.shape[1]
    if kind == "uniform_box":
        lo, hi = Xn.min(0) - scale, Xn.max(0) + scale
        return rng.uniform(lo, hi, size=(M, D))
    if kind == "gaussian_jitter":
        return Xn[rng.integers(0, len(Xn), size=M)] + rng.normal(scale=1.5 * scale, size=(M, D))
    if kind == "global_shell":
        return shell_pool(lambda n, r: Xn[r.integers(0, len(Xn), size=n)], Xn, M, rng, 1.0 * scale, 6.0 * scale)
    if kind == "per_mode_shell":
        km = KMeans(n_clusters=max(1, k), n_init=3, random_state=0).fit(Xn)
        cen = km.cluster_centers_
        def anch(n, r):
            base = cen[r.integers(0, len(cen), size=n)]
            return base + r.normal(size=(n, D)) * (1.2)   # spread around each mode center
        return shell_pool(anch, Xn, M, rng, 1.0 * scale, 6.0 * scale)
    raise ValueError(kind)


def student(X, y, seed):
    return MLPRegressor(hidden_layer_sizes=(64, 32), activation="tanh", solver="adam",
                        learning_rate_init=5e-3, max_iter=1000, random_state=seed,
                        tol=1e-6, n_iter_no_change=30).fit(X, y)


SAMPLERS = ["uniform_box", "gaussian_jitter", "global_shell", "per_mode_shell"]


def run_cell(D, k, seed):
    Xn, Xa = gen_data(D, k, seed)
    teach = teacher_of(Xn); sb = lambda P: np.stack([teach(P)], axis=-1)
    pct = make_percentile_maps(sb(Xn)); t_an = pct(sb(Xa))[:, 0]
    scale = shell_scale(Xn)
    rows = []
    for kind in SAMPLERS:
        rng = _rng(kind, D, k, seed)
        Xg = make_queries(kind, Xn, rng, scale, k)
        Xall = np.concatenate([Xn, Xg]) if len(Xg) else Xn
        st = student(Xall, pct(sb(Xall))[:, 0], seed)
        s_an = st.predict(Xa)
        rows.append({"D": D, "k": k, "seed": seed, "sampler": kind,
                     "fid": spearmanr(s_an, t_an).statistic, "n_queries": len(Xg)})
    return rows


def _work(task):
    D, k, seed = task
    try:
        return run_cell(D, k, seed)
    except Exception as e:
        return [{"__fail__": f"D{D}/k{k}/s{seed}: {e}"}]


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--dims", default="8,32,128,512")
    ap.add_argument("--modes", default="1,8")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir", default="results_manifold_crossover")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(exist_ok=True)
    Ds = [int(x) for x in a.dims.split(",")]; Ks = [int(x) for x in a.modes.split(",")]
    tasks = [(D, k, s) for D in Ds for k in Ks for s in range(a.seeds)]
    print(f"dims={Ds} modes={Ks} seeds={a.seeds} cells={len(tasks)} workers={a.workers} budget M={M}", flush=True)
    fields = ["D", "k", "seed", "sampler", "fid", "n_queries"]
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
    # quick crossover summary
    import pandas as pd
    d = pd.read_csv(out / "results.csv")
    print("\n=== fidelity by D x k x sampler (mean over seeds) ===", flush=True)
    piv = d.groupby(["k", "D", "sampler"])["fid"].mean().unstack("sampler")
    print(piv.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
