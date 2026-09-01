"""The good scenario: distilling a multivariate residual-MONITORING pipeline.

A realistic condition-monitoring pipeline is a chain, not a scalar:

    x -> autoencoder -> per-signal residual r_j(x)          (d outputs)
      -> per-signal percentile calibration -> R_j(x)        (calibrated residual field)
      -> aggregate score S(x) = mean_j R_j(x) ; threshold S > tau   (the alarm)
      -> attribution alpha_j(x) = R_j / sum_k R_k           (which signal broke: root cause)

The residual FIELD R(x) is the primitive; score, threshold, and attribution are
deterministic post-processing. So a student that copies R(x) copies the whole
pipeline. Crucially, on the normals every residual is small and the attribution is
meaningless -- all the interesting, operationally valuable behavior (WHICH signals
deviate, and by how much) lives OFF the manifold, exactly where we have no labels.

Hypothesis (pre-stated):
  H1  The aggregate score S distills tolerably from normals alone (roughly monotone
      in total residual): normals_only score-fidelity already decent.
  H2  The ATTRIBUTION alpha does NOT distill from normals: normals_only attribution
      fidelity is poor, and off-manifold SAMPLING is required to recover it.
  H3  An attribution-aware sampler (query the shell where the residual field is most
      STRUCTURED = high inter-signal variance) recovers attribution best.

We distill R(x) with a multi-output student and report, on held-out real anomalies:
  * score fidelity     : Spearman(student S, teacher S)
  * attribution cosine : mean cosine(student alpha, teacher alpha)
  * attribution top-1  : fraction where argmax signal agrees (same root cause)
  * decision agreement : F1 of student alarm vs teacher alarm at tau
Samplers: normals_only, uniform_shell, disagreement_shell, shell_box_mix.
Label-free: no anomaly labels for training/acquisition/calibration; labels only
enter the final evaluation split.
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
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from experiment import make_percentile_maps

DATA_DIR = Path(__file__).parent / "data" / "adbench"
M = 1000


def _seed_of(*p):
    return zlib.crc32("|".join(map(str, p)).encode()) & 0x7FFFFFFF


def all_datasets(min_anom=20, min_normal=400, min_d=6, max_d=64):
    keep = []
    for p in sorted(DATA_DIR.glob("*.npz")):
        try:
            a = np.load(p); y = a["y"].astype(int); X = a["X"]
            if (y == 1).sum() >= min_anom and (y == 0).sum() >= min_normal \
                    and min_d <= X.shape[1] <= max_d:
                keep.append(p.stem)
        except Exception:
            pass
    return keep


def load(name, seed, max_train=2500):
    a = np.load(DATA_DIR / f"{name}.npz")
    X, y = a["X"].astype(np.float64), a["y"].astype(int)
    normals, anoms = X[y == 0], X[y == 1]
    rng = np.random.default_rng(seed); rng.shuffle(normals)
    n_tr = min(max_train, len(normals) // 2)
    Xtr = normals[:n_tr]
    Xval = normals[n_tr:n_tr + min(1500, max(400, len(normals) // 4))]   # capped val
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xval), sc.transform(anoms)


class ResidualPipeline:
    """AE -> per-signal residual field. R(x) = per-signal percentile-calibrated
    squared residual (fit on normals). Score/threshold/attribution derive from R."""
    def __init__(self, X, seed):
        d = X.shape[1]
        self.ae = MLPRegressor(hidden_layer_sizes=(64, max(2, d // 2), 64), activation="tanh",
                               solver="adam", learning_rate_init=3e-3, max_iter=600, batch_size=64,
                               random_state=seed, tol=1e-5, n_iter_no_change=25).fit(X, X)
        self.pct = make_percentile_maps(self.resid(X))          # per-signal calibration
        R = self.R(X); self.tau = float(np.quantile(R.mean(1), 0.95))

    def resid(self, P):
        return (P - self.ae.predict(P)) ** 2                     # (n, d) raw residual

    def R(self, P):
        return self.pct(self.resid(P))                           # (n, d) calibrated field

    @staticmethod
    def score(R):
        return R.mean(1)

    @staticmethod
    def attribution(R):
        Rp = np.clip(R, 0, None); s = Rp.sum(1, keepdims=True) + 1e-12
        return Rp / s


def shell_scale(X, seed):
    nn = NearestNeighbors(n_neighbors=2).fit(X); rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(500, len(X)), replace=False)
    return float(np.median(nn.kneighbors(X[idx])[0][:, 1]))


def shell_pool(X, need, rng, rmin, rmax):
    d = X.shape[1]; nn = NearestNeighbors(n_neighbors=1).fit(X); pts = []; tries = 0
    while len(pts) < need:
        anchors = X[rng.integers(0, len(X), size=6000)]
        dirs = rng.normal(size=(6000, d)); dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
        cand = anchors + dirs * rng.uniform(rmin, rmax, size=(6000, 1))
        dd = nn.kneighbors(cand)[0][:, 0]
        pts.extend(cand[(dd > rmin) & (dd <= rmax)].tolist()); tries += 1
        if tries > 100:
            break
    return np.array(pts[:need]) if pts else np.zeros((0, d))


def make_queries(kind, X, pipe, rng, scale):
    d = X.shape[1]
    if kind == "normals_only":
        return np.zeros((0, d))
    if kind == "uniform_shell":
        return shell_pool(X, M, rng, 1.0 * scale, 6.0 * scale)
    if kind == "shell_box_mix":
        half = M // 2; lo, hi = X.min(0) - scale, X.max(0) + scale
        box = rng.uniform(lo, hi, size=(M - half, d)); sh = shell_pool(X, half, rng, scale, 6 * scale)
        return np.concatenate([sh, box]) if len(sh) else box
    if kind == "disagreement_shell":
        # attribution-aware: query the shell where the residual FIELD is most
        # STRUCTURED (high inter-signal variance = a few signals dominate = a
        # sharp root-cause pattern). Needs no anomaly labels, no student proxy.
        pool = shell_pool(X, M * 8, rng, 1.0 * scale, 6.0 * scale)
        if len(pool) < 10:
            return shell_pool(X, M, rng, scale, 6 * scale)
        R = pipe.R(pool); struct = R.var(axis=1)
        w = struct / (struct.sum() + 1e-12)
        return pool[rng.choice(len(pool), size=M, replace=True, p=w)]
    raise ValueError(kind)


def student(X, Y, seed):
    return MLPRegressor(hidden_layer_sizes=(32, 16), activation="tanh", solver="adam",
                        learning_rate_init=5e-3, max_iter=1000, random_state=seed,
                        tol=1e-6, n_iter_no_change=30).fit(X, Y)


SAMPLERS = ["normals_only", "uniform_shell", "disagreement_shell", "shell_box_mix"]


def run_cell(name, seed):
    Xtr, Xval, Xan = load(name, seed)
    pipe = ResidualPipeline(Xtr, seed)
    R_an = pipe.R(Xan); S_an = pipe.score(R_an); A_an = pipe.attribution(R_an)
    R_val = pipe.R(Xval); S_val = pipe.score(R_val)
    tau = pipe.tau
    teach_alarm = np.r_[S_val, S_an] > tau
    scale = shell_scale(Xtr, seed)
    rows = []
    for kind in SAMPLERS:
        rng = np.random.default_rng(seed * 1_000_003 + _seed_of(kind))
        Xg = make_queries(kind, Xtr, pipe, rng, scale)
        Xall = np.concatenate([Xtr, Xg]) if len(Xg) else Xtr
        st = student(Xall, pipe.R(Xall), seed)
        Rs_an = st.predict(Xan).reshape(len(Xan), -1)
        Rs_val = st.predict(Xval).reshape(len(Xval), -1)
        Ss_an = pipe.score(Rs_an); As_an = pipe.attribution(Rs_an)
        Ss_val = pipe.score(Rs_val)
        # metrics on anomalies
        score_fid = spearmanr(Ss_an, S_an).statistic
        cos = float(np.mean((As_an * A_an).sum(1) /
                    (np.linalg.norm(As_an, axis=1) * np.linalg.norm(A_an, axis=1) + 1e-12)))
        top1 = float(np.mean(As_an.argmax(1) == A_an.argmax(1)))
        # top-3 root-cause overlap (Jaccard of the 3 highest-attributed signals)
        kk = min(3, R_an.shape[1])
        j = []
        for i in range(len(Xan)):
            ta = set(np.argsort(A_an[i])[-kk:]); sa = set(np.argsort(As_an[i])[-kk:])
            j.append(len(ta & sa) / len(ta | sa))
        top3_jac = float(np.mean(j))
        # per-anomaly rank correlation of the full attribution vector
        arank = [spearmanr(As_an[i], A_an[i]).statistic for i in range(len(Xan))]
        attr_rank = float(np.nanmean(arank))
        stud_alarm = np.r_[Ss_val, Ss_an] > np.quantile(Ss_val, 0.95)
        dec_f1 = f1_score(teach_alarm, stud_alarm, zero_division=0)
        rows.append({"dataset": name, "d": Xtr.shape[1], "seed": seed, "sampler": kind,
                     "score_fid": score_fid, "attr_cosine": cos, "attr_top1": top1,
                     "attr_top3_jac": top3_jac, "attr_rank": attr_rank, "decision_f1": dec_f1})
    return rows


def _work(task):
    name, seed = task
    try:
        return run_cell(name, seed)
    except Exception as e:
        return [{"__fail__": f"{name}/s{seed}: {e}"}]


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir", default="results_pipeline_scenario")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(exist_ok=True)
    datasets = all_datasets()
    tasks = [(n, s) for n in datasets for s in range(a.seeds)]
    print(f"datasets={len(datasets)} seeds={a.seeds} cells={len(tasks)} workers={a.workers}", flush=True)
    print("  " + ", ".join(datasets), flush=True)
    fields = ["dataset", "d", "seed", "sampler", "score_fid", "attr_cosine", "attr_top1",
              "attr_top3_jac", "attr_rank", "decision_f1"]
    done, fails, n = 0, 0, 0
    f = (out / "results.csv").open("w", newline="")
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    with mp.Pool(a.workers) as pool:
        for res in pool.imap_unordered(_work, tasks, chunksize=1):
            if res and res[0].get("__fail__"):
                fails += 1; print(f"[FAIL] {res[0]['__fail__']}", flush=True); continue
            w.writerows(res); f.flush(); n += len(res); done += 1
            if done % 20 == 0:
                print(f"  ... {done}/{len(tasks)} cells ({fails} fails)", flush=True)
    f.close()
    print(f"\nWrote {out/'results.csv'} ({n} rows, {done} cells, {fails} fails)", flush=True)


if __name__ == "__main__":
    main()
