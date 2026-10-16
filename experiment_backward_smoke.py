"""Smoke test: data-free unification of a multi-output anomaly pipeline by
BACKWARD probing, vs replay, uniform, latent-decode, and the novelty-check
baselines (Wang 2021 random-direction boundary search; Vernekar 2019 random
null-space steps). Runs on TSB-AD-M multivariate series.

Teacher pipeline (per series): standardized windows (w x C) -> autoencoder ->
per-channel residual -> per-channel calibration CDF (u_c) -> fused score
S = max_c u_c -> threshold tau (99% of calibration S) -> alarm.
Outputs distilled into ONE student: u_1..u_C and S (alarm = S > tau).

Arms (each builds a transfer set of N windows labelled by the REAL pipeline):
  replay   : full training windows + multi-scale Gaussian jitter   (data: full)
  oracle   : the test windows themselves (ceiling; not a valid method)
  uniform  : uniform in standardized window space                 (data: none)
  decode   : z ~ N(mu,Sigma) -> D(z)          (data: latent moments; control)
  vernekar : D(z) + random-size step along a random decoder-normal direction
  wang     : D(z) + bisection along a random isotropic ray to the alarm flip
  backward : D(z) + bisection along a channel-targeted decoder-normal direction
             to a sampled target level u* of that channel's calibrated score
  backward0: as backward but z ~ N(0, I) (no retained statistics)
  backward1: as backward but anchors are 1% of real training windows
Pre-registered: decode probes are almost never alarms (circularity); oracle
gives the lowest regret; uniform fails the refresh floor on the fused head.
Kill: backward fails the fused-head floor where replay passes.
One CSV row per (series, seed, arm); resumable.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, csv, glob, re, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
torch.set_num_threads(1)

W = 8                 # window length
Q_BUDGET = 60000      # pipeline queries per arm (every queried point becomes training data)
SEARCH_COST = 18      # queries per search probe (<=3 expansions + 14 bisections + final)
MAX_TRAIN_WIN = 20000
MAX_TEST_WIN = 20000
ARMS = ["replay", "oracle", "uniform", "decode", "vernekar", "wang",
        "backward", "backward0", "backward1"]


# ---------------------------------------------------------------- data
def load_series(path):
    df = pd.read_csv(path).dropna()
    X = df.iloc[:, :-1].to_numpy(np.float64); y = df.iloc[:, -1].to_numpy().astype(int)
    m = re.search(r"_tr_(\d+)_1st_(\d+)", Path(path).name)
    tr = int(m.group(1)) if m else int(0.3 * len(X))
    first = int(m.group(2)) if m else tr
    tr = max(200, min(tr, first, len(X) - 200))
    mu = X[:tr].mean(0); sd = X[:tr].std(0) + 1e-8
    Z = (X - mu) / sd
    keep = sd > 1e-6                      # drop constant channels
    return Z[:, keep], y, tr


def windows(Z, start, end, step=1):
    idx = np.arange(start, end - W + 1, step)
    return np.stack([Z[i:i + W] for i in idx]).reshape(len(idx), -1), idx


# ---------------------------------------------------------------- teacher
class AE(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, k))
        self.dec = nn.Sequential(nn.Linear(k, 128), nn.Tanh(), nn.Linear(128, d))

    def forward(self, x):
        return self.dec(self.enc(x))


def fit_ae(X, d, k, seed, epochs=60):
    torch.manual_seed(seed)
    ae = AE(d, k); opt = torch.optim.Adam(ae.parameters(), 2e-3)
    T = torch.tensor(X, dtype=torch.float32); n = len(T)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for b in range(0, n, 256):
            xb = T[perm[b:b + 256]]
            opt.zero_grad(); loss = ((ae(xb) - xb) ** 2).mean(); loss.backward(); opt.step()
    ae.eval(); return ae


class Pipeline:
    """The teacher. Built only from normal training windows."""
    def __init__(self, Xn, C, seed, rng):
        self.C = C; self.d = Xn.shape[1]; k = int(min(16, max(4, C)))
        perm = rng.permutation(len(Xn)); cut = int(0.8 * len(Xn))
        Xfit, Xcal = Xn[perm[:cut]], Xn[perm[cut:]]
        self.ae = fit_ae(Xfit, self.d, k, seed)
        with torch.no_grad():
            Zl = self.ae.enc(torch.tensor(Xfit, dtype=torch.float32)).numpy()
        self.z_mu = Zl.mean(0); self.z_cov = np.cov(Zl.T) + 1e-6 * np.eye(k)   # latent-moment certificate
        R = self.resid(Xcal)
        self.sorted = [np.sort(R[:, c]) for c in range(C)]
        self.tail = [max(s[-1] - s[int(0.9 * (len(s) - 1))], 1e-9) for s in self.sorted]
        self.tau = float(np.quantile(self.outputs(Xcal)[1], 0.99))

    def resid(self, X):
        with torch.no_grad():
            Xr = self.ae(torch.tensor(X, dtype=torch.float32)).numpy()
        return ((X - Xr) ** 2).reshape(len(X), W, self.C).mean(1)

    def calib(self, R):
        U = np.empty_like(R)
        for c in range(self.C):
            s = self.sorted[c]; n = len(s)
            u = np.searchsorted(s, R[:, c], side="right") / n
            over = R[:, c] > s[-1]
            u[over] = 1.0 + np.log1p((R[over, c] - s[-1]) / self.tail[c])   # monotone tail, no ties
            U[:, c] = u
        return U

    def outputs(self, X):
        U = self.calib(self.resid(X)); S = U.max(1)
        return U, S

    def alarm(self, X):
        return self.outputs(X)[1] > self.tau


# ---------------------------------------------------------------- probing
def decoder_normal_dirs(pipe, Z, rng, chan=None):
    """Random unit directions in the normal space of the decoder at latent Z
    (orthogonal complement of the decoder Jacobian's column space). If chan is
    given, the Gaussian draw is restricted to that channel's coordinates before
    projection, so the step mostly raises that channel's residual."""
    dec = pipe.ae.dec
    Zt = torch.tensor(Z, dtype=torch.float32)
    J = torch.func.vmap(torch.func.jacrev(dec))(Zt).detach().numpy()  # (n, d, k)
    n, d, _ = J.shape
    G = rng.normal(size=(n, d))
    if chan is not None:
        mask = np.zeros((n, W, pipe.C)); mask[np.arange(n), :, chan] = 1.0
        G = G * mask.reshape(n, d)
    V = np.empty_like(G)
    for i in range(n):
        Q, _ = np.linalg.qr(J[i])
        v = G[i] - Q @ (Q.T @ G[i]); V[i] = v / (np.linalg.norm(v) + 1e-12)
    return V


def bisect(f_val, X0, V, target, t_hi, iters=14):
    """Vectorised 1-D bisection on t so that f_val(X0 + t V) crosses target
    (f assumed increasing in t). Returns t and the number of pipeline queries."""
    n = len(X0); lo = np.zeros(n); hi = np.full(n, t_hi); q = 0; seen = []
    for _ in range(3):                                   # expand hi until crossing
        Xe = X0 + hi[:, None] * V; seen.append(Xe); val = f_val(Xe); q += n
        need = val < target
        if not need.any():
            break
        hi[need] *= 2.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        Xe = X0 + mid[:, None] * V; seen.append(Xe); val = f_val(Xe); q += n
        up = val >= target
        hi[up] = mid[up]; lo[~up] = mid[~up]
    t = 0.5 * (lo + hi); seen.append(X0 + t[:, None] * V)
    return t, q + n, np.vstack(seen)


def sample_latent(pipe, n, rng, moments=True):
    k = len(pipe.z_mu)
    return (rng.multivariate_normal(pipe.z_mu, pipe.z_cov, size=n) if moments
            else rng.normal(size=(n, k)))


def decode(pipe, Z):
    with torch.no_grad():
        return pipe.ae.dec(torch.tensor(Z, dtype=torch.float32)).numpy().astype(np.float64)


def encode(pipe, X):
    with torch.no_grad():
        return pipe.ae.enc(torch.tensor(X, dtype=torch.float32)).numpy()


def backward_probe(pipe, X0, Z0, rng, n):
    """Backward probing: target a level u* of one channel's calibrated score and
    reach it by bisection along a channel-targeted decoder-normal direction."""
    idx = rng.integers(0, len(X0), size=n); X0 = X0[idx]; Z0 = Z0[idx]
    pts = []; q = 0
    chans = rng.integers(0, pipe.C, size=n)
    for c in range(pipe.C):
        sel = np.where(chans == c)[0]
        if len(sel) == 0:
            continue
        V = decoder_normal_dirs(pipe, Z0[sel], rng, chan=c)
        target = rng.uniform(0.6, 1.8, size=len(sel))    # below, at and beyond the alarm region
        f = lambda X, c=c: pipe.outputs(X)[0][:, c]
        t, qq, seen = bisect(f, X0[sel], V, target, t_hi=0.5 * np.sqrt(pipe.d))
        pts.append(seen); q += qq
    return np.vstack(pts), q


def make_transfer(arm, pipe, Xtrain, Xtest, rng):
    """Returns (X_transfer, pipeline_queries_used)."""
    d = pipe.d; N = Q_BUDGET
    on_frac = 0.3                                         # share of on-surface (normal-region) anchors
    if arm == "replay":
        base = Xtrain[rng.integers(0, len(Xtrain), size=N)]
        sig = rng.choice([0.0, 0.1, 0.3, 0.6, 1.0, 2.0], size=(N, 1))
        return base + sig * rng.normal(size=base.shape), N
    if arm == "oracle":
        return Xtest[rng.integers(0, len(Xtest), size=N)], N
    if arm == "uniform":
        return rng.uniform(-4, 4, size=(N, d)), N
    if arm == "backward1":
        anchors = Xtrain[rng.choice(len(Xtrain), size=max(20, len(Xtrain) // 100), replace=False)]
        Za = encode(pipe, anchors)
    else:
        Za = sample_latent(pipe, N, rng, moments=(arm != "backward0"))
        anchors = decode(pipe, Za)
    if arm == "decode":
        return anchors, N
    n_on = int(on_frac * N); n_off = N - n_on
    on = anchors[rng.integers(0, len(anchors), size=n_on)]
    n_search = max(1, n_off // SEARCH_COST)               # search arms: same total query budget
    if arm == "vernekar":
        idx = rng.integers(0, len(anchors), size=n_off)
        V = decoder_normal_dirs(pipe, Za[idx], rng)
        beta = rng.uniform(0.1, 1.0, size=(n_off, 1)) * 2.0 * np.sqrt(d)
        return np.vstack([on, anchors[idx] + beta * V]), N
    if arm == "wang":
        idx = rng.integers(0, len(anchors), size=n_search)
        V = rng.normal(size=(n_search, d)); V /= np.linalg.norm(V, axis=1, keepdims=True)
        f = lambda X: pipe.outputs(X)[1]
        t, q, seen = bisect(f, anchors[idx], V, np.full(n_search, pipe.tau), t_hi=0.5 * np.sqrt(d))
        X = np.vstack([on, seen]); return X, len(X)
    if arm in ("backward", "backward0", "backward1"):
        Xb, q = backward_probe(pipe, anchors, Za, rng, n_search)
        X = np.vstack([on, Xb]); return X, len(X)
    raise ValueError(arm)


# ---------------------------------------------------------------- student
class Student(nn.Module):
    def __init__(self, d, C):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, C + 1))

    def forward(self, x):
        return self.net(x)


def fit_student(X, Y, seed, epochs=40):
    torch.manual_seed(seed)
    st = Student(X.shape[1], Y.shape[1] - 1); opt = torch.optim.Adam(st.parameters(), 1e-3)
    T = torch.tensor(X, dtype=torch.float32); Yt = torch.tensor(np.clip(Y, 0, 50), dtype=torch.float32)   # no low clip: it tied saturated windows
    n = len(T); g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for b in range(0, n, 512):
            i = perm[b:b + 512]
            opt.zero_grad(); loss = ((st(T[i]) - Yt[i]) ** 2).mean(); loss.backward(); opt.step()
    st.eval(); return st


def regret(s_t, s_s, frac):
    k = max(1, int(round(frac * len(s_t))))
    A = set(np.argsort(s_t)[-k:]); B = set(np.argsort(s_s)[-k:])
    return 1.0 - len(A & B) / k


def prec_at_k(s, y, frac):
    k = max(1, int(round(frac * len(s)))); return float(y[np.argsort(s)[-k:]].mean())


# ---------------------------------------------------------------- cell
def run_series(path, seed, n_rep=2):
    rng = np.random.default_rng(seed * 7919 + 11)
    Z, y, tr = load_series(path); C = Z.shape[1]
    Xtrain, _ = windows(Z, 0, tr, step=max(1, (tr - W) // MAX_TRAIN_WIN + 1))
    step = max(1, (len(Z) - tr - W) // MAX_TEST_WIN + 1)
    Xtest, tidx = windows(Z, tr, len(Z), step=step)
    ytest = np.array([y[i:i + W].max() for i in tidx])
    pipe = Pipeline(Xtrain, C, seed, rng)
    Ut, St = pipe.outputs(Xtest); At = St > pipe.tau
    # refresh-churn floor: teacher retrained (new seed) on a bootstrap of the training windows
    floors = {0.01: [], 0.05: []}
    for r in range(n_rep):
        boot = Xtrain[rng.integers(0, len(Xtrain), size=len(Xtrain))]
        S2 = Pipeline(boot, C, seed + 100 + r, rng).outputs(Xtest)[1]
        for f in floors:
            floors[f].append(regret(St, S2, f))
    base = {"series": Path(path).stem, "seed": seed, "C": C, "n_train_win": len(Xtrain),
            "n_test_win": len(Xtest), "test_anom_frac": float(ytest.mean()),
            "teacher_alarm_rate": float(At.mean()),
            "floor_k1": float(np.mean(floors[0.01])), "floor_k5": float(np.mean(floors[0.05])),
            "p_teacher_k1": prec_at_k(St, ytest, 0.01), "p_teacher_k5": prec_at_k(St, ytest, 0.05)}
    rows = []
    for arm in ARMS:
        t0 = time.time()
        Xtr_s, queries = make_transfer(arm, pipe, Xtrain, Xtest, rng)
        Ut_s, St_s = pipe.outputs(Xtr_s)
        st = fit_student(Xtr_s, np.column_stack([Ut_s, St_s]), seed)
        with torch.no_grad():
            P = st(torch.tensor(Xtest, dtype=torch.float32)).numpy()
        Us, Ss = P[:, :C], P[:, C]; As = Ss > pipe.tau
        tp = (As & At).sum(); f1 = 2 * tp / max(1, As.sum() + At.sum())
        row = dict(base, arm=arm, queries=int(queries),
                   probe_alarm_frac=float((St_s > pipe.tau).mean()),
                   probe_S_mean=float(St_s.mean()),
                   reg_k1=regret(St, Ss, 0.01), reg_k5=regret(St, Ss, 0.05),
                   alarm_f1=float(f1),
                   rho_S=float(spearmanr(St, Ss).statistic),
                   rho_U=float(np.nanmean([spearmanr(Ut[:, c], Us[:, c]).statistic for c in range(C)])),
                   p_student_k1=prec_at_k(Ss, ytest, 0.01), p_student_k5=prec_at_k(Ss, ytest, 0.05),
                   secs=round(time.time() - t0, 1))
        row["pass_k1"] = row["reg_k1"] <= base["floor_k1"] + 1e-9
        row["pass_k5"] = row["reg_k5"] <= base["floor_k5"] + 1e-9
        rows.append(row)
    return rows


FIELDS = ["series", "seed", "C", "n_train_win", "n_test_win", "test_anom_frac", "teacher_alarm_rate",
          "floor_k1", "floor_k5", "p_teacher_k1", "p_teacher_k5", "arm", "queries", "probe_alarm_frac",
          "probe_S_mean", "reg_k1", "reg_k5", "alarm_f1", "rho_S", "rho_U", "p_student_k1",
          "p_student_k5", "pass_k1", "pass_k5", "secs"]


def pick_series(data_dir, n_series, max_c=40, min_len=3000):
    """One series per source dataset (prefix after the index), deterministic."""
    files = sorted(glob.glob(str(Path(data_dir) / "**" / "*.csv"), recursive=True))
    chosen, seen = [], set()
    for f in files:
        name = Path(f).name; src = name.split("_")[1] if "_" in name else name
        if src in seen:
            continue
        try:
            head = pd.read_csv(f, nrows=5)
            C = head.shape[1] - 1
            if C < 2 or C > max_c:
                continue
            n = sum(1 for _ in open(f)) - 1
            if n < min_len:
                continue
        except Exception:
            continue
        seen.add(src); chosen.append(f)
        if len(chosen) >= n_series:
            break
    return chosen


def _work(task):
    path, seed = task
    try:
        return run_series(path, seed)
    except Exception as e:
        import traceback
        return [{"__fail__": f"{Path(path).name} s{seed}: {e!r} | {traceback.format_exc()[-400:]}"}]


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--series", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--outdir", default="results_backward_smoke")
    ap.add_argument("--tiny", action="store_true", help="1 series, 1 seed, small sizes: code-path check")
    a = ap.parse_args()
    global Q_BUDGET, MAX_TRAIN_WIN, MAX_TEST_WIN
    if a.tiny:
        Q_BUDGET, MAX_TRAIN_WIN, MAX_TEST_WIN = 3000, 2000, 2000
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True); path = out / "results.csv"
    series = pick_series(a.data, 1 if a.tiny else a.series)
    print(f"series ({len(series)}): " + ", ".join(Path(s).name[:40] for s in series), flush=True)
    seen = set()
    if path.exists():
        with path.open() as fh:
            seen = {(r["series"], r["seed"]) for r in csv.DictReader(fh)}
    tasks = [(s, sd) for s in series for sd in range(1 if a.tiny else a.seeds)
             if (Path(s).stem, str(sd)) not in seen]
    new = not path.exists()
    fh = path.open("a", newline=""); w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
    if new:
        w.writeheader(); fh.flush()
    fails = 0
    with mp.Pool(min(a.workers, max(1, len(tasks)))) as pool:
        for res in pool.imap_unordered(_work, tasks, chunksize=1):
            if res and "__fail__" in res[0]:
                fails += 1; print(f"[FAIL] {res[0]['__fail__']}", flush=True); continue
            w.writerows(res); fh.flush()
            r0 = res[0]
            print(f"[done] {r0['series'][:40]} s{r0['seed']} C={r0['C']} floor_k5={r0['floor_k5']:.3f} | "
                  + " ".join(f"{r['arm']}={r['reg_k5']:.2f}" for r in res), flush=True)
    fh.close(); print(f"Wrote {path} ({fails} fails)", flush=True)
    if fails and a.tiny:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
