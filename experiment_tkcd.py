"""Cycle-3 pivot smoke: Top-K Contrastive Distillation (TKCD).

Distill the teacher's ALARM SET, not its whole score geometry. Same teacher-labelled
query set, same student, same eval as cycle-2. Loss = raw-score MSE + lambda * tail
contrastive term that pushes teacher-top-5% query points above near-cutoff
(85-95th pct) query points:
    L = MSE(f(Xq), z)  +  lambda * mean softplus(m - (f(x_pos) - f(x_neg)))
Raw baseline = same trainer with lambda=0 (controls for the torch reimplementation).
Metric: alarm-set regret at 5% budget on held-out (val normals + anomalies).
Pre-registered POSITIVE LEAD: median (R_raw - R_tkcd) >= +0.05 AND TKCD wins >=14/18.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, csv, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from experiment_realbench_v2 import load, TEACHERS, shell_scale

torch.set_num_threads(1)
M_BOX = 1000
ALARM = 0.05
MARGIN = 1.0
STEPS = 800
N_PAIRS = 256


def zstd(v):
    med = np.median(v); mad = np.median(np.abs(v - med)) + 1e-9
    return (v - med) / (1.4826 * mad)


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train(Xq, z, pos_idx, neg_idx, lam, seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    X = torch.tensor(Xq, dtype=torch.float32); Y = torch.tensor(z, dtype=torch.float32)
    P = torch.tensor(pos_idx); N = torch.tensor(neg_idx)
    m = MLP(Xq.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    sp = nn.Softplus()
    for _ in range(STEPS):
        opt.zero_grad()
        f = m(X)
        loss = ((f - Y) ** 2).mean()
        if lam > 0 and len(P) > 0 and len(N) > 0:
            pi = P[torch.randint(len(P), (N_PAIRS,), generator=g)]
            ni = N[torch.randint(len(N), (N_PAIRS,), generator=g)]
            loss = loss + lam * sp(MARGIN - (f[pi] - f[ni])).mean()
        loss.backward(); opt.step()
    m.eval()
    return m


def alarm_regret(ts, ss, frac=ALARM):
    k = max(1, int(round(frac * len(ts))))
    at = set(np.argsort(ts)[-k:]); as_ = set(np.argsort(ss)[-k:])
    return 1.0 - len(at & as_) / len(at)


def run_cell(name, tname, seed, lams=(0.0, 0.1, 0.3, 1.0)):
    Xtr, Xval, Xan = load(name, seed)
    teacher = TEACHERS[tname](Xtr, seed); scale = shell_scale(Xtr, seed)
    rng = np.random.default_rng(seed * 131 + 7)
    d = Xtr.shape[1]; lo, hi = Xtr.min(0) - scale, Xtr.max(0) + scale
    box = rng.uniform(lo, hi, size=(M_BOX, d))
    Xq = np.concatenate([Xtr, box]).astype(np.float64)
    sq = teacher(Xq); z = zstd(sq)
    # operational tail sets on the QUERY scores (label-free)
    cut_hi = np.quantile(sq, 0.95); cut_lo = np.quantile(sq, 0.85)
    pos_idx = np.where(sq >= cut_hi)[0]
    neg_idx = np.where((sq >= cut_lo) & (sq < cut_hi))[0]
    Xe = np.concatenate([Xval, Xan]).astype(np.float64); te = teacher(Xe)
    Xe_t = torch.tensor(Xe, dtype=torch.float32)
    out = {"dataset": name, "teacher": tname, "seed": seed,
           "n_pos": len(pos_idx), "n_neg": len(neg_idx)}
    reg = {}
    for lam in lams:
        m = train(Xq, z, pos_idx, neg_idx, lam, seed)
        with torch.no_grad():
            ss = m(Xe_t).numpy()
        reg[lam] = alarm_regret(te, ss)
    out["R_raw"] = reg[0.0]
    # TKCD: best over the small lambda grid (excluding 0)
    grid = {l: reg[l] for l in lams if l > 0}
    best_lam = min(grid, key=grid.get)
    out["R_tkcd"] = grid[best_lam]; out["best_lam"] = best_lam
    out["R_tkcd_l0.3"] = reg.get(0.3, np.nan)
    out["deltaR"] = out["R_raw"] - out["R_tkcd"]
    out["deltaR_l0.3"] = out["R_raw"] - reg.get(0.3, np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="6_cardio,40_vowels,38_thyroid")
    ap.add_argument("--teachers", default="knn,ae")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--outdir", default="results_tkcd")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(exist_ok=True); rows = []
    for name in a.datasets.split(","):
        for tname in a.teachers.split(","):
            for s in range(a.seeds):
                try:
                    r = run_cell(name, tname, s); rows.append(r)
                    print(f"[tkcd] {name:<13}{tname:<4}s{s} R_raw={r['R_raw']:.3f} "
                          f"R_tkcd={r['R_tkcd']:.3f}(lam{r['best_lam']}) dR={r['deltaR']:+.3f} "
                          f"| l0.3 dR={r['deltaR_l0.3']:+.3f} pos/neg={r['n_pos']}/{r['n_neg']}", flush=True)
                except Exception as e:
                    print(f"[tkcd] {name}/{tname}/s{s} FAIL {e}", flush=True)
    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    dR = np.array([r["deltaR"] for r in rows]); dR3 = np.array([r["deltaR_l0.3"] for r in rows])
    print(f"\n=== TKCD (best-lam): median dR={np.median(dR):+.3f} wins {int((dR>0).sum())}/{len(dR)} "
          f"| mean R_raw={np.mean([r['R_raw'] for r in rows]):.3f} R_tkcd={np.mean([r['R_tkcd'] for r in rows]):.3f} ===", flush=True)
    print(f"=== TKCD (fixed lam0.3): median dR={np.median(dR3):+.3f} wins {int((dR3>0).sum())}/{len(dR3)} ===", flush=True)
    print(f"LEAD? need median dR>=+0.05 AND wins>=14/18 (best-lam is optimistic; report both)", flush=True)


if __name__ == "__main__":
    main()
