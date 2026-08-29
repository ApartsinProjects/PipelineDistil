"""pipedistil-sweep: run the higher-D sampler sweep on Modal.

Fans out 20 containers ((d, seed) in {5, 10} x {0..9}), each running the six
samplers (S0-S5) once at that (d, seed), then aggregates results locally.

Usage:
    modal run modal_pipedistil.py                    # d=5,10 x seeds 0..9
    modal run modal_pipedistil.py --dims 5,10 --seeds 0,1,2
"""
from __future__ import annotations
import csv
import io
import json
import time
from pathlib import Path

import modal

HERE         = Path(__file__).parent.resolve()
RESULTS_DIR  = HERE / "results_highd"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Image: slim CPU-only python + our two source files
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "scipy==1.17.1", "scikit-learn==1.8.0")
    .add_local_file(str(HERE / "experiment.py"), "/app/experiment.py")
    .add_local_file(str(HERE / "experiment_highd.py"), "/app/experiment_highd.py")
)

app = modal.App("pipedistil-sweep", image=image)

results_vol = modal.Volume.from_name("pipedistil-results", create_if_missing=True)


# ---------------------------------------------------------------------------
# Remote worker: one (d, seed) unit, all six samplers
# ---------------------------------------------------------------------------
@app.function(
    timeout=1800,       # 30 min per shard
    memory=4096,
    cpu=2.0,
    volumes={"/results": results_vol},
)
def run_shard(d: int, seed: int) -> list[dict]:
    """Run all six SAMPLERS at one (d, seed). Return per-metric rows."""
    import sys, time
    sys.path.insert(0, "/app")

    import numpy as np
    from experiment import (
        SAMPLERS, fit_teachers, score_batch, normalize_stats,
        make_percentile_maps, sampler_none, sampler_langevin,
    )
    from experiment_highd import (
        gen_normals_gmm, gen_anomalies_highd,
        make_d_dim_uniform_sampler, make_d_dim_mixed_sampler,
        run_one_dsamp, N_TRAIN, N_VAL,
    )

    t_total = time.time()
    X_train = gen_normals_gmm(N_TRAIN, d, seed=0)  # data seed 0, fixed across seeds
    X_val   = gen_normals_gmm(N_VAL,   d, seed=1)
    anoms   = gen_anomalies_highd(X_train, d, seed=0)

    teachers, kde = fit_teachers(X_train, seed=0)
    S_train = score_batch(teachers, X_train)
    mu, sigma = normalize_stats(S_train)
    pct_map = make_percentile_maps(S_train)

    # Per-d uniform + mixed samplers (bounding box from X_train)
    u_fn = make_d_dim_uniform_sampler(X_train)
    m_fn = make_d_dim_mixed_sampler(u_fn)
    local_samplers = dict(SAMPLERS)
    local_samplers["S2_uniform"] = u_fn
    local_samplers["S4_mixed"]   = m_fn

    rows = []
    for cond in local_samplers:
        t0 = time.time()
        _, _, m = run_one_dsamp(
            seed, cond, X_train, X_val, anoms,
            teachers, mu, sigma, kde, pct_map,
            local_samplers, M=2000,
        )
        wall = time.time() - t0
        for metric, value in m.items():
            rows.append({"d": d, "seed": seed, "cond": cond,
                         "metric": metric, "value": float(value)})
        rows.append({"d": d, "seed": seed, "cond": cond,
                     "metric": "wall_s", "value": float(wall)})
        print(f"[d={d} seed={seed}] {cond}: {wall:.1f}s", flush=True)

    # Persist per-shard file too so a failure is recoverable.
    shard_path = Path(f"/results/d{d}_seed{seed}.json")
    shard_path.write_text(json.dumps(rows))
    results_vol.commit()

    print(f"[d={d} seed={seed}] DONE, {len(rows)} rows, "
          f"total {time.time() - t_total:.1f}s", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Local entrypoint: fan out over (d, seed) pairs, aggregate to CSV
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(dims: str = "5,10", seeds: str = "0,1,2,3,4,5,6,7,8,9"):
    dim_list  = [int(s) for s in dims.split(",")]
    seed_list = [int(s) for s in seeds.split(",")]
    shards    = [(d, s) for d in dim_list for s in seed_list]
    print(f"[main] {len(shards)} shards: dims={dim_list} seeds={seed_list}", flush=True)

    t0 = time.time()
    all_rows: list[dict] = []
    for shard_rows in run_shard.starmap(shards):
        if shard_rows:
            all_rows.extend(shard_rows)
    print(f"[main] all shards complete in {time.time() - t0:.1f}s; "
          f"{len(all_rows)} rows total", flush=True)

    # Write aggregated CSV in the same format experiment_highd.py uses.
    out_csv = RESULTS_DIR / "results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["d", "seed", "cond", "metric", "value"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"[main] wrote {out_csv} ({len(all_rows)} rows)", flush=True)
