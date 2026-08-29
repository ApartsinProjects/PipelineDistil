"""RunPod driver: fan out real-data shards across pod vCPUs.

Runs all (dataset, seed) shards concurrently via a multiprocessing pool,
writes per-shard JSON to /tmp, aggregates to results/results.csv.
Uses [train] output convention so the runner's monitor picks up progress.
"""
from __future__ import annotations

import os

# Pin BLAS/OMP to a single thread PER WORKER, before numpy imports. With one
# multiprocessing worker per CPU and each sklearn/BLAS call otherwise spawning
# threads for every core, 15 workers x N-core BLAS oversubscribes badly (2-5x
# slowdown from cache thrash). Single-thread-per-worker is also the numerical
# reference. Must be set before the first numpy import (fork workers inherit
# the parent's already-initialized BLAS, so an initializer would be too late).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import json
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# Make our source dir importable and results/ writable.
HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

# The RunPod runner uploads files flat to /root/data/, but
# experiment_real.py expects them at HERE/data/adbench/. Set up the expected
# layout by searching wherever the runner put them.
_DATA_TARGET = HERE / "data" / "adbench"
_DATA_TARGET.mkdir(parents=True, exist_ok=True)
for _name in ["32_shuttle.npz", "30_satellite.npz", "26_optdigits.npz"]:
    if (_DATA_TARGET / _name).exists():
        continue
    for _src in [HERE / _name, *HERE.rglob(_name)]:
        if _src.exists() and _src.resolve() != (_DATA_TARGET / _name).resolve():
            shutil.copy(_src, _DATA_TARGET / _name)
            print(f"[setup] linked {_src} -> {_DATA_TARGET / _name}", flush=True)
            break
    if not (_DATA_TARGET / _name).exists():
        print(f"[setup] WARNING: {_name} not found under {HERE}", flush=True)

DATASETS = ["shuttle", "satellite", "optdigits"]
SEEDS = list(range(5))
M_QUERIES = 1500


def _worker(shard):
    """One (dataset, seed) shard. Returns per-metric rows."""
    dataset, seed = shard
    from experiment_real import run_shard  # imported per-worker for cleanliness
    t = time.time()
    rows = run_shard(dataset, seed, M=M_QUERIES)
    wall = time.time() - t
    (RESULTS / f"shard_{dataset}_seed{seed}.json").write_text(json.dumps(rows))
    print(f"[train]   done {dataset} seed={seed}: {len(rows)} rows in {wall:.1f}s", flush=True)
    return rows


def main():
    shards = [(d, s) for d in DATASETS for s in SEEDS]
    n_workers = min(len(shards), os.cpu_count() or 4)
    print(f"[train] GPU: (CPU-only workload, no GPU used)", flush=True)
    print(f"[train] Loading model...", flush=True)
    print(f"[train] Model loaded: PipeDistil-real (samplers x {len(shards)} shards)", flush=True)
    print(f"[train] Starting work...", flush=True)
    print(f"[train]   dispatching {len(shards)} shards over {n_workers} workers", flush=True)

    t0 = time.time()
    all_rows: list[dict] = []
    with mp.Pool(processes=n_workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(_worker, shards), start=1):
            all_rows.extend(rows)
            print(f"[train]   {i}/{len(shards)} shard complete "
                  f"(elapsed {time.time() - t0:.0f}s, rows={len(all_rows)})", flush=True)

    out_csv = RESULTS / "results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "d", "seed", "cond", "metric", "value"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"[train] wrote {out_csv} ({len(all_rows)} rows) in {time.time() - t0:.1f}s", flush=True)
    print(f"[train] === DONE ===", flush=True)


if __name__ == "__main__":
    main()
