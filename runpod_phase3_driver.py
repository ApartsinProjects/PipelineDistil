"""RunPod driver for Phase 1 floor map (CPU-only sklearn; GPU unused, only satisfies
the bootstrap CUDA assert). Unpacks the bundle, ensures data/adbench/, runs
experiment_phase3.py across the pod's many vCPUs with incremental+resumable CSV.
"""
from __future__ import annotations
import os, sys, subprocess, shutil, glob, time
from pathlib import Path

HERE = Path(__file__).parent.resolve()
RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)


def sh(cmd):
    print(f"[train] $ {cmd}", flush=True); return subprocess.call(cmd, shell=True)


def main():
    print(f"[train] === Phase-3 (E3 properly trained + strong surrogates) === cpus={os.cpu_count()}", flush=True)
    for tgz in glob.glob(str(HERE / "bundle.tar.gz")):
        sh(f"tar xzf {tgz} -C {HERE}")
    sh(f"{sys.executable} -m pip install -q scikit-learn scipy numpy 2>/dev/null")
    dd = HERE / "data" / "adbench"; dd.mkdir(parents=True, exist_ok=True)
    for npz in glob.glob(str(HERE / "*.npz")):
        shutil.move(npz, str(dd / Path(npz).name))
    n = len(list(dd.glob("*.npz")))
    print(f"[train] data ready: {n} npz", flush=True); assert n >= 10, "upload failed"
    workers = max(2, (os.cpu_count() or 4) - 2)
    out = RESULTS / "phase3"
    t0 = time.time()
    rc = sh(f"{sys.executable} -u {HERE/'experiment_phase3.py'} --seeds 3 --workers 24 --outdir {out}")
    print(f"[train] phase3 rc={rc} in {time.time()-t0:.0f}s -> {out}", flush=True)
    print("[train] === DONE ===", flush=True)


if __name__ == "__main__":
    main()
