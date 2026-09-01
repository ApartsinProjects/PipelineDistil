"""RunPod driver: run the multi-output pipeline-distillation studies on the pod's
many vCPUs. CPU-only sklearn/numpy work; the pod GPU is unused (only satisfies the
bootstrap CUDA assert). Unpacks the bundle, ensures data/adbench/ exists, then runs
the three studies with high worker counts, writing every CSV under results/.
"""
from __future__ import annotations
import os, sys, subprocess, shutil, glob, time
from pathlib import Path

HERE = Path(__file__).parent.resolve()
RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)


def sh(cmd):
    print(f"[train] $ {cmd}", flush=True)
    return subprocess.call(cmd, shell=True)


def setup():
    # unpack bundle if present
    for tgz in glob.glob(str(HERE / "bundle.tar.gz")):
        sh(f"tar xzf {tgz} -C {HERE}")
    # make sure deps exist on the pod
    sh(f"{sys.executable} -m pip install -q scikit-learn scipy pandas numpy 2>/dev/null")
    # data path: scripts expect ./data/adbench/*.npz relative to their own dir
    dd = HERE / "data" / "adbench"; dd.mkdir(parents=True, exist_ok=True)
    moved = 0
    for npz in glob.glob(str(HERE / "*.npz")):
        shutil.move(npz, str(dd / Path(npz).name)); moved += 1
    n = len(list(dd.glob("*.npz")))
    print(f"[train] data ready: {n} npz in {dd} (moved {moved})", flush=True)
    return n


def run_study(script, outdir, extra=""):
    workers = max(2, (os.cpu_count() or 4) - 2)
    out = RESULTS / outdir
    cmd = (f"{sys.executable} -u {HERE/script} --seeds 10 --workers {workers} "
           f"--outdir {out} {extra}")
    t0 = time.time()
    rc = sh(cmd)
    print(f"[train] {script} rc={rc} in {time.time()-t0:.0f}s -> {out}", flush=True)
    return rc


def main():
    print(f"[train] === pipeline distillation studies === cpus={os.cpu_count()}", flush=True)
    n = setup()
    assert n >= 10, f"only {n} datasets found; upload failed"
    # core thesis first (so a time cap still yields it), then the supporting studies
    print("[train] STUDY 1/3: residual-attribution pipeline (core)", flush=True)
    run_study("experiment_pipeline_scenario.py", "pipeline_scenario")
    print("[train] STUDY 2/3: single-detector placement straw-man", flush=True)
    run_study("experiment_realbench_v2.py", "realbench_v2")
    print("[train] STUDY 3/3: multi-head ensemble distillation", flush=True)
    run_study("experiment_multioutput.py", "multioutput")
    print("[train] === DONE ===", flush=True)


if __name__ == "__main__":
    main()
