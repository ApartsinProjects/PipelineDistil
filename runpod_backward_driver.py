"""RunPod driver: download TSB-AD-M on the pod, run a tiny code-path check, then
the backward-probing smoke. CPU torch work; GPU only satisfies the bootstrap."""
from __future__ import annotations
import os, sys, subprocess, time
from pathlib import Path

HERE = Path(__file__).parent.resolve()
RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)
DATA = HERE / "tsbad"


def sh(cmd):
    print(f"[train] $ {cmd}", flush=True); return subprocess.call(cmd, shell=True)


def main():
    print(f"[train] === backward-probing smoke === cpus={os.cpu_count()}", flush=True)
    sh(f"{sys.executable} -m pip install -q pandas scipy 2>/dev/null")
    DATA.mkdir(exist_ok=True)
    if not list(DATA.rglob("*.csv")):
        z = DATA / "m.zip"
        rc = sh(f"curl -sSL --retry 3 -o {z} https://www.thedatum.org/datasets/TSB-AD-M.zip")
        size = z.stat().st_size if z.exists() else 0
        print(f"[train] download rc={rc} size={size} bytes", flush=True)
        import zipfile                                  # the image has no `unzip` binary
        with zipfile.ZipFile(z) as zf:
            zf.extractall(DATA)
        z.unlink()
    n = len(list(DATA.rglob("*.csv")))
    print(f"[train] TSB-AD-M csv files: {n}", flush=True)
    if n == 0:
        raise SystemExit(2)
    script = HERE / "experiment_backward_smoke.py"
    t0 = time.time()
    rc = sh(f"{sys.executable} -u {script} --data {DATA} --tiny --outdir {RESULTS/'tiny'}")
    print(f"[train] tiny rc={rc} in {time.time()-t0:.0f}s", flush=True)
    if rc != 0:
        print("[train] tiny check FAILED, aborting before the full run", flush=True)
        raise SystemExit(rc)
    t0 = time.time()
    rc = sh(f"{sys.executable} -u {script} --data {DATA} --series 20 --seeds 3 --workers 14 "
            f"--outdir {RESULTS/'backward_smoke'}")
    print(f"[train] smoke rc={rc} in {time.time()-t0:.0f}s", flush=True)
    print("[train] === DONE ===", flush=True)


if __name__ == "__main__":
    main()
