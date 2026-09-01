"""Analysis of the residual-pipeline scenario study.

Central question: for a multi-output monitoring pipeline, does off-manifold
sampling matter MORE for the root-cause ATTRIBUTION output than for the aggregate
SCORE? And which sampler recovers attribution best?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

CSV = Path(sys.argv[1] if len(sys.argv) > 1 else "results_pipeline_scenario/results.csv")
METRICS = ["score_fid", "attr_top1", "attr_top3_jac", "attr_rank", "decision_f1"]
SHELL = ["uniform_shell", "disagreement_shell", "shell_box_mix"]


def paired(a, b):
    d = (np.asarray(a) - np.asarray(b))
    d = d[~np.isnan(d)]
    if len(d) < 3 or np.allclose(d, 0):
        return float(np.mean(d)) if len(d) else np.nan, np.nan, len(d), int((d > 0).sum())
    try:
        p = wilcoxon(d, alternative="two-sided").pvalue
    except Exception:
        p = np.nan
    return float(np.mean(d)), p, len(d), int((d > 0).sum())


def main():
    df = pd.read_csv(CSV)
    cell = df.groupby(["dataset", "sampler"])[METRICS].mean().reset_index()
    piv = {m: cell.pivot(index="dataset", columns="sampler", values=m) for m in METRICS}
    n = piv["score_fid"].shape[0]
    print(f"datasets={n}\n")

    print(f"{'metric':<16}{'normals':>9}{'unif_shell':>12}{'disagree':>10}{'box_mix':>9}{'best_shell':>12}")
    for m in METRICS:
        P = piv[m]
        best = P[SHELL].max(1)
        print(f"{m:<16}{P['normals_only'].mean():>9.3f}{P['uniform_shell'].mean():>12.3f}"
              f"{P['disagreement_shell'].mean():>10.3f}{P['shell_box_mix'].mean():>9.3f}{best.mean():>12.3f}")

    print("\n--- Does sampling help each output? best-shell vs normals_only (two-sided) ---")
    for m in METRICS:
        P = piv[m]; best = P[SHELL].max(1)
        md, p, k, w = paired(best, P["normals_only"])
        star = "  *" if (p is not None and p < 0.05) else ""
        print(f"  {m:<16} lift={md:+.3f}  p={p:.4f}  wins={w}/{k}{star}")

    print("\n--- KEY: does ATTRIBUTION benefit MORE than SCORE? (per-dataset lifts) ---")
    Ps, Pa = piv["score_fid"], piv["attr_top1"]
    lift_score = (Ps[SHELL].max(1) - Ps["normals_only"])
    lift_attr = (Pa[SHELL].max(1) - Pa["normals_only"])
    md, p, k, w = paired(lift_attr, lift_score)
    print(f"  mean score lift  = {lift_score.mean():+.3f}")
    print(f"  mean attr  lift  = {lift_attr.mean():+.3f}  (top-1 root cause)")
    print(f"  attr_lift - score_lift = {md:+.3f}  two-sided p={p:.4f}  (attr>score in {w}/{k} datasets)")
    print(f"  normals-only attr_top1 = {Pa['normals_only'].mean():.3f} (chance ~= 1/mean_d); "
          f"sampled = {Pa[SHELL].max(1).mean():.3f}")

    print("\n--- Which sampler wins attribution (attr_top1) per dataset? ---")
    Pa_all = cell.pivot(index="dataset", columns="sampler", values="attr_top1")
    win = Pa_all[["normals_only"] + SHELL].idxmax(1).value_counts()
    for k_ in ["normals_only"] + SHELL:
        print(f"  {k_:<20} wins {int(win.get(k_,0))}/{n}")

    print("\n--- disagreement_shell vs uniform_shell on attribution (the new sampler) ---")
    for m in ["attr_top1", "attr_top3_jac", "attr_rank"]:
        P = piv[m]; md, p, k, w = paired(P["disagreement_shell"], P["uniform_shell"])
        star = "  *" if (p is not None and p < 0.05) else ""
        print(f"  {m:<16} disagree-unif = {md:+.3f}  p={p:.4f}  wins={w}/{k}{star}")


if __name__ == "__main__":
    main()
