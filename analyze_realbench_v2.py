"""Analysis of the decisive real-data study (corrected protocol).

Answers, with two-sided tests and teacher-quality gating:
  A. Does SHELL placement beat OTHER off-manifold placements (jitter, box),
     not just normals-only? (the straw-man test)
  B. Per teacher family: is the shell lift significant across datasets?
  C. What predicts the lift -- coverage or teacher radiality? Does the
     coverage correlation survive dropping the shuttle leverage point?
  D. Which placement wins each cell as a function of radiality/coverage?
  E. Same story under the operational top-k agreement metric.
  F. Method improvement: does wide/multiscale shell recover the failure cells?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr

CSV = Path(sys.argv[1] if len(sys.argv) > 1 else "results_realbench_v2/results.csv")
AUROC_FLOOR = 0.70
SAMPLERS = ["normals_only", "gaussian_jitter", "uniform_box",
            "uniform_shell", "wide_shell", "multiscale_shell"]


def cell_means(df, metric="fid"):
    """Mean over seeds -> one row per (dataset, teacher), columns per sampler."""
    g = (df.groupby(["dataset", "teacher", "sampler"])[metric].mean()
           .unstack("sampler"))
    meta = (df.groupby(["dataset", "teacher"])[["teacher_auroc", "coverage",
            "coverage_wide", "radiality", "d"]].mean())
    return g.join(meta).reset_index()


def paired(a, b):
    """Two-sided paired Wilcoxon; returns (mean_diff, p, n, wins)."""
    d = (a - b).dropna().values
    if len(d) < 3 or np.allclose(d, 0):
        return float(np.mean(d)) if len(d) else np.nan, np.nan, len(d), int((d > 0).sum())
    try:
        p = wilcoxon(d, alternative="two-sided").pvalue
    except Exception:
        p = np.nan
    return float(np.mean(d)), p, len(d), int((d > 0).sum())


def report(metric="fid"):
    df = pd.read_csv(CSV)
    cm = cell_means(df, metric)
    gated = cm[cm.teacher_auroc >= AUROC_FLOOR].copy()
    print(f"\n{'='*70}\nMETRIC: {metric}   (cells={len(cm)}, gated AUROC>= {AUROC_FLOOR}: {len(gated)})\n{'='*70}")

    print("\n--- A. Shell vs other placements (gated cells, two-sided paired Wilcoxon) ---")
    for base in ["normals_only", "gaussian_jitter", "uniform_box"]:
        md, p, n, w = paired(gated["uniform_shell"], gated[base])
        star = "  *" if (p is not None and p < 0.05) else ""
        print(f"  uniform_shell - {base:<16} mean={md:+.3f}  p={p:.4f}  n={n}  wins={w}/{n}{star}")
    print("  (method improvements vs plain shell:)")
    for imp in ["wide_shell", "multiscale_shell", "shell_box_mix"]:
        md, p, n, w = paired(gated[imp], gated["uniform_shell"])
        star = "  *" if (p is not None and p < 0.05) else ""
        print(f"  {imp:<16} - uniform_shell  mean={md:+.3f}  p={p:.4f}  n={n}  wins={w}/{n}{star}")

    print("\n--- B. Per teacher family: best-shell(=max of shell variants) vs normals_only ---")
    for t in sorted(gated.teacher.unique()):
        sub = gated[gated.teacher == t]
        best_shell = sub[["uniform_shell", "wide_shell", "multiscale_shell"]].max(1)
        md, p, n, w = paired(best_shell, sub["normals_only"])
        star = "  *" if (p is not None and p < 0.05) else ""
        print(f"  {t:<8} shell-best - none  mean={md:+.3f}  p={p:.4f}  n={n}  wins={w}/{n}{star}")

    print("\n--- C. What predicts the shell lift? (gated) ---")
    lift = (gated["uniform_shell"] - gated["normals_only"])
    for cov_name in ["coverage", "coverage_wide", "radiality"]:
        r, p = spearmanr(gated[cov_name], lift)
        print(f"  corr({cov_name:<14}, shell_lift) = {r:+.3f}  p={p:.4f}")
    # drop the extreme leverage points (shuttle) and recheck coverage
    for drop in ["32_shuttle"]:
        sub = gated[gated.dataset != drop]
        l2 = sub["uniform_shell"] - sub["normals_only"]
        r, p = spearmanr(sub["coverage"], l2)
        rr, pr = spearmanr(sub["radiality"], l2)
        print(f"  [drop {drop}] corr(coverage,lift)={r:+.3f} p={p:.4f} | "
              f"corr(radiality,lift)={rr:+.3f} p={pr:.4f}  (n={len(sub)})")

    print("\n--- D. Which placement wins each gated cell? (count of argmax over samplers) ---")
    cols = ["normals_only", "gaussian_jitter", "uniform_box", "uniform_shell",
            "wide_shell", "multiscale_shell"]
    win = gated[cols].idxmax(1).value_counts()
    for k in cols:
        print(f"  {k:<18} wins {int(win.get(k,0)):>3} / {len(gated)} cells")
    # split by radiality
    hi = gated[gated.radiality > 0.6]; lo = gated[gated.radiality <= 0.6]
    for label, sub in [("radiality>0.6 (radial teachers)", hi),
                       ("radiality<=0.6 (structured teachers)", lo)]:
        if len(sub):
            w = sub[cols].idxmax(1).value_counts()
            top = ", ".join(f"{k}:{int(w.get(k,0))}" for k in cols if w.get(k, 0) > 0)
            print(f"    {label} (n={len(sub)}): {top}")


def perdataset():
    df = pd.read_csv(CSV)
    cm = cell_means(df, "fid")
    print(f"\n{'='*70}\nPER-CELL DETAIL (fid means; * = teacher AUROC < {AUROC_FLOOR}, excluded)\n{'='*70}")
    print(f"  {'dataset':<18}{'teach':<8}{'Tauroc':>7}{'radial':>7}{'cov':>6}"
          f"{'none':>7}{'jit':>7}{'box':>7}{'shell':>7}{'wide':>7}")
    for _, r in cm.sort_values(["teacher", "radiality"]).iterrows():
        flag = "" if r.teacher_auroc >= AUROC_FLOOR else " *"
        print(f"  {r.dataset:<18}{r.teacher:<8}{r.teacher_auroc:>7.3f}{r.radiality:>7.2f}"
              f"{r.coverage:>6.2f}{r.normals_only:>7.2f}{r.gaussian_jitter:>7.2f}"
              f"{r.uniform_box:>7.2f}{r.uniform_shell:>7.2f}{r.wide_shell:>7.2f}{flag}")


if __name__ == "__main__":
    report("fid")
    report("topk")
    perdataset()
