# PipelineDistil

**Uncertainty-Guided Sampling for Unsupervised Distillation of Blackbox Anomaly-Detection Pipelines.**

- 📄 Paper (live): https://apartsinprojects.github.io/PipelineDistil/
- 📄 [paper_uncertainty_sag_kd.md](paper_uncertainty_sag_kd.md) (Markdown source)
- 📄 [docs/paper.pdf](docs/paper.pdf) | [docs/paper.docx](docs/paper.docx)

## What this is

Compress one or more blackbox unsupervised anomaly-detection **pipelines**
(preprocessing + detector + postprocessing) into a small student model deployable
under tight compute budgets. Teachers are only queryable; no labeled anomalies
are available at any point in training. The distillation objective is a plain
regression on teacher scores — the interesting problem is *where* to place the
synthetic training queries.

The paper proposes **uncertainty-guided sampling** (a Langevin walk climbing an
inter-teacher percentile-disagreement potential plus a drift toward the anomalous
side), compares it against normals-only, Gaussian jitter, uniform, and a mixed
half-uncertainty-half-uniform sampler, and stress-tests the coverage argument
across `d ∈ {2, 5, 10}` on synthetic benchmarks with three unsupervised teachers
(KDE, Isolation Forest, kNN-distance).

## Headline results

- **2-D two-moons, 10 seeds.** Uncertainty-guided S3 wins close AUROC
  (0.987 ± 0.004) and medium (0.986 ± 0.007); best off-manifold score-fidelity
  RMSE on boundary/close/medium; best rank calibration on the hardest set.
- **d = 10, 3 seeds.** S3 wins **all four anomaly bands**
  (boundary 0.773, close 0.919, medium 0.965, far 0.970). Uniform augmentation
  collapses as predicted by the coverage argument (boundary AUROC drops from
  0.92 at d = 2 to 0.65 at d = 10).
- The mixed sampler S4 (half uncertainty-guided, half uniform) is a robust
  runner-up at every d.

## Reproduce

Everything runs on a single CPU. No cloud, no GPU.

```bash
# 2-D sweep (10 seeds, 4 conditions, ~7 min)
python experiment.py --seeds 10

# Higher-D stress test (d=5 and d=10, 3 seeds each, ~30 min)
python experiment_highd.py --dims 5 10 --seeds 3

# Rebuild paper (Markdown -> HTML + DOCX with native OMML equations)
python build_paper.py

# Rebuild the standalone HTML for hosting (pre-renders math to MathML,
# inlines figures as data: URIs)
python build_artifact.py
```

## Files

- [experiment.py](experiment.py) — 2-D two-moons sweep, sanity invariants, main figure
- [experiment_highd.py](experiment_highd.py) — d=5, d=10 stress test on a 3-cluster GMM
- [paper_uncertainty_sag_kd.md](paper_uncertainty_sag_kd.md) — full paper source (Markdown + LaTeX math)
- [build_paper.py](build_paper.py) — one-command Markdown → styled HTML + single-column DOCX
- [build_artifact.py](build_artifact.py) — HTML → self-contained artifact (MathML inline, images as data: URIs)
- [results/results.csv](results/results.csv) — 2-D per-seed × per-condition × per-metric
- [results_highd/results.csv](results_highd/results.csv) — d=5, d=10 per-seed × per-condition × per-metric
- [docs/](docs) — GitHub Pages source (rendered paper + DOCX + PDF + figures)

## Dependencies

- Python 3.10+
- `numpy`, `scipy`, `scikit-learn`, `matplotlib` for the experiments
- `python-markdown`, `python-docx`, `pypandoc`, `pywin32`, `pymupdf` for the paper build
- Node.js + `katex` for the math → MathML pre-render (only needed for `build_paper.py` and `build_artifact.py`)
- Microsoft Word (for the DOCX → PDF render-verify step)

## License

MIT.
