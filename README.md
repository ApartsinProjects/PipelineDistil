# PipelineDistil

**Where to Query a Blackbox Anomaly Pipeline: Shell Sampling for Label-Free Distillation.**

- 📄 Paper (live): https://apartsinprojects.github.io/PipelineDistil/
- 📄 [paper_shell_sampling.md](paper_shell_sampling.md) (Markdown source)
- 📄 [docs/paper.pdf](docs/paper.pdf) | [docs/paper.docx](docs/paper.docx)

## What this is

Distill a complicated, blackbox, unsupervised anomaly-detection **pipeline**
(preprocessing, one or more detectors, and a fusion step) into a single small
student model for resource-constrained deployment. The pipeline is only
queryable, its internals are opaque, and there are **no labeled anomalies** at
any point: only normal operating samples. The student must therefore learn the
pipeline's behavior on the anomalous region from synthetic query points we place
there and label with the pipeline's own scores.

The paper's question is **where a fixed budget of synthetic queries should go**,
and its answer:

- The informative queries lie in the **low-density shell** just off the normal
  manifold (not on the manifold, not in the saturated far field).
- Within the shell, the right target follows the pipeline's off-manifold
  **shape**: for *growing* detectors (autoencoder reconstruction error) sample
  toward high score; for *saturating* detectors (one-class SVM) sample where the
  score has a large gradient.
- A single shell sampler weighted by **both** score and score-gradient magnitude
  is automatically shape-robust and needs no manual choice or anomaly labels. A
  cheap growth-signature probe matches but does not beat it, and flags when a
  pipeline is beyond a small student's reach.

Success is measured as student-teacher **fidelity** on held-out anomalies (rank
agreement), not standalone detection accuracy.

## Headline results

- **Teacher-pipeline spectrum (fidelity, Spearman).** Every shell sampler lifts
  fidelity from the normals-only baseline (~0.2-0.3) to 0.7-0.97 on distillable
  pipelines. The combined sampler wins the hard one-class SVM (0.708) and ties
  the best elsewhere; score-only wins the autoencoder (0.911), gradient-only
  wins one-class SVM (0.562) - the shape dependence, measured.
- **Mechanism (complex 3-step pipeline, 5 seeds).** Normals-only reproduces the
  pipeline on anomalies at 0.245; the adaptive shell sampler reaches 0.939.
- **Honest limits.** Discontinuous (max-fusion) pipelines are undistillable by a
  small student regardless of sampling; real high-dimensional fine-fidelity is an
  open problem (coarse detection there needs no sampling).

## Reproduce

CPU-only. No cloud, no GPU.

```bash
python experiment_edge.py --seeds 3         # teacher-pipeline spectrum (shell samplers)
python experiment_complex.py --seeds 5      # complex-pipeline mechanism study
python experiment_spectrum.py --seeds 5     # growth-signature / shape index
python experiment_real.py --datasets shuttle satellite --seeds 5   # real-data scope
python build_paper.py                       # Markdown -> styled HTML + DOCX
python build_artifact.py                    # self-contained HTML (MathML + inlined figures)
```

## Files

- [experiment.py](experiment.py) — core: samplers, teachers, extended-percentile map, student, batched finite-difference gradient
- [experiment_edge.py](experiment_edge.py) — shell samplers (score / gradient / combined), growth-signature probe, spectrum comparison
- [experiment_complex.py](experiment_complex.py) — complex 3-step pipeline teacher + mechanism figure
- [experiment_spectrum.py](experiment_spectrum.py) — teacher spectrum + non-monotonicity / growth index
- [experiment_real.py](experiment_real.py) — real ADBench tabular datasets
- [paper_shell_sampling.md](paper_shell_sampling.md) — full paper source
- [build_paper.py](build_paper.py) — Markdown -> styled HTML + single-column DOCX (with deck-term canary)
- [docs/](docs) — GitHub Pages source (rendered paper + DOCX + PDF + figures)

## Dependencies

Python 3.10+, `numpy scipy scikit-learn matplotlib`; plus `python-markdown
python-docx pypandoc pywin32 pymupdf` and Node + `katex` for the paper build.

## License

MIT.
