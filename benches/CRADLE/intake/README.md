# Chora/Cora: A Multi-Agent Concept Space Architecture

## Overview

This repository contains the research paper and source files for **Chora/Cora**, a multi-agent collaboration framework grounded in a dual-layer concept space architecture.

**Paper Title:** Chora/Cora: A Multi-Agent Concept Space Architecture Based on Gaussian Process Mean-Shift Dynamics

## Architecture Summary

Chora/Cora introduces a dual-layer concept space architecture:

- **The Oracle** — A shared common-sense concept space (collective consciousness center)
- **The Maze** — Individualized self-concept spaces (personalized agent identity)

Each agent maintains a Gaussian process posterior over a high-dimensional concept manifold, with its self-identity formalized as the mean of a consciousness distribution. Inter-agent coordination emerges through Bayesian conditioning on shared inducing points, while concept dimensions spontaneously emerge from interaction dynamics.

## Files

- `ms.tex` — LaTeX source of the paper
- `ChoraCora_paper.pdf` — Pre-compiled PDF of the paper
- `.github/workflows/build-paper.yml` — GitHub Actions workflow for automatic compilation

## Building the Paper

### Option 1: GitHub Actions (Recommended)

Push to `main` or `master` branch. The GitHub Actions workflow will automatically compile `ms.tex` and generate `ChoraCora_paper.pdf` as a build artifact. You can also enable GitHub Pages deployment via the workflow.

### Option 2: Local Compilation

```bash
# Using Overleaf (recommended for collaboration)
# Upload ms.tex and any supporting files to Overleaf and compile

# Or locally with TeX Live:
pdflatex ms.tex
bibtex ms.aux  # if using BibTeX
pdflatex ms.tex
pdflatex ms.tex  # run twice for references
```

### Option 3: Using the pre-compiled PDF

Simply download `ChoraCora_paper.pdf` from the repository releases or artifacts.

## Theoretical Foundations

This framework is grounded in five theoretical pillars:

1. **GP + Multi-Agent** — Decentralized Gaussian process learning (Kontoudis & Stilwell, 2025; Viseras et al., 2016)
2. **Conceptual Spaces** — Abstract coordinate systems for concept representation (Harvard, 2024; Subjective Conceptual Spaces survey, 2026)
3. **Dimension Emergence** — Form Transformation Theory theorems (Dimension Complementarity & Full Spectrum Suppression)
4. **Mean-Shift Convergence** — Rigorous convergence proofs (Fukunaga & Hostetler, 1975)
5. **Multi-Agent Coordination Spaces** — Joint probability framework (Sterling et al., 1998)

## License

This work is shared as open research. Feel free to fork, cite, and build upon it.

## Citation

```
@misc{choracora2026,
  title={Chora/Cora: A Multi-Agent Concept Space Architecture Based on Gaussian Process Mean-Shift Dynamics},
  author={Independent Researcher},
  year={2026},
  url={https://github.com/[YOUR_USERNAME]/[YOUR_REPO]}
}
```
