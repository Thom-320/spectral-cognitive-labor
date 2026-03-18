# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Academic research project analyzing whether humans are intuitive spectral optimizers. Compares human-generated partitions from the SODCL experiment ("Seeking the Unicorn", Andrade-Lotero & Goldstone, PLOS ONE 2021) against Fiedler bisection using conductance as the metric. The core graph is an 8x8 grid with 8-connectivity (64 nodes, 210 edges).

**Key finding:** lambda2 has multiplicity 2 in the 8x8 square grid (2D Fiedler eigenspace). Axis-aligned cuts (LR/TB) achieve h=22/210, beating the naive Fiedler median cut (h=28/210). Human dyads with clear spatial splits converge to these axis-aligned optima.

**Author:** Thomas Chisica - Universidad del Rosario, Teoria de Grafos 2026-I

## Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Interactive menu (handles venv automatically)
./run.sh

# Run individual scripts (venv must be active)
python src/00_spectral_grid.py            # Base spectral analysis -> spectral_results.npz
python src/01_single_dyad_analysis.py     # Single dyad example (requires spectral_results.npz)
python src/02_full_comparison.py          # All 29 dyads comparison
python src/03_power_iteration.py          # Power iteration with axis-aligned bias (pedagogical)
python src/03_power_iteration_fiedler.py  # Fiedler vector via power iteration + deflation
python src/04_temporal_dynamics.py        # Temporal h(S) tracking over 60 rounds
python src/05_counterexample_P6xP8.py    # Rectangular grid counterexample (no degeneracy)
```

Scripts must run from the project root directory. Script 00 must run before 01/02 (produces `spectral_results.npz`). Scripts 03-05 are standalone.

## Architecture

**Pipeline:** Scripts are numbered and sequential. Each reads from `data/raw/` or `data/results/`, produces outputs to `data/results/` (NPZ/CSV) and `figures/` (PNG).

**Shared computational patterns across scripts (not factored into a shared module):**
- `build_grid_graph(n=8)` - 8x8 grid with 8-connectivity (king moves)
- `compute_laplacian(A)` - Combinatorial Laplacian L = D - A
- `spectral_analysis(L)` - Eigendecomposition via `scipy.linalg.eigh`
- `conductance(S, A, degrees)` - h(S) = cut(S,S') / min(vol(S), vol(S'))
- `fiedler_bisection()` - Partition by Fiedler vector (2nd eigenvector)
- `is_connected(S, A)` - BFS connectivity check

Each script redefines these functions locally. If adding a new script, copy the pattern from an existing one.

**Key metric:** eta = h(S_Fiedler) / h(S_obs) -- efficiency ratio comparing spectral optimum to human partition.

**Data flow:** `humans_only_absent.csv` (raw experiment data) -> scripts extract visit frequencies per cell per dyad -> derive observed partition S_obs -> compare against Fiedler bisection.

**Root path convention:** Scripts use `ROOT = Path(__file__).parent.parent` to resolve paths relative to project root.

## Conventions

- All code comments are in **Spanish without accent marks** (tildes)
- All file paths must be **relative** to project root (no absolute paths -- this was a past bug, see `docs/CORRECCIONES.md`)
- No test framework; validation is through figure inspection and CSV output
- No linter configured
- LaTeX manuscript uses IEEEtran class (`paper/` directory)
