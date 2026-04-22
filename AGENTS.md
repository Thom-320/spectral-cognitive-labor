# AGENTS.md

This file provides guidance to Codex when working in this repository.

## Project overview

Research/coursework project on SODCL / Seeking the Unicorn using graph theory.

Working thesis:

- `P_8 ⊠ P_8` has a degenerate Fiedler eigenspace.
- That degeneracy makes the naive Fiedler cut basis-dependent.
- Several human dyads break that symmetry into stable axial roles.
- An early geometric signal from the first absent rounds helps predict that later stabilization.

Non-thesis:

- Do not frame this as proof that humans are generic spectral optimizers.
- Do not imply that conductance replaces `DLIndex`, `Similarity`, or `Consistency`.
- Do not imply that the current lens fully explains `ALL/NOTHING/RS` or the whole original phenomenon.

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
./run.sh

python src/00_spectral_grid.py
python src/01_single_dyad_analysis.py
python src/02_full_comparison.py
python src/03_power_iteration.py
python src/03_power_iteration_fiedler.py
python src/04_temporal_dynamics.py
python src/05_counterexample_P6xP8.py
python src/06_partition_robustness.py
python src/07_entropy_analysis.py
python src/08_early_prediction.py
```

## Pipeline notes

- `data/raw/humans_only_absent.csv`
  - primary absent-only dataset
  - used for the spectral, robustness, entropy, and early-prediction analyses
- `data/raw/performances.csv`
  - full 60-round dataset
  - used for transfer-to-performance summaries

- `src/02_full_comparison.py` produces:
  - `data/results/spectral_analysis_audit.csv`
  - `data/results/spectral_comparison_results.csv`
- `src/06_partition_robustness.py` produces:
  - `data/results/partition_stability_summary.csv`
- `src/08_early_prediction.py` produces:
  - `data/results/early_prediction_features.csv`
  - `data/results/early_prediction_summary.csv`
  - `data/results/performance_transfer_summary.csv`
  - `data/results/present_performance_increment.csv`

## Conventions

- Comments in code must be in Spanish without accents.
- Use relative paths from the project root.
- Keep auditability explicit:
  - no silent exclusion of dyads,
  - document reasons in outputs and docs.
- Current repo-level deliverables that matter:
  - `docs/RESULTADOS_PRELIMINARES.md`
  - `docs/EXCLUSIONES.md`
  - `docs/NOTA_ANDRADE.md`
  - `paper/entrega2.tex`
  - `paper/preprint_base.tex`
  - `figures/early_prediction_summary.png`
