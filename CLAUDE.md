# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project overview

This repo studies SODCL / Seeking the Unicorn through graph theory, but the working thesis is now:

- the square king-grid `P_8 ⊠ P_8` has a degenerate Fiedler eigenspace,
- successful dyads often break that symmetry into stable axial roles,
- and an early geometric signal helps predict later role specialization.

Do not frame the project as:

- "humans beat Fiedler"
- "conductance replaces the official metrics"
- or "the current analysis explains all focal strategies in the original paper"

## Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Interactive menu
./run.sh

# Main scripts
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

# Compile the second-delivery draft
cd paper && pdflatex entrega2.tex && pdflatex entrega2.tex && cd ..
```

## Data split

- `data/raw/humans_only_absent.csv`
  - absent rounds only
  - used for the spectral and early-prediction analyses
- `data/raw/performances.csv`
  - full 60-round dataset
  - used for transfer-to-performance summaries

## Core analysis structure

- `src/02_full_comparison.py`
  - audits all 45 dyads,
  - defines the primary conductance set,
  - compares `h(S_obs)` against naive Fiedler and axis-aligned baselines,
  - keeps `DLIndex`, `Similarity`, `Consistency`, `Joint`, and `Size_visited`.
- `src/06_partition_robustness.py`
  - derives stable late orientation from absent rounds.
- `src/07_entropy_analysis.py`
  - reports MI/JSD and builds the Andrade summary figure.
- `src/08_early_prediction.py`
  - builds the early geometric signal from the first 5 common absent rounds,
  - compares it against early official metrics,
  - summarizes transfer to full-task performance,
  - reports the incremental-value diagnostic for late `h_obs`.

## Important outputs

- `data/results/spectral_analysis_audit.csv`
- `data/results/spectral_comparison_results.csv`
- `data/results/partition_stability_summary.csv`
- `data/results/early_prediction_features.csv`
- `data/results/early_prediction_summary.csv`
- `data/results/performance_transfer_summary.csv`
- `data/results/present_performance_increment.csv`
- `figures/andrade_summary.png`
- `figures/early_prediction_summary.png`
- `paper/entrega2.tex`

## Conventions

- Code comments must be in Spanish without accents.
- Paths must remain relative to the project root.
- Keep the denominator explicit:
  - absent-only analyses are absent-only analyses,
  - do not write about "60 rounds per player" when using `humans_only_absent.csv`.
- Prefer explicit audit trails over silent filtering.
- When writing results, the correct thesis is:
  - symmetry breaking,
  - stable axial specialization,
  - early prediction,
  - and cautious comparison against official metrics.
