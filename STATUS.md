# Project status

**Last updated:** 2026-06-10

## Current scope

Reproducible graph-theoretic reanalysis of the SODCL / *Seeking the Unicorn*
experiment. Stable results, frozen pipeline (`make pipeline`), no outstanding
regressions on the late-window metrics.

## Active extensions

- **Extending to PsyNet `Coordinator_and_Foragers` data.** The current
  spectral reference is the strong-product graph `P_8 ☐ P_8`. The same
  conductance / MI / JSD family of metrics will be evaluated on the
  `Coordinator_and_Foragers` PsyNet experiment in collaboration with Prof.
  Edgar Andrade-Lotero (UC Davis), to test whether axial symmetry breaking
  in the Fiedler eigenspace generalizes beyond the 8x8 board.
- Status: scaffold in progress; first dataset pull planned for the week
  following the 2026-06-08 review.

## Open questions (tracked, not blocking)

- Whether the early geometric signal (AUC_LOOCV = 0.860 combined) is robust
  under leave-one-block-out (the current LOOCV unit is a single dyad).
- How to handle the `ALL`, `NOTHING`, and `RS` non-bipartite strategies in
  the `Coordinator_and_Foragers` setting, where mixed strategies are
  expected to be more common.
- Reproducibility of the published conductance numbers on a fresh re-run
  (audit table: `data/results/spectral_analysis_audit.csv`).

## Frozen reference numbers

| Metric | Value | Source |
| --- | --- | --- |
| Axial dyad conductance (late window, n=29) | 0.142 | `data/results/spectral_comparison_results.csv` |
| Mixed dyad conductance (late window, n=29) | 0.714 | same |
| MI axial vs. mixed | 0.839 vs. 0.260 | `data/results/entropy_analysis_results.csv` |
| JSD axial vs. mixed | 0.842 vs. 0.284 | same |
| Early geometric signal AUC (LOOCV) | 0.804 | `data/results/early_prediction_summary.csv` |
| Combined AUC (LOOCV) | 0.860 | same |

## Recent review trail

- 2026-06-01 — README visual framing improved; figures promoted to top of
  the document so the result is the first thing a reader sees.
- 2026-05-24 — Audit table regenerated; frozen reference numbers above
  match the current HEAD.
- 2026-03-18 — Initial public release.
