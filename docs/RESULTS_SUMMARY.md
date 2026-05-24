# Results Summary

This note summarizes the numerical outputs used by the final paper.

## Board Geometry

- Board graph: `P_8 boxtimes P_8`
- Vertices: `64`
- Edges: `210`
- Degenerate Fiedler eigenvalue:
  `lambda_2 = lambda_3 approx 0.4164`
- Naive Fiedler cut:
  `h = 28/210 = 0.1333`
- Axis-aligned left-right / top-bottom reference:
  `h = 22/210 = 0.1048`

The value `22/210` is used as an axis-aligned reference within the evaluated
family. It is not presented as a proof of the global conductance optimum.

## Late Spectral Comparison

Primary set:

- Audited dyads: `45`
- Included in the hard-partition conductance set: `29`
- Axial dyads: `21`
- Mixed dyads: `8`

Group means:

| Metric | Axial | Mixed |
|---|---:|---:|
| `h_obs` | `0.142` | `0.714` |
| `eta` | `1.120` | `0.196` |
| `DLIndex` | `0.919` | `0.514` |
| `MI` | `0.839` | `0.260` |
| `JSD` | `0.842` | `0.284` |

Mann-Whitney comparisons:

| Metric | p-value |
|---|---:|
| `h_obs` | `3.07e-05` |
| `DLIndex` | `1.33e-04` |
| `MI` | `4.23e-04` |
| `JSD` | `3.50e-04` |

The spectral lens separates axial and mixed dyads, but it is strongly aligned
with the original `DLIndex` measure:

- Spearman `eta` vs. `DLIndex_mean`: `rho = 0.894`, `p = 6.23e-11`

This is why the paper treats conductance as an interpretable geometric lens,
not as a replacement for the original behavioral metrics.

## Early Prediction

The early geometric signal uses the first five common absent rounds for each
dyad.

| Model | AUC in-sample | AUC LOOCV |
|---|---:|---:|
| Geometry only | `0.834` | `0.804` |
| Official early metrics | `0.808` | `0.736` |
| Geometry + official metrics | `0.890` | `0.860` |

The main additional contribution is the early, symmetry-aware geometric signal.

## Descriptive Transfer to Performance

Using the stable orientation inferred from absent rounds and then inspecting
`performances.csv`:

| Regime | Group | Accuracy | Score |
|---|---|---:|---:|
| Unicorn absent | Axial | `0.977` | `21.789` |
| Unicorn absent | Mixed | `0.900` | `9.189` |
| Unicorn present | Axial | `0.936` | `24.430` |
| Unicorn present | Mixed | `0.688` | `-6.142` |

This is descriptive, not causal. The dyad is the relevant experimental unit.
