# Spectral analysis of cognitive labor

**What does a spectral representation add to behavioral measures and simple spatial geometry?**

This project investigates that question through a computational reanalysis of
*Seeking the Unicorn*, the experiment reported by **Edgar Andrade-Lotero and
Robert L. Goldstone (2021)** in [Self-organized division of cognitive labor](https://doi.org/10.1371/journal.pone.0254532).
The reanalysis is by **Thomas Chisica**. Experimental design and data collection
belong to the original study; participants were recruited at Indiana University.

**Current stage:** temporal reconstruction and representation audit completed;
behavioral outcome and predictive evaluation protocol pending scientific agreement.

[Temporal audit](docs/TEMPORAL_REPAIR.md) · [Literature](docs/LITERATURE_EXPANSION_AND_AUDIT_REVIEW.md) · [Project status](STATUS.md) · [Source experiment](https://github.com/EAndrade-Lotero/SODCL)

## Read in five minutes

1. **Question:** does a spectral representation add information beyond simple geometry?
2. **My contribution:** reconstruct past-only behavioral features and compare basis-invariant spectral and coordinate representations.
3. **Evidence:** start with the [cohort](audit/temporal_repair_v1/early_cohort.csv), [numerical summary](audit/temporal_repair_v1/summary.json), and [integrity checks](docs/TEMPORAL_REPAIR.md).
4. **Boundary:** the model-free audit does not validate historical early-prediction claims. No training is necessary to read its results.

**Analysis history:** the [explored-AUC register](docs/EXPLORED_AUC_REGISTER.md)
also records retrospective fits performed elsewhere in the project, including
an evaluation on the reconstructed cohort. "Zero models" describes the
model-free audit, not a claim that this cohort has never been explored.
Future predictive analyses on it are not blind confirmation.

## The question

Pairs searching the same board can develop complementary spatial roles. A spectral
representation offers a way to describe those patterns, but its usefulness must be
assessed against the original behavioral measures and simpler coordinate-based descriptions.

The project distinguishes three contributions:

1. **Description:** characterize spatial organization with well-defined, invariant quantities.
2. **Prediction:** assess whether those quantities help anticipate later specialization.
3. **Mechanism:** identify a behavioral process with predictions that distinguish it from alternatives.

The current audit establishes computational and mathematical properties. It does
not establish a predictive advantage or a cognitive mechanism.

## What the audit found

| Finding | Evidence and interpretation |
| --- | --- |
| A past-only cohort can be reconstructed | All **45 dyads** have five target-absent opportunities ending between **rounds 6 and 19**. This equalizes opportunities, not elapsed rounds. |
| Original metrics can be recovered | DLIndex and Similarity match all **1,244 shared historical rows**; Consistency matches **1,194 values and 50 missing entries**. This checks implementation consistency. |
| A single Fiedler vector is not canonical | The spatial graph has a **two-dimensional Fiedler eigenspace**. A sign cut of one solver-selected vector depends on the choice of basis. |
| Spectral and coordinate energies are close | Principal angles are approximately **6.59046°**; energy correlation across the reconstructed cohort is **0.999486**. The representations are distinct but highly redundant in this sample. |
| Predictive value remains unresolved | The audit fits **zero models**. High energy correlation does not establish predictive equivalence, and high spatial energy does not necessarily imply axial specialization. |

See the [recorded numerical summary](audit/temporal_repair_v1/summary.json),
[cohort table](audit/temporal_repair_v1/early_cohort.csv), and
[definitions and checks](docs/TEMPORAL_REPAIR.md).

## Representation

The 8 × 8 board is modeled as the unweighted strong product $P_8 \boxtimes P_8$:
64 vertices, 210 edges, and combinatorial Laplacian $L=D-A$. Eight-neighbor
adjacency is a **spatial proximity hypothesis**, not a claim that players can
only select neighboring cells.

Both representations use the same Euclidean space of 64 cells and orthonormal
bases of rank two:

- $Q_F$: the complete Fiedler eigenspace, with $\lambda_2=\lambda_3\approx0.4164$.
- $Q_{xy}$: centered horizontal and vertical coordinates.

For the early difference in cell-visit counts between players, $m$, compare

$$
E_F(m)=\frac{m^TQ_FQ_F^Tm}{m^Tm},\qquad
E_{xy}(m)=\frac{m^TQ_{xy}Q_{xy}^Tm}{m^Tm}.
$$

These energies are invariant to changes of orthonormal basis within each
subspace and to swapping players. A zero margin is flagged explicitly; the
implementation records zero energy by convention because the ratio is undefined.
No zero margins occur in the reconstructed cohort.

## Next scientific contrast

A candidate question is whether information available after five target-absent
opportunities anticipates **sustained late axial specialization**. Before fitting
models, the team must define persistence, the late window, sufficient observations,
and the distinction between legitimate nonaxial behavior and unclassifiable cases.

| Candidate model | Information available at the same early cutoff |
| --- | --- |
| A | Original behavioral metrics |
| B | The same metrics + coordinate energy $E_{xy}$ |
| C | The same metrics + Fiedler energy $E_F$ |

All comparisons require the same dyads, preprocessing and evaluation procedure.
B–A asks whether this spatial summary helps. C–B asks whether substituting the
spectral representation helps under that procedure; it does not establish a
mechanism or additional information conditional on both energies. The uncertainty
procedure and a materially useful improvement remain to be specified.

## Reproduce the model-free audit

Use **Python 3.11 or newer**; Python 3.12 is recommended for the pinned environment.
The recorded run used Python 3.12.14 and NumPy 2.3.5.

```bash
python3 -m venv .venv-audit
.venv-audit/bin/python -m pip install -r requirements-audit.txt
.venv-audit/bin/python -B -m unittest discover -s tests -p 'test_temporal_repair.py' -v
```

To reconstruct the cohort and write a separate snapshot:

```bash
.venv-audit/bin/python -B scripts/audit_temporal_repair.py --output /tmp/sodcl-audit-new
```

The output directory **must not already exist**. The script leaves source files
unchanged and writes reconstructed tables, projectors, checks, hashes and an
archive of the input checkout. It runs locally without fitting predictive models.
The existing [audit snapshot](audit/temporal_repair_v1/) records the earlier checkout;
subsequent documentation corrections remain traceable through Git.

CI runs the six integrity tests and checks stored historical artifacts. Passing
those checks verifies computational integrity, not the scientific validity of the
historical predictive analysis.

## Historical results

The original pipeline and outputs are retained for traceability. Its descriptive
late-window analysis used **29 dyads**; this is not the population of every AUC.
The geometric and combined AUCs of **0.804 and 0.860** are historical evaluations.
Their early selection used an absent-followed-by-absent filter and reaches round
30 or later in **15 of 45 dyads**, overlapping candidate outcome periods.
Furthermore, its `dominant_score` uses coordinate templates rather than eigenvectors.
These values do not establish validated early prediction or spectral advantage.

The solver-selected Fiedler cut and the favorable axial reference are likewise
historical comparisons, not a canonical spectral partition or proof of global
conductance optimality.

- [Historical result tables](data/results/)
- [Historical figures](figures/)
- [Preserved archive and provenance](audit/temporal_repair_v1/provenance.json)
- [Historical reproduction instructions](docs/REPRODUCIBILITY.md)

`make pipeline` and `make all` execute the historical pipeline, including model
fitting and writes to result/figure paths. Use the separate audit command above
for the current reconstruction. Older methodological documents describe the
historical analysis and should be read alongside the temporal audit.

## Literature and scientific context

Relevant work spans collaborative visual search, emergent roles, joint action,
hierarchical planning and graph-based representations. The reviewed selection
is not an exhaustive search and does not establish novelty by absence of a match.

- [Initial annotated review](docs/LITERATURE_REVIEW_2026-09-10.md)
- [Expanded literature and assessment of external audit claims](docs/LITERATURE_EXPANSION_AND_AUDIT_REVIEW.md)
- [Paper catalog with source URLs and download hashes](docs/literature/paper_catalog.json)
- [Prompt for independent research](docs/CHATGPT_PRO_RESEARCH_PROMPT.md)

PDFs are not redistributed in this repository. There is no established transfer
to Coordinator-and-Foragers/PsyNet; task and outcome comparability would require
its own assessment.

## Data and repository guide

| Location | Contents |
| --- | --- |
| `data/raw/performances.csv` | Source table used to reconstruct all target-absent opportunities |
| `data/raw/humans_only_absent.csv` | Historical filtered table; not the complete set of absent trials |
| `audit/temporal_repair_v1/` | Reconstructed cohort, round metrics, projectors and provenance |
| `scripts/audit_temporal_repair.py` | Current model-free reconstruction |
| `tests/test_temporal_repair.py` | Metric encoding, temporal integrity and invariance tests |
| `src/`, `data/results/`, `figures/` | Historical pipeline and outputs |
| `docs/` | Methodology, audit reports and literature |

Source data and protocol are documented in the
[original SODCL repository](https://github.com/EAndrade-Lotero/SODCL) and
[study publication](https://doi.org/10.1371/journal.pone.0254532).
The software license does not grant ownership of the original data or replace
their source conditions. Citation metadata for this reanalysis and the original
study are in [CITATION.cff](CITATION.cff).

## Resumen en español

Este proyecto pregunta qué aporta el análisis espectral al estudio de la división
espontánea del trabajo cognitivo. La reconstrucción temporal recupera 45 díadas
con información disponible tras cinco oportunidades de búsqueda sin objetivo.
Las energías de Fiedler y de coordenadas son distintas, pero casi redundantes en
esta muestra. Falta acordar el resultado conductual y evaluar su utilidad
predictiva; las AUC históricas no resuelven esa pregunta.
