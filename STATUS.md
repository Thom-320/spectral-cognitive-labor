# Project status

**Last updated:** 2026-09-21

## Current scientific scope

Computational and spectral reanalysis by Thomas Chisica of the experiment by
Edgar Andrade-Lotero and Robert L. Goldstone (2021), PLOS ONE,
[doi:10.1371/journal.pone.0254532](https://doi.org/10.1371/journal.pone.0254532).
Participants were recruited at Indiana University. This repository does not
attribute experimental design, collection, or ownership of the source data to Thomas.

The temporal-repair deliverable reconstructs 45 dyads using five absent
opportunities per dyad, ending at rounds 6–19. It preserves historical results,
recovers original metrics, and compares rank-two coordinate and Fiedler projectors.
The two energies are highly correlated (0.999486), not identical. No corrected
predictive AUC or spectral advantage has been established.

## Pending decisions

- Behavioral outcome: late axiality, persistence of player territories, or another
  scientifically justified target; distinguish legitimate nonaxial from unclassifiable.
- Late window and minimum observations; elapsed-round control; evaluation and uncertainty.
- A small matched comparison of behavioral metrics, + coordinate energy, + Fiedler energy.
- Scientific interpretation with Andrade. A mathematical review by Esteban Vargas Bernal
  is a proposal supported by his documented network-science work, not an agreed role.
- Whether the first contrast is an out-of-dyad re-evaluation of the published
  MBIASES/WSLS/FRA ladder or the latent-state models of the mechanism proposal.
  Either way it is blocked on a parametric rewrite: see the design check below.
- Transfer to Coordinator-and-Foragers/PsyNet is not established. Earlier extension
  plans and institutional attributions in this file were not verified and are withdrawn
  from current project status; the earlier version remains in Git and the audit archive.

## Historical evaluation — preserved, not revalidated

| Metric | Historical value | Source |
| --- | --- | --- |
| Axial / mixed conductance (descriptive n=29) | 0.142 / 0.714 | data/results/spectral_comparison_results.csv |
| MI axial / mixed | 0.839 / 0.260 | data/results/entropy_analysis_results.csv |
| JSD axial / mixed | 0.842 / 0.284 | same |
| Geometric AUC, former cohort | 0.804 | data/results/early_prediction_summary.csv |
| Combined AUC, former cohort | 0.860 | same |

The former predictor reaches round >=30 in 15/45 dyads. `dominant_score` uses
coordinate templates, not eigenvectors. Historical pipeline reproducibility does
not establish temporal validity. The naive Fiedler reference is basis dependent.
The axial reference is certified as the exact global minimum conductance of the
8×8 eight-neighbor graph (22/210, attained only by the LR and TB halves); see
`audit/optimum_certificate/`. This is a property of the graph, not of participants.

## Design check before the next contrast

A synthetic power and recovery study, on 2,554 absent-to-absent transitions in 45
dyads with 200 replicates, shows that the mechanism proposal's rival models written
as frequency tables cannot be told apart by this design: the richest model is
recovered 73% of the time and a dyad-clustered interval excludes zero only 11% of
the time at the effect size the data themselves suggest. The minimum detectable
effect is roughly twice that. The models were therefore rewritten parametrically,
in the shape of the published ladder, and the study repeated. With 4, 6 and 8 free
parameters the same design recovers the generating model in every replicate and
detects a mechanism about six times weaker than the fitted one, so the contrast is
viable on the original 45 dyads. The type-I error is the binding constraint:
picking the model with the lowest out-of-dyad log-loss is wrong 27% of the time
when the extra mechanism is false, against 2% when a dyad-clustered interval is
required to exclude zero. No model comparison on participants has been run.
See [audit/model_recovery/PARAMETRIC.md](audit/model_recovery/PARAMETRIC.md).

## Evidence and next action

- [Temporal reconstruction and mathematical checks](docs/TEMPORAL_REPAIR.md)
- [Power and model recovery of the published models](audit/model_recovery/PARAMETRIC.md)
- [The same study for the table-based version](audit/model_recovery/README.md)
- [Certificate of the global conductance minimum](audit/optimum_certificate/README.md)
- [Edge-weight sensitivity of the spatial optimum](audit/edge_weights/README.md)
- [Register of explored AUCs](docs/EXPLORED_AUC_REGISTER.md)
- [Directed literature review](docs/LITERATURE_REVIEW_2026-09-10.md)
- [Expanded search and external-audit assessment](docs/LITERATURE_EXPANSION_AND_AUDIT_REVIEW.md)
- [Research prompt for independent review](docs/CHATGPT_PRO_RESEARCH_PROMPT.md)

Historical snapshot: commit 3f3b70c29be44d4b9d9e485642e54cc1807c870a, archived in
`audit/temporal_repair_v1/historical_evaluation.tar.gz`. Its provenance hashes describe
the historical files, not subsequent corrections to README or this status document.
The quoted external audit's new AUCs and bootstrap intervals have not been verified
from executable artifacts in this repository and are not adopted as project results.
