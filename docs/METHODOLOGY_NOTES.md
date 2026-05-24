# Methodology Notes

## Primary Analysis Set

The primary conductance analysis uses:

```text
data/raw/humans_only_absent.csv
```

The late-window partition is computed on absent rounds with `Round >= 40`.

A dyad enters the primary hard-partition set only if:

- both players are present in the selected window,
- the observed partition is defined by
  `S_obs = {v : f_1(v) > f_2(v)}`,
- `10 <= |S_obs| <= 54`, and
- the observed conductance is finite and nonzero.

## Exclusions

Out of 45 audited dyads, 29 enter the primary set.

Excluded by very small observed partition:

- `216-713`
- `261-970`
- `313-199`
- `356-137`
- `379-897`
- `462-640`
- `475-186`
- `880-349`

Excluded by very large observed partition:

- `352-425`
- `359-904`
- `416-710`
- `483-710`
- `487-811`
- `590-286`
- `636-625`
- `938-219`

These exclusions are not arbitrary noise. They mostly correspond to strategies
that are not clean hard bipartitions, such as `ALL`, `NOTHING`, or `RS`.

## Scope of the Graph Model

The model is strongest for stable axial left-right and top-bottom
specialization. It is weaker for strategies that do not induce a clear
bipartition of the board.

The final paper therefore frames the result as a partial, symmetry-aware
reanalyis of SODCL, not as a complete explanation of every strategy observed in
the original experiment.
