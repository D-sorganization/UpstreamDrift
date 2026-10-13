# Paired Change in Relative Shot-Shape Benefit

A table of separate curve-versus-Straight benefits does not provide an interval
for the change in that benefit between conditions. Add an explicitly paired
contrast for shaft-twist SD 1°→2° and curve scale 1→2.

For curve C (Draw or Fade), conditions A and B, and shared sample index i, define

`D_i = (SG_C,B,i − SG_Straight,B,i) − (SG_C,A,i − SG_Straight,A,i)`.

Estimate the mean of `D_i` on the intersection of supported indices across all
six populations (Straight, Draw, Fade at each condition). Positive values mean
the curve's SG benefit relative to Straight increases in B. Resample those same
indices jointly for the paired bootstrap; do not subtract two independently
computed confidence limits or separately conditioned means. Use the existing
bootstrap convention, seed, and replicate count and record them explicitly.

The six-population intersection gives Draw and Fade comparisons the same
conditioning population. Report `common_n / N` and excluded counts by pattern
and condition. If any samples are unsupported, this is a sensitivity conditional
on support in both conditions, not a full-population expected-score change.
Intervals describe Monte Carlo sampling, not empirical player uncertainty.
Multiple grid contrasts are descriptive; they are not independent replications
or multiplicity-adjusted discoveries.

## Pairing Contract

`sample_paired_noise` constructs the covariance root from the correlation
matrix, draws a seeded standardized matrix, then multiplies each column by its
SD. At a fixed correlation matrix and seed, changing only shaft SD preserves
its underlying standardized error draw. Require:

- Identical club/lean, noise profile and correlation, seed, sample count, target,
  scoring baseline/course geometry, and qualified physical model.
- SD contrast: shaft errors divided by their nonzero SDs agree at every index;
  all other primitive errors agree. Only the declared shaft SD may differ.
- Curve-scale contrast: all primitive errors agree at every index. Face/path
  and speed are independently rematched nominal inputs, so compare their
  **errors after subtracting each nominal**, not the delivered raw values.
- Constant zero-SD dimensions stay zero; never divide them by zero. Validate
  all attempted rows, including unsupported ones, before support filtering.
- Reject changed seed, correlation, unrelated SDs, missing/duplicate indices,
  incompatible source projections, targets, or scoring definitions rather than
  silently pairing them by row position.

The same construction can be applied to radial-error contrasts with clearly
stated units and desirable sign (lower error is better). Reuse public scoring
and paired-statistics pathways; avoid a second SG formula.

## Interpretation

These contrasts compare complete matched-delivery conditions. Increasing curve
strength can change solved nominal face, path, and speed while holding raw
landing target fixed. Therefore the result is not an isolated causal effect of
curvature. Increasing shaft SD is an assumed perturbation-distribution change,
not evidence about how a golfer would adapt their swing. Preserve source
protocol identifiers when original and supplemental conditions are combined.
