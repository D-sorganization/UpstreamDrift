# Low Priority Issue: Incorrect Units for Dimensionless Jerk

**Status:** Open
**Priority:** Low
**Labels:** physics-error, low
**Date Identified:** 2026-01-27
**Author:** PHYSICS AUDITOR

## Description
In `src/shared/python/statistical_analysis.py`, the `dimensionless_jerk` metric is computed as:
`dim_jerk = peak_jerk / peak_acc`

Dimensionally, this is $[L/T^3] / [L/T^2] = [1/T] = \text{Frequency} (Hz)$. It is not dimensionless.

Standard "Dimensionless Jerk" metrics are usually calculated using the root-mean-square (RMS) jerk normalized by the movement duration and amplitude, for example:
$$ DJ = - \ln \left( \frac{\int j(t)^2 dt \cdot D^5}{A^2} \right) $$
or simply:
$$ DJ = \frac{\text{Jerk}_{rms} \cdot D^2}{V_{peak}} $$ (This formula is dimensionless: $[L/T^3] \cdot [T^2] / [L/T] = 1$)

## Impact
- **Interpretation:** Users expecting a dimensionless index will be confused by the magnitude and unit dependence.
- **Comparability:** The metric cannot be used to compare smoothness across movements of different speeds or durations validly.

## Affected Files
- `src/shared/python/statistical_analysis.py`

## Recommended Fix
1.  Rename the metric to `jerk_to_accel_ratio` OR
2.  Implement a standard dimensionless jerk formula, such as:
    ```python
    # For discrete time:
    # Integral approx sum(j**2) * dt
    int_j2 = np.sum(jerk**2) * dt
    scaling = (duration**5) / (amplitude**2)
    dim_jerk = -np.log(int_j2 * scaling)
    ```
