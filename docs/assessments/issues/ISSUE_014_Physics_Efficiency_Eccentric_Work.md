# Medium Priority Issue: Efficiency Score Omits Eccentric Work

**Status:** Open
**Priority:** Medium
**Labels:** physics-gap, medium
**Date Identified:** 2026-01-31
**Author:** PHYSICS AUDITOR

## Description

The `efficiency_score` calculation in `src/shared/python/statistical_analysis.py` (`compute_swing_profile`) calculates "Input Work" by summing only _positive_ power (concentric work).

```python
pos_power = np.maximum(power, 0)
total_work = ... trapz(pos_power ...)
```

It ignores negative power (eccentric work), where the body absorbs energy (e.g., decelerating the club, controlling the downswing). Eccentric muscle action has a significant metabolic cost. Ignoring it results in an "efficiency" ratio (Output KE / Input Work) that artificially inflates the score for swings that rely heavily on braking forces, and fails to represent true biomechanical cost.

## Impact

- **Metric Validity:** The "Efficiency" score is not a true measure of mechanical or metabolic efficiency.
- **Optimization:** May encourage swing styles that minimize concentric work but ignore high eccentric loads, potentially increasing injury risk.

## Affected Files

- `src/shared/python/statistical_analysis.py`

## Recommended Fix

1.  Update the definition of `total_work` in `efficiency_score` to include the absolute value of negative work: `total_work = integral(|power| dt)`.
2.  Alternatively, define separate `concentric_efficiency` and `total_efficiency` metrics.
3.  Rename the metric if it is intended to represent only "Propulsive Efficiency".
