---
title: "Physics Error: Dimensionless Jerk Metric has Units of Frequency"
labels: ["physics-error", "medium"]
assignees: ["analytics-lead"]
---

## Description
The `dimensionless_jerk` metric in `src/shared/python/statistical_analysis.py` is calculated as `peak_jerk / peak_accel`.

*   Peak Jerk units: $[L][T]^{-3}$
*   Peak Accel units: $[L][T]^{-2}$
*   Result units: $[T]^{-1}$ (Frequency, e.g., $Hz$)

This is not a dimensionless quantity. A true dimensionless jerk metric must account for the duration of the movement to be comparable across different time scales.

## Impact
The metric value scales with the speed/duration of the swing. A faster swing with the same smoothness profile will have a higher "dimensionless jerk" score simply because it is faster ($T$ is smaller, so $T^{-1}$ is larger). This invalidates comparisons between players with different swing speeds.

## Affected Files
*   `src/shared/python/statistical_analysis.py`: `compute_jerk_metrics`

## Recommended Fix
Update the formula to one of the standard dimensionless forms:
1.  **Simple:** $J_{dim} = \frac{\text{PeakJerk} \cdot \text{Duration}}{\text{PeakAccel}}$
2.  **Integrated (Cost Function):** $J_{dim} = -\ln \left( \frac{\int j(t)^2 dt \cdot D^5}{A^2} \right)$ (where $D$ is duration, $A$ is amplitude)
