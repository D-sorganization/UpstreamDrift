# Critical Issue: Efficiency Score Uses Hardcoded 1.0kg Clubhead Mass

**Status:** Open
**Priority:** High
**Labels:** physics-error, high
**Date Identified:** 2026-02-16
**Author:** JULES (Gap Analysis)

## Description

The `efficiency_score` calculation in `src/shared/python/statistical_analysis.py` (`compute_swing_profile`) uses a hardcoded mass of 1.0 kg for the clubhead when calculating Kinetic Energy (KE).

```python
            # Energy metrics
            ke = 0.5 * 1.0 * (self.club_head_speed**2)  # Dummy mass 1kg
```

A typical driver head weighs approximately 200g (0.2 kg). Using 1.0 kg inflates the calculated Kinetic Energy by a factor of 5, leading to a grossly overestimated efficiency score (Output KE / Input Work).

## Impact

- **Metric Accuracy:** Efficiency scores will be artificially high (potentially exceeding 100% physically impossible levels if work calculation was correct).
- **Misleading Feedback:** Users may believe their swing is highly efficient when it is not.
- **Comparison:** Makes it impossible to compare efficiency across different club types if the mass is always 1.0 kg.

## Affected Files

- `src/shared/python/statistical_analysis.py` (`compute_swing_profile`)

## Recommended Fix

1.  Update `StatisticalAnalyzer` to accept a `club_mass` parameter in `__init__` or `compute_swing_profile`.
2.  Use a realistic default mass (e.g., 0.2 kg for driver, or specific to the club used).
3.  Ideally, retrieve the club mass from the simulation configuration if available.
