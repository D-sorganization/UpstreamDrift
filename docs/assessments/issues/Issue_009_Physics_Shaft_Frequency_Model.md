# High Priority Issue: Incorrect Natural Frequency for Tapered Shafts

**Status:** Open
**Priority:** High
**Labels:** physics-gap, high
**Date Identified:** 2026-01-22
**Author:** PHYSICS AUDITOR

## Description
The `ModalShaftModel` in `shared/python/flexible_shaft.py` calculates the natural frequency of the shaft using a formula for a uniform cantilever beam, substituting *average* stiffness ($$EI$$) and *average* mass ($$\mu$$).

Golf shafts are tapered structures where $$EI$$ can vary by a factor of 2-5x from tip to butt. Using average properties in the uniform beam equation ($$\omega_n = \beta_n^2 \sqrt{EI_{avg}/(\mu_{avg} L^4)}$$) leads to significant errors in the natural frequency estimation (up to 20%).

## Impact
- **Feel:** The simulated "kick" timing will not match real shafts.
- **Matching:** Cannot accurately simulate specific shaft profiles (e.g., "Tip Stiff" vs "Butt Stiff" designs) since averaging washes out the distribution.

## Affected Files
- `shared/python/flexible_shaft.py`

## Recommended Fix
1.  Implement a numerical frequency solver (e.g., Rayleigh-Ritz method or Finite Element Analysis) to compute eigenvalues for the specific non-uniform $$EI(x)$$ and $$\mu(x)$$ profiles.
2.  Alternatively, implement a Transfer Matrix Method for frequency determination.
