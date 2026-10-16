# F-UD-016

**Orientation cost uses `2·acos(|q₁·q₂|)`, ill-conditioned exactly where the optimiser converges**

| Severity  | Category | Confidence | Effort | Status   |
| --------- | -------- | ---------- | ------ | -------- |
| 🟨 Medium | numerics | high       | S      | proposed |

**Locations:** `src/shared/python/motion_matching/_geodesic.py:41`, `src/shared/python/motion_matching/cost.py:186`

## Finding

- Severity: Medium - Category: numerics
- Confidence: high
- Location: src/shared/python/motion_matching/\_geodesic.py:41-43; src/shared/python/motion_matching/cost.py:186-187
- Evidence:
  ```python
  dots = np.abs(np.einsum("ij,ij->i", q1, q2))
  dots = np.clip(dots, -1.0, 1.0)
  return 2.0 * np.arccos(dots)
  ```
- Failure scenario: for near-identical orientations `dot = 1 − θ²/8`; in float64 the smallest representable angle is
  `2·acos(1 − 2⁻⁵³) ≈ 3e-8 rad`, and `d(acos)/dx → ∞`, so the orientation term has a ~1e-15 rad² noise floor with a gradient
  that is pure rounding noise in the last decade of convergence. Gradient-based drivers (`knot_gradient_optimiser.py`,
  `multi_shooting_fit.py`) see jitter there; the cost's MATLAB-parity claim (1e-9) is only meaningful away from the optimum.
  No unit-norm check: a quaternion resampled by `slerp_series` but not renormalised gives `|dot| > 1` → clipped to 0 angle.
- Why a newer engineer misses this: `acos` of a dot product is the textbook formula; its conditioning near 0 is a numerical
  analysis footnote.
- Proposed change: `theta = 4*np.arctan2(np.linalg.norm(q1 - s*q2, axis=1), np.linalg.norm(q1 + s*q2, axis=1))` with
  `s = sign(dot)`, after normalising both rows.
- Verification: `quaternion_geodesic_angles(q, quat_exp(1e-9*axis) ⊗ q) ≈ 1e-9` to 1e-3 relative.
- Effort: S

## Draft Pull Request

**Title:** `fix(motion-matching): use atan2 orientation geodesic, not 2*acos`

### Why This Matters (The Pitch)

The orientation cost computes the geodesic angle as 2·acos(|q1·q2|), which is ill-conditioned exactly where the optimiser converges. For near-identical orientations dot = 1 − θ²/8, the smallest representable angle in float64 is about 3e-8 rad, and the derivative of acos blows up, so the last decade of convergence has a ~1e-15 rad² noise floor and a gradient that is pure rounding noise. Gradient-based drivers such as knot_gradient_optimiser.py and multi_shooting_fit.py see jitter there, and the MATLAB-parity claim of 1e-9 only holds away from the optimum. There is also no unit-norm check, so a quaternion resampled by slerp_series and not renormalised gives |dot| > 1, which clips to zero angle. The fix is a one-line formula with arctan2 on the norms of the sum and difference, after normalising both rows, and a test with a 1e-9 rotation. Size S, and the optimiser's final digits become trustworthy.

### Proposed PR Body

## Problem

`src/shared/python/motion_matching/_geodesic.py:41-43` returns `2.0 * np.arccos(np.clip(np.abs(einsum(q1,q2)), -1, 1))` (also used at `cost.py:186-187`). Near convergence this has a ~3e-8 rad resolution floor and a divergent derivative; non-normalised inputs (e.g. from `slerp_series`) clip to zero angle.

## Change

- [ ] Use `theta = 4*np.arctan2(np.linalg.norm(q1 - s*q2, axis=1), np.linalg.norm(q1 + s*q2, axis=1))` with `s = sign(dot)`
- [ ] Normalise both rows first

## Verification

`quaternion_geodesic_angles(q, quat_exp(1e-9*axis) ⊗ q) ≈ 1e-9` to 1e-3 relative.

## Risk and Rollback

Low: values away from the optimum are unchanged to rounding; MATLAB-parity tests should be re-run. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-UD-016

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
