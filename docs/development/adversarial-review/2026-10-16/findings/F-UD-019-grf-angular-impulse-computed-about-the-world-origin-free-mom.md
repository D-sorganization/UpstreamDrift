# F-UD-019

**GRF angular impulse computed about the world origin; free moment never integrated; silent COM fallback**

| Severity  | Category | Confidence | Effort | Status   |
| --------- | -------- | ---------- | ------ | -------- |
| 🟨 Medium | numerics | high       | S      | proposed |

**Locations:** `src/shared/python/physics/ground_reaction_forces.py:398`, `src/shared/python/physics/ground_reaction_forces.py:195`, `src/shared/python/physics/ground_reaction_forces.py:453`, `src/shared/python/physics/ground_reaction_forces.py:88`

## Finding

- Severity: Medium - Category: numerics
- Confidence: high
- Location: src/shared/python/physics/ground_reaction_forces.py:398-401; src/shared/python/physics/ground_reaction_forces.py:195-262;
  src/shared/python/physics/ground_reaction_forces.py:453-460; src/shared/python/physics/ground_reaction_forces.py:88-96
- Evidence:
  ```python
  # Angular impulse (about origin if no COM specified)
  ref_point = np.zeros(3)
  angular_impulse = compute_angular_impulse(forces, cops, timestamps, ref_point)
  ```
  `compute_angular_impulse` integrates `r × F` only; `GRFTimeSeries.moments` (the plate's free moment `Tz`) is never read;
  `analyze()` uses `self.golfer_com_trajectory[0]` when lengths differ.
- Failure scenario: `ImpulseMetrics.angular_impulse_magnitude` changes when the force plate is moved in the lab frame — it is
  not a property of the swing. The vertical angular impulse (the number golf GRF papers care about) is under-reported by the
  free-moment integral, which for a driver swing is the same order as `r × F_horizontal`. A COM trajectory sampled at a
  different rate silently collapses to its first sample.
- Why a newer engineer misses this: `np.zeros(3)` is the obvious default for "a point"; free moment is the plate channel most
  people never plot.
- Proposed change: require a reference point (no default) or raise; add `moments` to the integrand
  (`L = ∫ (r × F + M_free) dt`); raise on COM length mismatch instead of taking `[0]`.
- Verification: translate `cops` by `[1, 0, 0]` and assert `angular_impulse_about_golfer_com` is invariant; synthetic pure
  free-moment plate data must give non-zero vertical angular impulse.
- Effort: S

## Draft Pull Request

**Title:** `fix(grf): reference angular impulse to golfer COM; add free moment`

### Why This Matters (The Pitch)

ImpulseMetrics.angular_impulse_magnitude changes when you move the force plate in the lab frame, so it is not a property of the swing. The analysis defaults the reference point to np.zeros(3), integrates only r × F, and never reads the plate's free moment channel. The vertical angular impulse, the number golf GRF papers care about, is therefore under-reported by the free-moment integral, which for a driver swing is the same order as r × F_horizontal. On top of that, a COM trajectory sampled at a different rate silently collapses to its first sample. Requiring a reference point, adding the free moment to the integrand, and raising on length mismatch makes the metric physically meaningful and reproducible across labs. Size S, and anyone publishing GRF numbers from this module stops publishing numbers that depend on where the plate was bolted down.

### Proposed PR Body

## Problem

`ground_reaction_forces.py:398-401` uses `ref_point = np.zeros(3)` for `compute_angular_impulse` (`:195-262`), which integrates `r × F` only; `GRFTimeSeries.moments` (free moment `Tz`) is never read; `analyze()` (`:453-460`) uses `golfer_com_trajectory[0]` when lengths differ.

## Change

- [ ] Require a reference point (no default) or raise
- [ ] Add `moments` to the integrand: `L = ∫ (r × F + M_free) dt`
- [ ] Raise on COM length mismatch instead of taking `[0]`

## Verification

Translate `cops` by `[1, 0, 0]` and assert `angular_impulse_about_golfer_com` is invariant; synthetic pure free-moment plate data must give non-zero vertical angular impulse.

## Risk and Rollback

Medium: reported angular impulse values change, and callers relying on the default origin will now error. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-UD-019

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
