# F-UD-026

**Dead configuration knobs: `BallProperties.cl0/cl1/cl2`, `CostOptions.q_orientation_repr/time_alignment/resample_to_hz`**

| Severity | Category    | Confidence | Effort | Status   |
| -------- | ----------- | ---------- | ------ | -------- |
| ⬜ Low   | correctness | high       | S      | proposed |

**Locations:** `src/shared/python/physics/ball_properties.py:47`, `src/shared/python/physics/ball_properties.py:64`, `src/shared/python/motion_matching/cost.py:81`, `src/shared/python/motion_matching/cost.py:186`

## Finding

- Severity: Low - Category: correctness
- Confidence: high
- Location: src/shared/python/physics/ball_properties.py:47-49; src/shared/python/physics/ball_properties.py:64-66;
  src/shared/python/motion_matching/cost.py:81-83; src/shared/python/motion_matching/cost.py:186
- Evidence: `calculate_cl` ignores `cl0/cl1/cl2` and calls the Penner fit; `compute_cost` never reads `q_orientation_repr`
  (`"rotmat"` is accepted and silently treated as quaternion), `time_alignment`, `resample_to_hz`.
- Failure scenario: a user tunes `BallProperties(cl1=0.5)` and gets identical trajectories; `CostOptions(q_orientation_repr=
"rotmat")` promises a representation switch that does not exist, so the MATLAB-parity claim depends on which option MATLAB used.
- Why a newer engineer misses this: frozen dataclasses with defaults look complete; nothing fails when a field is unused.
- Proposed change: delete the fields or `raise NotImplementedError` for non-default values; add a `vulture`-style dead-attribute
  check to CI.
- Verification: `vulture src/shared/python/physics/ball_properties.py` reports no unused attributes.
- Effort: S

## Draft Pull Request

**Title:** `chore(physics): remove dead BallProperties lift and CostOptions knobs`

### Why This Matters (The Pitch)

Setting BallProperties(cl1=0.5) changes nothing, because calculate_cl ignores cl0, cl1 and cl2 and calls the Penner fit. CostOptions accepts q_orientation_repr='rotmat' and silently treats it as a quaternion, and time_alignment and resample_to_hz are never read either. A frozen dataclass with defaults looks complete, so users tune knobs that do nothing and the MATLAB-parity claim depends on which option MATLAB used. Deleting the fields, or raising NotImplementedError for non-default values, is a size-S change that makes the API say what it does.

### Proposed PR Body

## Problem

`ball_properties.py:47-49,64-66`: `calculate_cl` ignores `cl0/cl1/cl2` and uses the Penner fit. `motion_matching/cost.py:81-83,186`: `compute_cost` never reads `q_orientation_repr` (`"rotmat"` is accepted and treated as quaternion), `time_alignment` or `resample_to_hz`.

## Change

- [ ] Delete the dead fields, or raise `NotImplementedError` for non-default values
- [ ] Add a `vulture`-style dead-attribute check to CI

## Verification

`vulture src/shared/python/physics/ball_properties.py` reports no unused attributes.

## Risk and Rollback

Low: callers passing the dead fields will need updating. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-UD-026

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
