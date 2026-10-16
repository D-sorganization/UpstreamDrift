# F-UD-041

**`air_density`/`cd_dimpled_sphere` reject NumPy scalar types with `TypeError`**

| Severity | Category    | Confidence | Effort | Status   |
| -------- | ----------- | ---------- | ------ | -------- |
| ⬜ Low   | correctness | high       | S      | proposed |

**Locations:** `src/shared/python/physics/atmosphere.py:114`, `src/shared/python/physics/atmosphere.py:352`

## Finding

- Severity: Low - Category: correctness
- Confidence: high
- Location: src/shared/python/physics/atmosphere.py:114-121; src/shared/python/physics/atmosphere.py:352-355
- Evidence: `if not isinstance(altitude_m, (int, float)) or isinstance(altitude_m, bool): raise TypeError`.
- Failure scenario: `np.float32`, `np.int64`, `np.float16` values from a vectorised caller (`ball_enhanced_simulator` passes
  `position[2]`, a `np.float64` which happens to subclass `float`; a float32 trajectory would not) raise `TypeError("altitude_m
must be a real number")`.
- Why a newer engineer misses this: `np.float64` subclasses `float`, so the common case works and hides the rule.
- Proposed change: `isinstance(x, numbers.Real)` or `float(x)` with a `math.isfinite` check.
- Verification: `air_density(np.float32(100.0))` returns.
- Effort: S

## Draft Pull Request

**Title:** `fix(physics): accept NumPy scalars in air_density and cd_dimpled_sphere`

### Why This Matters (The Pitch)

air_density and cd_dimpled_sphere reject anything that is not a Python int or float, so np.float32, np.int64 and np.float16 from a vectorised caller raise TypeError. np.float64 subclasses float, so the common case works and hides the rule. Using numbers.Real, or float(x) with an isfinite check, is a size-S fix.

### Proposed PR Body

## Problem

`src/shared/python/physics/atmosphere.py:114-121` and `:352-355` use `isinstance(altitude_m, (int, float))` and raise `TypeError("altitude_m must be a real number")`.

## Change

- [ ] Use `isinstance(x, numbers.Real)` or `float(x)` with a `math.isfinite` check

## Verification

`air_density(np.float32(100.0))` returns.

## Risk and Rollback

Negligible: widens accepted input types. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-UD-041

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
