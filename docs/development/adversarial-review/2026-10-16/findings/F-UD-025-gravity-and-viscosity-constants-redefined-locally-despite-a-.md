# F-UD-025

**Gravity and viscosity constants redefined locally despite a `PhysicalConstant` single source**

| Severity  | Category | Confidence | Effort | Status   |
| --------- | -------- | ---------- | ------ | -------- |
| 🟨 Medium | numerics | high       | S      | proposed |

**Locations:** `src/shared/python/core/physics_constants.py:52`, `src/shared/python/optimization/casadi_backend.py:78`, `src/shared/python/motion_matching/dynamics_filter.py:34`, `src/shared/python/motion_matching/pipeline/centroidal_filter.py:47`, `src/bunkershot3d/solvers/envelope.py:71`, `src/shared/python/physics/ball_simulator.py:158`

## Finding

- Severity: Medium - Category: numerics
- Confidence: high
- Location: src/shared/python/core/physics_constants.py:52-54; src/shared/python/optimization/casadi_backend.py:78;
  src/shared/python/motion_matching/dynamics_filter.py:34; src/shared/python/motion_matching/pipeline/centroidal_filter.py:47;
  src/bunkershot3d/solvers/envelope.py:71; src/shared/python/physics/ball_simulator.py:158; src/shared/python/core/physics_constants.py:193-195
- Evidence: `GRAVITY_M_S2 = PhysicalConstant(9.80665, ...)` vs `_GRAVITY = np.array([0.0, 0.0, -9.81])`, `GRAVITY_M_S2 = 9.81`
  (×2), `GRAVITY_M_S2 = 9.81`; grep: 73 occurrences of `9.80665`, 37 of `9.81`, 4 of `9.807`, 3 of `9.8`. `ball_simulator.py`
  hard-codes `viscosity=1.81e-5` while `AIR_VISCOSITY_KG_M_S = 1.789e-5` (ISO 2533) exists.
- Failure scenario: the CasADi trajectory-optimisation backend and the MuJoCo XML (`_golf_swing_full_body_xml.py:20`) disagree
  on g by 3.4e-4 relative; cross-engine parity gates at 1e-6..1e-12 then measure the constant, not the engine. Re at launch
  shifts 1.2 % from the viscosity alone, moving Cd on the crisis curve.
- Why a newer engineer misses this: 9.81 is muscle memory; a `PhysicalConstant` wrapper that is not a `float` subclass needs
  `float(...)` at every use, which nudges people to type the literal.
- Proposed change: `ruff`-style custom check (there is already a `check_hardcoded_style_ratchet.py` pattern) forbidding
  `9\.8\d*` literals in `src/` outside `physics_constants.py`; import the constants.
- Verification: `grep -rE "9\.8[0-9]*" src --include=*.py | grep -v physics_constants.py | wc -l` → 0.
- Effort: S

## Draft Pull Request

**Title:** `fix(constants): use physics_constants for gravity and air viscosity`

### Why This Matters (The Pitch)

The repo has a PhysicalConstant single source for standard gravity, 9.80665, and then redefines it locally: \_GRAVITY = [0, 0, -9.81] in the CasADi backend, GRAVITY_M_S2 = 9.81 twice in the motion-matching filters and again in bunkershot3d. A grep finds 73 occurrences of 9.80665, 37 of 9.81, 4 of 9.807 and 3 of 9.8. ball_simulator.py also hard-codes viscosity 1.81e-5 where AIR_VISCOSITY_KG_M_S = 1.789e-5 (ISO 2533) exists. The CasADi optimiser and the MuJoCo XML disagree on g by 3.4e-4 relative, so cross-engine parity gates at 1e-6 to 1e-12 end up measuring the constant instead of the engine, and Reynolds number at launch shifts 1.2 % from viscosity alone, moving Cd on the crisis curve. Importing the constants and adding a ratchet-style check for 9.8x literals outside physics_constants.py removes a class of phantom discrepancies. Size S.

### Proposed PR Body

## Problem

`core/physics_constants.py:52-54` defines `GRAVITY_M_S2 = PhysicalConstant(9.80665, ...)`, but `optimization/casadi_backend.py:78`, `motion_matching/dynamics_filter.py:34`, `motion_matching/pipeline/centroidal_filter.py:47` and `bunkershot3d/solvers/envelope.py:71` use 9.81 literals, and `physics/ball_simulator.py:158` hard-codes `viscosity=1.81e-5` vs `AIR_VISCOSITY_KG_M_S = 1.789e-5` (`physics_constants.py:193-195`). The g mismatch is 3.4e-4 relative; the viscosity shifts Re by 1.2 %.

## Change

- [ ] Replace the local gravity and viscosity literals with imports from `physics_constants.py`
- [ ] Add a check (modelled on `check_hardcoded_style_ratchet.py`) forbidding `9\.8\d*` literals in `src/` outside `physics_constants.py`

## Verification

`grep -rE "9\.8[0-9]*" src --include=*.py | grep -v physics_constants.py | wc -l` → 0.

## Risk and Rollback

Medium: numerical outputs shift slightly (3.4e-4 relative in g), so golden values and tolerances may need review. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** none

Refs: adversarial review 2026-10-16, F-UD-025

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
