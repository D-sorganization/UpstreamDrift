# F-UD-034

**`PinocchioWrapper` energy test asserts `KE >= 0` on a zero-velocity fixed-base model; wrapper has no `src` consumer**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| ⬜ Low   | tests    | high       | S      | proposed |

**Locations:** `tests/unit/engines/mujoco/test_pinocchio_interface.py:120`, `tests/unit/engines/mujoco/test_pinocchio_interface.py:20`

## Finding

- Severity: Low - Category: tests
- Confidence: high
- Location: tests/unit/engines/mujoco/test_pinocchio_interface.py:120-134; tests/unit/engines/mujoco/test_pinocchio_interface.py:20-25
- Evidence: fixture `DOUBLE_PENDULUM_XML`, `v = data.qvel.copy()` (zeros), asserts `isinstance(ke, float)`, `ke >= 0`,
  `np.isfinite(ke)`. `grep -rn PinocchioWrapper src` finds only its own module.
- Failure scenario: F-UD-003 could not be caught by this test even if Pinocchio were installed. The class is dead code in
  production and alive only to be tested.
- Why a newer engineer misses this: "non-negative and finite" feels like a physics assertion.
- Proposed change: either delete `pinocchio_interface.py` (600 lines) or test it against MuJoCo's own `mj_energyVel`/`qfrc_bias`
  with a free joint and non-zero velocity.
- Verification: the new test fails before the F-UD-003 fix and passes after.
- Effort: S

## Draft Pull Request

**Title:** `test(mujoco): test or delete PinocchioWrapper (energy test is vacuous)`

### Why This Matters (The Pitch)

The only PinocchioWrapper energy test uses a fixed-base double pendulum at zero velocity and asserts KE is a float, finite and non-negative, which passes with KE = 0. It could not have caught the free-joint bug in F-UD-003. A grep finds no consumer of PinocchioWrapper in src, so the class is alive only to be tested. Delete the roughly 600-line module and its test, or test it against mj_energyVel and qfrc_bias with a free joint and non-zero velocity.

### Proposed PR Body

## Problem

`tests/unit/engines/mujoco/test_pinocchio_interface.py:120-134` (fixture `DOUBLE_PENDULUM_XML`, `v = data.qvel.copy()` zeros) asserts `isinstance(ke, float)`, `ke >= 0`, `np.isfinite(ke)`; `grep -rn PinocchioWrapper src` finds only its own module.

## Change

- [ ] Either delete `pinocchio_interface.py` (600 lines) and its test, or
- [ ] Test it against MuJoCo's `mj_energyVel`/`qfrc_bias` with a free joint and non-zero velocity

## Verification

The new test fails before the F-UD-003 fix and passes after.

## Risk and Rollback

Low: removing an unused module or adding a test. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** Follow-ups: F-UD-003.

Refs: adversarial review 2026-10-16, F-UD-034

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
