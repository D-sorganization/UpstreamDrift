# F-UD-042

**MOSAIC regressor uses central differences (2 solves/column) for an exactly linear map, with a fixed absolute step**

| Severity | Category    | Confidence | Effort | Status   |
| -------- | ----------- | ---------- | ------ | -------- |
| ⬜ Low   | performance | medium     | S      | proposed |

**Locations:** `src/shared/python/estimation/mosaic/mujoco_regressor.py:503`, `src/shared/python/estimation/mosaic/mujoco_regressor.py:524`

## Finding

- Severity: Low - Category: performance
- Confidence: medium
- Location: src/shared/python/estimation/mosaic/mujoco_regressor.py:503-504; src/shared/python/estimation/mosaic/mujoco_regressor.py:524-544
- Evidence: docstring: "inverse dynamics is strictly linear in the stacked link spatial parameters"; the code perturbs
  `pi ± h_step` (`h_step=1e-6`) and divides by `2h`, for 10 parameters × bodies, each requiring `mj_setConst` + `mj_inverse`.
- Failure scenario: a 20-body model costs 400 `mj_setConst`+`mj_inverse` calls per node (800 k for a 2 kHz × 1 s trial) where
  200 (forward difference) or `n_bodies` (set each body's pseudo-inertia to a basis element) suffice because the map is linear;
  an absolute `h=1e-6` on `I_xy ≈ 1e-4 kg·m²` vs `m ≈ 50 kg` gives 5 orders of magnitude of relative step variation, so column
  accuracy is roundoff-limited unevenly (`eps·|τ|/h ≈ 1e-8` relative — acceptable, but `h` is a silent tuning knob).
- Why a newer engineer misses this: central differences are the safe default for nonlinear functions.
- Proposed change: forward differences with `h_k = 1e-4 · max(|pi_k|, scale_k)`; or compute `Y` columns as `τ(pi + h e_k) −
τ(pi)` once per column and assert linearity by a residual check in tests (the oracle test already compares to 1e-7).
- Verification: `test_mujoco_regressor.py` oracle tests unchanged; wall time halves.
- Effort: S

## Draft Pull Request

**Title:** `perf(mosaic): forward-difference regressor columns with scaled steps`

### Why This Matters (The Pitch)

The MOSAIC regressor perturbs each parameter with central differences at a fixed absolute step of 1e-6, although inverse dynamics is strictly linear in the stacked link spatial parameters. A 20-body model costs 400 mj_setConst plus mj_inverse calls per node, and 200 would suffice. The absolute step also varies by orders of magnitude relative to parameters like I_xy and mass. Using forward differences with a scaled step halves wall time with the oracle tests unchanged.

### Proposed PR Body

## Problem

`src/shared/python/estimation/mosaic/mujoco_regressor.py:503-504` and `:524-544` perturb `pi ± h_step` (`h_step=1e-6`) and divide by `2h` for 10 parameters × bodies, each needing `mj_setConst` + `mj_inverse`. Confidence is medium.

## Change

- [ ] Use forward differences with `h_k = 1e-4 · max(|pi_k|, scale_k)`, or compute `Y` columns as `τ(pi + h e_k) − τ(pi)`, with a linearity residual check in tests

## Verification

`test_mujoco_regressor.py` oracle tests unchanged; wall time halves.

## Risk and Rollback

Low: the oracle tests at 1e-7 guard accuracy. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** none

Refs: adversarial review 2026-10-16, F-UD-042

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
