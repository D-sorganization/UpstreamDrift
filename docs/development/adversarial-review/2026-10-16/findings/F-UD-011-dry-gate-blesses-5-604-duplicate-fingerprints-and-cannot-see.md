# F-UD-011

**DRY gate blesses 5,604 duplicate fingerprints and cannot see semantic duplicates (11 quat→matrix, 17 RK4)**

| Severity  | Category     | Confidence | Effort | Status   |
| --------- | ------------ | ---------- | ------ | -------- |
| 🟨 Medium | architecture | high       | L      | proposed |

**Locations:** `scripts/config/dry_duplication_baseline.json:1`, `scripts/config/dry_duplication_quarantine.json:1`, `src/shared/python/pose_interchange/se3.py:23`, `src/shared/python/motion_matching/diagnostics/forward_kinematics.py:51`, `src/shared/python/estimation/mosaic/mujoco_regressor.py:72`, `src/shared/python/estimation/dime_observation_factors.py:76`

## Finding

- Severity: Medium - Category: architecture
- Confidence: high
- Location: scripts/config/dry_duplication_baseline.json:1; scripts/config/dry_duplication_quarantine.json:1;
  src/shared/python/pose_interchange/se3.py:23-50; src/shared/python/motion_matching/diagnostics/forward_kinematics.py:51-67;
  src/shared/python/estimation/mosaic/mujoco_regressor.py:72; src/shared/python/estimation/dime_observation_factors.py:76;
  src/shared/python/loop_nullspace/configuration_manifold.py:73; src/shared/python/motion_matching/named_state.py:406;
  src/shared/python/motion_matching/counterfactual.py:383; src/shared/python/physics/ball_enhanced_simulator.py:246
- Evidence: 5,001 baseline + 603 quarantined "duplicated production logic fingerprints"; `_rx/_ry/_rz/_euler_xyz` are
  byte-identical in `se3.py` and `forward_kinematics.py`; grep finds 11 `quat*→matrix` functions and 17 `rk4` steppers in `src/`
  (plus the Rust one), all textually different, all the same math.
- Failure scenario: a sign-convention fix in one copy (the AGENTS.md warning about "divergent re-implementation when the shared
  module changes its sign convention" is literal here) leaves ten others wrong; the gate reports no growth because the hash is
  over normalised text.
- Why a newer engineer misses this: duplication detectors measure syntax; the costly duplication in numerical code is
  semantic (same formula, different variable names). A gate that is green at 5,604 known duplicates has stopped measuring DRY.
- Proposed change: create `src/shared/python/math_utils/{rotations,integrators}.py` as the single home (there is already
  `math_utils/quaternion.py` with `rotmat_to_quat`), delete the copies, and turn the DRY gate into a count that must go down.
- Verification: `grep -rn "def .*quat.*to.*mat\|def .*rk4" src | wc -l` → ≤ 2.
- Effort: L

## Draft Pull Request

**Title:** `refactor(math): consolidate quaternion and RK4 helpers into math_utils`

### Why This Matters (The Pitch)

The DRY gate is green while blessing 5,001 baseline and 603 quarantined duplicate fingerprints, and it cannot see the duplication that matters in numerical code: the same formula under different variable names. A grep finds 11 quaternion-to-matrix functions and 17 RK4 steppers in src/, all textually different, all the same math, plus the Rust one. The \_rx/\_ry/\_rz/\_euler_xyz helpers are byte-identical in se3.py and forward_kinematics.py. AGENTS.md warns about divergent re-implementation after a sign-convention change, and this is that warning made literal: fix one copy and ten others stay wrong, and the gate reports no growth. Creating math_utils/rotations.py and integrators.py next to the existing quaternion.py, deleting the copies and turning the DRY gate into a must-go-down count reduces what has to be cloned, linted and reviewed, and gives physicists one place to audit. Size L, so worth slotting deliberately.

### Proposed PR Body

## Problem

`scripts/config/dry_duplication_baseline.json:1` and `dry_duplication_quarantine.json:1` bless 5,001 + 603 fingerprints. `_rx/_ry/_rz/_euler_xyz` are byte-identical in `pose_interchange/se3.py:23-50` and `motion_matching/diagnostics/forward_kinematics.py:51-67`; grep finds 11 `quat*→matrix` and 17 `rk4` implementations in `src/` (e.g. `estimation/mosaic/mujoco_regressor.py:72`, `loop_nullspace/configuration_manifold.py:73`, `physics/ball_enhanced_simulator.py:246`).

## Change

- [ ] Create `src/shared/python/math_utils/{rotations,integrators}.py` as the single home (alongside existing `math_utils/quaternion.py`)
- [ ] Migrate callers and delete the duplicate copies
- [ ] Turn the DRY gate into a count that must go down

## Verification

`grep -rn "def .*quat.*to.*mat\|def .*rk4" src | wc -l` → ≤ 2.

## Risk and Rollback

Medium: sign-convention differences between copies may surface during consolidation; do it package by package. Rollback is reverting per package.

## Handoff

Follow-ups: F-UD-038 (ball-simulator integrator parity).

Refs: adversarial review 2026-10-16, F-UD-011

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
