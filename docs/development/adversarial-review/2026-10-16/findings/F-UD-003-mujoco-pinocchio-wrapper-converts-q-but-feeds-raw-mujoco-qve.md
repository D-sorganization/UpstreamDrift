# F-UD-003

**MuJoCo→Pinocchio wrapper converts `q` but feeds raw MuJoCo `qvel` to Pinocchio (free-flyer frame mismatch)**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| 🟧 High  | numerics | high       | M      | proposed |

**Locations:** `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/pinocchio_interface.py:168`, `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/pinocchio_interface.py:294`, `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/pinocchio_interface.py:338`, `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/pinocchio_interface.py:548`, `src/shared/python/pose_interchange/adapters/pinocchio.py:77`, `src/shared/python/pose_interchange/adapters/mujoco.py:155`

## Finding

- Severity: High - Category: numerics
- Confidence: high
- Location: src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/pinocchio_interface.py:168-203;
  src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/pinocchio_interface.py:294-299;
  src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/pinocchio_interface.py:338-342;
  src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/pinocchio_interface.py:548-557;
  src/shared/python/pose_interchange/adapters/pinocchio.py:77-80; src/shared/python/pose_interchange/adapters/mujoco.py:155-157
- Evidence:
  ```python
  if v is None:
      v = self.data.qvel.copy()            # no frame conversion
  return pin.rnea(self.pin_model, self.pin_data, q, v, a)
  ...
  m_matrix = self.compute_mass_matrix(q)    # Pinocchio M(q) in Pinocchio's tangent frame
  ke = 0.5 * v.T @ m_matrix @ v             # v in MuJoCo's tangent frame
  ```
  while the repo's own adapters document: `"""Pinocchio free-flyer: linear and angular velocity both LOCAL"""` vs
  `"""MuJoCo free joint: linear world, angular local"""`.
- Failure scenario: any model with a free joint (the humanoid golfer). MuJoCo's base linear velocity is expressed in the world
  frame, Pinocchio's in the body frame; with the pelvis rotated 90° about Z, `vx` and `vy` swap. `compute_kinetic_energy`,
  `compute_inverse_dynamics` and `compute_forward_dynamics` all return wrong numbers whenever the base is rotated and moving —
  silently, with plausible magnitudes. The only test (`tests/unit/engines/mujoco/test_pinocchio_interface.py:120-134`) uses a
  fixed-base double pendulum at `qvel = 0`, so `KE = 0 >= 0` passes.
- Why a newer engineer misses this: the quaternion reorder is the famous gotcha and it was handled; the velocity frame
  convention is the quieter one because both libraries call the quantity "qvel"/"v" of length `nv`.
- Proposed change: route through `pose_interchange` (`DynamicMapsMixin` already knows both layouts), or add
  `_mujoco_v_to_pinocchio_v` that rotates `v[0:3]` by `R_base.T` for every free joint and apply it wherever `self.data.qvel` is
  read (lines 299, 341, 556, and the Coriolis/gravity helpers).
- Verification: a MuJoCo model with a free joint, base quaternion = 90° about Z, `qvel = [1,0,0, 0,0,0, ...]`; assert
  `wrapper.compute_kinetic_energy()` equals MuJoCo's `mj_energyVel` to 1e-9, and `compute_inverse_dynamics(a=0)` equals
  `data.qfrc_bias` after frame mapping.
- Effort: M

## Draft Pull Request

**Title:** `fix(mujoco): map free-joint qvel into Pinocchio's frame in the wrapper`

### Why This Matters (The Pitch)

The MuJoCo-to-Pinocchio wrapper does the famous quaternion reorder correctly and then hands Pinocchio the raw MuJoCo velocity. MuJoCo's free-joint linear velocity is in the world frame; Pinocchio's is in the body frame. With the pelvis rotated 90 degrees about Z, vx and vy swap, and compute_kinetic_energy, compute_inverse_dynamics and compute_forward_dynamics all return wrong numbers with plausible magnitudes. The repo's own pose_interchange adapters document exactly this mismatch, so the knowledge is in the building, just not at this call site. The only test uses a fixed-base double pendulum at qvel = 0, where KE = 0 satisfies KE >= 0 forever. Anyone cross-checking dynamics for the humanoid golfer through this path gets silently incorrect answers, and the cross-engine parity story built on it inherits them. The fix is a bounded frame rotation applied wherever qvel is read, with a test that would actually have failed.

### Proposed PR Body

## Problem

`pinocchio_interface.py:168-203` converts `q` but passes `self.data.qvel.copy()` straight to `pin.rnea` and uses it in `0.5 * v.T @ m_matrix @ v` (`:294-299`, `:338-342`, `:548-557`). The repo's adapters (`pose_interchange/adapters/pinocchio.py:77-80`, `adapters/mujoco.py:155-157`) document Pinocchio free-flyer velocity as LOCAL vs MuJoCo linear-world/angular-local. Any model with a free joint is wrong whenever the base is rotated and moving.

## Change

- [ ] Route through `pose_interchange` (`DynamicMapsMixin` knows both layouts), or add `_mujoco_v_to_pinocchio_v` that rotates `v[0:3]` by `R_base.T` for every free joint
- [ ] Apply it wherever `self.data.qvel` is read (lines 299, 341, 556, and the Coriolis/gravity helpers)

## Verification

MuJoCo model with a free joint, base quaternion = 90° about Z, `qvel = [1,0,0, 0,0,0, ...]`; assert `wrapper.compute_kinetic_energy()` equals MuJoCo's `mj_energyVel` to 1e-9 and `compute_inverse_dynamics(a=0)` equals `data.qfrc_bias` after frame mapping.

## Risk and Rollback

Medium: results for free-joint models will change to the correct values; fixed-base models are unaffected. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** Follow-ups: F-UD-034 proposes deleting this wrapper entirely, since `src` has no consumer; decide between the two.

Refs: adversarial review 2026-10-16, F-UD-003

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
