# ADR 0012: Canonical Pose Interchange

- Status: Accepted
- Date: 2026-05-09
- Decision Makers: UpstreamDrift core maintainers
- Related Issues/PRs: EPIC [#4895](https://github.com/D-sorganization/UpstreamDrift/issues/4895),
  this ADR closes the design portion of [#4896](https://github.com/D-sorganization/UpstreamDrift/issues/4896);
  builds on tracking issues [#4513](https://github.com/D-sorganization/UpstreamDrift/issues/4513)
  (cross-engine `fit_swing` parity) and [#4475](https://github.com/D-sorganization/UpstreamDrift/issues/4475)
  (multi-source motion targets).

## Context

Every supported physics engine in this project carries a different pose
convention:

- **Drake** reads URDF; joint axes are RPY, free-flyer `q` is
  `[xyz, rpy]`.
- **MuJoCo** reads MJCF; free-joint `qpos` is `[xyz, quat_wxyz]`
  (quaternion **w-first**).
- **Pinocchio** uses a free-flyer `q` of `[xyz, quat_xyzw]`
  (quaternion **w-last** — the opposite of MuJoCo).
- **OpenSim** (`.osim`) reports BodyKinematics in **XYZ** Euler whereas
  the existing canonical FK in
  `src.shared.python.motion_matching.diagnostics.forward_kinematics`
  uses intrinsic XYZ degrees too — so this one we already match — but
  several `.osim` models invert Y for shoulder external rotation.
- **Simscape** parameter-bus joints carry Simulink.Parameter names
  like `HipStartPositionX`, `LSStartPositionZ`, in **degrees**.

A naive Pose Studio would have the engine picker round-trip each pose
through an N×N matrix of pairwise converters. Every pair is a
sign-bug factory. AGENTS.md §G already calls out sign-convention
drift as the historical bug source for this codebase:

- Wiffle xlsx CM-vs-inches confusion (closed by the MATLAB loader as
  the source of truth).
- Simscape's torso-twist joint between spine and hub.
- The shared FK's hand-height asymmetry with the reference Address
  pose.

We need a single canonical convention and one adapter per engine — N
adapters total, not N² converters.

## Decision

We adopt a **canonical pose convention** that mirrors the convention
already used by `forward_kinematics` and `reference_golfer_setup`,
since those modules are the de-facto canonical pose representation in
the codebase today. Specifically:

1. **Pelvis pose** is an SE(3) transform represented as
   `(translation_m, rotation_xyz_deg)` where `rotation_xyz_deg` is
   intrinsic XYZ Euler in **degrees**.
2. **Joint angles** are a flat dict mapping the
   :func:`reference_golfer_setup` field names (e.g. `HipStartPositionX`,
   `LSStartPositionZ`) to **degrees**.
3. **Velocities are out of scope** for the interchange. Pose Studio
   always sets v=0 when materialising an initial state; subtask 6
   (#4900) handles engine-side state files.
4. The convention version is tagged `"canonical-v1"`; future bumps
   bump the tag and adapters refuse unfamiliar tags.

The implementation lives under
:mod:`src.shared.python.pose_interchange`:

- :class:`CanonicalPose` — frozen, DbC-validated dataclass.
- :class:`PoseConventionAdapter` — runtime-checkable Protocol with
  `to_canonical`, `from_canonical`, `joint_layout` methods.
- :class:`JointSlot` — frozen dataclass describing one joint's slot
  in an engine's `q` vector (start index, length, units, sign,
  limits).
- SE(3) helpers in `pose_interchange/se3.py`.

Per-engine adapters land in subtask 2 (#4897), one file each under
:mod:`pose_interchange.adapters`.

## Alternatives Considered

1. **Quaternion-based canonical**. Cleaner mathematically; avoids the
   gimbal-lock question entirely. **Rejected** because the existing
   FK and reference pose use intrinsic XYZ Euler in degrees and we
   would have to convert at every boundary. The canonical golfer pose
   stays comfortably away from gimbal lock (`y` rotation never
   approaches ±90°), so the loss is theoretical.
2. **Per-engine ad-hoc converters**. The current state of the
   codebase, partially. **Rejected** for the N²-converter and
   sign-drift reasons above.
3. **Reuse `pose_editor.core.JointInfo`** directly as the canonical
   record. **Rejected**: `JointInfo` is a _runtime_ per-joint editing
   record (carries `current_position`, `current_velocity`, UI
   metadata). The interchange wants a _value type_ with no runtime
   state.

## Consequences

- **Positive:**
  - Single source of truth for "what does pose X look like in
    engine Y's `q` vector?".
  - Round-trip parity test (`from_canonical(to_canonical(q)) == q`)
    is the gate for every adapter PR.
  - The canonical convention is a strict superset of what the matcher
    and `fit_swing` pipeline already consume, so no Simscape-side
    work is needed for adoption.
  - JSON serialisation is human-readable and stable for diffs.
- **Negative:**
  - Singularity at `y = ±90°` (gimbal lock) — out of range for the
    golfer pose envelope but not for arbitrary humanoid poses, so a
    future humanoid extension may need a quaternion-based v2.
  - Adapters incur a per-call conversion cost; not a concern for
    interactive editing (≤10 Hz of canonical updates) but to be
    revisited if motion-matching cost-function evaluation needs to
    consume `CanonicalPose` directly (it currently does not).
- **Follow-ups:**
  - Subtask 2 (#4897) lands the per-engine adapters.
  - Subtask 3 (#4898) lands `LiveKinematicsService`, which uses the
    adapter to set engine state from a canonical pose.
  - Subtask 6 (#4900) lands `pose_io` — saves the canonical pose as
    each engine's native initial-state file.

## Validation

Tests in `tests/unit/pose_interchange/`:

- `test_canonical_pose.py` — DbC validation, immutability, JSON
  round-trip.
- `test_se3.py` — SE(3) helpers; bijection with random rotations to
  1e-12 tolerance.
- `test_reference_pose_alignment.py` — `canonical_from_reference_setup()`
  end-effectors match `forward_kinematics(reference_golfer_setup())`
  exactly.

Per-adapter parity tests land in subtask 2 (#4897); each adapter has
a `from_canonical(to_canonical(q)) == q` test to 1e-9 plus a
reference-pose end-effector test to 5 mm against the canonical FK
output (the existing `CROSS_ENGINE_PARITY_SPEC.md` tolerance).
