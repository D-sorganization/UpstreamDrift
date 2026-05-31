# `canonical-v2` — the dynamic state contract

> **Status:** Frozen (see [ADR-0026](../adr/0026-canonical-dynamic-state-v2.md)).
> Every analysis module, estimator, and engine adapter in the Canonical Core
> ([EPIC #6772](https://github.com/D-sorganization/UpstreamDrift/issues/6772))
> routes through this contract. Per-engine conversions happen **only** at the
> adapter boundary and are verified by the cross-engine conformance suite (CC-7).

`canonical-v2` is a **strict superset** of `canonical-v1`
([ADR-0012](../adr/0012-canonical-pose-interchange.md)). `canonical-v1`
(`CanonicalPose`, intrinsic-XYZ-degrees pose, `v = 0`) remains valid and
unchanged for pose-only callers (Pose Studio, the matcher, `fit_swing`).
`canonical-v2` adds the velocity/acceleration channels and a singularity-free
quaternion floating base needed for cross-engine **dynamics**.

---

## 1. Units & frames

| Quantity       | Unit / convention                                                                                 |
| -------------- | ------------------------------------------------------------------------------------------------- |
| Length         | metre (m)                                                                                         |
| Mass           | kilogram (kg)                                                                                     |
| Time           | second (s)                                                                                        |
| Force / torque | newton (N) / newton-metre (N·m)                                                                   |
| Angle          | radian (rad) — **note:** `canonical-v1` pose angles are degrees; v2 state is radians              |
| World frame    | **Z-up**, right-handed                                                                            |
| Gravity        | `g = [0, 0, -9.80665]` m/s²                                                                       |
| Segment frames | ISB recommendations where defined; otherwise documented per segment in the canonical model (CC-3) |

## 2. State layout

A `canonical-v2` state is `(q, v, a, t)` plus metadata
`convention="canonical-v2"`, `frame="world_Zup"`, `units="SI"`.

For a model with a floating base and `n_j` internal joint coordinates:

```
q  (configuration, length nq = 7 + n_j):
    [ base_x, base_y, base_z,            # base position, metres, world frame
      base_qw, base_qx, base_qy, base_qz,# base orientation, UNIT quaternion (w,x,y,z)
      j_0, j_1, ..., j_{n_j-1} ]         # internal joint coordinates, radians

v  (generalized velocity, length nv = 6 + n_j):
    [ base_vx, base_vy, base_vz,         # base linear velocity, world frame, m/s
      base_wx, base_wy, base_wz,         # base angular velocity, BODY (local) frame, rad/s
      dj_0, ..., dj_{n_j-1} ]            # joint velocities, rad/s

a  (generalized acceleration, length nv):  same layout as v, time-derivative
```

Key facts:

- **`nq = nv + 1`.** The quaternion carries one redundant coordinate, so the
  configuration manifold's tangent space is one dimension smaller. **Never** do
  vector arithmetic (`q + dq`) on `q`; use `integrate(q, dq)` (§4).
- **Base quaternion ordering is `(w, x, y, z)`** (scalar-first), unit norm.
- **Base angular velocity is in the body (local) frame** — the Pinocchio
  free-flyer LOCAL convention. Base _linear_ velocity is in the world frame.

## 3. Per-engine conversion table (adapter boundary only)

Each adapter (CC-9 Pinocchio, CC-10 MuJoCo/OpenSim, CC-28 Drake, CC-29 MJX,
CC-30 MyoSuite) converts between its native layout and `canonical-v2` and cites
this table rather than re-deriving it. **The conformance suite (CC-7) verifies
each row** via the round-trip and ID↔FD checks — if a quaternion order or
angular-velocity frame is wrong, those tests fail.

| Engine           | Native base quaternion             | Native base layout                                                                 | Angular-velocity frame                                                                    | Notes                                                                                         |
| ---------------- | ---------------------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| **Pinocchio**    | `(x, y, z, w)` — **w-last**        | `[xyz, quat_xyzw]` free-flyer                                                      | LOCAL (body)                                                                              | Reference engine; canonical ang-vel frame matches it. Only the quaternion order is reordered. |
| **MuJoCo / MJX** | `(w, x, y, z)` — **w-first**       | `qpos = [xyz, quat_wxyz]` free joint                                               | free-joint `qvel` angular part — **verify frame per adapter** against `mj_objectVelocity` | Quaternion order matches canonical; confirm ang-vel frame in CC-10/CC-29.                     |
| **MyoSuite**     | `(w, x, y, z)` — **w-first**       | MuJoCo MJCF `qpos = [xyz, quat_wxyz]` free joint                                   | body-local free-joint `qvel` angular part, matching the canonical MyoSuite adapter        | Activation-driven; declares `MUSCLES`, `FORWARD_DYN`, `CONTACT`, and no joint-torque ID.      |
| **Drake**        | `(w, x, y, z)` — w-first           | `QuaternionFloatingJoint`: `[quat_wxyz, xyz]` (or URDF RPY floating: `[xyz, rpy]`) | spatial velocity `V_WB` — **verify frame per adapter**                                    | Base block ordering differs (quat first); RPY-floating URDFs convert via `se3` Euler path.    |
| **OpenSim**      | coordinate-based (3 rot + 3 trans) | `FreeJoint` coordinates, often XYZ Euler                                           | per-coordinate speeds                                                                     | Some `.osim` models invert Y for shoulder external rotation (see ADR-0012).                   |
| **Simscape**     | parameter-bus, **degrees**         | `*StartPosition*` Simulink params                                                  | n/a (pose import)                                                                         | Degrees → radians at the boundary.                                                            |

> **Rule:** any cell marked _verify per adapter_ must be pinned down in that
> adapter's PR with a unit test, and the result recorded back here. The frame of
> a free-joint angular velocity is the classic silent-divergence trap; the
> contract picks body-local and the adapter is responsible for converting to it.

## 4. Manifold operations (mandatory on the base)

The base orientation lives on the unit-quaternion manifold `S³`, not in a flat
vector space. State updates use the tangent space:

- `to_tangent` / `from_tangent` — map a base perturbation between the `S³`
  manifold and `ℝ³` via `quat_log` / `quat_exp`.
- `integrate(q, dq)` — apply a tangent-space increment `dq` (length `nv`) to a
  configuration `q` (length `nq`): position and joint coordinates add directly;
  the base quaternion is updated by `q_base ⊗ exp(½ · dq_ang)`.

**Round-trip invariant (CC-2 tests, CC-7 conformance):**

```
from_tangent(to_tangent(s)) == s         # to < 1e-9 (rigid) / < 1e-6 (quaternion)
from_canonical(to_canonical(s)) == s     # per adapter, same tolerance
```

Forbidden: `q_base_new = q_base + dq_ang` (naive addition leaves the manifold
and is a sign/scale bug). Adapters that expose a native `integrate` (Pinocchio,
MuJoCo) should delegate to it.

## 5. Provenance

Every materialised state and result is stamped (CC-6 `ProvenanceStamp`) with the
`convention`, `frame`, and `units` tags above, so a consumer can reject a state
whose convention it does not recognise and a result can be reproduced exactly.

## 6. Double-pendulum AffineDrift coupling

The analysis-layer helper
`src.shared.python.analysis.affine_drift_coupling.couple_trace_to_affine_drift`
samples the golf double-pendulum drift/control split on estimated
`canonical-v2` kinematics. It accepts a `Trace` whose `q`/`v` arrays are either:

- a native two-coordinate double-pendulum trace, where all `q`/`v` columns are
  the pendulum state; or
- a `canonical-v2` floating-base trace, where the helper deterministically uses
  the last two internal-joint columns from `q` and `v` unless explicit
  `q_indices`/`v_indices` are supplied.

For each sample, the result exposes the AffineDrift state
`x = [q0, q1, v0, v1]`, drift vector
`[v, solve(M(q), -bias(q, v))]`, and control matrix whose lower block is
`M(q)^-1`. The calculation is pointwise on the measured trajectory; it does not
forward-integrate a zero-torque or zero-velocity counterfactual.

## 7. Migration from `canonical-v1`

- A `CanonicalPose` (v1) maps to a `canonical-v2` `q` by converting the pelvis
  SE(3) (`euler_xyz_deg` → quaternion `wxyz`) and joint degrees → radians, with
  `v = a = 0`.
- Pose-only callers need no change; they keep using `CanonicalPose`.
- `pose_interchange.__version__` is bumped to `2.0.0` when CC-2 lands the
  `CanonicalState` type and manifold ops.
