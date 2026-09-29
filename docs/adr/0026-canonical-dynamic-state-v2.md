# ADR-0026: Canonical dynamic state (`canonical-v2`)

- Status: Accepted
- Date: 2026-05-30
- Decision Makers: UpstreamDrift core maintainers
- Related Issues/PRs: Unified Biomechanics Platform / Canonical Core EPIC
  [#6772](https://github.com/D-sorganization/UpstreamDrift/issues/6772);
  this ADR is the design portion of
  [#6773](https://github.com/D-sorganization/UpstreamDrift/issues/6773) (CC-1)
  and the contract that
  [#6774](https://github.com/D-sorganization/UpstreamDrift/issues/6774) (CC-2)
  implements. Builds on [ADR-0012](0012-canonical-pose-interchange.md)
  (canonical pose) and [ADR-0002](0002-physics-engine-plugin-architecture.md)
  (engine plugin architecture + capability taxonomy).

## Context

[ADR-0012](0012-canonical-pose-interchange.md) established `canonical-v1`: a
single engine-agnostic **pose** convention (pelvis SE(3) as
`(translation_m, rotation_xyz_deg)` intrinsic-XYZ Euler in degrees, joint
angles in degrees) so per-engine adapters convert through one canonical form
instead of an N×N matrix of pairwise converters. It deliberately scoped
**velocities out** — Pose Studio always materialises an initial state with
`v = 0` — and noted a quaternion-based `canonical-v2` would be needed once the
interchange had to carry full dynamic state for arbitrary humanoid poses. The
public-API note in `pose_interchange/__init__.py` (issue #5917) records the same
upgrade path: "introduce `canonical-v2` instead when possible" for a breaking
superset.

Cross-engine **dynamics** comparison — the core thesis of EPIC #6772 — needs
more than a pose. To run one golf swing through MuJoCo, Drake, Pinocchio,
OpenSim, MJX and MyoSuite and compare inverse-dynamics torques, ZTCF/ZVCF
decompositions, and ground-reaction wrenches, every engine must agree on:

- a **generalized velocity** `v` and **acceleration** `a`, not just `q`;
- a **floating base** representation that is singularity-free (Euler `y = ±90°`
  gimbal lock is acceptable for the golfer pose envelope but not for general
  humanoid motion or for differentiating through the base);
- the **frame** in which the base angular velocity is expressed (engines
  disagree: Pinocchio's free-flyer velocity is LOCAL/body-frame, MuJoCo's free
  joint differs), and the **ordering** of the base quaternion (MuJoCo is
  **w-first** `[w,x,y,z]`; Pinocchio is **w-last** `[x,y,z,w]` — the opposite).

Without one frozen contract these differences become silent sign-bug factories,
exactly the failure mode ADR-0012 was created to prevent — now in the velocity
and acceleration channels as well as pose.

## Decision

Adopt **`canonical-v2`**, a dynamic state convention that is a strict superset
of `canonical-v1`. It is documented in full in
[`docs/conventions/canonical-v2.md`](../conventions/canonical-v2.md) and
implemented by CC-2 (#6774) as a frozen `CanonicalState` value type plus
manifold operations added to `pose_interchange/se3.py`.

1. **Units & frames.** SI throughout (m, kg, s, N, N·m, rad). World frame is
   **Z-up**; gravity is `[0, 0, -9.80665]` m/s². These match the existing
   simulation backends.

2. **State.** `(q, v, a, t)` with explicit `convention="canonical-v2"`,
   `frame="world_Zup"`, `units="SI"` metadata.

3. **Floating base = free-flyer.** Configuration `q` is laid out as
   `[base_xyz (3) | base_quat_wxyz (4) | joint_coords (n_j)]`, so
   `nq = 7 + n_j`. The base orientation is a **unit quaternion ordered
   `(w, x, y, z)`**. Generalized velocity `v` and acceleration `a` share the
   layout `[base_lin (3) | base_ang (3) | joint (n_j)]`, so `nv = 6 + n_j`
   (`nq = nv + 1` because the quaternion has one redundant coordinate). The
   **base angular velocity is expressed in the body (local) frame**, matching
   the Pinocchio free-flyer LOCAL convention (Pinocchio is the CC-9 reference
   engine).

4. **Manifold operations are mandatory on the base.** Naive `q += dq` on the
   quaternion block is forbidden. State updates use
   `integrate(state, dq)` built on `quat_exp`/`quat_log` (CC-2), and adapters
   use the engine's native `integrate`/manifold op where available. This is the
   single most common source of cross-engine garbage and is enforced by the
   CC-7 round-trip conformance check.

5. **Backward compatibility.** `canonical-v1` (`CanonicalPose`) remains valid
   and unchanged for the pose-only / `v = 0` path (Pose Studio, the matcher,
   `fit_swing`). `canonical-v2` adds the dynamic channels; a `CanonicalPose`
   maps to a `canonical-v2` `q` with `v = a = 0`. The version tag is bumped and
   adapters refuse states whose `convention` tag they do not recognise.

6. **Per-engine conversions live only at the adapter boundary.** The
   conversion table (quaternion order, Euler↔quaternion, deg↔rad, sign flips,
   angular-velocity frame) is frozen in the conventions doc; CC-9/CC-10/CC-28
   adapters cite it rather than re-deriving it, and the conformance suite (CC-7)
   verifies each adapter's round-trip and ID↔FD self-consistency.

## Alternatives Considered

1. **Extend the Euler-XYZ-degrees pose convention with velocities.** Rejected:
   the Euler parameterisation is singular at `y = ±90°` and its time-derivative
   relationship to angular velocity is frame- and order-dependent and
   ill-conditioned near the singularity — unsuitable for differentiating
   through the base (the estimator, CC-18/19) or for general humanoid motion.

2. **A quaternion ordered `(x, y, z, w)` (w-last, Pinocchio-native).** Rejected
   for the _canonical_ ordering: `(w, x, y, z)` is the more common scalar-first
   convention (MuJoCo, Drake, most quaternion libraries) and keeps the canonical
   form readable. The Pinocchio adapter converts at its boundary (CC-9).

3. **Use each engine's native state and convert pairwise on demand.** Rejected
   for the same N²-converter / sign-drift reasons ADR-0012 already rejected for
   pose.

4. **Defer the angular-velocity-frame decision to each adapter.** Rejected: an
   unspecified frame is precisely the silent-divergence trap. The contract picks
   body-local; adapters convert; conformance verifies.

## Consequences

- **Positive**
  - One dynamic value type that every analysis (ZTCF/ZVCF, wrench, comparison)
    and the estimator consume, independent of engine.
  - Singularity-free base suitable for differentiation and arbitrary humanoid
    poses, unblocking the estimation milestone (M3).
  - The round-trip parity gate `from_canonical(to_canonical(s)) == s`
    generalises ADR-0012's pose gate to full dynamic state and becomes a CC-7
    conformance check.
  - Aligns the canonical layout with Pinocchio (the reference engine), reducing
    conversion surface for the most-used rigid backend.
- **Negative**
  - `nq ≠ nv` (quaternion redundancy) means consumers must use the manifold
    ops, not raw vector arithmetic, on the base — a learning cost mitigated by
    making `integrate` the only supported update path.
  - A second live convention (`v1` + `v2`) until pose-only callers migrate;
    bounded by keeping `v1` a strict subset.
- **Follow-ups**
  - CC-2 (#6774): `CanonicalState` + `quat_exp`/`quat_log`/`integrate` in
    `se3.py`, property-based round-trip tests, `pose_interchange.__version__`
    bump to `2.0.0`.
  - CC-5 (#6777): the adapter capability boundary carries
    `to_canonical`/`from_canonical` for `CanonicalState` — coordinated with the
    JaxSim capability-taxonomy work (#6647) so there is one taxonomy.
  - CC-9/CC-10 (#6782/#6783): reference and second adapter implement the v2
    remap and enter the conformance suite.

## Validation

- `docs/conventions/canonical-v2.md` is the authoritative contract; all CC
  issues reference it by path.
- CC-2 ships property-based (`hypothesis`) tests for the manifold round-trips
  (`from_tangent(to_tangent(s)) == s`) to `< 1e-9` (rigid DOF) / `< 1e-6`
  (quaternion DOF), including near-gimbal and large-rotation cases, with 100%
  branch coverage on the manifold ops.
- CC-7 adds the cross-engine round-trip and ID↔FD self-consistency checks that
  catch a wrong quaternion order or angular-velocity frame in any adapter.
