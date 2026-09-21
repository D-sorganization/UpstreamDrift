# ADR-0051: One Motion-Matching Abstraction Stack

- Status: Accepted
- Date: 2026-09-20
- Decision Makers: repo owner (Dieter), antigravity (local)
- Related Issues/PRs: #10331 (MS-12), #8867, #10043, #8864, #10337 (MS-30), #10338 (MS-31), #10340 (MS-41), #10156 (HO-2), #10522 (workspace handoff)

## Context

Over successive milestones, UpstreamDrift developed multiple concurrent representations and entrypoints for motion matching and trajectory optimization:

1. **The Modern Full-Body Pipeline (`MatchingPlant` + Receipts, MS-10)**:
   A ground-support pipeline built on `MatchingPlant` (`src/shared/python/motion_matching/pipeline/plant.py`), executing full-body IK, trajectory smoothing, dynamic tracking, and physical acceptance evaluation (`acceptance.py`), writing attested JSON receipts (`receipt.json`) validated against schema contracts (`receipt_schema.py`).
2. **The Provider Registry (`FitSwingProvider`)**:
   The engine-agnostic adapter interface (`src/shared/python/motion_matching/provider.py`) enabling dispatch across registered physics engines (`mujoco`, `drake`, `pinocchio`, `opensim`, `myosuite`, `pendulum`). Historically, providers handled club-only targets and returned `supports_body_target() == False`.
3. **The CIR Motion Pipeline (`src/shared/python/motion_pipeline/`)**:
   A legacy pipeline stack containing `ik/` and `matching/` solvers. Several backends (`ik/opensim`, `ik/mujoco`, `ik/drake`, `matching/cmc`, `matching/rra`) were placeholders or stubs that raised unconditional errors or remained unintegrated with the physical evidence pipeline.
4. **Representation Divergence (#8867)**:
   The CIR representation (`SkeletonRig`, `JointTrajectory` in `motion_pipeline/contracts.py`) and the canonical pose interchange (`CanonicalPose` in `pose_interchange/canonical.py`) lacked clean bidirectional bridging, impeding reuse across stages.
5. **Chart Metadata Ambiguity (#10043)**:
   Different visualization and export tools made conflicting assumptions about quaternion component ordering (`[w, x, y, z]` vs `[x, y, z, w]`), Euler angle conventions (intrinsic vs extrinsic), and coordinate axis definitions.

To deliver Milestone MS-12, UpstreamDrift requires a single, unified abstraction stack for motion matching with clear boundaries and retired legacy stubs.

---

## Decision

### 1. One Canonical Stack: Pipeline `MatchingPlant` + Receipts

The pipeline `MatchingPlant` protocol (MS-10) combined with attested execution receipts (`receipt.json`) is the **single canonical product path** for full-body matched swing optimization and simulation across all physics engines.

### 2. Provider Delegation & Body Target Support

- `FitSwingProvider.fit_swing(target, opts)` delegates to the full-body matching pipeline whenever a body target is provided (`has_body_target(target)`).
- `supports_body_target()` returns `True` for engines with verified, callable full-body fitting routes and evidence:
  - `mujoco` (ground-support pipeline, MS-10),
  - `drake` (native trajectory optimization, MS-30 #10337),
  - `pinocchio` (native Crocoddyl full-body fitting, MS-31 #10338).
- `supports_body_target()` returns `False` for engines without full-body lanes (`opensim`, `myosuite`, `pendulum`). Passing body targets to these engines fails closed with an explicit `ValueError`.

### 3. CanonicalFitResult & Receipts Contract

`CanonicalFitResult` (`src/shared/python/motion_matching/fit_result.py`) carries an explicit `receipt_path: Path | str | None = None` field.
When a body target fit executes, `receipt_path` is populated with the path to the written receipt JSON and is guaranteed to exist on disk.

### 4. Retirement of Legacy CIR Solvers

The unmaintained placeholder stubs in `src/shared/python/motion_pipeline/` are formally retired:

- `make_ik_solver` raises an actionable `NotImplementedError` referencing ADR-0051 when called for `opensim`, `mujoco`, or `drake`. Active IK solvers are `geometric` (dependency-free) and `pinocchio`.
- `make_matching_solver` marks `cmc` and `rra` as retired under ADR-0051. Active production matching solvers are `drake_trajopt`, `mujoco_torque`, and `pinocchio_inverse_dyn`.

### 5. Web Surface Routing & `motion_pipeline/api.py` (#8864 / MS-85)

`src/shared/python/motion_pipeline/api.py` is retained and reserved for MS-85 (general observation ingestion and web API routing), while tour and full-body matched swing workloads route through `MatchingPlant` and workspace session handoff (`model_match_handoff.py`).

### 6. Representation Parity (#8867)

`src/shared/python/pose_interchange/cir_bridge.py` provides lossless bidirectional conversion:

- `canonical_pose_to_joint_trajectory(pose, rig)`: Maps `CanonicalPose` SE(3) pelvis transform and joint angles into a CIR `JointTrajectory`.
- `joint_trajectory_to_canonical_pose(trajectory)`: Reconstructs a frozen `CanonicalPose` from CIR `JointTrajectory` frames.

### 7. Chart Metadata Contract (#10043)

`pose_interchange` is declared the single authoritative conversion provider. The chart metadata contract establishes:

- **Quaternions**: Standardized order is `[w, x, y, z]` (scalar real part first) with positive real part convention ($w \ge 0$).
- **Euler Angles**: Intrinsic XYZ Euler rotation in degrees.
- **Coordinate System**: Right-handed Cartesian with $+Z$ pointing vertically upward (gravity along $-Z$), $+X$ along the target line/forward, $+Y$ leftward.
- **Expression Frames**: Explicit declaration required (`"world"`, `"pelvis"`, `"ground"`, `"camera"`).

---

## Alternatives Considered

1. **Maintain Separate Club and Body Dispatch Hierarchies**:
   Rejected. Maintaining parallel provider registries for club and body targets caused code duplication, split test suites, and drift between engine capabilities.
2. **Keep CIR IK/Matching Stubs Active with Silent Fallbacks**:
   Rejected. Silent fallbacks (e.g. returning neutral poses) mask missing solver implementations and violate Design by Contract (DbC). Loud, actionable errors citing ADR-0051 direct developers to the real pipeline.
3. **Ad-Hoc Coordinate Conversions in UI/Visualizers**:
   Rejected. Ad-hoc conversions were historically a prime source of sign bugs and axis-inversion errors. Funneling all conversions through `pose_interchange` guarantees cross-engine consistency.

---

## Consequences

- **Positive**:
  - A single entrypoint for motion matching across all physics engines.
  - Provable qualification and reproducibility via on-disk receipt references (`receipt_path`).
  - Clear separation between active production solvers and retired placeholders.
  - Seamless interchange between CIR trajectories and canonical poses.
- **Negative / Trade-offs**:
  - Legacy callers attempting to instantiate retired IK/matching solvers will encounter immediate `NotImplementedError` or `ValueError` rather than quiet stubs.

---

## Validation

- Unit tests in `tests/unit/motion_matching/test_body_target_provider.py` verify `supports_body_target()` and receipt generation across all engines.
- Unit tests in `tests/unit/motion_pipeline/test_motion_pipeline_retirement.py` verify actionable errors referencing ADR-0051.
- Unit tests in `tests/unit/motion_matching/test_pose_interchange_parity.py` verify representation parity between `CanonicalPose` and `JointTrajectory`.
- CI architecture and lint checks verify conformance.
