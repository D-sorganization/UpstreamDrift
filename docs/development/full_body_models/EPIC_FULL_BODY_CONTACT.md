# Epic: Full-Body Models With Lower Limbs and Ground Contact (MuJoCo, Drake, Pinocchio)

Status: FB-1 and FB-2 implemented 2026-09-13 (see HANDOFF.md beside this file;
spec SHA256 06272a18…); GitHub epic #10062 and children #10063–#10070 filed. Owner
lane: Claude for the shared contracts (FB-0, FB-1, FB-2); cheaper agents for
per-engine builders and verifiers once the contracts and red tests exist.

## Outcome

Three engines (MuJoCo, Drake, Pinocchio) gain a full-body variant of the golf
model: the qualified upper-body native model plus pelvis floating base, hips,
knees, ankles and feet, with rigid foot-ground contact, matching the 38-marker
tour-average capture (`data/C3D_TA_Driver.c3d`, 654 frames, 360 Hz, Y-up,
metres). The existing upper-body native models and their Simscape R2025b
parity are preserved untouched: the full-body variant is a separate model
specification with its own identity, never an edit of the qualified one.
Simscape stays upper-body (block-count licence limit) and remains the
reference for the upper-body subset. OpenSim's full-body golf humanoid (epic
#10003) consumes the same capture contract and marker inventory.

Acceptance for any engine's full-body variant is the same as the native lane:
an uninterrupted forward replay from the original initial state over the full
capture with global degree-six actuator polynomials, the marker gates on the
38-label set with masks, closure/contact audits, and same-input replay across
the other two engines. Kinematic IK fits, prefix fits and pose fits are
milestones, not acceptance.

## Shared Contracts (Do First, Test First)

### FB-0 Capture Contract and Full-Body Marker Inventory

Done in `src/shared/python/motion_matching/tour_capture_contract.py`: frozen
file identity, clock, units, axis, 38 labels grouped by segment, validated
loader. Every engine lane imports this; none re-reads the C3D on its own.
Remaining: none (tests `tests/opensim/test_tour_capture_contract.py`).

### FB-1 Full-Body Model Specification Schema

Extend the canonical native specification (`native_spec.py`, hashed JSON with
`coordinate_order`, `joints`, solids, weld sidecar) with lower-limb joints and
a `contact` block. Requirements:

- Coordinate order: existing 27 upper-body coordinates first, unchanged names
  and primitives; lower-limb coordinates appended (`pelvis_*` floating base,
  `hip_{flex,add,rot}_{l,r}`, `knee_{l,r}`, `ankle_{l,r}`, `subtalar_{l,r}`,
  `mtp_{l,r}`), so the upper-body subset of any full-body state is a valid
  native upper-body state (a test must assert this slice identity).
- Inertial data from the anthropometrics pipeline (`src/shared/python/anthropometrics`)
  or the Rajagopal segment table, with source and units recorded.
- `contact` block: foot contact geometry (heel/toe spheres per foot), the
  contact law family name, parameters and their provenance. The block is
  descriptive; each engine's adapter must implement the same law or fail.
- Marker attachments: the 38 labels mapped to bodies (reuse
  `MARKER_SEGMENTS`); offsets calibrated per engine in FB-4 from the same
  capture frames and stored with the spec hash they belong to.
- DbC: validator rejects unknown labels, duplicated coordinates, non-SPD
  inertias, missing contact provenance. DRY: one validator used by all
  engine builders; LoD: builders receive the parsed spec object only.

Tests (RED first): schema roundtrip, slice identity with the upper-body spec,
validator rejections, hash stability across key order.

### FB-2 Contact Law Contract and Cross-Engine Parity Harness

Define one contact law that all three engines can realise exactly enough to
compare: rigid ground plane with compliant Hunt-Crossley normal force and
regularised Coulomb friction, parameters in the spec. Provide a shared
`ContactSample` contract (per-sphere penetration, normal force, friction
force, world point) and a parity harness that drives the three adapters with
identical states and reports force differences. Stock engine contact
solvers (MuJoCo soft constraints, Drake hydroelastic/point, Pinocchio has
none) are NOT equivalent by default; each adapter must either implement the
shared law as an explicit external force or document the exact deviation and
its magnitude on the parity harness.

Tests: analytic penetration cases per engine (one sphere, known depth and
velocity, expected force), and a cross-engine parity test on random states.

## Per-Engine Work (Cheaper Agents, After FB-1/FB-2 Exist)

### FB-3-P Pinocchio Full-Body Builder

Extend `native_model.py`'s builder to consume the full-body spec, keeping the
scalar upper-body builder path untouched (add, do not modify). Add the
contact law as external forces in `accelerations`; keep the weld closure
solver. Tests: upper-body slice reproduces the qualified model to 1e-12 in
mass matrix and FK; full-body FK against the spec's frames; contact force
at analytic states. Real-engine tests skip without Pinocchio.

### FB-3-M MuJoCo Full-Body MJCF Export and Adapter

Extend `export_native_mjcf` with the lower-limb joints and contact geoms; the
adapter must apply the shared contact law (explicit `xfrc_applied`) rather
than MuJoCo's soft contact unless the parity harness shows equivalence within
a stated bound. Same slice-identity and FK tests; compile nq/nv checks.

### FB-3-D Drake Full-Body URDF and KKT Adapter

Extend the URDF/sidecar bundle with lower limbs; contact as externally applied
spatial forces in the existing KKT adapter. Same tests.

### FB-4 Marker Calibration and IK per Engine

Reuse the OpenSim lane's alternating calibration
(`opensim/python/tour_matching/marker_calibration.py`, pure Python with
injected FK/IK) to place the 38 markers on each engine's full-body model from
the capture frames. Each engine supplies FK and a least-squares IK callable;
the calibration algorithm is not re-implemented. Output: per-engine
calibrated marker offsets with spec hash, marker RMS per frame, an IK
trajectory for the full capture. This is a kinematic milestone.

### FB-5 Full-Body Forward-Dynamics Matching

Extend the two-window shooting fitter to the full-body coordinate set with
contact. Start from the FB-4 IK trajectory for node initialisation (nodes
only, never state resets during replay), the qualified upper-body polynomial
as the initial upper-body control, and zero lower-limb controls. Use the
native lane's receipts discipline: measured derivative floors, once-only
shared boundaries, budgets, uninterrupted original-state acceptance.

### FB-6 Cross-Engine Full-Body Parity and Visual Review

Same-input replay of the accepted candidate in all three engines with the
shared contact law; marker overlay animations per engine; parity receipts.

## Rules Every Child Follows

- TDD: failing test first, committed with the implementation.
- DbC: validate inputs in every public function; document postconditions.
- LoD: adapters take spec objects and arrays, never reach into engine
  internals of another module.
- DRY: capture loading, marker inventory, contact law parameters and the
  calibration algorithm live once in shared modules.
- Receipts: every numerical run archives inputs, hashes, environment, logs
  and a HANDOFF; failed runs stay archived.
- Never edit the qualified upper-body specification, its sidecar, or its
  MJCF/URDF outputs; full-body artefacts carry distinct hashes and names.

## Tracking

| Item   | Issue                   | Owner       | Depends on |
| ------ | ----------------------- | ----------- | ---------- |
| Epic   | #10062                  | claude      | —          |
| FB-0   | done in-tree (this doc) | claude      | —          |
| FB-1   | #10063                  | claude      | FB-0       |
| FB-2   | #10064                  | claude      | FB-1       |
| FB-3-P | #10065                  | codex/jules | FB-1, FB-2 |
| FB-3-M | #10066                  | codex/jules | FB-1, FB-2 |
| FB-3-D | #10067                  | codex/jules | FB-1, FB-2 |
| FB-4   | #10068                  | codex/jules | FB-3       |
| FB-5   | #10069                  | claude      | FB-4       |
| FB-6   | #10070                  | codex/jules | FB-5       |
