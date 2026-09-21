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
Completed in PR for #10070: all three engines (MuJoCo, Pinocchio, Drake)
replayed the 654-frame candidate uninterrupted with RK45 integration and
passed step-size convergence verification. Pairwise whole marker RMSE diff:
MuJoCo vs Pinocchio 1.9 cm; MuJoCo vs Drake 34.5 cm; Pinocchio vs Drake 36.4 cm.
Visual overlay animations and cryptographic receipts archived under `evidence/fb6_parity/`.

## GS Ground Support (User Direction 2026-09-13)

The user's direction supersedes the earlier "kinematic milestone" reading of
FB-4: whenever legs are shown, the golfer must be supported by the ground
through the shared contact law, hold dynamic balance, and the legs must
match the c3d leg markers under full physics; upper-body-only models are
exempt. Gates, each with its own receipt, in order:

- GS-0 Lower-limb geometry is anatomical: hip and knee axes verified by
  `test_lower_limb_axes.py` (v2 document); hip joints at functional centres
  from the capture (sphere sd below 5 mm); femur and tibia lengths scaled to
  the golfer by the pinned IK objective. Done in `evidence/ground_support`.
- GS-1 Ground and stance: ground height from the lowest toe markers with a
  stated standoff; per-frame stance spheres from marker heights relative to
  address; both feet flat at address.
- GS-2 Address pose: full 41-coordinate IK at frame 0 with stance spheres
  pinned, grip closed, CoM inside the support polygon; report per-segment
  RMS. Done: 3.1 mm whole, arms 0.5 mm, legs 5.4/5.2 mm.
- GS-3 Reference trajectory: full-capture IK with stance pins, calibrated
  leg offsets, zero-phase smoothing and a consistency re-solve; the
  reference must keep every stance sphere within 5 mm of the plane and the
  closure within 5 mm; whole RMS reported against the upper-body-only floor
  (26 mm mean with the qualified attachments).
- GS-4 Ground-supported dynamics: forward simulation with the shared contact
  law, unactuated root, joint torques only; over the full capture the
  weight fraction stays within [0.2, 3], the centre of pressure inside the
  support polygon for at least 90 % of stance frames, no sphere below the
  plane by more than the static penetration times five, root within 0.05 m
  of the reference, and marker RMS of the simulated motion reported. This
  is the "legs support the body weight" claim; nothing weaker counts.
  Status 2026-09-14: met to 1.0 s (root within 14 mm, weight fraction 0.36
  to 1.70), not met through the downswing (root 0.15 m at 1.3 s).
- GS-5 Cross-engine: the same controls replayed in Pinocchio and Drake with
  the shared contact law and closure (parity receipts) on the same document.
- GS-6 Matching: the FB-5 shooting fit on the full body with contact,
  accepted only by the uninterrupted replay gates of the native lane.
- AN-1 Anthropometric geometry (#10099): neutral address by a static
  trial, de Leva masses, subject lengths from the swing. Status
  2026-09-14: neck, anatomical wrist (cock plus flexion, neutral-grip
  offset), setup windows, CoM balance, typical clubs from the club
  database, both captures matched (IK 24.0 / 24.1 mm), torso and club
  visuals, human ranges flagged, setup parity for both, Motion Matching
  launcher tile; continued as epic #10113 (MM-1 to MM-10). Evidence
  `evidence/anthropometry/REVIEW.md` 9 to 11, `evidence/setup_parity/`.

## Direction (User, 2026-09-14)

The MuJoCo, Drake, Pinocchio and OpenSim full-body models are the showpiece
and may be improved beyond the block-limited (1000-block) Simscape model
wherever that makes them better golfers; Simscape remains the
cross-validation lane and an additional feature. Parity between the four
showpiece models is maintained through the shared document (one
`coordinate_order`, poses by coordinate name, `evidence/setup_parity`) and
the shared contact, closure and range-of-motion modules; every departure
from Simscape is recorded in the document (`closure_fit`, `subject`,
`visual_hints`) and in REVIEW.md.

MM-2 (#10104) resolved 2026-09-14: the constant hand-to-club rotation is
fitted from the matches (`grip_fit.py`), the wrists are bounded to the human
ranges in the matching, and both captures hold (REVIEW.md 14).

MM-7 (#10109) advanced 2026-09-14: the downswing replay holds to 38 mm
at the pelvis through impact with a compliant sole and a band-limited tracked
reference (was 178 mm, airborne); the reference zero-moment point diagnostic
shows the composite capture is not dynamically consistent for the model, so
the cart-table dynamics filter was tried and rejected; the FB-5 shooting
fit (iterative-learning pelvis command, replay in the loop) was implemented and diverges on both captures (pelvis yaw lag is a ground yaw-moment limit), so MM-7b became the MJX differentiable trajectory optimisation, which transfers within its 1.5 s cost window (shared-plant replay 56.1 to 40.2 mm, pelvis yaw lag at 1.4 s 12.3 to 2.6 deg) but not yet over the follow-through (REVIEW.md 17).

Handoff (2026-09-14): the remaining work is epic #10162 (HO-1 to HO-10,
tiered cheap/moderate/expert with implementation instructions per child).

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
