# Full-Body Models Handoff (Epic #10062)

Updated 2026-09-13 (claude; leases on #10063 and #10064). Branch
`docs/10003-opensim-matching-epic`, worktree
`C:/Users/diete/Repositories/Worktrees/UpstreamDrift-opensim-10003`, PR not
created. Design: [EPIC_FULL_BODY_CONTACT.md](EPIC_FULL_BODY_CONTACT.md).
Copy-ready prompt for the next agent: [NEXT_AGENT_PROMPT.md](NEXT_AGENT_PROMPT.md).

## Done (Test-First, No Engine Required)

FB-1 (#10063) and FB-2 (#10064) are implemented and their child issues can be
closed by the PR that carries this branch.

- `src/shared/python/motion_matching/full_body_spec.py`: `BodySpec`,
  `JointSpec`, `LowerLimbExtension`, `ContactSpec`, `ContactSphere`,
  `MarkerAttachment`; `derive_full_body_spec`, `validate_full_body_spec`,
  `upper_body_slice`, `canonical_sha256`, `save/load_full_body_spec`,
  `pelvis_alignment` (Kabsch reuse). The validator enforces that the embedded
  upper-body slice is byte-identical to the qualified native spec, that every
  body is reachable from world through exactly one joint (reusing
  `order_native_tree`), SPD inertias, positive masses, the shared contact
  parameter set, tracked capture labels only, and provenance. Tests:
  `tests/unit/motion_matching/test_full_body_spec.py` (14).
- `src/shared/python/motion_matching/contact_law.py`: the shared law
  (`f_n = k d (1 + c d_dot)` clipped at zero; regularised Coulomb friction with
  static/dynamic/viscous terms and `tanh` transition), `ContactParameters`,
  `GroundPlane`, `ContactSample`, `sphere_ground_contact`,
  `random_contact_states`, `contact_parity_report`. Tests:
  `tests/unit/motion_matching/test_contact_law.py` (5) with closed-form values.
- Committed document `full_body_spec_v1.json` (canonical SHA256
  `06272a18eadda9fb89b2bca2c1948f7c57963ebb8e635193bf10947943c67a5f`), built
  by `build_full_body_spec.py` with `build_receipt.json`: 41 coordinates (the
  27 qualified upper-body coordinates first, then hip flexion/adduction/
  rotation, knee, ankle, subtalar and MTP per side), 24 bodies (14 upper plus
  femur, tibia, talus, calcn, toes per side from Rajagopal 2016 via the
  packaged golf_humanoid.osim), 23 joints, four foot contact spheres, and 34
  marker attachments (25 with the native lane's qualified offsets, 9 lower-limb
  and `RShoulderTop` attachments with offsets left `null` for FB-4).
- Pelvis alignment between the native `Hip` frame and the Rajagopal pelvis
  frame by rigid registration of the four waist markers: RMS residual 5.8 mm,
  transform recorded in the receipt.

## Simplifications Recorded in the Document

Patella bodies and patellofemoral coupler constraints dropped; the walker
knee reduced to a hinge about its primary axis (coupled spline translations
ignored); OpenSim body inertias interpreted as about the mass centre in the
body frame; contact stiffness, dissipation, friction and sphere placements
are placeholders sized from the calcaneus geometry; ground height is
uncalibrated (`calibrated: false`) until FB-4 sets it from toe markers in the
native world frame. The hip primitive sequence Rx, Ry, Rz maps onto OpenSim's
Z, X, Y flexion/adduction/rotation order through a recorded permutation; FB-3
forward-kinematics parity tests against OpenSim decide whether that mapping
and the Euler convention (SimTK body-fixed XYZ) are right, and must fail
loudly if not.

## FB-3-M Completed (#10066)

- `src/engines/physics_engines/mujoco/python/full_body_mjcf.py`: `export_full_body_mjcf`
  converts `full-body-v1` documents to MJCF XML preserving all 41 scalar joints
  (27 upper-body joints + 14 lower-limb joints), upper-body kinematic transforms
  and aggregated inertias, 16 frame sites, 2 weld closure sites, and adds 4
  calcaneus contact sphere geoms and sites (`contact_heel_r`, `contact_forefoot_r`,
  `contact_heel_l`, `contact_forefoot_l`). Stock MuJoCo contact solver is disabled
  (`flag contact="disable"`).
- `src/engines/physics_engines/mujoco/python/full_body_model.py`: `NativeMujocoFullBodyModel`
  provides full-body kinematics, `evaluate_contact_samples`, and `accelerations`.
  Applies the shared FB-2 Hunt-Crossley and regularized Coulomb contact law
  (`sphere_ground_contact`) via spatial wrenches in `data.xfrc_applied` and
  generalized force projection `jac_pos.T @ f_contact`. Resolves the dual-grip
  closure constraint via an explicit rigid solve.
- `tests/unit/motion_matching/test_full_body_mujoco.py`: 5 live simulation tests
  verifying compilation ($nq=41, nv=41$), upper-body slice inertia and FK parity
  to 1e-12, frame poses parity against the qualified native model, and exact 0.0 N
  contact force parity on the shared FB-2 harness.

## FB-3-P Completed (#10065)

FB-3-P is implemented and verified on ControlTower with Pinocchio 4.1.0:

- `src/engines/physics_engines/pinocchio/python/native_model.py`: extended with
  `FullBodyPinocchioModel` and `build_full_body_pinocchio_model` without touching
  the qualified `NativePinocchioModel`.
- Foot contact spheres added as operational frames to the Pinocchio model before
  `createData()`.
- Shared contact law evaluated via `contact_forces(...)` using
  `sphere_ground_contact` from `src.shared.python.motion_matching.contact_law`.
- Contact forces converted to generalized joint torques via spatial Jacobian
  projection $\tau_{\text{contact}} = \sum J^T F_{\text{contact}}$ and combined with
  actuator efforts into `pin.constraintDynamics(...)` retaining the weld loop
  closure constraint solver.
- Gate results:
  - Gate (a): Upper-body slice reproduces qualified model mass matrix and FK to
    $0.0$ max difference (< $10^{-12}$) across 20 random states.
  - Gate (b): Full-body FK matches spec frames to $0.0$ max difference (< $10^{-12}$)
    across 20 random states.
  - Gate (c): Contact forces at analytic states match the FB-2 reference with $0.0$
    difference in normal force, friction force, and penetration depth.
  - Gate (d): Closure residuals unchanged on the upper-body chain to $0.0$ max
    difference (< $10^{-12}$) across 20 random positions and velocities.
- Evidence archived under `docs/development/full_body_models/evidence/fb3_pinocchio/`:
  - Driver: `verify_pinocchio_full_body.py` (auto-dispatches to ControlTower if
    local Pinocchio is absent).
  - Receipt: `receipt.json`.
- Tests: `tests/unit/motion_matching/test_pinocchio_full_body_builder.py` (6 passed
  on ControlTower in 0.46s; clean skip on Windows).

## Next

- FB-3-D (#10067 Drake builder) consuming `full_body_spec_v1.json` with the 1e-12 upper-body slice parity test and shared contact forces.
- Cross-engine same-input replay and comparison across MuJoCo FB-3-M and Pinocchio FB-3-P.
- FB-4 ground height and marker calibration, and FB-5 fitting per Epic #10062.
