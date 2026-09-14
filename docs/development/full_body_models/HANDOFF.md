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

## FB-3-D Completed (#10067)

FB-3-D is implemented and verified on ControlTower with Drake 1.57.0:

- `src/engines/physics_engines/drake/python/full_body_urdf.py`: exports `full-body-v1`
  documents to URDF XML preserving all 41 scalar joints (27 upper-body joints + 14
  lower-limb joints), physical solid inertias, 16 upper-body frames, 2 weld closure
  frames, and 4 foot contact sphere links (`contact_heel_r`, `contact_forefoot_r`,
  `contact_heel_l`, `contact_forefoot_l`). Sidecar metadata emitted with `schema_version: 1`,
  `requires_sidecar: True`, and `representation: native-full-body-urdf-v1`.
- `src/engines/physics_engines/drake/python/full_body_model.py`: `FullBodyDrakeModel`
  provides full-body kinematics, `contact_forces`, and `accelerations`.
  Applies the shared FB-2 Hunt-Crossley and regularized Coulomb contact law
  (`sphere_ground_contact`) via spatial contact wrenches and generalized force
  projection $J_{\text{trans}}^T F_{\text{contact}}$ into the rigid continuous
  constrained KKT solve (`solve_weld_acceleration`) enforcing the 6-DOF dual-grip weld
  closure without numerical relaxation.
- Gate results:
  - Gate (a): Upper-body slice reproduces qualified model mass matrix and FK to
    $0.0$ max difference (< $10^{-12}$) across 20 random states, with exact link mass
    and inertia parity.
  - Gate (b): Full-body FK matches spec frames to $0.0$ max difference (< $10^{-12}$)
    across 20 random states.
  - Gate (c): Contact forces at analytic states match the FB-2 reference with $0.0$
    difference in normal force, friction force, and penetration depth across all 4 spheres.
  - Gate (d): Closure residuals unchanged on the upper-body chain to $0.0$ max position
    difference and $2.22 \times 10^{-16}$ max velocity difference (< $10^{-12}$) across
    20 random positions and velocities.
  - Accelerations: 41 coordinates, all finite under combined contact forces and weld closure.
- Evidence archived under `docs/development/full_body_models/evidence/fb3_drake/`:
  - Driver: `verify_drake_full_body.py` (auto-dispatches to ControlTower if local Drake is absent).
  - Receipt: `receipt.json` (`059ed50e48b65e0d84e3d7c7cfc7bfd0a99262470e28387e5f9b8a5ee0447001`).
- Tests: `tests/unit/motion_matching/test_full_body_drake.py` (6 passed on ControlTower
  in 0.98s; 1 passed, 5 cleanly skipped on Windows without pydrake).

## Visual Skeleton Layer Completed (Step 1 of the Visuals Plan)

- `src/shared/python/motion_matching/visual_skeleton.py`: one engine-agnostic
  visual description derived from any native or full-body spec (capsule per
  body, COM and frame spheres, ground plane opposite gravity with a
  `calibrated` flag, world-segment mapping for viewers); 5 unit tests.
- `src/engines/physics_engines/mujoco/python/visual_layer.py` and
  `export_full_body_mjcf(..., visual=True)`: massless non-colliding visual
  geoms, floor, lights, camera; physics proven identical to the plain export
  (masses, inertias, xpos, qacc) in `tests/unit/motion_matching/test_mujoco_visual_layer.py`.
- Evidence `evidence/visual_layer/` (default pose PNG, returned81 kinematic
  playback GIF, MJCF, receipt). Steps 3 to 6 (viewer tile, cross-engine replay
  videos, OpenSim real horizon, FB-3 re-checks) are specified for lower-level
  agents in [VISUALS_HANDOFF.md](VISUALS_HANDOFF.md).

## FB-4 Marker Calibration and IK per Engine Completed (#10068)

- Shared alternating marker calibration moved to `src/shared/python/motion_matching/marker_calibration.py`
  with backward-compatible re-export in `src/engines/physics_engines/opensim/python/tour_matching/marker_calibration.py`.
- Shared trajectory IK solver and marker RMS evaluator in `src/shared/python/motion_matching/full_body_ik.py`
  (`solve_full_body_ik_trajectory`, `compute_marker_rms_trajectory`).
  Features warm-started Levenberg-Marquardt optimization across frames, explicit dual-grip weld loop closure residual enforcement,
  and regularisation to guarantee $m \ge n$.
- Engine adapters implemented:
  - `src/engines/physics_engines/mujoco/python/full_body_ik.py` (`MujocoFullBodyIK`)
  - `src/engines/physics_engines/pinocchio/python/full_body_ik.py` (`PinocchioFullBodyIK`)
  - `src/engines/physics_engines/drake/python/full_body_ik.py` (`DrakeFullBodyIK`)
- Automated verification driver: `docs/development/full_body_models/evidence/fb4_calibration/verify_full_body_calibration.py`
  (supports `--engine mujoco|pinocchio|drake|all`, automatic dispatch to WSL Ubuntu on deskcomputer when local engine is missing).
- Evidence generated and archived for all three engines under `docs/development/full_body_models/evidence/fb4_calibration/`:
  - `mujoco/`: `calibrated_offsets.json`, `ik_trajectory.npz`, `receipt.json`
    - Subsample RMS (33 frames, 3 iters): 174.49 mm (best iter 3)
    - Full trajectory RMS (654 frames): 159.47 mm (mean frame RMS 157.67 mm, max 241.57 mm)
    - Weld closure max error: 0.37 mm
  - `pinocchio/`: `calibrated_offsets.json`, `ik_trajectory.npz`, `receipt.json`
    - Subsample RMS (33 frames, 3 iters): 240.51 mm (best iter 2)
    - Full trajectory RMS (654 frames): 178.52 mm (mean frame RMS 176.32 mm, max 297.48 mm)
    - Weld closure max error: 0.00 mm
  - `drake/`: `calibrated_offsets.json`, `ik_trajectory.npz`, `receipt.json`
    - Subsample RMS (33 frames, 3 iters): 317.16 mm (best iter 3)
    - Full trajectory RMS (654 frames): 176.47 mm (mean frame RMS 172.21 mm, max 544.30 mm)
    - Weld closure max error: 0.00 mm
- Head-marker limitation documented:
  - Three head markers (`HeadTop`, `HeadFront`, `HeadSide`) are attached to the single rigid trunk/head segment (`Hub`).
    Because the skeletal specification does not include an articulated cervical neck joint, independent head motions relative to the thorax produce higher rigid residual on `Hub`.
- Tests: `tests/unit/motion_matching/test_marker_calibration.py` (3 passed), `tests/unit/motion_matching/test_full_body_marker_calibration.py` (5 passed, 2 cleanly skipped on Windows).

## Next

- FB-5 Full-Body Forward-Dynamics Matching (#10069): Extend two-window shooting fitter to full-body coordinates with contact, initializing node trajectory from the FB-4 IK trajectories.
- FB-6 Cross-Engine Full-Body Parity and Visual Review (#10070).
