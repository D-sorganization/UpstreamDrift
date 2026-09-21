# Full-Body Models Handoff (Epic #10062)

## Global Polynomial Step Prerequisite (#10265)

The name-safe effort mapping and native RK4 step preserve the global degree-six
control contract and all six unactuated root coordinates. Derivatives apply
the chain rule through every RK4 stage and substep. Qualification here concerns
local numerical derivatives; it does not establish a fitted physical swing.
Independent acceptance replay must explicitly retain the optimization ground:
the historical forward runner currently auto-calibrates a near-zero plane.

## Qualified Solver Integration (#10254)

Resume with [Worker Turnover](../../plans/qualified_motion_integration/TURNOVER.md)
and [Pink Pipeline Packets](../../plans/qualified_motion_integration/PINK_TURNOVER.md).
Storage recovery is complete and native runtime probes pass. Adapter/derivative
PRs remain under review; production Pink selection and full-body Crocoddyl
qualification are not complete. The turnover records separate implementation,
CI and scientific gates.

The [integration plan](../../plans/qualified_motion_integration/README.md)
defines coordinate, closure, underactuation, timing and evidence boundaries.
First slice #10255 corrects the missing contact-force chain rule in full-body
Pinocchio derivatives. Independent real-engine differences reproduce failure
before the change and pass afterward; contact kinks remain explicitly nonsmooth.
No fit receipts were regenerated. The #10250 refresh is merged; #10271 receipt integrity and #10162's
physical acceptance remain prerequisites. #10256 and #10257 are separately
owned viewer and Pink adapter slices, not completed product integration.

## Optional Viewer Adapter Status (#10256, #10254)

MeshCat and Gepetto adapters now own persistent native visualizers, validate
configuration dimensions and preserve separate visual/collision models. Scene
cleanup is scoped to each adapter. Real MeshCat dispatch is exercised in the
opt-in heavy integration test; live Gepetto qualification remains pending.
These wrappers are prerequisites for replay integration, not evidence that the
full-body product already routes through them or that any model fit is valid.

## Reproducible Numerical Runtime (#10262)

Use the optional runtime manifest/lock and checker documented in
`docs/engines/pinocchio.md`. Real Crocoddyl descent and Pink hard-equality /
infeasible-QP probes pass in the consistent conda-forge stack. Preserve the
receipt's explicit source-freshness status; a runtime pass is not model or
physical acceptance. Linux lock requires x86-64-v3. Viewer qualification
remains separate.

## Pink Adapter Status (#10257, #10254)

The low-level Pink adapters now share validated state/time handling, retain
collision geometry, forward hard constraints and limits, refresh cached
kinematics and propagate infeasibility. Four real-native tests cover equality,
infeasibility, sequential state refresh and free-flyer `nq != nv` behavior.
Mixed unit/native collection isolates the real tests from unit mocks and
checks their executed results. This does not implement the full-body marker,
stance or weld tasks; those must use the qualified #10260 closure derivative
and explicit physical-time contract before production selection.

Finite weld correction #10260 differentiates the six-component pose error away
from closure with the correct SE(3) log Jacobian. Trajectory acceleration
linearization retains the separate constraint velocity Jacobian. Eleven real
closure regression tests and the 11 contact derivative tests pass on Pinocchio
3.8 and 4.1; no motion-fit evidence or acceptance thresholds changed.
The principal-log branch at rotation pi is rejected for derivative evaluation;
trajectory finite differences also reject steps that could cross that cut.

Updated 2026-09-16 (claude; HO-9 #10111). Branch
`feat/10111-de-leva-table`. Design: [EPIC_FULL_BODY_CONTACT.md](EPIC_FULL_BODY_CONTACT.md).
Design Decisions: [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md).
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
- Capsule radius policy (user review 2026-09-13: club and segments looked
  uniformly thick): radius of a uniform-density cylinder,
  `sqrt(m / (pi * 1500 kg/m^3 * L))`, clamped to 6 to 50 mm, and one capsule per
  child joint plus one to the farthest centre of mass. Club shaft 17 mm,
  forearms 36 to 40 mm, torso and thighs 50 mm (`address_segments.json`).
- Address-pose evidence at the returned81 candidate's q0 (the Simscape
  qualified start, legs at zero): `address_{front,side,top}.png` with frame-0
  capture markers (black) and native marker predictions (blue) overlaid
  (`render_address_views.py`, frame-0 valid marker RMS 2.9e-13 m), the
  labelled joint-to-joint Simscape skeleton `simscape_skeleton_address.png`
  with `address_posture.json` (`render_simscape_skeleton.py`), and
  `render_playback.py` for the default pose and playback GIF. No MATLAB is
  runnable on DeskComputer (`C:/Program Files/MATLAB/R2025b` has no
  `bin/matlab.exe`), so the Simscape skeleton is drawn from the ported
  geometry whose frame parity against R2025b is receipted at 2.1e-13 m
  (`native_frame_parity_r2025b.json`).
- Posture facts the user should decide on (model and start pose, not
  rendering): Simscape geometry inputs UpperArmLength 14.5 in (0.368 m) and
  LowerArmLength 12 in (0.305 m elbow to wrist); hub to shoulder 0.254 m;
  the hip, torso and spine joints coincide, so the pelvis-to-rod bend of
  -16.9 deg (SpineInputX) at q0 is what the frame-0 marker fit needed; the
  shoulders sit 0.09 m (L) and 0.19 m (R) below the hub at q0 (scapula
  coordinates 30/-40 deg and 35/38 deg). A straight spine or higher
  scapulae at address means refitting q0 (or re-calibrating attachments)
  with those coordinates constrained, then re-qualifying.

## Tour Matching Viewer Launcher Tile Completed (Step 3 of the Visuals Plan)

- Implemented pure-data engine-agnostic replay and forward kinematics reader
  `src/tools/tour_matching_viewer/core.py` (`load_replay`, `body_poses_from_state`,
  `viewer_frame`). Supports native `returned-replay.npz` and OpenSim IK `.mot`
  formats. Forward kinematics derived pure-Python from `full_body_spec_v1.json`
  evaluates to $< 10^{-12}$ error against MuJoCo forward kinematics across all 24
  bodies over 20 random states without importing `mujoco`.
- Implemented PyQt6/matplotlib 3D viewer `TourMatchingViewerWidget` and
  `TourMatchingViewerWindow` (`src/tools/tour_matching_viewer/gui.py`) with scrub
  slider, playback controls, target/model markers, visual skeleton segments,
  ground wireframe, and per-frame valid marker RMS readout.
- Embedded launcher tile via `_TourMatchingViewerEmbedAdapter` (`_embed_adapter.py`)
  registered into `EMBEDDABLE_TOOL_REGISTRY` and configured in `src/config/models.yaml`
  (`tool_id: tour_matching_viewer`, category `tool`, status `beta`).
- 10 unit tests passing in `tests/unit/tools/test_tour_matching_viewer_core.py` and
  `tests/unit/tools/test_tour_matching_viewer_adapter.py`.
- Rendered offscreen verification evidence archived under
  `docs/development/full_body_models/evidence/viewer/`:
  - `screenshot_returned81.png`
  - `screenshot_mot.png`
  - `receipt.json`
  - `returned81_replay.npz`
  - `opensim_os3b_ik.mot`

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

## FB-5 Full-Body Forward-Dynamics Matching Completed (#10069)

- Derivative floor and resolution utilities implemented in `src/shared/python/motion_matching/derivative_resolution.py`:
  - `measure_derivative_floor`, `compute_finite_difference_step_vector`, `DerivativeResolutionResult`.
  - Determines scale-aware perturbation steps and measures truncation vs. roundoff noise floors on stiff contact objectives.
- Multiple-shooting fitter in `src/shared/python/motion_matching/multi_shooting_fit.py` extended:
  - `shared_boundary_policy`: `"both" | "once"`, preventing sample duplication at shooting window interfaces.
  - `node_mode`: `"joint" | "fixed_nodes" | "nodes_only"`.
- Shared metrics extracted to `src/shared/python/motion_matching/tour_metrics.py`:
  - Standardized calculation of `whole_marker_rmse_m`, `early_marker_rmse_m`, `terminal_marker_rmse_m`, `club_marker_rmse_m`, `pelvis_yaw_rmse_rad`.
  - Backward-compatible re-export in `src/engines/physics_engines/opensim/python/tour_matching/metrics.py`.
- Native MuJoCo KKT solver hardened in `src/engines/physics_engines/mujoco/python/native_model.py`:
  - Damped regularization $W = J M^{-1} J^T + \epsilon I$ with `lstsq` fallback to prevent numerical constraint singularity failures.
- Uninterrupted forward dynamics simulation in `src/shared/python/motion_matching/full_body_forward_dynamics.py`:
  - Supports continuous `solve_ivp(method="rk45")` and semi-implicit Euler integration across all 41 coordinates.
  - Auto-calibrates ground plane height from lowest contact sphere at address.
  - Full ground contact force and penetration audit (`ContactAuditResult`).
- Automated verification runner `docs/development/full_body_models/evidence/fb5_matching/verify_full_body_matching.py`:
  - Measures derivative resolution floor: resolved at $h=1.00\times 10^{-6}$ with relative error $2.93\times 10^{-9}$ (noise floor $1.00\times 10^{-12}$).
  - Demonstrates two-window shooting fit setup with `shared_boundary_policy="once"`, node mode `nodes_only`, and nodes initialized from FB-4 IK.
  - Simulates uninterrupted original-state replay over all 654 frames from $(q_0, \dot{q}_0=0)$ with qualified native polynomial controls.
- Evidence archived under `docs/development/full_body_models/evidence/fb5_matching/`:
  - `forward_trajectory_mujoco.npz` (`time_s`, `q`, `qd`, `predicted_markers_m`).
  - `receipt_mujoco.json` containing environment, input SHA256 hashes, derivative floor resolution, multi-shooting setup, 5 shared metrics, and contact audit.
- Tests: `tests/unit/motion_matching/test_derivative_resolution.py` (4 passed), `tests/unit/motion_matching/test_multi_shooting_fit.py` (7 passed), `tests/unit/motion_matching/test_full_body_forward_dynamics.py` (2 passed).

## FB-6 Cross-Engine Full-Body Parity and Visual Review Completed (#10070)

- Shared cross-engine replay and visual review module implemented in `src/shared/python/motion_matching/cross_engine_replay.py`:
  - `CrossEngineReplayConfig`, `StepSizeConvergenceResult`, `compute_step_size_convergence`, `EngineReplayOutcome`, `CrossEngineComparisonReport`, `compare_engine_replays`.
  - 3D marker overlay frame generator and GIF animation renderer: `generate_overlay_frame`, `render_marker_overlay_animation` with target (green) vs. model (blue/red) marker visualization, candidate hash prefix in filenames, and configurable frame stride.
- Unified physics engine adapters:
  - `src/engines/physics_engines/mujoco/python/full_body_model.py`: unified `.coordinate_order` property and `evaluate_contact_samples`.
  - `src/engines/physics_engines/pinocchio/python/native_model.py`: unified `.coordinate_order` property and `evaluate_contact_samples`.
  - `src/engines/physics_engines/drake/python/full_body_model.py`: unified `.coordinate_order` property and `evaluate_contact_samples`; fixed sphere frame offset in address ground height calibration.
  - `src/shared/python/motion_matching/full_body_forward_dynamics.py`: adaptive RK45 integration across all three engines with configurable `rtol`/`atol` options and universal ground height auto-calibration for `ground_plane` (Drake), `ground` (Pinocchio), and `_ground_plane` (MuJoCo).
- Automated verification driver: `docs/development/full_body_models/evidence/fb6_parity/verify_cross_engine_parity.py`
  - Replays the accepted candidate (`returned-candidate.json`) over all 654 frames across all three engines (MuJoCo, Pinocchio, Drake).
  - Step-size convergence: verified on initial settling window via tolerance refinement (`rtol=1e-5, atol=1e-7` vs `rtol=1e-6, atol=1e-8`). All engines achieved `is_converged=True` (MuJoCo: 3.17e-3 rad, Pinocchio: 2.58e-5 rad, Drake: 1.06e-2 rad, all $\le 0.05$ rad tolerance).
  - Full-body parity metrics across 654 frames:
    - MuJoCo: whole marker RMSE 2.738 m, terminal RMSE 3.438 m, yaw RMSE 2.003 rad, max closure residual 2.675 m.
    - Pinocchio: whole marker RMSE 2.757 m, terminal RMSE 3.411 m, yaw RMSE 1.885 rad, max closure residual 2.467 m.
    - Drake: whole marker RMSE 2.393 m, terminal RMSE 3.521 m, yaw RMSE 1.626 rad, max closure residual 1.909 m.
    - Pairwise whole marker RMSE diff: MuJoCo vs Pinocchio: 0.019 m (1.9 cm); MuJoCo vs Drake: 0.345 m; Pinocchio vs Drake: 0.364 m.
- Evidence archived under `docs/development/full_body_models/evidence/fb6_parity/`:
  - Visual review animations:
    - `replays/overlay_mujoco_3f94aa92f28a.gif` (131 frames)
    - `replays/overlay_pinocchio_3f94aa92f28a.gif` (131 frames)
    - `replays/overlay_drake_3f94aa92f28a.gif` (131 frames)
  - Cryptographic receipts and parity comparison report:
    - `receipt_mujoco.json`
    - `receipt_pinocchio.json`
    - `receipt_drake.json`
    - `parity_report.json`
  - Compressed forward trajectories:
    - `forward_trajectory_mujoco.npz`
    - `forward_trajectory_pinocchio.npz`
    - `forward_trajectory_drake.npz`
- Tests: `tests/unit/motion_matching/test_full_body_parity.py` (4 passed), `tests/unit/motion_matching/test_full_body_forward_dynamics.py` (2 passed).

## Visuals Handoff Step 4 Completed: Same-Input Replays in MuJoCo, Pinocchio, and Drake (#10062)

- Replayed returned81 candidate trajectory (`returned-candidate.json`, SHA256 `dfafdff1cdec1a7fa15c41a34d41f054ef45d8a7df0a1faf8fab7898ab855776`) across all three physics engines (Pinocchio, MuJoCo, Drake).
- Shared 5-metric evaluator and standardized NPZ reader/writer implemented in `src/shared/python/motion_matching/replay_metrics.py`.
- Evaluated 307 frames on candidate coordinates:
  - MuJoCo: whole_rms_m 0.026366, early_rms_m 0.011427, terminal_rms_m 0.046305, club_cluster_rms_m 0.015955, pelvis_yaw_error_pct 13.923%. Max marker error vs Pinocchio: $2.35 \times 10^{-15}$ m.
  - Pinocchio: whole_rms_m 0.026366, early_rms_m 0.011427, terminal_rms_m 0.046305, club_cluster_rms_m 0.015955, pelvis_yaw_error_pct 13.923%.
  - Drake: whole_rms_m 0.026365, early_rms_m 0.011426, terminal_rms_m 0.046305, club_cluster_rms_m 0.015953, pelvis_yaw_error_pct 13.923%. Max marker error vs Pinocchio: $1.62 \times 10^{-5}$ m.
- Evidence archived under `docs/development/full_body_models/evidence/replays/`:
  - Three animated visual GIFs: `mujoco_returned81.gif`, `pinocchio_returned81.gif`, `drake_returned81.gif`.
  - Three standardized NPZ replays: `mujoco_returned81_replay.npz`, `pinocchio_returned81_replay.npz`, `drake_returned81_replay.npz`.
  - Engine receipts: `mujoco_receipt.json`, `pinocchio_receipt.json`, `drake_receipt.json`.
  - Combined cross-engine receipt: `receipt.json`.
- Tests: `tests/unit/motion_matching/test_returned81_cross_engine_replay.py` (3 passed).

## Ground Support Program (User Direction 2026-09-13, in Progress)

User direction: when the legs are shown, the golfer must be carried by the
ground through modelled contact, keep dynamic balance, and the legs must
match the c3d leg markers under full physics in MuJoCo, Drake and Pinocchio.
Gates: [EPIC_FULL_BODY_CONTACT.md](EPIC_FULL_BODY_CONTACT.md) section
"GS Ground Support". Branch feat/10062-visual-skeleton-layer, PR #10087.

Shared contracts (all with unit tests under `tests/unit/motion_matching`):

- `src/shared/python/motion_matching/ground_support.py`: capture to native
  world `(x, y, z) -> (x, -z, y)` (capture frame 0 is native t=0 exactly),
  ground height from the lowest toe markers minus a standoff, support report
  (weight fraction, centre of pressure, convex support polygon).
- `hip_calibration.py`: functional hip centres by sphere fit of the knee
  markers in the pelvis frame (sd 2 mm over 649 frames), anatomical pelvis
  axes (right = hip line, up = the Hip frame's +z, forward = up x right),
  rewrite of a document's hip joints. `segment_scaling.py`: length scaling
  of named bodies. `marker_calibration.calibrate_marker_offsets` gained an
  anatomical prior (`prior_offsets`, `prior_weight`).

MuJoCo lane:

- `full_body_markers.py`: marker FK on spec frames and bodies (lower-limb
  offsets mapped through `adapter.body_frames`, because MJCF bodies sit at
  the joint follower frame), projected Levenberg-Marquardt pose IK with grip
  closure, one-sided ground penalty, stance pins, planted-sphere anchors
  (`plant_stance`), CoM-over-support rows, joint bounds, locks, prior
  trajectory, body poses for the shared calibration.
- `full_body_simulation.py`: RK4 over the adapter's closure-constrained
  accelerations with an unactuated root; affine dynamics `a = A tau + b`
  from one KKT solve; least-norm inverse dynamics (closure removes six
  directions, singular values below 1e-2 dropped); computed-torque hold and
  tracking controllers with per-joint natural frequencies, optional CoM
  balance and root regulation; feet preload; support record. Standing hold
  on the balanced address posture: weight fraction 1.000, CoM drift 1.5 mm
  over 2 s.

Specification defects found and fixed:

- v1 had the hip and knee permutation matrices swapped in
  `build_full_body_spec.py` (hip flexion turned about the femur's long axis,
  the knee about the femur's y axis). `full_body_spec_v2.json`
  (`build_receipt_v2.json`) fixes it; `test_lower_limb_axes.py` guards it.
  v1 and every FB-3 receipt built on it are superseded for lower-limb use.
- The v1/v2 pelvis alignment (from the OS-3 pelvis offsets, a 0.2 m RMS fit)
  put the hips mirrored and about 0.15 m off; the driver relocates them to
  the functional centres (`full_body_spec_hipcal.json`).
- Heel and metatarsal spheres alone leave the address centre of mass 3 to
  6 cm ahead of the support polygon (the model tips forward); the driver
  adds toe spheres at calcaneus x = 0.23 m and raises the placeholder
  contact stiffness from 5e4 to 2e5 N/m (`full_body_spec_hipcal_scaled.json`).

Evidence `evidence/ground_support/` (`run_ground_support.py`, `receipt.json`,
`ik_trajectory.npz`, `dynamics_record.npz`, playback GIFs and frame
montages), spec SHA in the receipt:

- Address (GS-2): whole 3.1 mm, legs 5.4/5.2 mm, arms 0.5/0.4 mm, pelvis
  4.3 mm; hip flexion 44/59 deg, hip rotation -17/+7 deg; feet flat.
- Leg calibration: 23 to 16 mm on 109 frames; femur scale 0.97 chosen from a
  {0.94, 0.97, 1.00}^2 grid; after scaling 20 to 13 mm.
- Reference (GS-3): full-capture IK 28.8 mm whole (legs 12/14 mm, arms 31/26,
  pelvis 44, head 47, club 10) against an upper-body-only floor of 26 mm;
  smoothed 12 Hz plus consistency re-solve 28.5 mm, stance spheres within
  2.4 mm of the plane, planted-sphere drift 2.1 mm, closure 0.9 mm.
- Dynamics (GS-4, open): computed-torque tracking, unactuated root, joints
  to 0.0035 rad. To 1.0 s (address and backswing): root error max 14 mm,
  marker RMS 22 mm, weight fraction 0.36 to 1.70. Downswing and impact
  diverge: root error 21 mm at 1.1 s, 65 at 1.2, 151 at 1.3, 313 mm at
  1.75 s; whole-run marker RMS 174 mm, inside-polygon 85 %, peak torque
  4375 N m. Root regulation through the legs and CoM balance terms did not
  help (receipted in the session, not adopted). Joint bounds are the
  Rajagopal ranges widened 2x because the hip zero twist is not calibrated.

## Anthropometry and Posture Review (User Direction 2026-09-14)

The user asked why the torso looks arched at address and how the body
dimensions were determined. Answer in
[evidence/anthropometry/REVIEW.md](evidence/anthropometry/REVIEW.md) with
`receipt.json` (`review_anthropometry.py`): the golfer's markers show a hip
hinge with a modest trunk lean, no spinal arch; the model's bend is a 7 deg
extension plus a 21 deg side bend at its only trunk joint, which sits at the
base of the neck 0.515 m above the hips, with the hub 0.14 m above the
shoulder centre so the clavicle links point 21 and 47 deg downward. Upper
arms are 33 % and forearms 15 % too long, the trunk-plus-head 70 % too heavy
(model 108 kg versus about 78 kg). Shared modules added: `anthropometry.py`
(de Leva table, transcribed, verify against the paper before qualification),
`posture_metrics.py`, `anthropometric_candidate.py` (de Leva lengths,
masses and inertias on the current topology; unqualified). The candidate
run (`ground_support/candidate_anthro/`) shows scaling alone makes the fit
worse (48 mm), so the trunk topology must change; section 8 of the review
is the ready-to-file child issue. The driver `run_ground_support.py` gained
`--anthropometric`, `--recalibrate-upper`, `--out` and posture metrics in
its receipt.

## AN-1 Iteration 1: Anthropometric Native Geometry (#10099, in Progress)

Findings and receipts in [evidence/anthropometry/REVIEW.md](evidence/anthropometry/REVIEW.md)
section 9. Code (all with unit tests, 15 in the touched files):

- `src/shared/python/motion_matching/anthropometric_geometry.py`:
  `build_upper_body(native, stature_m, mass_kg, trunk_scale, arm_scale,
shoulder_scale)`; 27 native coordinate names, native body/joint/frame
  names, club and closure verbatim; pelvis, trunk to the shoulder centre,
  hub at the shoulders, scapula `Rx` elevation + `Rz` protraction (the
  native `Ry` was a pure spin), upper arms forward at zero pose so the
  shoulder gimbal stays away from its singularity, one-sided elbows;
  `COORDINATE_RANGES_DEG` and `ADDRESS_SEED_DEG` travel in the document.
- `docs/development/full_body_models/build_anthropometric_spec.py` writes
  `full_body_spec_anthro_<club>.json` (+ `build_receipt_anthro_<club>.json`) with
  the Rajagopal legs on a fixed pelvis alignment, toe spheres, 2e5 N/m.
  Canonical scales: trunk 1.15, arm 1.10, shoulder 1.00 at 1.71 m, 78 kg.
- `marker_calibration.static_marker_offsets` (static-trial placement);
  `full_body_markers.solve_pose(marker_weights=...)` and
  `solve_trajectory(restarts, restart_threshold_m)`.
- Driver `run_ground_support.py`: `--static-seeds` (neutral address with
  locked scapulae and a bounded spine, offsets from the first 24 frames),
  document bounds and seed for anthropometric documents, head markers at
  weight 0.1 in every IK (`HEAD_MARKER_WEIGHT`), four restarts above 30 mm.
  These last two change the baseline numbers of every rerun: report
  body-only and all-marker RMS side by side.
- `evidence/anthropometry/scan_geometry.py` ranks (trunk, arm, shoulder)
  scales by the decimated-swing body-marker RMS with static offsets and no
  calibration (`scan_geometry_receipt.json`).

State against the #10099 acceptance (full driver run, REVIEW.md 9.1,
`evidence/ground_support/anthro_driver/receipt.json`): address 2.1 mm with
spine bend 0.3 deg forward, 6.3 deg lateral and clavicle links within 5 deg
of horizontal (met); full-capture IK 26.2 mm on the non-head markers, equal
to the qualified geometry's 26.2 mm, 40.0 mm with the deweighted head (no
neck; 104 mm); tracking to 1.0 s root error 5 mm (qualified 14 mm),
whole-run root RMS 49 mm (qualified 172 mm), weight fraction still reaches
0 (GS-4 open). Solver additions for this: `continuous_branches` (Euler
branch and 2 pi continuity before smoothing), per-coordinate
`prior_weights` (0.1 on the three collinear spins), restart margin 3 mm.
Renders: `evidence/visual_layer/*_driver.*` and `*_iron.*` (scripts take `--spec
--trajectory --suffix`). Unqualified until Simscape carries the geometry.

### Neck and Address Arms (User Direction 2026-09-14)

Head body on a three-axis neck (`NeckInputX/Y/Z`, 30 upper coordinates;
Simscape has none, accepted), head markers at full weight; address elbows
bounded in the static trial with weak elbow-pit axis rows
(`solve_pose(axis_targets=...)`) and restarts in the address fit; lane
settings shared by driver and scan (`configure_lane`). Receipt
`evidence/ground_support/anthro_driver/receipt.json`: full-capture IK 30.2 mm
over all markers (head 28 mm), address 11.4 mm neutral, left elbow -25 deg,
right -2 deg, backswing root error 7 mm. The left elbow pit still faces
outward: forcing it inward breaks the swing fit (REVIEW.md 9.2 lists the
four attempts); the wrist frame copied from the native document is the
suspect. Renders `evidence/visual_layer/*_driver.*` and `*_iron.*` refreshed.

### Wrist Axis, Setup Position, Parity, Centre of Mass (2026-09-14)

Wrist cock axis rolled onto the elbow axis (`WRIST_ROLL`), lead scapula
retraction and elbow windows in every address fit, centre-of-mass overlay
(`visual_layer.add_com_markers`, driver playback and address views,
`centre_of_mass` in the address receipt), cross-engine setup parity
(`evidence/setup_parity/`: Pinocchio 9e-16 m, Drake 6.5e-06 m on the
same document; poses translate by coordinate name). Receipt (then anthro_v1, now anthro_driver):
IK 30.4 mm all markers, address 8.3 mm, elbows -7.5/-4.7 deg,
backswing root 5 mm, CoM inside the polygon at address: True.
Open: the left elbow pit faces outward (REVIEW.md 9.3).

### Clubs, Two Captures, Torso Visuals, Ranges of Motion (2026-09-14)

`club_models.py` (driver, 7-iron; `apply_club`), documents
`full_body_spec_anthro_driver.json` and `full_body_spec_anthro_iron7.json`
(builder `--club`), the 7-iron capture registered in the contract
(`--capture iron`), `visual_hints` for torso ellipsoids, clavicles, shaft and
club heads, `range_of_motion.py` with receipt flags, address balance rows.
Receipts `evidence/ground_support/anthro_driver/` and `anthro_iron/`:
IK 24.8 / 34.1 mm, address 7.1 / 15.5 mm, CoM
inside the polygon in both; setup parity `evidence/setup_parity/receipt_*.json`.
REVIEW.md section 10.

### Anatomical Wrist, Visual Realism, Launcher Tool, Epic #10113 (2026-09-14)

Wrist Rz flexion with a 25 deg neutral-grip offset (driver IK 24.0 mm,
7-iron 24.1 mm; wrists flagged, not bounded: bounding collapses the fit
because the left humerus roll is wrong, MM-2/MM-5); thicker legs, slimmer
torso, hidden marker spheres; `club_models.from_database`; the "Motion
Matching" launcher tile (`src/tools/motion_matching`, tests under
`tests/tools/motion_matching`). Epic #10113 tracks MM-1 to MM-10; REVIEW.md
section 11 carries the table.

### Marker-Driven Elbow Pits, Grip Roll, Club Mesh Epic (2026-09-14)

Elbow pits follow the markers per frame (`posture_metrics.elbow_pit_direction`,
`Lane.pit_targets_per_frame`); `GRIP_ROLL_DEG` / `--grip-roll` exposes the
hand roll about the shaft and `evidence/anthropometry/scan_grip_roll.py`
calibrates it (roll 0 deg is best with a total excursion of 224 deg (lead cock 122 deg beyond its range), +45 deg 277, -45 deg 333, -90 deg 303, +90 deg 451, so no roll brings the wrists within human ranges and the roll is not the lever). Receipts (wrists flagged): driver IK
26.0 mm, 7-iron 24.1 mm. Bounding the wrists still costs the
fit until the roll is calibrated (REVIEW.md 12, MM-2 #10104). Club meshes
are epic #10120.

### Closure Fitted From the Address, Showpiece Direction (2026-09-14)

User direction: MuJoCo, Drake, Pinocchio and OpenSim full-body models are
the showpiece and may improve beyond the block-limited Simscape model;
Simscape stays the cross-validation lane. `closure_fit.py` and the driver's
`--fit-closure` fit the two-hand weld from the address with anatomical
wrists. Driver result: open-chain address fit 43.7 mm (hands held on the grip point, weld orientation free, trail wrist locked at ulnar -10 / flexion 0 / pronation 30 deg, lead wrist bounded), weld turned 65.6 deg and moved 1.2 mm, address with the fitted weld and bounded wrists 41.1 mm, full-capture IK 67.8 mm (`anthro_driver_fit/receipt.json`); the lead cock sits at its +25 deg radial limit at address. Verdict: fitting the weld from the address does not make the human wrist ranges reachable either; with the pits fixed by the markers the lead wrist still needs +50 to +65 deg of cock at address and 110 deg of travel through the swing (the unbounded receipts), about twice a human radial-ulnar range. The C3D carries no hand markers, so the wrist axes are observed only through the club: the remaining hypothesis is that the cock coordinate is absorbing motion that belongs to flexion/extension and pronation because the wrist base axes are still rolled relative to the golfer's hand, and the test for it is to fit the hand frame from the club orientation at three swing phases (address, top, impact) and solve the constant hand-to-club rotation that minimises the wrist excursions jointly, rather than the shaft roll alone. The driver keeps the wrists flagged (driver 26.0 mm, 7-iron 24.1 mm) and `--fit-closure` stays available as an experiment with its receipt. Details REVIEW.md 13.

### Hand-to-Club Rotation Fitted, Wrists Bounded (MM-2, 2026-09-14)

`src/shared/python/motion_matching/grip_fit.py` fits the constant rotation
of each hand on its wrist from the matched swings (the one unobserved
constant of the hand-club chain; the C3D has no hand markers). Fitted over
the driver and 7-iron together: lead (-89.7, 46.7, 0.0) deg, trail
(-34.2, 46.0, 62.0) deg, wrist excursions beyond the human ranges 40.6 ->
3.9 deg RMS (lead) and 17.7 -> 1.2 deg (trail). These are the builder
defaults (`GRIP_ROTATION_DEG`) and the driver now bounds the wrists and
forearms by default for fitted documents. Receipts: driver address 5.1 mm,
IK 27.3 mm; 7-iron 4.4 / 28.6 mm; no wrist flags; parity Drake 6e-6 m,
Pinocchio 1e-15 m. Details REVIEW.md 14; derivation
`evidence/anthropometry/fit_grip_rotation_receipt.json`.

### Downswing Dynamics: Compliant Sole, Tracked Reference, ZMP (MM-7, 2026-09-14)

`evidence/ground_support/downswing_experiment.py` replays the dynamics
stage of a finished run with one setting changed (32 receipts under
`anthro_driver/downswing_*.json`). Reference jerk, friction creep, friction
coefficients and root regulation were ruled out; the feet were being
unloaded by sub-degree root tilt on a 200 kN/m sole. Adopted: anthropometric
documents carry a 50 kN/m, 2 s/m sole (`build_anthropometric_spec.py`) and
the driver tracks the re-solved reference through a 12 Hz low-pass
(`TRACKING_CUTOFF_HZ`). Driver: root error 38 mm through impact
(was 178), never airborne, peak torque 476 N m (was 2523), whole-run
marker RMS 74.6 mm; 7-iron root 31 (to 1.5 s; 133 at 1.75 s in the follow-through) mm, marker RMS
112.3 mm. `full_body_simulation.reference_zmp` (in every receipt as
`dynamics.reference_zmp`) shows the composite reference's zero-moment point
outside the feet on 77 % of the downswing frames: the remaining
3 to 4 cm is the reference, not the controller. Details REVIEW.md 15.

### Cart-Table Dynamics Filter Tried (MM-7B, Not Adopted, 2026-09-14)

`src/shared/python/motion_matching/dynamics_filter.py`, centre-of-mass rows
in the marker solver (`com_target`, `com_targets_per_frame`) and the driver's
`--zmp-filter` implement the cart-table zero-moment-point correction. On the
driver it moves the centre of mass 112 to 278 mm, raises the reference's
marker error from 27 to 98 mm and makes the replay worse (497 mm); the
excursions come from the arm-club angular momentum the cart table ignores.
Kept as an experiment (`anthro_driver_zmp/receipt.json`); MM-7b now means a
whole-body shooting fit (FB-5). Details REVIEW.md 15.

### Contact-Aware Shooting Fit (FB-5, MM-7B, 2026-09-14)

`run_ground_support.py --shooting-fit N` replays the tracked reference,
moves the pinned pelvis command against the replay's drift (iterative
learning) and re-solves the joints against the markers
(`solve_trajectory(locked_per_frame=...)`); the best replay is kept.
On both captures every gain diverges from iteration 0 (driver 74.6 to 134.8 mm at gain 0.7, to 103.9 mm at gain 0.25; 7-iron 112.3 to 192.9 mm), so the dynamics stage keeps the unmodified reference. The replay error is pelvis yaw lag (72.7 of 74.6 mm), a ground yaw-moment limit for this composite reference; the next form is a differentiable-simulator trajectory optimisation (JaxSim #6647 or MJX). Details REVIEW.md 16.

### Differentiable Trajectory Optimisation With MJX (FB-5, MM-7B, 2026-09-14)

`evidence/ground_support/export_mjx_package.py --run <run>` then, in the
MJX environment, `mjx_trajectory_optimisation.py --run <run> --iterations N
--learning-rate 5e-4` (`--diagnose` replays only, `--init` warm-starts,
`--horizon` sets the cost window). Environment recipe (Windows, CPU):
`python -m venv ~/.venv-mjx && ~/.venv-mjx/Scripts/pip install "jax[cpu]"
mujoco-mjx defusedxml numpy scipy` (JAX 0.11.1, MuJoCo 3.13, MJX 3.13; the
main environment keeps MuJoCo 3.3.4). Validate any optimised reference in
the shared-law plant with `downswing_experiment.py --run <run> --reference
<npz>`. Driver result: to 1.5 s the shared-plant replay drops from 56.1 to 40.2 mm (pelvis yaw lag at 1.4 s 12.3 to 2.6 deg); the uncosted follow-through collapses from iteration 8, so iterations 4 to 6 are the whole-swing choice (79.6 mm against 74.6). The MJX plant's own follow-through diverges after 1.6 s (not the weld: a five-times stiffer one is identical), and the joint-by-joint comparison shows it is the pelvis yaw response of the contact (joints track equally; MJX lags 50.6 deg in the follow-through, the shared plant 34.4, and less than the shared plant through the downswing), so the contact integration is the next thing to reconcile before a full-horizon solve. Details REVIEW.md 17.

### Completion Plan (Handoff Epic #10162, 2026-09-14)

The remaining work is organised as epic #10162 with tiered, self-contained
children (each carries files, TDD steps, DbC/LoD/DRY constraints, commands,
acceptance criteria and receipts): expert HO-5 (#10159, MJX plant
reconciliation, full horizon, 7-iron); moderate HO-4 (#10158, tile stages),
HO-8 (#10108, hip zero twist), HO-10 (#10112, dynamics replay parity);
cheap HO-1 (#10155, pipeline package under src, first), HO-2 (#10156,
receipt schema), HO-3 (#10157, MJX environment and JAX-gated tests), HO-6
(#10160, 7-iron receipts), HO-7 (#10161, design-decision record), HO-9
(#10111, de Leva verification); blocked #10110 (Simscape lane). Order:
HO-1, then HO-2 and HO-3, HO-7 and HO-9 any time, HO-4/HO-8/HO-10 after
HO-1, HO-6 after HO-5. Definition of done is on the epic.

**HO-0 (#10186, expert) comes before all of them (2026-09-15).** The 22
commits after PR #10087 (`11cccded9` to `ee418af35`) are on this branch
only, and `main` took a parallel lane on the same program (FB-4 #10089,
FB-5 #10092, FB-6 #10094, viewer tile #10090, fixes #10164/#10165): a
merge conflicts in 22 files (dry run 2026-09-15). HO-0 lands the branch on
`main` with one merge commit resolved by the per-file rules in the issue
(branch side for the 20 visual-layer files, main side for SPEC.md and the
divergence inventory, hand-merge of this file and the development log,
`static_marker_offsets` ported into the shared `marker_calibration`
module), records a "Two Implementations, One Program" table (main's
`full_body_ik.py`/`full_body_forward_dynamics.py`/`cross_engine_replay.py`
against this lane's `full_body_markers.py`/`full_body_simulation.py`/
`verify_setup_parity.py`), and repoints the epic and children at `main`.
Until HO-0 merges, child branches must start from this branch and will
not merge; nobody should start HO-1 before HO-0 is done.

### Two Implementations Coexist (Read This First)

Two implementations of the full-body program currently coexist in the repository: the FB-4/5/6 lane landed on main (#10089, #10092, #10094) and the Ground Support lane landed in this PR (#10186, #10113, #10162). Until unified in HO-1 (#10155), see the [Two Implementations, One Program](#two-implementations-one-program) reconciliation table below for canonical assignments and receipts.

### How to Continue (Read This First)

1. Run the pipeline from the launcher tile "Motion Matching" (`python -m src.tools.motion_matching`) or headlessly via `python docs/development/full_body_models/evidence/ground_support/run_ground_support.py --spec <doc> ...`. The launcher tile provides a tabbed interface:
   - **Matching Tab**: Full pipeline execution with granular stage controls (free/bound wrists, fit closure, cart-table ZMP filter, whole-body shooting fit iterations and gain, cutoff frequency).
   - **Downswing Experiment Tab**: Run parameter sweep and variant comparison experiments against reference fits with custom torque/gain overrides.
   - **MJX Tab**: Export MJX optimization packages and validate optimized trajectory references against the shared contact-law plant.
     Headless pipeline runs: `python docs/development/full_body_models/build_anthropometric_spec.py ... --club driver|iron7`
     then `python docs/development/full_body_models/evidence/ground_support/run_ground_support.py --spec <doc> --skip-hip-calibration --static-seeds [--free-wrists] --capture driver|iron --out <run>` (wrists bounded by default for fitted documents; refit the hand rotation with `evidence/anthropometry/fit_grip_rotation.py --run <free-wrist run> ...`).
2. Read the receipt (`<run>/receipt.json`): `address.calibrated`, `ik`
   (full-capture IK, `range_of_motion_flags`, `attachments_m`), `dynamics`
   (`root_error_timeline_m`, `weight_fraction`, `reference_zmp`,
   `backswing_to_1s`); playback GIFs. Dynamics-only variants:
   `evidence/ground_support/downswing_experiment.py --run <run> --name <n> [...]`.
3. Cross-engine: `evidence/setup_parity/verify_setup_parity.py --run <run>`
   (Drake and Pinocchio on ControlTower over SSH; poses by coordinate name).
4. Never rerun with `--recalibrate-upper`; keep one static-trial round;
   ranges act on the matching only (`range_of_motion.py`).
5. Epics: #10162 (handoff, HO-0 landing #10186 first, then HO-1 to HO-10, tiered), #10113 (MM-1 to MM-10)
   and #10120 (CM-1 to CM-6) hold every open item with acceptance
   criteria; update DL-#10062 and this handoff in every implementation
   commit.

## Two Implementations, One Program

The following table records the canonical module assignments to be executed in HO-1 (#10155):

| Concern                       | On `main` (FB-4/5/6 lane)                                                                                                   | On the branch (ground-support lane)                                                                                                                  | Canonical Choice for HO-1                                                                                                                                 | Validating Receipt / Rationale                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Marker IK                     | `src/shared/python/motion_matching/full_body_ik.py` (240), `src/engines/physics_engines/mujoco/python/full_body_ik.py` (92) | `src/engines/physics_engines/mujoco/python/full_body_markers.py` (775; static seeds, planted stance, axis rows, CoM rows, `locked_per_frame`)        | **Consolidated** (`full_body_markers.py` retired to deprecation shim; shared IK in `full_body_ik.py`, MuJoCo adapter in `mujoco/python/full_body_ik.py`)  | Validated by tour capture receipts: driver address 5.1 mm (`docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json#address.calibrated.marker_rms_m`), IK 27.3 mm (`docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json#ik.marker_rms_m`); 7-iron address 4.4 mm (`docs/development/full_body_models/evidence/ground_support/anthro_iron_zmp/receipt.json#address.calibrated.marker_rms_m`), IK 28.6 mm (`docs/development/full_body_models/evidence/ground_support/anthro_iron_zmp/receipt.json#ik.marker_rms_m`). Primary uncalibrated baselines: driver address 42.2 mm (`docs/development/full_body_models/evidence/ground_support/anthro_driver/receipt.json#address.calibrated.marker_rms_m`), IK 52.3 mm (`docs/development/full_body_models/evidence/ground_support/anthro_driver/receipt.json#ik.marker_rms_m`); 7-iron address 70.2 mm (`docs/development/full_body_models/evidence/ground_support/anthro_iron/receipt.json#address.calibrated.marker_rms_m`), IK 72.0 mm (`docs/development/full_body_models/evidence/ground_support/anthro_iron/receipt.json#ik.marker_rms_m`). Carries static trial placement, planted stance, and anatomical joint limits. |
| Forward dynamics and tracking | `src/shared/python/motion_matching/full_body_forward_dynamics.py` (785; zero-feedback, polynomial control, contact audit)   | `src/engines/physics_engines/mujoco/python/full_body_simulation.py` (692; computed torque with root free, acceleration feedforward, `reference_zmp`) | **Consolidated** (`full_body_simulation.py` retired to deprecation shim; forward dynamics and tracking consolidated into `full_body_forward_dynamics.py`) | Validated by whole-run marker RMS 74.6 mm (`docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json#dynamics.marker_rms_m`), 89.1 mm (`docs/development/full_body_models/evidence/ground_support/anthro_driver/receipt.json#dynamics.marker_rms_m`), 116.0 mm (`docs/development/full_body_models/evidence/ground_support/anthro_iron/receipt.json#dynamics.marker_rms_m`), and 144.0 mm (`docs/development/full_body_models/evidence/ground_support/anthro_iron_zmp/receipt.json#dynamics.marker_rms_m`) with unactuated floating root, compliant sole, and computed-torque tracking.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| MJCF and native model         | `src/engines/physics_engines/mujoco/python/full_body_mjcf.py` (172), `native_model.py` (226, on both)                       | `src/engines/physics_engines/mujoco/python/full_body_model.py` (211; anthropometric documents, toe spheres, closure sites)                           | **Ground Support Lane** (`full_body_model.py` to be consolidated into `full_body_mjcf.py` / `native_model.py`)                                            | Carries anthropometric documents, toe contact spheres, and dual-grip closure sites required for ground support.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Cross-engine replay           | `src/shared/python/motion_matching/cross_engine_replay.py` (366), `derivative_resolution.py` (193)                          | `docs/development/full_body_models/evidence/setup_parity/verify_setup_parity.py` (poses by coordinate name over SSH)                                 | **Main Lane** (`cross_engine_replay.py`, `derivative_resolution.py`)                                                                                      | Validated by `evidence/fb6_parity/parity_report.json` and step-size convergence analysis across MuJoCo, Pinocchio, and Drake; replaces the SSH script in #10112.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Contact law                   | `src/shared/python/motion_matching/contact_law.py`                                                                          | same module (imported by the branch's simulation)                                                                                                    | **Identical**                                                                                                                                             | Same shared module used across both implementations; no conflict.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Evidence                      | `evidence/fb4_calibration`, `fb5_matching`, `fb6_parity`, `viewer`                                                          | `evidence/ground_support`, `anthropometry`, `setup_parity`, `visual_layer`                                                                           | **Both Retained**                                                                                                                                         | Both sets of evidence are retained in the tree under `docs/development/full_body_models/evidence/`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |

## Status and Next Steps

- **HO-0 (#10186)**: Landed on `main` reconciling ground-support and FB-4/5/6 lanes.
- **HO-1 (#10155)**: Landed on `main` in PR #10218 (`afea5e0a8`). Ground-support pipeline package established under `src/shared/python/motion_matching/pipeline/` with CLI wrapper in `docs/development/full_body_models/evidence/ground_support/run_ground_support.py`.
- **HO-2 (#10156)**: Implemented Pydantic V2 receipt schema and validator (`src/shared/python/motion_matching/pipeline/receipt_schema.py`), unit tests validating all committed receipts and rejection paths (`test_receipt_schema.py`), generated markdown documentation (`docs/development/full_body_models/RECEIPTS.md`), and freshness test (`test_receipts_markdown_freshness.py`).
- **HO-7 (#10161)**: Consolidated seventeen sections of findings from `evidence/anthropometry/REVIEW.md` into authoritative design decision record `docs/development/full_body_models/DESIGN_DECISIONS.md`. Established structured parser, validator with DbC contracts, and full test suite `tests/unit/motion_matching/pipeline/test_design_decisions.py`. Merged to `main` in PR #10235 (`27b271fe2`).
- **HO-4 (#10158)**: Exposed all pipeline stages in the Motion Matching launcher tile (`src/tools/motion_matching/gui.py`, `pipeline.py`). Tabbed interface across Matching (with granular Stages options for free/bound wrists, fit closure, ZMP filter, shooting fit, cutoff frequency), Downswing experiment, and MJX tabs, with asynchronous `RunWorker` and strict LoD. Headless PyQt6 and pipeline test suites in `tests/tools/motion_matching/`. Added `tools.motion_matching` to `src/config/feature_parity.json` (tracking gap #10106) and regenerated `docs/development/feature_parity_matrix.md`. Merged to `main` in PR #10236 (`db4fe88c4`).
- **HO-9 (#10111, MM-9)**: Verified de Leva (1996) male segment table line-by-line against Table 4 of the published paper. Pinning unit test in `tests/unit/motion_matching/test_de_leva_table.py` asserts each segment against literal paper values (mass %, CoM %, and principal radii %). Corrected shank `com_fraction` from 0.4459 to 0.4395 and radii from (0.255, 0.249, 0.103) to (0.251, 0.246, 0.102) in `src/shared/python/motion_matching/anthropometry.py`, eliminating bony-landmark carryover from Zatsiorsky-Seluyanov 1985. Committed verification receipt `docs/development/full_body_models/evidence/anthropometry/de_leva_verification.json`.
- **HO-3 (#10157)**: Delivered cross-platform MJX virtualenv setup scripts (`scripts/setup_mjx_env.ps1`, `scripts/setup_mjx_env.sh`) reading pinned versions from `scripts/config/mjx_env_pins.json`, refactored differentiable trajectory optimization into pure testable functions with `SiteState` and `WeldGains` parameter tuples complying with architecture parameter budgets, added `requires_jax` marker to `pyproject.toml`, and implemented 6 unit tests in `tests/unit/motion_matching/test_mjx_optimisation.py` covering knot basis, tail freezing mask, contact force parity (< 1e-6 N), weld wrench equilibrium, initial preload, and finite gradients on truncated 12-frame horizon. Merged to `main` in PR #10228 (`09346c9dd`).
- **HO-11 (#10250)**: Rebuilt anthropometric documents (`full_body_spec_anthro_driver.json`, `..._iron7.json`) and ground-support receipts with canonical `de_leva_table_sha256` and `anthropometry` blocks after HO-9 de Leva shank fix. Added document freshness test suite `tests/unit/motion_matching/test_document_freshness.py` with 9 unit/contract tests enforcing byte-level agreement against `DE_LEVA_MALE`. Updated Pydantic receipt schema (`receipt_schema.py`), pipeline receipt builder (`receipt.py`), and regenerated `RECEIPTS.md`. Fixed local engine imports in `pipeline/lane.py` and `pipeline/address.py`. Re-executed driver and 7-iron captures, validating sub-millimeter shift (< 1e-4 m) between unedited and rebuilt models. Merged to `main` in PR #10261 (`81ea27bfb`).
- **HO-8 (#10108, MM-6)**: Delivered hip coordinate zero-twist functional calibration (`src/shared/python/motion_matching/hip_calibration.py`) using knee medial markers or shank-thigh ankle plane normal fallback, post-multiplying `parent_to_base` by $R_z(\theta)$ to rotate coordinate zero without translating hip centers. Set `BOUND_WIDENING: float = 1.0` in `src/shared/python/motion_matching/pipeline/constants.py`. Regenerated receipts for `anthro_driver`, `anthro_driver_shoot`, `anthro_iron`, and `anthro_iron_shoot`, verifying 0 lower-limb `range_of_motion_flags` on the IK reference for both captures. Re-verified cross-engine setup parity across MuJoCo, Drake, and Pinocchio.
- **Review 2026-09-16 (expert agent)**: receipts on `main` unchanged through
  the landing on canonical calibrated runs (driver address 5.1 mm (`docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json#address.calibrated.marker_rms_m`), IK 27.3 mm (`docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json#ik.marker_rms_m`), dynamics 74.6 mm (`docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json#dynamics.marker_rms_m`)). Two gaps found and filed: **HO-11 (#10250)** the anthropometric
  documents and every receipt were stale since HO-9 changed the de Leva
  shank row; **HO-12 (#10251)** `run_ground_support.py` was
  779 lines on `main` against HO-1's acceptance of under 400.
- **MS-03 (#10324)**: Reconciled headline tour numbers with primary receipts on `main`. Formally designated canonical calibrated reference runs (`anthro_driver_shoot_g025`, `anthro_iron_zmp`) vs primary uncalibrated baselines (`anthro_driver`, `anthro_iron`), produced `CANONICAL_RUN.md` and `bisect_receipt.json` attributing the 27.3 mm (`docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json#ik.marker_rms_m`) to 52.3 mm (`docs/development/full_body_models/evidence/ground_support/anthro_driver/receipt.json#ik.marker_rms_m`) shift to omission of `--static-seeds` during HO-8 regeneration, and added document freshness test suite `tests/unit/motion_matching/test_handoff_numbers_match_receipts.py`.
- **MS-11 (#10330)**: Finished HO-1 IK and forward dynamics consolidation into shared modules. Retired `full_body_markers.py` (813 -> 44 lines) and `full_body_simulation.py` (709 -> 55 lines) to backward-compatible deprecation shims, eliminating 1,423 duplicate lines across engine files while preserving zero MuJoCo imports in `src/shared/python/motion_matching/`. Consolidated ground-support IK into `BaseFullBodyIK` in `src/shared/python/motion_matching/full_body_ik.py` and `FullBodyMarkerKinematics` in `src/engines/physics_engines/mujoco/python/full_body_ik.py`. Consolidated computed torque, reference ZMP, foot preloading, and RK4 rollout into `src/shared/python/motion_matching/full_body_forward_dynamics.py`. Added comprehensive consolidation test suite `tests/unit/motion_matching/test_full_body_consolidation.py` verifying import boundaries, deprecation warnings, and strict numerical parity.
- **Doc hotspot rule**: `DEVELOPMENT_LOG.md` (`DL-#10062`) and `HANDOFF.md` "Status and Next Steps" are edited by every child, so they conflict whenever another child merges first. Resolve once, as the last step before merge: `git merge origin/main`, take `main`'s copy of both files (`git checkout --theirs`), re-apply only your own lines (your DL "Last verified"/"Next step", your one status bullet), commit, push. Never push repeated "Merge branch 'main'" commits with unresolved hunks. PR title prefix must match the diff (`test:`/`docs:`/`refactor:` when nothing under `src/` changes) or phantom-guard fails. `equivalence (3.11)` and `Trivy Container Scan` are not required and fail on `main` already; name them as pre-existing in the PR body, do not fix them in a child PR. Run `python scripts/ci/check_architecture_budget.py` before pushing (8-parameter and function-size budgets on changed files).
- **Next**: Land HO-12 PR (#10251), then proceed to HO-10 (#10160), HO-5 (#10159), and HO-6 (#10121) in epic #10162.
