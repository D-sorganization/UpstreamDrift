# Tour Baselines Coverage Matrix and Non-Golf Tool Exclusions

Parent Epic: [#10584](https://github.com/D-sorganization/UpstreamDrift/issues/10584)  
Governing Issue: [#10585](https://github.com/D-sorganization/UpstreamDrift/issues/10585) (TB-00)

---

## 1. Two-Capture Coverage Matrix

Every registered model is mapped against both canonical tour captures:

- **Driver Capture:** 360.0 Hz, 654 frames, 38 markers, SHA-256 `545405ccdbae87a297d16951487b501d5d76f5a2ab253cfc6d797744184943ba`
- **7-Iron Capture:** 359.0 Hz, 657 frames, 38 markers, SHA-256 `395deb1f91006586819020fc85180409e716f07e1c680f9fb2ca114759f80845`

| Model ID                         | Capture    | Supported | Observation Set                                           | Ownership     | Status                  | Existing Artifact / Blocker                                                                       | Governing Issue |
| -------------------------------- | ---------- | --------- | --------------------------------------------------------- | ------------- | ----------------------- | ------------------------------------------------------------------------------------------------- | --------------- |
| `reconstruction_golfer`          | **Driver** | Yes       | Upper/lower body anatomical markers (no club)             | UpstreamDrift | 🏛️ Historical Reference | `docs/development/reference_fitting_epic.md`                                                      | #9914           |
| `reconstruction_golfer`          | **Iron**   | Yes       | Upper/lower body anatomical markers (no club)             | UpstreamDrift | 🏛️ Historical Reference | `docs/development/reference_fitting_epic.md`                                                      | #9914           |
| `reconstruction_double_pendulum` | **Driver** | Yes       | Shoulder and hands markers (wrists midpoint)              | UpstreamDrift | 🏛️ Historical Reference | `docs/development/reference_fitting_epic.md`                                                      | #9914           |
| `reconstruction_double_pendulum` | **Iron**   | Yes       | Shoulder and hands markers (wrists midpoint)              | UpstreamDrift | 🏛️ Historical Reference | `docs/development/reference_fitting_epic.md`                                                      | #9914           |
| `reconstruction_triple_pendulum` | **Driver** | Yes       | Shoulder, elbow, and hands markers                        | UpstreamDrift | 🏛️ Historical Reference | `docs/development/reference_fitting_epic.md`                                                      | #9914           |
| `reconstruction_triple_pendulum` | **Iron**   | Yes       | Shoulder, elbow, and hands markers                        | UpstreamDrift | 🏛️ Historical Reference | `docs/development/reference_fitting_epic.md`                                                      | #9914           |
| `driven_double_pendulum`         | **Driver** | Yes       | Projected 2D swing plane (shoulder pivot + clubhead/grip) | Tools         | ⏳ Unqualified          | Pending TB-04 bounded dynamic fitting campaign                                                    | #10589          |
| `driven_double_pendulum`         | **Iron**   | Yes       | Projected 2D swing plane (shoulder pivot + clubhead/grip) | Tools         | ⏳ Unqualified          | Pending TB-04 bounded dynamic fitting campaign                                                    | #10589          |
| `driven_triple_pendulum`         | **Driver** | Yes       | Projected 2D swing plane (shoulder pivot + clubhead/grip) | Tools         | ⏳ Unqualified          | Pending TB-05 bounded dynamic fitting campaign                                                    | #10590          |
| `driven_triple_pendulum`         | **Iron**   | Yes       | Projected 2D swing plane (shoulder pivot + clubhead/grip) | Tools         | ⏳ Unqualified          | Pending TB-05 bounded dynamic fitting campaign                                                    | #10590          |
| `constrained_upper_body_golfer`  | **Driver** | Yes       | 3D upper torso, bilateral arms, and clubhead trajectory   | Tools         | ⏳ Unqualified          | Pending TB-06 upper-body constrained dynamic fitter                                               | #10591          |
| `constrained_upper_body_golfer`  | **Iron**   | Yes       | 3D upper torso, bilateral arms, and clubhead trajectory   | Tools         | ⏳ Unqualified          | Pending TB-06 upper-body constrained dynamic fitter                                               | #10591          |
| `full_body_mujoco`               | **Driver** | Yes       | Full 38 C3D markers + dual GRF plates                     | UpstreamDrift | ✅ G1 Passed            | `docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json` | #10378          |
| `full_body_mujoco`               | **Iron**   | Yes       | Full 38 C3D markers + dual GRF plates                     | UpstreamDrift | ⚙️ Native Candidate     | Pending dual-club G3 cross-engine qualification                                                   | #10378          |
| `full_body_pinocchio`            | **Driver** | Yes       | Full 38 C3D markers + dual GRF plates                     | UpstreamDrift | ⚙️ Native Candidate     | `evidence/matched/driver_g1_crocoddyl_rk45_b100/` (REJECTED)                                      | #10378          |
| `full_body_pinocchio`            | **Iron**   | Yes       | Full 38 C3D markers + dual GRF plates                     | UpstreamDrift | ⚙️ Native Candidate     | Pending dual-club G3 cross-engine qualification                                                   | #10378          |
| `full_body_drake`                | **Driver** | Yes       | Full 38 C3D markers + dual GRF plates                     | UpstreamDrift | ⚙️ Native Candidate     | IK 47 mm / tracking 382 mm REJECTED                                                               | #10378          |
| `full_body_drake`                | **Iron**   | Yes       | Full 38 C3D markers + dual GRF plates                     | UpstreamDrift | ⚙️ Native Candidate     | Pending dual-club G3 cross-engine qualification                                                   | #10378          |
| `full_body_opensim`              | **Driver** | Yes       | Full 38 C3D markers + dual GRF plates                     | UpstreamDrift | ⚙️ Native Candidate     | Pending Moco full-swing tracking convergence under MS-102                                         | #10378          |
| `full_body_opensim`              | **Iron**   | Yes       | Full 38 C3D markers + dual GRF plates                     | UpstreamDrift | ⚙️ Native Candidate     | Pending dual-club G3 cross-engine qualification                                                   | #10378          |
| `full_body_simscape`             | **Driver** | Yes       | Full 38 C3D markers + dual GRF plates                     | UpstreamDrift | ⚙️ Native Candidate     | Terminal error 40.3 mm > 35 mm gate; pinned to MATLAB R2025b                                      | #9921           |
| `full_body_simscape`             | **Iron**   | Yes       | Full 38 C3D markers + dual GRF plates                     | UpstreamDrift | ⚙️ Native Candidate     | Pending dual-club G3 cross-engine qualification                                                   | #9921           |
| `full_body_myosuite`             | **Driver** | No        | Full 38 C3D markers + dual GRF plates                     | UpstreamDrift | ❌ Blocked              | Requires MyoSuite environment & retargeting (fail-closed per MS-50)                               | #10378          |
| `full_body_myosuite`             | **Iron**   | No        | Full 38 C3D markers + dual GRF plates                     | UpstreamDrift | ❌ Blocked              | Requires MyoSuite environment & retargeting (fail-closed per MS-50)                               | #10378          |
| `reference_pinocchio_urdf`       | **Driver** | Yes       | Mapped subset of anatomical markers                       | UpstreamDrift | 🏛️ Historical Reference | `docs/development/reference_fitting_epic.md` (167.60 mm RMS)                                      | #9914           |
| `reference_pinocchio_urdf`       | **Iron**   | Yes       | Mapped subset of anatomical markers                       | UpstreamDrift | 🏛️ Historical Reference | `docs/development/reference_fitting_epic.md` (171.85 mm RMS)                                      | #9914           |
| `reference_pinocchio_urdf_ik`    | **Driver** | Yes       | Mapped subset of anatomical markers                       | UpstreamDrift | 🏛️ Historical Reference | `docs/development/reference_fitting_epic.md` (167.60 mm RMS)                                      | #9914           |
| `reference_pinocchio_urdf_ik`    | **Iron**   | Yes       | Mapped subset of anatomical markers                       | UpstreamDrift | 🏛️ Historical Reference | `docs/development/reference_fitting_epic.md` (171.85 mm RMS)                                      | #9914           |
| `reference_drake_urdf`           | **Driver** | Yes       | Mapped subset of anatomical markers                       | UpstreamDrift | 🏛️ Historical Reference | `docs/development/reference_fitting_epic.md` (131.69 mm RMS)                                      | #9914           |
| `reference_drake_urdf`           | **Iron**   | Yes       | Mapped subset of anatomical markers                       | UpstreamDrift | 🏛️ Historical Reference | `docs/development/reference_fitting_epic.md` (138.96 mm RMS)                                      | #9914           |
| `reference_simple_humanoid`      | **Driver** | Yes       | Mapped subset of anatomical markers (no club)             | UpstreamDrift | 🏛️ Historical Reference | `docs/development/reference_fitting_epic.md` (196.30 mm RMS)                                      | #9914           |
| `reference_simple_humanoid`      | **Iron**   | Yes       | Mapped subset of anatomical markers (no club)             | UpstreamDrift | 🏛️ Historical Reference | `docs/development/reference_fitting_epic.md` (188.12 mm RMS)                                      | #9914           |
| `reference_human_subject`        | **Driver** | Yes       | Mapped subset of anatomical markers (no club)             | UpstreamDrift | 🏛️ Historical Reference | `docs/development/reference_fitting_epic.md` (101.11 mm RMS)                                      | #9914           |
| `reference_human_subject`        | **Iron**   | Yes       | Mapped subset of anatomical markers (no club)             | UpstreamDrift | 🏛️ Historical Reference | `docs/development/reference_fitting_epic.md` (96.52 mm RMS)                                       | #9914           |
| `reference_mujoco_humanoid`      | **Driver** | Yes       | Mapped subset of anatomical markers (no club)             | UpstreamDrift | 🏛️ Historical Reference | `docs/development/reference_fitting_epic.md` (236.01 mm RMS)                                      | #9914           |
| `reference_mujoco_humanoid`      | **Iron**   | Yes       | Mapped subset of anatomical markers (no club)             | UpstreamDrift | 🏛️ Historical Reference | `docs/development/reference_fitting_epic.md` (226.31 mm RMS)                                      | #9914           |
| `myosuite_body`                  | **Driver** | No        | None (placeholder)                                        | UpstreamDrift | ❌ Blocked              | Bundled myobody assets are placeholders, not MyoSuite anatomy                                     | #9914           |
| `myosuite_body`                  | **Iron**   | No        | None (placeholder)                                        | UpstreamDrift | ❌ Blocked              | Bundled myobody assets are placeholders, not MyoSuite anatomy                                     | #9914           |
| `opensim_golfer`                 | **Driver** | No        | None (placeholder)                                        | UpstreamDrift | ❌ Blocked              | Native OpenSim custom-joint/muscle constraints require OpenSim adapter                            | #9914           |
| `opensim_golfer`                 | **Iron**   | No        | None (placeholder)                                        | UpstreamDrift | ❌ Blocked              | Native OpenSim custom-joint/muscle constraints require OpenSim adapter                            | #9914           |

---

## 2. Non-Golf Tool Exclusions

The following applications and tools registered in `src/config/models.yaml` are explicitly excluded from the Tour Baselines matrix with their specific rationale:

| Tool ID                  | Display Name              | Launcher Category | Exclusion Rationale                                                                                       |
| ------------------------ | ------------------------- | ----------------- | --------------------------------------------------------------------------------------------------------- |
| `bunkershot`             | Bunker Shot Simulation    | `special_app`     | Sand-trap particle/contact game and visualization; not a full tour swing trajectory.                      |
| `shot_tracer`            | Shot Tracer               | `simulation`      | Aerodynamic post-impact ball flight model (Waterloo, MacDonald, Nathan); not swing biomechanics.          |
| `cross_engine_dashboard` | Cross-Engine Dashboard    | `simulation`      | Engine-level perturbation and numerical robustness analyzer; not a biomechanical baseline model.          |
| `pose_studio`            | Pose Studio               | `tool`            | Interactive cross-engine kinematic pose authoring editor; not a dynamic golf swing fitter.                |
| `starting_pose_matcher`  | Starting-Pose Matcher     | `tool`            | Static address-frame alignment solver; does not track or simulate continuous swing dynamics.              |
| `force_plate_lab`        | Force Plate Lab           | `tool`            | Ground reaction force visualizer and COP diagnostics; consumes GRF traces without solving swing dynamics. |
| `swing_plane_analyzer`   | Swing Plane Analyzer      | `tool`            | Pure geometric plane fitting and club-shaft inclination utility; does not simulate body or club physics.  |
| `putting_green`          | Putting Green Simulator   | `simulation`      | Short-game pendulum putting surface and ball roll physics; excluded from full-swing tour baselines.       |
| `camera_setup`           | Camera Setup Wizard       | `tool`            | Multi-camera extrinsics and optical calibration tool; does not model golfer dynamics.                     |
| `coaching_drawings`      | Coaching Drawings Tool    | `tool`            | Video 2D overlay and angle measurement annotations for golf instruction; not a physics model.             |
| `model_explorer`         | Model Explorer            | `tool`            | 3D asset hierarchy and mesh inspector; does not simulate biomechanics.                                    |
| `calibration_wizard`     | Sensor Calibration Wizard | `tool`            | IMU and mocap sensor calibration utility; not a fittable biomechanical model.                             |
