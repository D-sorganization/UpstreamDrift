### Launcher Tools and Desktop Product Inventory

#### Assertion Classes

Every tile row below is assigned exactly one class. The class fixes the
pass/fail criterion so an agent never has to exercise judgement.

| Class | Applies to                        | PASS requires                                                                                                                                                                                             | FAIL is                                                                                          |
| ----- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **A** | `ready` tiles                     | Single launch → target window/tab appears within 30 s with no traceback → its documented primary action produces a changed, non-placeholder observable → close returns focus to a still-running launcher. | Silent exit, launcher termination, traceback, preview-only result, or a second duplicate window. |
| **B** | `beta` / `experimental` tiles     | Launch succeeds **or** an actionable named-dependency message appears; either way the launcher survives and unrelated tiles still launch.                                                                 | Raw crash dialog, silent no-op, false success toast, or launcher termination.                    |
| **C** | `deprecated` / legacy-alias tiles | The tile is hidden, or visibly marked legacy and does not present itself as new work.                                                                                                                     | A deprecated tile that is indistinguishable from a supported one.                                |
| **D** | `runtime_unavailable` providers   | Status is truthful before launch, and launching gives a named missing-runtime message.                                                                                                                    | Advertised as available, or launching produces an unnamed error.                                 |

`BLOCKED` is reserved for a dependency the product's own probe (`PF-04`,
`API-14`) demonstrates is absent. A present-but-broken dependency is class-A/B
**FAIL**, never `BLOCKED`.

#### Native Launcher Tile Inventory (`src/config/models.yaml`, 46 In-Repo Tiles)

Regenerate with `scripts/testing/enumerate_test_surface.py --native-table`.

| ID     | Tile id                     | Category        | Name                           | Status       | Class | Known issues                                                                             |
| ------ | --------------------------- | --------------- | ------------------------------ | ------------ | ----- | ---------------------------------------------------------------------------------------- |
| NAT-01 | `biomech_gait`              | biomechanics    | Gait Model                     | ready        | A     | #8068                                                                                    |
| NAT-02 | `biomech_sit_to_stand`      | biomechanics    | Sit-to-Stand Model             | ready        | A     |                                                                                          |
| NAT-03 | `canonical_core_comparison` | biomechanics    | Canonical-Core Comparison      | experimental | B     | #8081 (web)                                                                              |
| NAT-04 | `canonical_core_estimation` | biomechanics    | Canonical-Core Estimation      | experimental | B     | #8081 (web)                                                                              |
| NAT-05 | `movement_optimizer`        | biomechanics    | Movement Optimizer             | ready        | A     | #8074                                                                                    |
| NAT-06 | `pinn_hybrid`               | biomechanics    | PINN Hybrid                    | experimental | B     |                                                                                          |
| NAT-07 | `pinn_pure_rigid`           | biomechanics    | Pure Rigid Body                | experimental | B     |                                                                                          |
| NAT-08 | `library_tool`              | documentation   | Library                        | ready        | A     |                                                                                          |
| NAT-09 | `project_map`               | documentation   | Project Map                    | ready        | A     |                                                                                          |
| NAT-10 | `c3d_viewer`                | motion_capture  | C3D Viewer                     | ready        | A     | #8073, #8082, #8088                                                                      |
| NAT-11 | `mediapipe_analysis`        | motion_capture  | MediaPipe                      | ready        | A     | #8071                                                                                    |
| NAT-12 | `openpose_analysis`         | motion_capture  | OpenPose                       | ready        | A     |                                                                                          |
| NAT-13 | `video_analyzer`            | motion_capture  | Video Analyzer                 | ready        | A     |                                                                                          |
| NAT-14 | `video_processor`           | motion_capture  | Video Processor                | ready        | A     |                                                                                          |
| NAT-15 | `motion_target_preview`     | motion_matching | Motion-Match Preview           | ready        | A     | #8066                                                                                    |
| NAT-16 | `pose_studio`               | motion_matching | Pose Studio                    | beta         | B     | #8070, #8086                                                                             |
| NAT-17 | `pose_subscriber_demo`      | motion_matching | Pose Subscriber (demo)         | experimental | B     |                                                                                          |
| NAT-18 | `starting_pose_matcher`     | motion_matching | Starting-Pose Matcher (legacy) | deprecated   | C     | hidden in shared manifest                                                                |
| NAT-19 | `drake_golf`                | physics_engine  | Drake                          | ready        | A     | = `ENG-02`                                                                               |
| NAT-20 | `matlab_suite`              | physics_engine  | Simscape                       | ready        | A     | #8072; = `ENG-06`                                                                        |
| NAT-21 | `mujoco_unified`            | physics_engine  | MuJoCo                         | ready        | A     | = `ENG-01`                                                                               |
| NAT-22 | `myosim_suite`              | physics_engine  | MyoSuite                       | ready        | A     | = `ENG-05`                                                                               |
| NAT-23 | `opensim_golf`              | physics_engine  | OpenSim                        | ready        | A     | = `ENG-04`                                                                               |
| NAT-24 | `pinocchio_golf`            | physics_engine  | Pinocchio                      | ready        | A     | = `ENG-03`                                                                               |
| NAT-25 | `ball_flight_simulator`     | simulation      | Ball Flight Simulator          | ready        | A     | strict Rust kernel dependency                                                            |
| NAT-26 | `bunker_shot`               | simulation      | Bunker Shot 3D                 | experimental | B     |                                                                                          |
| NAT-27 | `bunkershot3d`              | simulation      | BunkerShot3D                   | experimental | B     | #8084 (import boundary)                                                                  |
| NAT-28 | `cross_engine_dashboard`    | simulation      | Cross-Engine Dashboard         | ready        | A     |                                                                                          |
| NAT-29 | `drake_models_shared`       | simulation      | Drake Models                   | ready        | A     | needs `Drake_Models` (`PF-07`)                                                           |
| NAT-30 | `golf_environment`          | simulation      | Golf Environment               | ready        | A     |                                                                                          |
| NAT-31 | `golf_simulation_suite`     | simulation      | Simulator                      | ready        | A     |                                                                                          |
| NAT-32 | `mujoco_models_shared`      | simulation      | MuJoCo Models                  | ready        | A     | needs `MuJoCo_Models`                                                                    |
| NAT-33 | `opensim_models_shared`     | simulation      | OpenSim Models                 | ready        | A     | needs `OpenSim_Models`                                                                   |
| NAT-34 | `pendulum_simulator`        | simulation      | Pendulum Simulator             | ready        | A     | #8065                                                                                    |
| NAT-35 | `pinocchio_models_shared`   | simulation      | Pinocchio Models               | ready        | A     | needs `Pinocchio_Models`                                                                 |
| NAT-36 | `putting_green`             | simulation      | Putting Green                  | ready        | A     | #8063; canonical tile                                                                    |
| NAT-37 | `putting_green_gui`         | simulation      | Putting Green                  | ready        | C     | #8090 — duplicate visible name/target on `origin/main`; must be hidden or differentiated |
| NAT-38 | `shot_tracer`               | simulation      | Shot Tracer                    | ready        | A     | #8069; web pair `WEB-16`                                                                 |
| NAT-39 | `swing_flight_pipeline`     | simulation      | Swing → Flight Pipeline        | ready        | A     |                                                                                          |
| NAT-40 | `terrain_engine`            | simulation      | Terrain Engine                 | ready        | A     | web pair `WEB-10`                                                                        |
| NAT-41 | `config_setup_wizard`       | tool            | Setup Wizard                   | beta         | B     | #8067                                                                                    |
| NAT-42 | `data_explorer`             | tool            | Data Explorer                  | ready        | A     | web pair `WEB-08`                                                                        |
| NAT-43 | `data_processor`            | tool            | Data Processor                 | ready        | A     |                                                                                          |
| NAT-44 | `model_explorer`            | tool            | Model Explorer                 | ready        | A     | web pair `WEB-05`                                                                        |
| NAT-45 | `sidekick`                  | tool            | Sidekick                       | beta         | B     | #8075; web pair `WEB-15`                                                                 |
| NAT-46 | `training_controller`       | tool            | Training Controller            | beta         | B     |                                                                                          |

#### Shared-Manifest Tiles With No Native `models.yaml` Entry (24)

These render on the web dashboard and are served by `/api/launcher/tiles` but
have **no native counterpart**. Each row's first assertion is `SH-10`: the tile
must either gain a native entry or be recorded in the documented web-only
exception list. `web_route` `—` means the tile advertises no reachable route,
which is itself the `#7461` `launcher.tile_web_reachability` parity gap.

| ID     | Tile id                 | Category        | Name                   | Status       | Web route                  | Class |
| ------ | ----------------------- | --------------- | ---------------------- | ------------ | -------------------------- | ----- |
| MAN-01 | `analysis_tools_api`    | analysis        | Analysis Tools         | ready        | `/tools/analysis`          | A     |
| MAN-02 | `perturbation_analysis` | analysis        | Perturbation Analysis  | experimental | —                          | B     |
| MAN-03 | `biomech_exercise`      | biomechanics    | Exercise Dashboard     | gui_ready    | —                          | A     |
| MAN-04 | `injury_analysis`       | biomechanics    | Injury Risk Analysis   | gui_ready    | —                          | A     |
| MAN-05 | `matlab_unified`        | external        | Matlab Models          | external     | —                          | B     |
| MAN-06 | `motion_pipeline`       | motion_matching | Motion Pipeline        | experimental | —                          | B     |
| MAN-07 | `drake_dashboard`       | physics_engine  | Drake Dashboard        | experimental | —                          | B     |
| MAN-08 | `mujoco_dashboard`      | physics_engine  | MuJoCo Dashboard       | experimental | —                          | B     |
| MAN-09 | `pinocchio_dashboard`   | physics_engine  | Pinocchio Dashboard    | experimental | —                          | B     |
| MAN-10 | `cross_engine`          | simulation      | Cross-Engine Dashboard | gui_ready    | —                          | A     |
| MAN-11 | `simulation_backends`   | simulation      | Simulation Backends    | gui_ready    | —                          | A     |
| MAN-12 | `swing_optimizer`       | simulation      | Swing Optimizer        | gui_ready    | —                          | A     |
| MAN-13 | `actuator_controls`     | tool            | Actuator Controls      | experimental | —                          | B     |
| MAN-14 | `aip`                   | tool            | AI Protocol (AIP)      | experimental | —                          | B     |
| MAN-15 | `character_builder`     | tool            | Character Builder      | gui_ready    | `/tools/character-builder` | A     |
| MAN-16 | `chat_assistant`        | tool            | AI Assistant           | gui_ready    | `/chat`                    | A     |
| MAN-17 | `dataset_generator`     | tool            | Dataset Generator      | gui_ready    | `/tools/dataset`           | A     |
| MAN-18 | `force_overlays`        | tool            | Force Overlays         | experimental | —                          | B     |
| MAN-19 | `motion_capture`        | tool            | Motion Capture         | utility      | `/tools/motion-capture`    | A     |
| MAN-20 | `pid_generator`         | tool            | P&ID Generator         | external     | —                          | B     |
| MAN-21 | `realtime_ws`           | tool            | Realtime WebSocket     | experimental | —                          | B     |
| MAN-22 | `robotics_module`       | tool            | Robotics Module        | experimental | —                          | B     |
| MAN-23 | `tools_calculator_hub`  | tool            | Tools Calculator Suite | external     | —                          | B     |
| MAN-24 | `unreal_integration`    | tool            | Unreal Integration     | external     | —                          | B     |

#### Provider Model-Pack Children (`PRV-*`, Environment-Dependent)

Provider packs contribute one child tile per exercise per engine. On a machine
with all four `*_Models` repos resolved this is 28 tiles
(`{mujoco,drake,opensim,pinocchio}_models-{bench_press,clean_and_jerk,deadlift,gait,sit_to_stand,snatch,squat}`).
They are tested **parametrically**, not one row each.

| ID     | Scope                                  | Steps                                                                                | Pass criterion                                                                                                                                                                   | Priority |
| ------ | -------------------------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| PRV-01 | Count and provenance                   | Compare the provider child set in `surface.json` with the resolved packs in `PF-07`. | Child count equals `resolved packs × exercises`; every child names its source pack.                                                                                              | P0       |
| PRV-02 | Handler resolution (all children)      | For every provider child, resolve its `ModelHandlerRegistry` handler.                | Every visible child has a handler mapped to its declared engine, or is hidden. Catches `#8087`.                                                                                  | P0       |
| PRV-03 | Capability truthfulness (all children) | Read `capabilities` for every visible child.                                         | Non-empty and consistent with the declared engine and exercise id. Catches `#8091`.                                                                                              | P1       |
| PRV-04 | Status truthfulness (all children)     | Compare each child's advertised status against an engine runtime probe.              | `provider_ready` only when the engine imports; otherwise `runtime_unavailable`. Baseline on `origin/main`: MuJoCo 7 ready, Drake/OpenSim/Pinocchio 7 each `runtime_unavailable`. | P0       |
| PRV-05 | Representative launch (one per engine) | Launch one child per engine — `*_models-squat` by default.                           | Class A for `provider_ready`, class D for `runtime_unavailable`. Four rows recorded, one per engine.                                                                             | P0       |
| PRV-06 | Absent-pack degradation                | Temporarily point one pack path at a non-existent directory and reload.              | The launcher starts, the pack's children disappear, and no other tile is affected. Restore afterwards.                                                                           | P1       |

#### Deep-Workflow Groups

| IDs        | Category / inventory                                                                                                                                                                                        | Primary functional assertion                                                                                   | Priority |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | -------- |
| ENG-01–06  | Physics engines: MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite, Simscape                                                                                                                                      | Load known compatible model, run a short simulation or documented minimal analysis, observe result/visualizer. | P0       |
| SIM-01–12  | Simulators: Golf Simulation Suite, Putting Green, model repositories, BunkerShot3D, Pendulum, Shot Tracer, Cross-Engine Dashboard, Terrain Engine, Golf Environment, Bunker Shot, Swing→Flight, Ball Flight | Launch, configure smallest valid case, run/calculate, verify output and reset.                                 | P0/P1    |
| MM-01–03   | Motion Matching: Motion-Match Preview, Pose Studio, Pose Subscriber demo                                                                                                                                    | Open target/pose fixture, edit/select pose, preview/propagate/revert as supported.                             | P1       |
| MC-01–05   | Motion Capture: C3D Viewer, Video Analyzer, Video Processor, OpenPose, MediaPipe                                                                                                                            | Open compatible fixture, scrub/process/estimate, inspect result and export only to temporary workspace.        | P1       |
| DATA-01–06 | Tools: Model Explorer, Data Explorer, Data Processor, Sidekick, Training Controller, Setup Wizard                                                                                                           | Complete documented minimal workflow and validate error/retry path.                                            | P0/P1    |
| DOC-01–02  | Documentation: Project Map, Library                                                                                                                                                                         | Open/document search or navigation; verify content opens and links are valid.                                  | P2       |
| BIO-01–06  | Biomechanics: Gait, Sit-to-Stand, Movement Optimizer, Canonical-Core Estimation/Comparison, Pure Rigid Body, PINN Hybrid                                                                                    | Launch and run a documented smallest supported analysis; preserve provenance.                                  | P1       |

### Data, Recovery, and Reliability Scenarios

| ID     | Scenario                     | Core procedure                                                                                                           | Expected observable result                                                                                       | Priority |
| ------ | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- | -------- |
| REL-01 | Validation                   | Submit invalid range/value and malformed fixture in each input-capable screen.                                           | Field-level or action-level explanation; app stays usable.                                                       | P0       |
| REL-02 | Cancellation/retry           | Cancel a long-enough safe operation, then repeat a valid one.                                                            | Cancellation is unambiguous; retry works and does not duplicate work.                                            | P0       |
| REL-03 | Network/backend interruption | Stop/restart the local API while a web screen is open; reconnect.                                                        | Connection loss is honest; recovery requires no refresh-loop or stale result.                                    | P0       |
| REL-04 | Persistence                  | Save an allowed disposable session/layout/profile, reopen product, then load it.                                         | State round-trips or clearly states persistence limitations.                                                     | P1       |
| REL-05 | Failure isolation            | Cause one optional tool/engine to fail or be unavailable.                                                                | Launcher and unrelated ready tools remain usable.                                                                | P0       |
| REL-06 | Repeated use                 | Repeat launch/run/close cycle three times for representative P0 flows.                                                   | No duplicate windows, leaked loading state, crash, or material degradation.                                      | P1       |
| REL-07 | Optional-import blast radius | Start the launcher with each optional engine package made unimportable in turn (rename on `sys.path`, do not uninstall). | Launcher reaches its main window in every case. Catches `#8084` (`WinError 1114` via `bunkershot3d` → `mujoco`). | P0       |
| REL-08 | Child-process containment    | Launch a tile whose target raises at import; observe the launcher.                                                       | The child dies, the launcher logs a named error and stays alive. No `#8066`/`#8070`-class launcher termination.  | P0       |
| REL-09 | Orphan sweep                 | After every campaign slice, diff the process table against the pre-slice snapshot.                                       | Zero surviving `python`/tool child processes owned by the launcher. Recorded as evidence, not judgement.         | P0       |
| REL-10 | Read-only output path        | Point the temporary workspace at a read-only directory and attempt one save/export.                                      | An actionable permission error appears; no partial/corrupt artifact and no silent success.                       | P1       |
| REL-11 | Concurrent launch            | Double-invoke the same tile within 1 s, and launch three different tiles simultaneously.                                 | Exactly one container per tile; no duplicated process, no deadlocked splash.                                     | P1       |

### Feature-Parity Registry (`PAR-*`)

`src/config/feature_parity.json` is the machine-readable ledger CI enforces.
Baseline on `origin/main`: 37 entries — 16 `parity`, 9 `gap`, 12 `exempt`.

| ID     | Check                    | Steps                                                                                  | Pass criterion                                                                                                                                       | Priority |
| ------ | ------------------------ | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| PAR-01 | Gap entries are real     | For each of the 9 `gap` entries, run the named PyQt6 workflow and its web counterpart. | The divergence still exists **and** the referenced issue (#7448, #7450, #7451, #7452, #7459, #7461) is open. A closed gap is a stale registry entry. | P1       |
| PAR-02 | Parity entries hold      | For each of the 16 `parity` entries, run both surfaces.                                | Both produce the same class of result. A `parity` entry that has regressed to a gap is a P0 registry defect.                                         | P0       |
| PAR-03 | Exempt entries justified | Read the 12 `exempt` entries' reasons against the observed product.                    | Each reason is still accurate (desktop-only capability, not a silent omission).                                                                      | P2       |
| PAR-04 | Registry covers tiles    | Cross-reference registry `tiles` fields against `NAT-*`/`MAN-*`.                       | Every visible tile is covered by ≥1 registry entry — this is the CI rule; a violation is a CI-gate hole, filed as such.                              | P1       |
