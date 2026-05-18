# Motion-matching adversarial audit

Date: 2026-05-08. Branch: `feat/motion-pipeline-matching-completeness`. Auditor: Claude Opus 4.7.

## TL;DR

- **Only 3 of 9 engines** (Drake, MuJoCo, Pinocchio) actually expose a registered `fit_swing` provider on the canonical `provider_registry`. OpenSim has all the supporting machinery (`fit_swing_opensim`, `simulate_with_coefficients`, multistart) but has **never been wrapped in a provider class** despite the parity spec, OPENSIM_PARITY_SPEC.md, and a tracked plan (#4513 / scratch issue 21). Simscape 2D, Simscape 3D, MyoSuite, Pendulum, Putting Green have **no `motion_matching/` directory at all** under `src/engines/`.
- **The matcher GUI has no engine dropdown.** `src/tools/starting_pose_matcher/` is a Qt app for skeleton/pose authoring; it has _zero_ references to `fit_swing`, `MultiSourceTarget`, `engine_combo`, or `provider_registry`. The five files under `src/tools/starting_pose_matcher/providers/` are **skeleton-extraction providers** (FK / marker mapping), not motion-matching `fit_swing` providers — they have been conflated. The user has no GUI workflow to "pick an engine and fit a swing" today.
- **No `reports/cross_engine_leaderboard.json` exists in the repo.** A workflow (`.github/workflows/cross-engine-leaderboard.yml`) and a runner (`scripts/run_cross_engine_leaderboard.py`) reference it but no checked-in artefact. This means the leaderboard has been a paper concept; nothing publishes to it.
- **C3D pipeline silently drops the entire `FORCE_PLATFORM` parameter group** (CORNERS, ORIGIN, CAL*MATRIX, TYPE), every `EVENT` parameter group, and screen-axis hints. Force-plate detection is done from \_analog channel name patterns*, not from the C3D parameter section — this means CoP cannot be transformed into lab frame, plate calibration matrices are ignored, and any C3D file that names channels non-conventionally is invisible to the loader. There is no Vicon vs Qualisys vs BTS vs Codamotion variant handling.
- **Test coverage is bottom-heavy**: provider-level unit tests exist only for Drake (7), MuJoCo (13), Pinocchio (6); zero for OpenSim provider (because it doesn't exist), zero for Simscape/MyoSuite/Pendulum/PuttingGreen. No integration test exercises the registry end-to-end across multiple engines. `tests/cross_engine/` contains exactly one file (`test_mujoco_vs_pinocchio.py`).

---

## 1. Engine coverage matrix

Source paths checked:

- `src/engines/physics_engines/{drake,mujoco,pinocchio,opensim,myosuite,pendulum,putting_green}/`
- `src/engines/Simscape_Multibody_Models/{2D,3D}_Golf_Model/`
- `src/engines/pendulum_models/`
- `src/shared/python/motion_matching/provider_registry.py`
- `src/tools/starting_pose_matcher/providers/`

| Engine                                                      | `motion_matching/` dir                                                                   | `fit_swing` provider class                                                                                                  | Auto-registers on import       | Driven by canonical contract | Consumes `MultiSourceTarget`                  | Consumes `BodyTarget`             | Consumes `ClubBallTarget`             |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------ | ---------------------------- | --------------------------------------------- | --------------------------------- | ------------------------------------- |
| **Drake**                                                   | yes (`physics_engines/drake/python/motion_matching/`)                                    | `DrakeFitSwingProvider` (`provider.py`)                                                                                     | yes (in package `__init__.py`) | yes                          | duck-typed via `target.club`                  | NO (`supports_body_target=False`) | partial (extracts `target.club` only) |
| **MuJoCo**                                                  | yes                                                                                      | `MujocoFitSwingProvider` (`provider.py`)                                                                                    | yes (at module bottom)         | yes                          | yes (typed `MultiSourceTarget \| ClubTarget`) | NO                                | NO                                    |
| **Pinocchio**                                               | yes                                                                                      | `PinocchioFitSwingProvider` (`provider.py`)                                                                                 | yes                            | yes                          | yes (via `club_target_adapter`)               | NO                                | partial                               |
| **OpenSim**                                                 | yes                                                                                      | **MISSING** — has `fit_swing_opensim` and `fit_swing_opensim_multistart` but no provider class, no `register_provider` call | NO                             | NO                           | NO                                            | NO                                | NO                                    |
| **MyoSuite**                                                | **NO** — no `motion_matching/` subpackage at all                                         | n/a                                                                                                                         | n/a                            | n/a                          | n/a                                           | n/a                               | n/a                                   |
| **Simscape 2D** (`Simscape_Multibody_Models/2D_Golf_Model`) | NO Python motion-matching surface; only MATLAB scripts                                   | n/a                                                                                                                         | n/a                            | n/a                          | n/a                                           | n/a                               | n/a                                   |
| **Simscape 3D** (`Simscape_Multibody_Models/3D_Golf_Model`) | only MATLAB (`matlab/motion_matching/shared/load_club_target_c3d.m`); no Python provider | n/a                                                                                                                         | n/a                            | n/a                          | n/a                                           | n/a                               | n/a                                   |
| **Pendulum** (`physics_engines/pendulum/`)                  | NO                                                                                       | n/a                                                                                                                         | n/a                            | n/a                          | n/a                                           | n/a                               | n/a                                   |
| **Pendulum (legacy `pendulum_models/`)**                    | NO; only MATLAB / arduino / python double-pendulum sim, no fit_swing                     | n/a                                                                                                                         | n/a                            | n/a                          | n/a                                           | n/a                               | n/a                                   |
| **Putting Green**                                           | NO; has stroke + green sim only                                                          | n/a                                                                                                                         | n/a                            | n/a                          | n/a                                           | n/a                               | n/a                                   |

**Smoke test result.** Importing all four python motion_matching packages with `available_engines()` afterwards on Python 3.14 yields `[]` because `MujocoFitSwingProvider` raises an `engine_name 'mujoco' is already registered` error during its synthesis-helper monkey-patch path (separate bug — see issue 8 below). When that path is bypassed, only drake/mujoco/pinocchio appear. **OpenSim never appears in the registry under any code path.**

`src/tools/starting_pose_matcher/providers/{drake,mujoco,pinocchio,opensim,mediapipe,openpose}.py` are **skeleton-extraction providers** (they map engine model bodies → matcher vocabulary `hip,spine,torso,hub,ls,rs,le,re,lw,rw,mp,ch`). They are unrelated to the canonical `fit_swing` registry and are not driven by the cross-engine matcher. The naming collision is itself a hazard.

---

## 2. Functional parity matrix

| Engine                       | `fit_swing(target, opts) -> FitResult`                                      | `simulate_with_coefficients`     | Theta validation                                                                         | Engine version queryable               | Per-frame error diagnostics           | Fit-quality card                      | Leaderboard row writer                                              | Sample / golden snapshot | Live entry in `reports/cross_engine_leaderboard.json` |
| ---------------------------- | --------------------------------------------------------------------------- | -------------------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------- | ------------------------------------- | ------------------------------------- | ------------------------------------------------------------------- | ------------------------ | ----------------------------------------------------- |
| **Drake**                    | yes                                                                         | yes (`drake/.../simulate.py`)    | partial — bounds in `default_theta_bounds`, validators in `validate_theta` shared module | unclear (no `engine_version()` getter) | yes (`plot_error_timecourse`, shared) | yes (`plot_fit_quality_card`, shared) | NO per-engine writer                                                | NO golden                | NO file                                               |
| **MuJoCo**                   | yes                                                                         | yes                              | partial                                                                                  | NO                                     | yes (shared)                          | yes (shared)                          | NO                                                                  | NO                       | NO file                                               |
| **Pinocchio**                | yes                                                                         | yes                              | yes (validators wired)                                                                   | NO                                     | yes                                   | yes                                   | yes — has dedicated `leaderboard_writer.py` (only engine that does) | NO golden                | NO file                                               |
| **OpenSim**                  | function exists (`fit_swing_opensim`) but **not behind canonical contract** | yes (per OPENSIM_PARITY_SPEC.md) | partial (in `coord_map`)                                                                 | NO                                     | yes (shared)                          | yes (shared)                          | NO                                                                  | NO                       | NO file                                               |
| **MyoSuite**                 | NO                                                                          | NO                               | NO                                                                                       | NO                                     | NO                                    | NO                                    | NO                                                                  | NO                       | NO                                                    |
| **Simscape 2D / 3D**         | MATLAB only; no Python `fit_swing`                                          | NO Python                        | NO                                                                                       | NO                                     | NO                                    | NO                                    | NO                                                                  | NO                       | NO                                                    |
| **Pendulum / Putting Green** | NO                                                                          | NO                               | NO                                                                                       | NO                                     | NO                                    | NO                                    | NO                                                                  | NO                       | NO                                                    |

**Critical observations.**

- The `reports/cross_engine_leaderboard.json` file is referenced by `scripts/run_cross_engine_leaderboard.py` and the GH workflow, but **no committed artefact exists**, and only Pinocchio has a per-engine leaderboard writer (`pinocchio/.../leaderboard_writer.py`). There is no shared `leaderboard.py:append_row(engine, fit_result)` API.
- "Per-frame error" / "fit-quality card" diagnostics live in shared modules (`src/shared/python/motion_matching/{plot_error_timecourse,plot_fit_quality_card,diagnostics}.py`), so any engine that produces a `CanonicalFitResult` can drive them — but no engine actually wires this into a leaderboard-publishing CI step.
- "Engine version queryable" is **not implemented anywhere**. Providers expose `engine_name` only. There is no `engine_version() -> str`. This makes the leaderboard non-reproducible.
- "Theta validation" is **inconsistent**: Pinocchio uses the shared `validate_theta`; Drake/MuJoCo only check via `default_theta_bounds`; OpenSim uses internal `coord_map` checks. There is no canonical assertion that bounds are equivalent across engines.

---

## 3. Test coverage per engine

Sources: `tests/unit/motion_matching/{drake,mujoco,pinocchio}/`, `tests/heavy_integration/`, `tests/integration/`, `tests/parity/`, `tests/motion_matching/`, `tests/cross_engine/`.

| Engine                       | Provider unit tests                                     | Other unit tests (engine-flavoured)                                                                                                                                                                                                        | Integration / heavy / parity | Cross-engine                                 | Verdict                      |
| ---------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------- | -------------------------------------------- | ---------------------------- |
| **Drake**                    | 7 (`tests/unit/motion_matching/drake/test_provider.py`) | scattered in `tests/unit/motion_pipeline/ik/test_drake_backend.py`, `tests/heavy_integration/test_drake_real_model_loading.py`, `tests/heavy_integration/test_phase1_drake_integration.py`, `tests/parity/test_simulate_contract_drake.py` | yes                          | yes (mujoco vs pinocchio, but no drake-vs-X) | OK provider, weak end-to-end |
| **MuJoCo**                   | 13                                                      | `tests/integration/test_mujoco_protocol.py`, `tests/parity/test_simulate_contract_mujoco.py`, `tests/cross_engine/test_mujoco_vs_pinocchio.py`                                                                                             | yes                          | partial                                      | OK                           |
| **Pinocchio**                | 6                                                       | `tests/heavy_integration/test_pinocchio_*.py` (8 files), `tests/parity/test_simulate_contract_pinocchio.py`, `tests/engines/test_pinocchio_*.py`                                                                                           | strongest of any engine      | yes                                          | OK                           |
| **OpenSim**                  | **0** — no provider, no provider tests                  | `tests/heavy_integration/test_opensim_muscles.py`, `test_opensim_myosuite_wiring.py`, `tests/parity/test_simulate_contract_opensim.py`                                                                                                     | partial                      | NO                                           | **GAP**                      |
| **MyoSuite**                 | 0                                                       | `tests/heavy_integration/test_myosuite_muscles.py`, `tests/integration/test_myosuite_muscles.py`                                                                                                                                           | minimal                      | NO                                           | **GAP**                      |
| **Simscape 2D / 3D**         | 0                                                       | `tests/integration/simscape/...`, `tests/integration/test_system_identification_simscape.py`                                                                                                                                               | partial                      | NO                                           | **GAP**                      |
| **Pendulum** (both lineages) | 0                                                       | `tests/analytical/test_pendulum_lagrangian.py`, `tests/parity/test_pendulum_simulation_parity.py`, `tests/engines/pendulum_models/.../test_double_pendulum_dynamics.py`                                                                    | OK for the math; NIL for fit | NO                                           | **GAP** for fit_swing        |
| **Putting Green**            | 0                                                       | `tests/integration/putting_green/`, `tests/heavy_integration/test_putting_green_engine.py`                                                                                                                                                 | OK for putt sim              | NO                                           | **GAP** for fit_swing        |

**Gaps below the "5 unit tests + 1 integration" bar:**

- OpenSim, MyoSuite, Simscape 2D, Simscape 3D, Pendulum, Putting Green all have **zero provider unit tests** and **zero canonical-contract integration**.
- `tests/cross_engine/` has only 1 test file. Calling this layer "cross-engine" is generous.

---

## 4. Playback / pose manipulation per surface

| Surface                         | Open the matcher / load a pose                         | Adjust starting pose (sliders/transform)                                                                                                                                                              | Save chosen pose to JSON/config the engine consumes | Playback a fitted swing                                                                              | Engine dropdown to switch backend                                                                                                                                   |
| ------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Drake**                       | via launcher tile (`src/launchers/drake_dashboard.py`) | NO standalone pose-tweak GUI                                                                                                                                                                          | partial (URDF/SDF model files)                      | scriptable; not from matcher GUI                                                                     | NO                                                                                                                                                                  |
| **MuJoCo**                      | launcher tile                                          | NO                                                                                                                                                                                                    | partial (MJCF)                                      | scriptable                                                                                           | NO                                                                                                                                                                  |
| **Pinocchio**                   | launcher tile                                          | NO                                                                                                                                                                                                    | partial (URDF)                                      | scriptable; vendor MeshCat hook in `pinocchio/.../viz/`                                              | NO                                                                                                                                                                  |
| **OpenSim**                     | launcher tile                                          | NO                                                                                                                                                                                                    | partial (.osim)                                     | scriptable                                                                                           | NO                                                                                                                                                                  |
| **MyoSuite**                    | NO                                                     | NO                                                                                                                                                                                                    | NO                                                  | NO                                                                                                   | NO                                                                                                                                                                  |
| **Simscape 2D**                 | launcher tile                                          | NO Python; MATLAB-only                                                                                                                                                                                | NO from Python                                      | NO from Python                                                                                       | NO                                                                                                                                                                  |
| **Simscape 3D**                 | launcher tile                                          | NO Python; MATLAB-only                                                                                                                                                                                | NO from Python                                      | NO from Python                                                                                       | NO                                                                                                                                                                  |
| **Pendulum**                    | NO                                                     | NO                                                                                                                                                                                                    | NO                                                  | NO                                                                                                   | NO                                                                                                                                                                  |
| **Putting Green**               | launcher tile                                          | NO                                                                                                                                                                                                    | NO                                                  | yes (own sim, putt only)                                                                             | NO                                                                                                                                                                  |
| **`starting_pose_matcher` GUI** | YES — Qt app loads xlsx/c3d/mat sources                | YES — has `SkeletonProvider` injection, animated marker preview (#4475 work), source-toggle panel (#4475 work), event-preset combo, sheet combo, but **no joint sliders, no rigid-transform sliders** | YES — `session_schema.py` round-trips JSON          | playback group exists in `gui_playback.py` but **does not call `fit_swing` and is not engine-aware** | **NO engine combo / dropdown.** Grep for `engine_combo`, `engineCombo`, `engine_dropdown`, `self.engine =` in `src/tools/starting_pose_matcher/` returns zero hits. |

**Bottom line.** There is no working "user picks engine, loads C3D, scrubs to starting pose, fires fit_swing, watches playback" workflow today. The matcher GUI is a pose / source authoring tool. The launcher tiles open per-engine dashboards but those don't host the canonical `fit_swing` either. No surface today drives `provider_registry.get_provider(...)`.

---

## 5. C3D pipeline gaps

Sources audited:

- `src/shared/python/motion_matching/loaders/c3d.py` (271 lines) — club-target loader
- `src/shared/python/motion_matching/loaders/c3d_body.py` (464 lines) — body-target loader
- `src/shared/python/motion_matching/loaders/_marker_clusters.py` — cluster heuristic
- `src/shared/python/upstream_drift_tools/lab/bio/c3d_reader.py` (260 lines) — canonical reader
- `src/shared/python/upstream_drift_tools/lab/bio/_c3d_io.py`, `_c3d_analog.py`, `_c3d_markers.py`, `_c3d_models.py`
- `src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/c3d_reader.py` — duplicate (52 lines, slated for collapse per #4475 child issue 9)
- `src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/_c3d_force_plates.py` — 13 lines stub

Findings (severity-ordered):

1. **`FORCE_PLATFORM` parameter group is silently dropped.** `grep -r "CORNERS\|CAL_MATRIX\|FORCE_PLATFORM\|ORIGIN" src/shared/python/upstream_drift_tools/lab/bio/` returns zero hits. The reader detects force plates by _analog channel name patterns_ (`_c3d_analog.py:detect_force_plate_channels`), not by parsing `FORCE_PLATFORM:CORNERS` / `ORIGIN` / `CAL_MATRIX` / `TYPE`. Consequences: (a) force-plate corners are unknown, so CoP cannot be transformed into lab frame; (b) calibration matrices for type-2/3/4 plates are ignored, so raw analog values are misinterpreted as already-calibrated forces; (c) channels with non-standard names (Kistler 9287, AMTI BP400600) are invisible.

2. **C3D `EVENT` group is never read.** No code reads `EVENT:USED`, `EVENT:LABELS`, `EVENT:TIMES`, `EVENT:CONTEXTS`. The matcher GUI has its own event-preset combo but cannot ingest events the lab already labelled in the C3D file. The user must re-label every file by hand.

3. **Marker-set heuristic is a substring match against ~6 candidate names.** `c3d.py:BUTT_CANDIDATES` and `HEAD_CANDIDATES` are tiny tuples (BUTT/GRIP/Grip/GripButt/ClubButt/BUTT_END/CLUB_BUTT and CH/ClubHead/CLUBHEAD/Clubhead/HEAD/ClubFace/CLUB_HEAD). Files using CGM2.4 or a 41-marker Plug-in-Gait convention with anatomical names (RTOE, LASI, CLAV) will be silently mismatched. There is no marker-set detection, no warning when `_pick_marker` returns `None` for a critical marker, and no fallback to MARKER_SET parameter group.

4. **No coordinate-frame detection.** The loader assumes Y-up by reading `_optical.y_up_to_z_up` deprecation forwarder; there is no inspection of `POINT.X_SCREEN` / `Y_SCREEN` (the standard C3D screen-axis hint) to confirm the file's convention. Vicon Nexus saves as Z-up; Cortex / Qualisys QTM as Y-up; Codamotion ODIN as Z-up — the loader applies a single hard-coded conversion.

5. **Three duplicate C3D readers.** Tracked as scratch issue 9 (`09_unify_c3d_readers.md`). Today: canonical `upstream_drift_tools/lab/bio/c3d_reader.py` (260 lines), Simscape app `apps/services/c3d_loader.py`, Simscape `src/c3d_reader.py` (52-line stub re-exporting), plus `_c3d_io.py` private module. Duplicates drift in features (force-plate handling diverges between Simscape's `_c3d_force_plates.py` stub and `_c3d_analog.detect_force_plate_channels`).

6. **`POINT.UNITS` and `ANALOG.UNITS` round-trip is partial.** `unit_scale(metadata.units, target_units)` only handles `m`/`mm` for points; `inch`, `cm`, custom analog units are ignored. The `target_units` plumb-through stops after `points_dataframe`.

7. **Variant handling.** No code path explicitly identifies Vicon vs Qualisys vs BTS vs Codamotion variants, even though they differ in: residual encoding (Vicon=fourth channel, BTS=separate), event group conventions, force-plate type defaults (Qualisys defaults to type-2, AMTI to type-4), and marker label casing. EzC3D papers over the binary differences but not the _parameter_ differences.

8. **No round-trip test.** There is no test that loads a C3D, writes it back, reloads and asserts metadata equivalence. `test_loaders_c3d.py` exists but only tests the points dataframe shape + the impact-detection heuristic.

9. **Event extraction for synthesised C3D.** No fixture exists in `tests/fixtures/motion_matching/` of a programmatically-built C3D with real EVENT entries to lock in extraction behaviour once it is added.

10. **Screen-axis ambiguity.** `_quaternion.rotmat_to_quat` is applied unconditionally without first asserting that the file's coordinate convention matches the code's expectation. If a Z-up file flows through, all club rotations are silently wrong.

---

## Outstanding issues to file

The four items already tracked under #4513, #4514, #4475, #4671, plus scratch issues 16–22 and 28–33, cover most of the **breadth**. The list below proposes **new** issues that those tracking parents do not already cover, ordered by severity.

### Issue 1 — feat(motion-matching): land OpenSim `fit_swing` provider (registry adapter)

**Description.** OpenSim has `fit_swing_opensim`, `fit_swing_opensim_multistart`, `simulate_with_coefficients`, and `coord_map` already shipped, plus a parity spec (`OPENSIM_PARITY_SPEC.md`). What is missing is a single ~80-line provider class and one `register_provider(...)` call in `__init__.py`, mirroring `MujocoFitSwingProvider`. The scratch note `21_opensim_provider.md` documents the contract; this issue should track shipping it. Without it, `available_engines()` will never list `"opensim"`.

**Acceptance criteria.**

- `src/engines/physics_engines/opensim/python/motion_matching/provider.py` defines `OpenSimFitSwingProvider` with `engine_name = "opensim"` and a `fit_swing(target, opts) -> CanonicalFitResult` method.
- Provider auto-registers at package import time (parallel structure to MuJoCo/Pinocchio).
- `_extract_club` accepts both `MultiSourceTarget` and bare `ClubTarget`.
- `supports_body_target() -> False` and `supports_ball_target() -> False` until #4520 lands those.
- New `tests/unit/motion_matching/opensim/test_provider.py` matching the depth of the MuJoCo provider tests (≥10 tests covering registration, target-extraction, options-projection, error surfaces).
- `tests/parity/test_simulate_contract_opensim.py` extended to assert canonical-fit roundtrip via the registry, not just the bare function.
- `available_engines()` in a clean interpreter returns `["drake", "mujoco", "opensim", "pinocchio"]` (sorted).

**Priority.** high. **Suggested labels.** `motion-matching`, `engine:opensim`, `parity`, `priority:high`.

### Issue 2 — feat(motion-matching): introduce `engine_version()` on the provider protocol

**Description.** `provider_registry` only requires `engine_name` and a callable `fit_swing`. There is no way to ask a provider what version of the underlying engine it wraps, which makes the planned `cross_engine_leaderboard.json` non-reproducible — two runs against different `pydrake` wheels produce indistinguishable rows. Add an `engine_version() -> str` method to the protocol with a default that returns `"unknown"` so existing providers stay compatible, and override it per-engine to query `pydrake.__version__`, `mujoco.__version__`, `pinocchio.__version__`, `opensim.GetVersion()`. Stamp the result into every leaderboard row.

**Acceptance criteria.**

- `FitSwingProvider` protocol in `src/shared/python/motion_matching/provider.py` adds `engine_version() -> str` (default `"unknown"`).
- All four shipped providers override it with the real version string.
- `CanonicalFitResult` (or the leaderboard row) gains an `engine_version` field.
- `tests/cross_engine/test_engine_version_advertised.py` asserts every registered provider returns a non-`"unknown"` value when the underlying engine is installed.
- Leaderboard runner stamps the value.

**Priority.** medium. **Labels.** `motion-matching`, `architecture`, `reproducibility`.

### Issue 3 — feat(starting-pose-matcher): wire the canonical `fit_swing` registry into the GUI; add an engine dropdown

**Description.** The matcher GUI (`src/tools/starting_pose_matcher/gui.py`) has zero references to `fit_swing`, `MultiSourceTarget`, or `provider_registry`. There is no path for a user to "load a C3D, choose an engine, run a fit, watch playback". The Qt app is currently a pose / source authoring tool. Add an engine-selection combo populated from `provider_registry.available_engines()`, a "Run fit" button that calls `get_provider(engine).fit_swing(target, opts)` on the GUI's currently-loaded `MultiSourceTarget`, and route the resulting `CanonicalFitResult` into the existing `gui_playback` pane.

**Acceptance criteria.**

- New combo widget shows the live registry; reacts to engines registered after import.
- "Run fit" button is disabled until a `MultiSourceTarget` is loaded and an engine is selected.
- Fit runs on a `QThread` with cancel; results display via existing `gui_playback`.
- "Save fit" button writes the `CanonicalFitResult` (theta, residuals, engine, engine_version, source-file sha256) to JSON via `session_schema`.
- Headless test asserts `gui._populate_engine_combo()` reflects `provider_registry.available_engines()`.
- Smoke test: in-process registry stub + one fake provider exercises the full button → fit → save path.

**Priority.** high. **Labels.** `motion-matching`, `gui`, `starting-pose-matcher`, `priority:high`.

### Issue 4 — feat(starting-pose-matcher): joint-slider + rigid-transform widget for starting pose

**Description.** Today the matcher loads a skeleton and lets the user pick event-frame snapshots, but provides no joint-angle sliders or rigid-transform handles to _adjust_ the loaded pose. Without this, the user cannot author the seed pose that `fit_swing` needs as `initial_pose`. Add a slider panel driven by the canonical 23-joint Drake humanoid coord set (or the provider's preferred set, queried from the provider) and a 6-DOF rigid handle for the root.

**Acceptance criteria.**

- Slider widget reads `provider.coord_names()` (new method) or falls back to a hard-coded canonical list.
- Slider updates re-render the live skeleton via existing `live_view_controller`.
- Pose is round-trippable via `session_schema` JSON.
- New `tests/integration/starting_pose_matcher/test_pose_authoring.py` covers slider → skeleton update → save → reload.

**Priority.** medium. **Labels.** `motion-matching`, `gui`, `starting-pose-matcher`.

### Issue 5 — feat(c3d): parse `FORCE_PLATFORM` parameter group (CORNERS, ORIGIN, CAL_MATRIX, TYPE)

**Description.** The canonical reader detects force plates from analog channel name regex only and never reads the C3D `FORCE_PLATFORM` group. This silently breaks CoP transformation to lab frame, drops calibration matrices for type-2/3/4 plates, and misses any plate whose channels are named non-conventionally. Read the parameter group, expose corners/origin/cal_matrix/type per plate via the metadata API, and apply the calibration when computing CoP.

**Acceptance criteria.**

- `C3DMetadata` gains a `force_plates: list[ForcePlateCalibration]` field with corners (4×3 m), origin (3,), cal_matrix (variable shape per type), type (1–4), and channel index ranges.
- Reader pulls these from the `FORCE_PLATFORM` parameter section using EzC3D's `parameters` API.
- `force_plate_dataframe(...)` honours type and applies cal_matrix when computing forces from raw voltages on type-2/3/4 plates.
- CoP columns are reported in lab frame, not plate-local.
- Synthesised-fixture test programmatically builds a C3D with known FORCE_PLATFORM values and asserts round-trip.
- Fixture using a real Kistler 9287 file (or a generic equivalent) added under `tests/fixtures/c3d/` confirms vendor-agnostic behaviour.

**Priority.** critical (silent data corruption). **Labels.** `c3d`, `motion-matching`, `priority:critical`, `data-integrity`.

### Issue 6 — feat(c3d): extract `EVENT` parameter group on load

**Description.** The matcher and the cost functions both reason about events (top-of-backswing, impact). The loader currently ignores `EVENT:USED`, `EVENT:LABELS`, `EVENT:TIMES`, `EVENT:CONTEXTS`. Files annotated by the lab arrive as if event-less, forcing manual relabelling. Read these parameters and surface them via `C3DMetadata.events: list[C3DEvent]` (the dataclass already exists).

**Acceptance criteria.**

- `build_metadata` populates `events`.
- `load_club_target_c3d` accepts an `event_label_for_alignment: str | None` and uses the matching `EVENT` time as the alignment frame instead of `detect_impact_index` heuristic when present.
- Fallback to heuristic logged at INFO when no events are present.
- Test with a synthesised C3D containing two events confirms extraction order, contexts, and times.

**Priority.** high. **Labels.** `c3d`, `motion-matching`, `priority:high`.

### Issue 7 — feat(c3d): marker-set detection (MARKER_SET / MODEL parameters)

**Description.** Marker discovery is a 6-candidate substring match. CGM2.4 / 41-marker Plug-in-Gait files break silently. Read `POINT:LABELS`, `SUBJECTS:MARKER_SETS`, and any `MODEL` parameter; pattern-match against a registry of known marker sets (Plug-in-Gait, CGM2.4, IOR, custom golf cluster); warn when the file's marker set is unrecognised; raise (not silent `None`) when a critical marker is missing.

**Acceptance criteria.**

- `MarkerSet` enum with at least: `CGM2_4`, `PLUG_IN_GAIT_41`, `IOR`, `GOLF_CLUSTER`, `UNKNOWN`.
- Detection function with a deterministic priority order; logs the chosen set.
- `load_club_target_c3d` raises `MarkerSetMismatchError` when the file is `UNKNOWN` and no override is provided, instead of returning a target with NaN club poses.
- Test fixtures: synthetic C3D with each known set; asserts detection.

**Priority.** high. **Labels.** `c3d`, `motion-matching`, `priority:high`.

### Issue 8 — fix(motion-matching): repeated import of MuJoCo motion-matching package raises during `synthesize` re-registration

**Description.** Importing `src.engines.physics_engines.mujoco.python.motion_matching` twice (e.g. once via the package, once via `provider.py`) prints `replacing existing 'mujoco' backend with <function synthesize_target_from_coefficients>` and then raises `ValueError: engine_name 'mujoco' is already registered`. This means the registry's last-writer-wins behaviour collides with `synthesize.py`'s own backend-replace logic. Symptom observed during this audit's smoke test (`available_engines()` returns `[]` after a multi-import sequence).

**Acceptance criteria.**

- Reproduce the error in a regression test.
- The MuJoCo package import is idempotent (re-import is a no-op for both the synthesis backend and the provider).
- Registry's idempotency is documented and asserted.

**Priority.** medium. **Labels.** `motion-matching`, `engine:mujoco`, `bug`.

### Issue 9 — feat(motion-matching): publish `reports/cross_engine_leaderboard.json` from CI

**Description.** `scripts/run_cross_engine_leaderboard.py` exists and `.github/workflows/cross-engine-leaderboard.yml` references it, but no committed `reports/cross_engine_leaderboard.json` exists in the repo, and only Pinocchio has a per-engine writer. Build a shared writer (`src/shared/python/motion_matching/leaderboard.py:append_row(engine, fit_result, engine_version)`), wire all four providers to it, and have CI commit the JSON back (or upload as a workflow artefact and a GitHub Pages page).

**Acceptance criteria.**

- Shared `append_row` API with deterministic column set: engine, engine_version, target_id, theta, residual_rms, wallclock, commit_sha.
- All four providers (drake, mujoco, opensim, pinocchio) call it from the canonical fit path.
- CI workflow produces the JSON on every PR; protected on main.
- New `tests/cross_engine/test_leaderboard_writer.py` validates schema.

**Priority.** medium. **Labels.** `motion-matching`, `ci`, `reproducibility`.

### Issue 10 — feat(motion-matching): land MyoSuite `fit_swing` provider OR explicitly mark the engine as fit-incapable

**Description.** MyoSuite is shipped as a tier-1 engine in `engines/loaders.py` but has no `motion_matching/` subpackage. Either implement the provider against the muscle-driven dynamics (probably wrapping MuJoCo's `fit_swing_mujoco` with a different model XML), or add a registry entry `MYOSUITE_FIT_INCAPABLE = True` and document that MyoSuite participates in forward-sim only. Today the silence is misleading.

**Acceptance criteria.**

- Either a `MyoSuiteFitSwingProvider` lands with provider tests, or
- `src/engines/physics_engines/myosuite/README.md` adds a "Why no fit_swing?" section and the engine catalog entry surfaces the limitation.

**Priority.** medium. **Labels.** `motion-matching`, `engine:myosuite`.

### Issue 11 — feat(motion-matching): pendulum + putting-green providers OR engine-catalog clarity

**Description.** Same shape as #10 for `pendulum`, `pendulum_models`, and `putting_green`. Pendulum already has analytic Lagrangian fitting in `tests/analytical/test_pendulum_lagrangian.py` — wrap that as a provider and the registry gains its first analytic baseline. Putting Green is genuinely fit-incapable for _swing_ but could expose a `fit_putt(stroke_target, opts)` on a sister registry; until then, the README should say so.

**Acceptance criteria.**

- Pendulum: provider class + register_provider; ≥5 unit tests.
- Putting Green: README clarification or sister-registry stub.
- Engine catalog (`src/engines/__init__.py`) exposes `fit_capable: bool` per engine.

**Priority.** low. **Labels.** `motion-matching`, `engine:pendulum`, `engine:putting-green`.

### Issue 12 — refactor(naming): disambiguate "skeleton provider" from "fit_swing provider"

**Description.** `src/tools/starting_pose_matcher/providers/{drake,mujoco,pinocchio,opensim}.py` are skeleton-extraction providers that map engine bodies → matcher vocabulary. They have nothing to do with `fit_swing` providers in `src/shared/python/motion_matching/provider_registry.py`. The shared identifier "provider" actively misleads. Rename one side.

**Acceptance criteria.**

- Pick: rename `src/tools/starting_pose_matcher/providers/` → `src/tools/starting_pose_matcher/skeleton_extractors/`, OR rename the canonical class to `FitSwingBackend` / similar.
- Update all imports and docstrings.
- Add a one-screen ADR covering the distinction.

**Priority.** low. **Labels.** `refactor`, `naming`, `motion-matching`.

---

## Test-coverage gaps to fill (separate from production gaps)

Targeted test work that should land **independent of** the production fixes above. Most of these are pure additions and don't require the production code to change.

1. **End-to-end registry test** (`tests/integration/motion_matching/test_registry_end_to_end.py`). After importing all four engine packages, assert `available_engines()` returns the expected list and each provider's `fit_swing` runs against a 2-second synthetic `ClubTarget` with `maxiter=5`. Today: missing.

2. **Round-trip C3D fidelity** (`tests/unit/motion_matching/test_c3d_roundtrip.py`). Build C3D via EzC3D with known POINT, ANALOG, FORCE_PLATFORM, EVENT, MARKER_SET parameters; load via canonical reader; assert every parameter survives.

3. **Synthesised C3D with FORCE_PLATFORM and EVENT** fixture under `tests/fixtures/motion_matching/`. Currently zero programmatic C3D builders in the repo. Build one and assert it loads.

4. **Marker-set detection coverage**. Once #7 lands, fixtures for at least three real-world marker conventions.

5. **Cross-engine fit-quality parity** (`tests/cross_engine/test_fit_quality_card_parity.py`). Run the same `ClubTarget` through every registered provider with `maxiter=20` and assert residual RMS within an agreed tolerance band. Today: only `tests/cross_engine/test_mujoco_vs_pinocchio.py` exists.

6. **Engine-version smoke** (`tests/unit/motion_matching/test_engine_version.py`). Once #2 lands, every registered provider returns a non-`"unknown"` version when the underlying SDK is importable.

7. **Headless GUI engine combo** (`tests/integration/starting_pose_matcher/test_engine_combo.py`). Once #3 lands, `pytest-qt` test that the combo reflects the registry.

8. **Pinocchio leaderboard writer regression** (`tests/unit/motion_matching/pinocchio/test_leaderboard_writer.py`). The writer currently exists only for Pinocchio with no test guarding its row schema.

9. **C3D screen-axis dispatch test**. Two synthetic files, one Y-up, one Z-up; assert the loader picks the correct frame conversion when `POINT:X_SCREEN`/`Y_SCREEN` is honoured (post-fix).

10. **Negative tests for OpenSim provider once it lands.** Mirror `tests/unit/motion_matching/mujoco/test_provider.py`'s structure verbatim (target-type errors, options-type errors, registration idempotency).

[did not get to] Detailed survey of `tests/motion_matching/test_cross_engine_equivalence.py` content — only the file path was confirmed; depth of coverage there is unverified within budget.
