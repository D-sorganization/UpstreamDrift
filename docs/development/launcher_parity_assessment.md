# Launcher Parity Assessment — February 2026

## Executive Summary

Two launcher implementations exist in UpstreamDrift:

| Launcher           | Technology              | Location                         | Status                                          |
| ------------------ | ----------------------- | -------------------------------- | ----------------------------------------------- |
| **PyQt Launcher**  | PyQt6 + Python          | `src/launchers/golf_launcher.py` | Primary, 2302 lines, tile-based grid (2×4)      |
| **Tauri Launcher** | React + Vite + Tauri v2 | `ui/`                            | Secondary, simulation-focused, missing features |

Both launchers are out of sync. The PyQt launcher has more tiles but broken launch paths. The Tauri launcher is more modern in design but missing most tiles and has no logos.

---

## 1. PyQt Launcher Assessment

### Tile Configuration (`src/config/models.yaml`)

| #   | ID               | Display Name   | Type              | Logo                     | Handler                    | Status      |
| --- | ---------------- | -------------- | ----------------- | ------------------------ | -------------------------- | ----------- |
| 1   | `mujoco_unified` | MuJoCo         | `custom_humanoid` | `mujoco_humanoid.png` ✅ | `HumanoidMuJoCoHandler` ✅ | **Working** |
| 2   | `drake_golf`     | Drake          | `drake`           | `drake.png` ✅           | `DrakeHandler` ✅          | **Working** |
| 3   | `pinocchio_golf` | Pinocchio      | `pinocchio`       | `pinocchio.png` ✅       | `PinocchioHandler` ✅      | **Working** |
| 4   | `opensim_golf`   | OpenSim        | `opensim`         | `opensim.png` ✅         | `OpenSimHandler` ✅        | **Working** |
| 5   | `myosim_suite`   | MyoSuite       | `myosim`          | `myosim.png` ✅          | `MyoSimHandler` ✅         | **Working** |
| 6   | `matlab_unified` | Matlab Models  | `special_app`     | `matlab_logo.png` ✅     | ❌ **No handler**          | **BROKEN**  |
| 7   | `motion_capture` | Motion Capture | `special_app`     | `c3d_icon.png` ⚠️        | ❌ **No handler**          | **BROKEN**  |
| 8   | `model_explorer` | Model Explorer | `special_app`     | `urdf_icon.png` ✅       | ❌ **No handler**          | **BROKEN**  |
| 9   | `putting_green`  | Putting Green  | `putting_green`   | ❌ **No logo**           | ❌ **No handler**          | **BROKEN**  |

### Key Issues

- **3 of 9 tiles broken** (special_app type has no handler in ModelHandlerRegistry)
- **1 tile has no logo** (putting_green)
- **Motion Capture logo** uses generic C3D icon, should represent all 3 tools (C3D + OpenPose + MediaPipe)
- **Missing tile:** Video Analyzer (backend exists but no launcher entry)
- **Status chips** don't cover `special_app` or `putting_green` types → shows "Unknown"
- **Help button** exists at line 1017 but visually buried in toolbar

### Logo → Tile Map (Canonical Reference)

```
src/launchers/assets/
├── mujoco_humanoid.png  → mujoco_unified ("MuJoCo")
├── drake.png            → drake_golf ("Drake")
├── pinocchio.png        → pinocchio_golf ("Pinocchio")
├── opensim.png          → opensim_golf ("OpenSim")
├── myosim.png           → myosim_suite ("MyoSuite")
├── matlab_logo.png      → matlab_unified ("Matlab Models")
├── c3d_icon.png         → motion_capture ("Motion Capture") ⚠️ misleading
├── urdf_icon.png        → model_explorer ("Model Explorer")
├── openpose.png/jpg     → (sub-item only, not primary tile)
├── [MISSING]            → putting_green ("Putting Green") ❌
└── [MISSING]            → video_analyzer ("Video Analyzer") ❌
```

---

## 2. Tauri/React Launcher Assessment

### Technology Stack

- React 19 + Vite 7 + TypeScript 5.9
- TailwindCSS 3
- Tauri v2 (Rust backend for process management)
- Recharts (live data plotting)
- React Three Fiber (3D visualization)
- React Query + WebSocket client

### Current Architecture

The Tauri launcher is a **simulation-focused** single-page app, NOT a tile-grid launcher:

- **Left sidebar**: Engine list + parameter panel + simulation controls
- **Center**: 3D viewport (Three.js)
- **Right sidebar**: Live analysis data
- **Bottom**: Live plot charts

### Engine Registry (`ui/src/api/useEngineManager.ts`)

| Name            | Display        | In PyQt?       | Logo?        |
| --------------- | -------------- | -------------- | ------------ |
| `mujoco`        | MuJoCo         | ✅             | ❌ Text only |
| `drake`         | Drake          | ✅             | ❌ Text only |
| `pinocchio`     | Pinocchio      | ✅             | ❌ Text only |
| `opensim`       | OpenSim        | ✅             | ❌ Text only |
| `myosuite`      | MyoSuite       | ✅             | ❌ Text only |
| `putting_green` | Putting Green  | ✅             | ❌ Text only |
| —               | —              | —              | —            |
| **MISSING**     | Motion Capture | ✅ in PyQt     | —            |
| **MISSING**     | Matlab Models  | ✅ in PyQt     | —            |
| **MISSING**     | Model Explorer | ✅ in PyQt     | —            |
| **MISSING**     | Video Analyzer | Backend exists | —            |

### Key Issues

- **4 tiles missing** compared to PyQt
- **0 logos** — all engine cards are text-only
- **No Help button** at all
- **Launch button below fold** — user must scroll sidebar to reach SimulationControls
- **No tile-grid dashboard** — jumps straight to simulation view
- **Only assets/react.svg** in `ui/src/assets/` — no engine logos

---

## 3. DRY Violations (Critical)

Model/tile definitions are duplicated in **4 places**:

1. `src/config/models.yaml` — PyQt tile definitions
2. `src/launchers/ui_components.py` `MODEL_IMAGES` dict — logo mappings
3. `ui/src/api/useEngineManager.ts` `ENGINE_REGISTRY` — Tauri tile definitions
4. `src/launchers/launcher_layout_manager.py` `default_ids` — default ordering

### Fix: Shared Launcher Manifest

Following the Gasification Model's `ServiceRegistry` pattern:

- Create `src/config/launcher_manifest.json` (single source of truth)
- Both launchers read from it
- Python side: `ModelRegistry` loads it
- TypeScript side: auto-generate types or fetch via API
- Test: parity tests validate both sides match

---

## 4. GitHub Issues Created

| #         | Title                                                                                        | Priority    |
| --------- | -------------------------------------------------------------------------------------------- | ----------- |
| **#1160** | Missing SpecialAppHandler — motion_capture, model_explorer, putting_green tiles don't launch | 🔴 Critical |
| **#1162** | Tauri/React launcher missing Motion Capture, Video Analyzer, Matlab, Model Explorer tiles    | 🔴 Critical |
| **#1163** | Create shared launcher manifest (DRY) — single source of truth for both PyQt and Tauri       | 🟡 High     |
| **#1164** | Logo/icon mapping audit — missing and mismatched logos across launchers                      | 🟡 High     |
| **#1165** | Tauri UI: launch button buried below fold — user must scroll to Start Simulation             | 🟠 Medium   |
| **#1167** | Video Analyzer tile missing — backend exists but inaccessible                                | 🟡 High     |
| **#1168** | PyQt status chip missing for special_app and putting_green types                             | 🟠 Medium   |
| **#1169** | Tauri EngineSelector has no logos/images — text-only engine cards                            | 🟡 High     |
| **#1170** | Help button not prominent — hard to find in top bar                                          | 🟠 Medium   |
| **#1171** | Tauri/React SimulationPage needs tile-based grid layout matching PyQt launcher               | 🟡 High     |
| **#1172** | Launcher parity tests — PyQt and Tauri must show identical tiles                             | 🟡 High     |
| **#1173** | MediaPipe/OpenPose GUIs are mock-only — no real estimator integration                        | 🟠 Medium   |

---

## 5. Recommended Execution Order

### Phase 1: Fix What's Broken (Critical)

1. **#1160** — Add SpecialAppHandler (unblocks 3 tiles)
2. **#1168** — Fix status chips for special_app/putting_green

### Phase 2: Shared Manifest (DRY Foundation)

3. **#1163** — Create `launcher_manifest.json`
4. **#1172** — Add parity tests

### Phase 3: Tauri Parity

5. **#1162** — Add missing tiles to Tauri
6. **#1169** — Add logos to Tauri EngineSelector
7. **#1171** — Add tile-grid dashboard to Tauri
8. **#1165** — Fix launch button below fold

### Phase 4: Polish

9. **#1164** — Complete logo audit (putting_green, video_analyzer)
10. **#1167** — Add Video Analyzer tile
11. **#1170** — Help button improvements
12. **#1173** — Wire MediaPipe/OpenPose to real estimators

---

## 6. Backend Integration Status (Video/Motion Capture)

All backend components **exist and are comprehensive**:

| Module                                                 | Lines | Status                                    |
| ------------------------------------------------------ | ----- | ----------------------------------------- |
| `shared/python/video_pose_pipeline.py`                 | 590   | ✅ Full pipeline                          |
| `shared/python/pose_estimation/mediapipe_estimator.py` | 385   | ✅ Production-ready                       |
| `shared/python/pose_estimation/openpose_estimator.py`  | 185   | ✅ Full BODY_25                           |
| `shared/python/pose_estimation/mediapipe_gui.py`       | 133   | ⚠️ Mock progress (not wired to estimator) |
| `shared/python/pose_estimation/openpose_gui.py`        | 133   | ⚠️ Mock progress (not wired to estimator) |
| `tools/video_analyzer/video_processor.py`              | 352   | ✅ Full implementation                    |
| `tools/video_analyzer/analyzer.py`                     | 573   | ✅ Full analyzer                          |
| `api/routes/video.py`                                  | 235   | ✅ REST API (sync + async)                |
| `launchers/motion_capture_launcher.py`                 | 79    | ✅ Sub-launcher with 3 tools              |
| `shared/python/help_system.py`                         | 721   | ✅ Full help dialog                       |

---

## 7. Implementation Status (Phase 1 Complete)

### ✅ Completed

| Change                   | File                                       | Status                                 |
| ------------------------ | ------------------------------------------ | -------------------------------------- |
| Shared launcher manifest | `src/config/launcher_manifest.json`        | ✅ 10 tiles, Model Explorer first      |
| Manifest loader (DBC)    | `src/config/launcher_manifest_loader.py`   | ✅ Typed, validated, frozen dataclass  |
| Manifest tests (TDD)     | `tests/config/test_launcher_manifest.py`   | ✅ 28 tests (27 pass, 1 expected skip) |
| SpecialAppHandler        | `src/launchers/launcher_model_handlers.py` | ✅ Fixes 3 broken tiles                |
| PuttingGreenHandler      | `src/launchers/launcher_model_handlers.py` | ✅ Fixes putting_green tile            |
| Handler tests (TDD)      | `tests/launchers/test_model_handlers.py`   | ✅ 19 tests, all pass                  |
| API endpoint             | `src/api/routes/launcher.py`               | ✅ /api/launcher/manifest route        |
| Default ordering         | `src/launchers/launcher_layout_manager.py` | ✅ Model Explorer first                |
| Server registration      | `src/api/server.py`                        | ✅ Router registered                   |

### 🔄 Remaining (tracked in GitHub issues)

Phase 2: Tauri UI updates (#1162, #1169, #1171, #1165)
Phase 3: Logo audit (#1164) — putting_green.png, data_explorer.png needed
Phase 4: Data Explorer page (#1178), Video export standardization (#1176)

---

## 8. Engine Capability Audit

### Force/Torque Vectors

| Engine            | Mass Matrix                | Jacobian                             | Force Viz           | Contact Forces                | Wrench/Screw                      |
| ----------------- | -------------------------- | ------------------------------------ | ------------------- | ----------------------------- | --------------------------------- |
| **MuJoCo**        | ✅ `compute_mass_matrix()` | ✅ `_compute_jacobian()`             | ✅ meshcat_adapter  | ✅ `compute_contact_forces()` | ✅ spatial_algebra + screw_theory |
| **Drake**         | ❓ Via MultibodyPlant      | ❓ Via MultibodyPlant                | ❓ MeshCat          | ❓                            | ❓                                |
| **Pinocchio**     | ✅ `compute_mass_matrix()` | ✅ `compute_end_effector_jacobian()` | ❓                  | ❓                            | ❓                                |
| **OpenSim**       | ✅ `compute_mass_matrix()` | ✅ `compute_jacobian()`              | ❓                  | ❓                            | ❓                                |
| **MyoSuite**      | ✅ `compute_mass_matrix()` | ✅ `compute_jacobian()`              | ❓                  | ❓                            | ❓                                |
| **Putting Green** | ❌ N/A (ball sim)          | ❌ N/A                               | ✅ friction vectors | ❌ N/A                        | ❌ N/A                            |

### Video Export

| Engine            | Video Export                                      | Dataset Export |
| ----------------- | ------------------------------------------------- | -------------- |
| **MuJoCo**        | ✅ `video_export.py` (VideoExporter, CV2/imageio) | ❓             |
| **Drake**         | ❌                                                | ❌             |
| **Pinocchio**     | ❌                                                | ❌             |
| **OpenSim**       | ❌                                                | ❌             |
| **MyoSuite**      | ❌                                                | ❌             |
| **Simscape**      | ✅ `golf_video_export.py`                         | ❌             |
| **Putting Green** | ❌                                                | ❌             |

---

## 9. Motion Capture Grouping

**Yes, MediaPipe, OpenPose, and C3D Reader all fall under Motion Capture.** This is confirmed by:

- `models.yaml`: motion_capture tile has description "C3D Viewer, OpenPose, and MediaPipe Analysis"
- `motion_capture_launcher.py`: sub-launcher with 3 items (C3D, OpenPose, MediaPipe)
- `launcher_manifest.json`: capabilities = ["c3d_viewer", "openpose", "mediapipe", "pose_estimation"]

They share the single "Motion Capture" tile with a sub-launcher that fans out to the 3 tools.
