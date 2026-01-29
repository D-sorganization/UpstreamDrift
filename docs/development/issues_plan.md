# Golf Modeling Suite - Upgrade Plan & Issues

This document outlines the upgrade plan for the Golf Modeling Suite. Each item represents a task to be tracked via GitHub Issues.

## 1. Unified MuJoCo Interface

**Issue Title:** Combine MuJoCo Models into Single Unified Interface
**Description:**
Currently, there are separate launcher tiles for "MuJoCo Humanoid" and "MuJoCo Dashboard". We need to combine these into a single "MuJoCo" interface that maintains the best features of both.

- The interface should allow selecting between the Humanoid simulation and the Dashboard analysis/visualization.
- Legacy versions should be preserved in an `archive` directory for reference.
- **Video Playback**: Ensure videos do not autoplay. The user must have play/pause/seek controls.

**Acceptance Criteria:**

- Single "MuJoCo" tile in the launcher.
- New GUI allows selecting simulation mode (Humanoid vs Dashboard).
- Video player has manual controls (No autoplay).
- Legacy files preserved.

**Feasibility:** High. Existing `mujoco_humanoid` and `dashboard` code can be imported into a wrapper GUI.

## 2. Unified Matlab Interface

**Issue Title:** Unified Launcher Interface for Matlab Models
**Description:**
Consolidate "Matlab Simscape 2D", "Matlab Simscape 3D", "Dataset Generator GUI", and "Golf Swing Analysis GUI" into a single "Matlab Models" launcher interface.

- Create a Python or Matlab-based GUI (Python preferred for consistency with the main launcher) that lists these available models/tools and launches them.
- Ensure integration with machine learning capabilities is considered (e.g., passing data between Python ML and Matlab).

**Acceptance Criteria:**

- Single "Matlab Models" tile in the launcher.
- Clicking it opens a sub-launcher (or tab) to choose the specific Matlab tool.
- Launch mechanisms for .slx and .m files remain functional.

**Feasibility:** High. Can use the same pattern as the Unified Launcher to create a "Matlab Launcher" window.

## 3. Model Explorer & URDF Generator Expansion [DONE]

**Issue Title:** Expand URDF Generator into Model Explorer
**Description:**
Rename "URDF Generator" to "Model Explorer". Expand its functionality to help users load and explore available models across categories:

- Humanoid
- Pendulum
- Robotic Manipulator
- Component
- Mechanical
- **Simplified Viewer**: Add a Python-based mesh viewer (using `trimesh` or `pyqtgraph` or `mescat`) as a fallback if the MuJoCo viewer is unavailable.

**Acceptance Criteria:**

- "Model Explorer" tile replaces "URDF Generator".
- UI lists models by category.
- Preview pane shows the simplified mesh viewer.
- Option to "Load" or "Generate" models.

**Feasibility:** High. `src/tools/urdf_generator` can be expanded.

## 4. Embed/Pop-out Screens [DONE]

**Issue Title:** Implement Embed/Pop-out Functionality for Interfaces
**Description:**
Enhance the main GUI (Unified Launcher) to allow running interfaces (like the MuJoCo simulation or Model Explorer) either:

1. **Embedded**: Within the main window dock area.
2. **Popped Out**: In a separate independent window.

- This applies to all integrated engines where feasible.

**Acceptance Criteria:**

- "Detach" / "Attach" button on the window title bars or toolbar.
- Window state persists or defaults to user preference.

**Feasibility:** Medium. Requires refactoring `GolfLauncher` to manage `QDockWidget` vs `QMainWindow` instances dynamically.

## 5. Screen Overlay & Controls [DONE]

**Issue Title:** Global Screen Overlay and Control Display
**Description:**
Implement an overlay system that can display controls and info on top of all screens for all engines.

- This might utilize a transparent top-level window or an overlay widget injection into the engine viewports.

**Acceptance Criteria:**

- Toggleable overlay layer.
- Works over MuJoCo and other engine windows (if they are embedded).

**Feasibility:** Medium. Embedding external windows (like MuJoCo Native UI) makes overlaying hard. If using Python bindings (rendering to image), it's easy.

## 6. Markerless Motion Capture Section (OpenPose & MediaPipe)

**Issue Title:** Marketless Motion Capture Integration (OpenPose & MediaPipe)
**Description:**
Create a dedicated section (or tool) for Markerless Motion Capture.

- Integrate `OpenPose` functionality (already present in `src/shared/python/pose_estimation`).
- **Add MediaPipe Support**: Integrate MediaPipe Pose as a permissive license alternative to OpenPose.
- Allow users to import video and extract motion data.
- **Note**: Ensure this is accessible via a new "Motion Capture" entry (replacing C3D Viewer or merging it) fitting into the simplified layout.

**Acceptance Criteria:**

- Interface to load video files.
- Option to choose **OpenPose** or **MediaPipe** backend.
- Run inference and visualize keypoints.
- Export to C3D or TRC format.

**Feasibility:** High. `openpose_estimator.py` exists. MediaPipe is easy to add.

## 7. Branding & Logo Update

**Issue Title:** Update Branding and Logo
**Description:**
Update the application logo and favicon with the provided asset.

- Asset path: `C:/Users/diete/.gemini/antigravity/brain/0e5dda19-c1e4-4b2b-963d-9e5edb8afe4a/uploaded_media_1769702712320.png`
- Apply to Window Icon, About Dialog, and Splash Screen.

**Acceptance Criteria:**

- New logo visible in all UI contexts.

**Feasibility:** Very High. Simple asset replacement.

## 8. Simplify Launcher Layout

**Issue Title:** Simplify Launcher to Two Rows of Tiles
**Description:**
Refactor the main launcher grid to display exactly 8 tiles in two rows of 4.
**Target Layout:**

1. MuJoCo (Unified)
2. Drake
3. Pinocchio
4. OpenSim
5. MyoSuite
6. Matlab Models (Unified)
7. C3D Viewer / Motion Capture
8. Model Explorer (URDF Gen)

- Hide or merge other items (e.g., Shot Tracer, individual Matlab models, Pendulum).
- Update `src/config/models.yaml` and `golf_launcher.py`.

**Acceptance Criteria:**

- Launcher shows exactly these 8 tiles.
- Grid layout is 2x4.

**Feasibility:** High. Configuration change.

## 9. Repository Cleanup & Professionalism

**Issue Title:** Repository Standardization and Cleanup
**Description:**
Improve the professional appearance and organization of the repository.

- **Root Cleanup**: Move loose `.md` files (plans, audits) into `docs/`.
- **README Update**: Rewrite the main README to be more professional, highlighting the new capabilities and engines.
- **GitHub Description**: Update the GitHub repository description (via `gh` CLI) to reflect the multi-engine nature.
- **Documentation**: Ensure `docs/` is organized.

**Acceptance Criteria:**

- Root directory is clean (only config files, src, docs, engines).
- README is modern and comprehensive.
- GitHub metadata is updated.

**Feasibility:** High. Organization task.
