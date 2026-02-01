# GitHub Issues for Pose Editor and Club Data Display System

This document contains GitHub issues to be created for tracking the deployment of the
pose editor and club data display systems across all physics engines.

---

## Issue 1: Integrate Pose Editor Tab into Drake GUI

**Title:** feat(drake): Integrate shared pose editor tab with full functionality

**Labels:** enhancement, drake, pose-editor

**Description:**

### Summary
Integrate the new shared pose editor system into the Drake GUI application to provide
comprehensive pose manipulation capabilities.

### Features to Integrate
- [ ] Add `DrakePoseEditorTab` as a new tab in `DrakeSimApp`
- [ ] Connect pose editor to plant and context
- [ ] Enable gravity toggle functionality
- [ ] Connect pose library save/load to visualization updates
- [ ] Add preset poses (Address, T-Pose, etc.)
- [ ] Ensure sliders sync with current model state

### Implementation Details
The pose editor tab is located at:
`src/engines/physics_engines/drake/python/src/pose_editor_tab.py`

Integration requires:
1. Import `DrakePoseEditorTab` in `drake_gui_app.py`
2. Add tab to the main tab widget
3. Connect `set_plant_and_context()` after model loading
4. Connect visualization update callback

### Testing
- [ ] Verify joint slider control works for all joints
- [ ] Test gravity toggle (model should not fall when disabled)
- [ ] Test pose save/load functionality
- [ ] Verify preset poses load correctly
- [ ] Test pose interpolation between saved poses

---

## Issue 2: Integrate Pose Editor Tab into Pinocchio GUI

**Title:** feat(pinocchio): Integrate shared pose editor tab with full functionality

**Labels:** enhancement, pinocchio, pose-editor

**Description:**

### Summary
Integrate the new shared pose editor system into the Pinocchio GUI application.

### Features to Integrate
- [ ] Add `PinocchioPoseEditorTab` as a new tab in `PinocchioGUI`
- [ ] Connect pose editor to model, data, q, and v references
- [ ] Enable gravity toggle functionality
- [ ] Connect pose library to visualization updates
- [ ] Add preset poses
- [ ] Replace/enhance existing kinematic controls with new system

### Implementation Details
The pose editor tab is located at:
`src/engines/physics_engines/pinocchio/python/pinocchio_golf/pose_editor_tab.py`

Integration requires:
1. Import `PinocchioPoseEditorTab` in `gui.py`
2. Add as a tab or replace existing kinematic controls
3. Connect `set_model_and_data()` after URDF loading
4. Connect visualizer and update callback

### Testing
- [ ] Verify all joint controls work
- [ ] Test gravity toggle
- [ ] Test pose library functionality
- [ ] Verify kinematic mode works correctly with pose editor

---

## Issue 3: Integrate Club Data Tab into MuJoCo GUI

**Title:** feat(mujoco): Add club data display and target trajectory visualization

**Labels:** enhancement, mujoco, club-data

**Description:**

### Summary
Add the shared club data tab to the MuJoCo GUI for displaying club specifications
and professional player target trajectories.

### Features to Integrate
- [ ] Add `ClubDataTab` to `AdvancedGolfAnalysisWindow`
- [ ] Load default Club_Data.xlsx on startup
- [ ] Implement target trajectory overlay rendering
- [ ] Connect target tracking error computation
- [ ] Display real-time tracking error metrics

### Implementation Details
The club data tab is located at:
`src/shared/python/club_data/club_data_tab.py`

Target overlay rendering requires:
1. Add rendering code to `MuJoCoSimWidget` for trajectory path
2. Render phase markers (address, top, impact, finish)
3. Compute and display tracking error during simulation

### Testing
- [ ] Verify Excel club data loads correctly
- [ ] Test player data loading
- [ ] Verify trajectory overlay renders correctly
- [ ] Test tracking error computation

---

## Issue 4: Integrate Club Data Tab into Drake GUI

**Title:** feat(drake): Add club data display and target trajectory visualization

**Labels:** enhancement, drake, club-data

**Description:**

### Summary
Add the shared club data tab to the Drake GUI for displaying club specifications
and professional player target trajectories.

### Features to Integrate
- [ ] Add `ClubDataTab` to `DrakeSimApp` main tabs
- [ ] Implement Meshcat target trajectory overlay
- [ ] Connect tracking error computation
- [ ] Display real-time metrics during simulation

### Implementation Details
Target overlay rendering in Meshcat requires:
1. Add trajectory line geometry to Meshcat scene
2. Add sphere markers for swing phases
3. Update overlay during simulation

### Testing
- [ ] Verify club data loads correctly
- [ ] Test trajectory overlay in Meshcat
- [ ] Verify tracking error display

---

## Issue 5: Integrate Club Data Tab into Pinocchio GUI

**Title:** feat(pinocchio): Add club data display and target trajectory visualization

**Labels:** enhancement, pinocchio, club-data

**Description:**

### Summary
Add the shared club data tab to the Pinocchio GUI for displaying club specifications
and professional player target trajectories.

### Features to Integrate
- [ ] Add `ClubDataTab` to `PinocchioGUI` main tabs
- [ ] Implement Meshcat target trajectory overlay
- [ ] Connect tracking error computation
- [ ] Display metrics during simulation

### Implementation Details
Similar to Drake, requires Meshcat overlay rendering.

### Testing
- [ ] Verify club data loading
- [ ] Test trajectory visualization
- [ ] Verify tracking metrics

---

## Issue 6: Add Click-and-Drag IK Manipulation to Drake Pose Editor

**Title:** feat(drake): Add click-and-drag IK manipulation for pose editing

**Labels:** enhancement, drake, pose-editor, IK

**Description:**

### Summary
Extend the Drake pose editor to support interactive click-and-drag manipulation
using inverse kinematics, similar to the MuJoCo implementation.

### Features to Implement
- [ ] Body selection via mouse ray-casting
- [ ] Damped least-squares IK solver
- [ ] Constraint system (fix bodies in space)
- [ ] Real-time visualization updates during drag
- [ ] Joint limit enforcement

### Reference Implementation
See MuJoCo's `InteractiveManipulator` class at:
`src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/interactive_manipulation.py`

---

## Issue 7: Add Click-and-Drag IK Manipulation to Pinocchio Pose Editor

**Title:** feat(pinocchio): Add click-and-drag IK manipulation for pose editing

**Labels:** enhancement, pinocchio, pose-editor, IK

**Description:**

### Summary
Extend the Pinocchio pose editor to support interactive click-and-drag manipulation
using inverse kinematics.

### Features to Implement
- [ ] Use `pink` IK solver or implement custom damped LS
- [ ] Body selection from Meshcat viewport
- [ ] Constraint system for fixed bodies
- [ ] Joint limit enforcement

### Reference
Pinocchio already has IK capabilities via the `pink_solver.py` module.

---

## Issue 8: Create Comprehensive Test Suite for Pose Editor System

**Title:** test: Add comprehensive tests for shared pose editor system

**Labels:** testing, pose-editor

**Description:**

### Summary
Create unit and integration tests for the shared pose editor system.

### Tests to Create
- [ ] Unit tests for `JointInfo` and `PoseEditorState`
- [ ] Unit tests for `PoseLibrary` save/load/export/import
- [ ] Unit tests for `PoseInterpolator`
- [ ] Integration tests for Drake pose editor
- [ ] Integration tests for Pinocchio pose editor
- [ ] Widget tests for PyQt6 components

### Files to Test
- `src/shared/python/pose_editor/core.py`
- `src/shared/python/pose_editor/library.py`
- `src/shared/python/pose_editor/widgets.py`

---

## Issue 9: Create Comprehensive Test Suite for Club Data System

**Title:** test: Add comprehensive tests for shared club data system

**Labels:** testing, club-data

**Description:**

### Summary
Create tests for the shared club data loading and display system.

### Tests to Create
- [ ] Unit tests for `ClubDataLoader`
- [ ] Unit tests for `ClubSpecification` and `ProPlayerData`
- [ ] Unit tests for `ClubTargetManager`
- [ ] Integration tests with sample Excel files
- [ ] Widget tests for `ClubDataTab`

### Files to Test
- `src/shared/python/club_data/loader.py`
- `src/shared/python/club_data/targets.py`
- `src/shared/python/club_data/club_data_tab.py`

---

## Issue 10: Documentation for Pose Editor and Club Data Systems

**Title:** docs: Add user guide documentation for pose editor and club data systems

**Labels:** documentation

**Description:**

### Summary
Create comprehensive documentation for the new pose editor and club data systems.

### Documentation to Create
- [ ] User guide for pose editor functionality
- [ ] Developer guide for extending pose editor
- [ ] User guide for club data loading
- [ ] API reference documentation
- [ ] Integration examples for each engine

### Files to Create/Update
- `docs/user_guide/pose_editor.md`
- `docs/user_guide/club_data.md`
- `docs/developer_guide/extending_pose_editor.md`

---

## Priority Order

1. **High Priority (Core Functionality)**
   - Issue 1: Drake Pose Editor Integration
   - Issue 2: Pinocchio Pose Editor Integration
   - Issue 3: MuJoCo Club Data Integration

2. **Medium Priority (Feature Completion)**
   - Issue 4: Drake Club Data Integration
   - Issue 5: Pinocchio Club Data Integration
   - Issue 6: Drake IK Manipulation
   - Issue 7: Pinocchio IK Manipulation

3. **Lower Priority (Quality)**
   - Issue 8: Pose Editor Tests
   - Issue 9: Club Data Tests
   - Issue 10: Documentation

---

## Created By
This document was auto-generated during the implementation of the pose editor
and club data display systems.

Date: 2026-01-31
