# Golf Modeling Suite - Development Readiness Plan (Feb 2026)

## Overview
This document outlines the plan to address the items identified in the Development Readiness Assessment (Feb 2026). The goal is to move the repository from its current beta state towards a more robust, feature-complete, and integrated platform.

## 1. Core Readiness & Reliability
**Goal:** Ensure all engines are properly validated at launch and diagnostics are consistent.

### 1.1 Engine Probes
- **Task**: Implement `MatlabProbe` in `shared/python/engine_probes.py`.
  - **Details**: Validate MATLAB installation, Python engine API, and existence of 2D/3D model directories.
- **Task**: Integrate `MatlabProbe` into `EngineManager`.
  - **Details**: Add `MATLAB_2D` and `MATLAB_3D` to the probes dictionary. Ensure `launch_golf_suite.py` checks these probes.

### 1.2 Telemetry & Logging
- **Task**: Replace `print` statements in `launch_golf_suite.py` with structured logging.
  - **Details**: Ensure the fallback status report uses the configured logger (or a format consistent with it) to avoid bypassing CI quality gates.

## 2. Feature Implementation: Pose Estimation
**Goal**: Establish the missing "Pose Estimator" capability referenced in project goals.

### 2.1 Package Structure
- **Task**: Create `shared/python/pose_estimation/` package.
  - **Files**: `__init__.py`, `estimator_interface.py`, `dummy_estimator.py` (initially).
  - **Interface**: Define `ingest_video(path)`, `get_joint_angles(frame)`, `get_confidence(frame)`.

### 2.2 Integration
- **Task**: Add `PoseEstimatorProbe` to `EngineManager`.
  - **Details**: Check for model weights (e.g., if using MediaPipe or OpenPose models eventually) and dependencies.
- **Task**: Expose estimator outputs to the MuJoCo GUI.
  - **Details**: Add a "Pose Estimation" tab or panel in `advanced_gui.py` that can load a video and drive the kinematic mode skeleton.

## 3. Feature Implementation: MuJoCo Enhancements
**Goal**: Harden the existing GUI features and make force/torque data visible.

### 3.1 Force/Torque Visualization
- **Task**: Connect existing UI toggles (`show_torques_cb`, `show_forces_cb`) to the rendering pipeline.
  - **Details**: In the simulation loop (`sim.py` or equivalent), read `mj_contactForce` and `mj_id` (inverse dynamics) and render vectors using `mjv_initGeom`.
  - **Task**: Add per-joint force plots to the "Plotting" tab.

### 3.2 Pose Library & Manipulation
- **Task**: Add automated readiness probe for Pose Library persistence.
  - **Details**: Verify read/write access to the pose library JSON file during startup.
- **Task**: Extend regression tests to cover the save/load flow.

## 4. Legacy Cleanup
**Goal**: Remove or integrate deprecated code.

### 4.1 Pinocchio Legacy
- **Task**: Audit `engines/physics_engines/pinocchio/python/legacy`.
- **Action**: Archive or delete if confirmed unused. If useful, refactor into `pinocchio_golf` package.

## Execution Phase 1 (Immediate)
1. Implement `MatlabProbe` and update `EngineManager`.
2. Fix `launch_golf_suite.py` logging.
3. Create `pose_estimation` package skeleton.

## Execution Phase 2 (Follow-up)
1. Implement force/torque rendering hooks in MuJoCo.
2. Build out the Pose Estimation implementation.
3. Legacy cleanup.
