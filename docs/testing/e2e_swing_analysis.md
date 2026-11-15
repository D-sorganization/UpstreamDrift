# End-to-End Swing Analysis Testing Guide

This guide documents the end-to-end (E2E) testing workflow for the Golf Modeling Suite's full swing analysis pipeline. These tests verify the system from raw input (video/mocap) to final biomechanical report.

## Overview

The E2E pipeline consists of:

1.  **Input Ingestion**: Loading video or C3D motion capture data.
2.  **Pose Estimation**: Extracting 3D keypoints (MediaPipe/Simi).
3.  **Kinematics**: Inverse kinematics to solve for joint angles.
4.  **Dynamics**: Inverse dynamics to compute torques and forces.
5.  **Reporting**: Generating metrics and visualization.

## Prerequisites

To run the E2E tests, you need:

- **Test Data**: Download the `golf_test_data.zip` (see [Getting Started](../user_guide/getting_started.md)).
- **Physics Engine**: At least one engine (MuJoCo recommended) installed.
- **Assets**: URDF models in `shared/models/`.

## Running the E2E Suite

The E2E tests are located in `tests/integration/test_end_to_end.py`.

```bash
# Run all E2E tests
pytest tests/integration/test_end_to_end.py

# Run specific swing phases
pytest tests/integration/test_end_to_end.py -k "downswing"
```

## Test Scenarios

### 1. Video-to-Biomechanics

**Files**: `tests/unit/test_video_analyzer_pipeline.py` (video-to-pose-to-joint-metrics pipeline), `tests/unit/test_video_pose_pipeline.py` (pose estimation stage), and `tests/unit/api/test_routes_video.py` (FastAPI video route contract)

- **Input**: Sample swing video (`swing_sample.mp4` from `golf_test_data.zip`)
- **Steps**:
  - Upload video to API.
  - Trigger `analyze_video` task.
  - Wait for completion.
  - Verify joint angles are within human limits.
- **Success Criteria**:
  - Pose confidence > 0.8
  - Processing time < 10s

### 2. Mocap-to-Torque

**Files**: `tests/integration/test_c3d_body_pipeline.py` (C3D loading and marker-to-model body registration), `tests/unit/motion_matching/test_match_pinocchio_c3d.py` (IK trajectory retargeting), and `tests/unit/engines/mujoco/test_inverse_dynamics.py` (inverse dynamics joint torque computation)

- **Input**: Sample C3D capture (`pro_driver.c3d` from `golf_test_data.zip`) and `tests/assets/simple_arm.urdf`
- **Steps**:
  - Load C3D file.
  - Run Inverse Kinematics (IK).
  - Run Inverse Dynamics (ID).
- **Success Criteria**:
  - Residual errors < 1cm
  - Computed torques < 300 Nm (human limit)

## Troubleshooting Failures

- **"Model not found"**: Ensure `shared/models` is populated.
- **"Video decoding failed"**: Install `ffmpeg` and `opencv-python`.
- **"Torque explosion"**: Check for discontinuities in input data (smoothing required).
