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
-   **Test Data**: Download the `golf_test_data.zip` (see [Data Setup](../development/data_setup.md)).
-   **Physics Engine**: At least one engine (MuJoCo recommended) installed.
-   **Assets**: URDF models in `shared/models/`.

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
**File**: `tests/integration/test_video_e2e.py` (Proposed)

*   **Input**: `tests/assets/swing_sample.mp4`
*   **Steps**:
    *   Upload video to API.
    *   Trigger `analyze_video` task.
    *   Wait for completion.
    *   Verify joint angles are within human limits.
*   **Success Criteria**:
    *   Pose confidence > 0.8
    *   Processing time < 10s

### 2. Mocap-to-Torque
**File**: `tests/integration/test_mocap_e2e.py`

*   **Input**: `tests/assets/pro_driver.c3d`
*   **Steps**:
    *   Load C3D file.
    *   Run Inverse Kinematics (IK).
    *   Run Inverse Dynamics (ID).
*   **Success Criteria**:
    *   Residual errors < 1cm
    *   Computed torques < 300 Nm (human limit)

## Troubleshooting Failures

*   **"Model not found"**: Ensure `shared/models` is populated.
*   **"Video decoding failed"**: Install `ffmpeg` and `opencv-python`.
*   **"Torque explosion"**: Check for discontinuities in input data (smoothing required).
