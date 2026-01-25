# Critical Issue: Video Pose Pipeline Model Fitting Stub

**Status:** Open
**Priority:** Critical
**Labels:** incomplete-implementation, critical
**Date:** 2026-01-25

## Description
The method `_convert_poses_to_markers` in `src/shared/python/video_pose_pipeline.py` is currently a stub (`pass`).
This method is essential for converting pose estimation results into a format suitable for biomechanical model fitting.
Currently, the `fit_to_model` method calls this stub (commented out) and returns a dummy `RegistrationResult` with `success=False`.

This effectively disables the model fitting capability of the video pose pipeline.

## Impact Analysis
*   **User Impact (5/5):** Users cannot perform model fitting on video data. Core feature of the pipeline is non-functional.
*   **Test Coverage (1/5):** The method is empty. Any tests relying on it are either mocked or failing/non-existent.
*   **Complexity (4/5):** Requires implementing coordinate transformations and data mapping between pose estimation keypoints (e.g., MediaPipe/OpenPose formats) and OpenSim/MuJoCo marker sets.

## Technical Details
*   **File:** `src/shared/python/video_pose_pipeline.py`
*   **Method:** `_convert_poses_to_markers`
*   **Caller:** `fit_to_model`

## Recommended Action
Implement `_convert_poses_to_markers` to correctly map keypoints to model markers.
Uncomment the call in `fit_to_model` and ensure `RegistrationResult` is computed using the `MarkerToModelMapper`.
