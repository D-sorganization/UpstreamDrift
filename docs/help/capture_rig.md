---
title: Capture Rig
tile_id: capture_rig
status: active
---

# Capture Rig

## Purpose

The Capture Rig workbench configures, calibrates, and monitors multi-camera spatial video and optical sensor arrays. It coordinates camera extrinsic positioning, intrinsic lens calibration, frame synchronization, and coordinate transformation into the global laboratory frame.

## Inputs

| Input | Units | Description |
| --- | --- | --- |
| Camera Stream Feeds | frames / s | Multi-view video streams from USB, GigE, or virtual cameras |
| Calibration Target | mm | Geometric dimensions of calibration wand or checkerboard |
| Shutter & Exposure | ms, EV | Sensor timing and lighting compensation settings |

## Outputs

| Output | Units | Description |
| --- | --- | --- |
| Intrinsic Matrices | pixels | Focal length, principal point, and radial distortion parameters |
| Extrinsic Matrices | m, rad | 6-DoF transformation from camera frames to laboratory origin |
| Reprojection Error | pixels | Mean Euclidean residual across calibration keypoints |
| Calibration Quality | categorical | Pass/fail verdict based on geometric tolerance budgets |

## Method

Multi-camera calibration uses bundle adjustment with Levenberg-Marquardt optimization to minimize 2D-to-3D reprojection errors across simultaneous views. Epipolar geometry constraints ensure rigorous 3D spatial reconstruction.

## Limitations

Requires adequate spatial coverage and non-coplanar views. Extreme wide-angle or fisheye lenses require specialized distortion modeling.

## See Also

- [Motion Capture](motion_capture.md)
- [Video Processor](video_processor.md)
