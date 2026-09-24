---
title: Video Analyzer
tile_id: video_analyzer
status: active
---

# Video Analyzer

## Purpose

The Video Analyzer extracts 2D biomechanical keypoints, swing planes, body angles, and club delivery paths from recorded high-speed video footage. It supports manual measurement annotations, automated pose estimation overlays, and side-by-side comparison of swings.

## Inputs

| Input | Units | Description |
| --- | --- | --- |
| Video Footage | mp4 / avi / mov | Recorded swing video (target 60-240 fps) |
| Spatial Scale | pixels / m | Known reference distance in the video plane |
| Analysis Keyframes | frames / ms | Selected address, top-of-swing, impact, and finish events |

## Outputs

| Output | Units | Description |
| --- | --- | --- |
| 2D Keypoint Trajectories | pixels | Frame-by-frame joint locations (shoulders, hips, hands) |
| Swing Plane Angles | deg | Shaft lean, spine angle, and shoulder tilt |
| Temporal Sequencing | ms | Interval durations between address, top, impact, and finish |

## Method

Video frames are decoded via OpenCV / FFmpeg. Kinematic joint detection applies trained neural pose estimation with temporal filtering to reduce jitter. Geometric measurements project pixel coordinates into metric dimensions using calibrated planar scales.

## Limitations

Single-camera video suffers from planar foreshortening and parallax errors when motion occurs out of plane. 3D spatial quantities should be evaluated with multi-camera motion capture.

## See Also

- [Video Processor](video_processor.md)
- [Pose Studio](pose_studio.md)
- [Motion Capture](motion_capture.md)
