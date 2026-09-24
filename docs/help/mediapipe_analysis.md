---
title: MediaPipe
tile_id: mediapipe_analysis
status: complete
---

# MediaPipe

## Purpose

Estimate body pose from a single ordinary video file using Google's
MediaPipe (BlazePose), entirely on your own machine, and write the
per-frame result out as JSON. Use it when you have one camera and no
markers, and you want a fast first look at joint angles rather than a
lab-grade measurement.

## Inputs

| Input | Unit / type | Notes |
| --- | --- | --- |
| Video file | `.mp4`, `.avi`, `.mov` | Chosen through a file dialog. A blank selection is rejected before any worker starts, so the progress bar cannot silently sit at 0 percent. |
| `min_detection_confidence` | dimensionless, 0.0-1.0 (default `0.5`) | Typed into the configuration box as `key=value`. |
| `min_tracking_confidence` | dimensionless, 0.0-1.0 (default `0.5`) | As above. |

The configuration box is parsed line by line; `#` starts a comment,
numeric values become floats and `true`/`false` become booleans. Frame
rate and resolution are read from the video itself (frames, Hz, pixels)
and are not settable here.

## Outputs

| Output | Unit / type | Notes |
| --- | --- | --- |
| `output/mediapipe_results.json` | JSON array, one object per processed frame | Written relative to the process working directory. |
| `timestamp` | seconds | Computed as `frame_index / fps` from the video's own frame rate. |
| `confidence` | dimensionless, 0.0-1.0 | Mean landmark visibility for the frame. `0.0` when no pose was detected. |
| `joint_angles` | **radians** | Per-joint mapping. The `PoseEstimationResult` contract defines joint angles in radians. |
| `keypoint_count` | count | Number of raw keypoints retained for the frame; present only when raw keypoints exist. |
| In-memory raw keypoints | normalized 3D and pixel 2D | Held during the run but **not** written to the JSON file. Normalized `x`/`y` are fractions of image width/height; `z` is MediaPipe's relative depth. The 2D copy is `x * width_px`, `y * height_px`. |

## Method

The tile is a PyQt6 front end,
[`mediapipe_gui.py`](../../src/shared/python/pose_estimation/mediapipe_gui.py),
which runs `MediaPipeEstimator`
([`mediapipe_estimator.py`](../../src/shared/python/pose_estimation/mediapipe_estimator.py))
on a `QThread` so the window stays responsive. The estimator wraps
MediaPipe Pose, which produces **33 landmarks**, maps landmark indices
to names via its `LANDMARK_MAP`, applies per-landmark Kalman temporal
smoothing, and converts the smoothed keypoints to joint angles. The
shared result contract is `PoseEstimationResult` in
[`interface.py`](../../src/shared/python/pose_estimation/interface.py).

MediaPipe is Apache-2.0 and runs locally; nothing is uploaded. Its role
as a source format for the wider pipeline (33 landmarks, normalized
coordinates, partial 3D via relative depth) is recorded in the
[format matrix](../motion_pipeline/formats.md).

## Limitations

- **Estimated landmarks are inference, not measurement.** A neural
  network is guessing where your joints are from pixels. Nothing here
  was measured. Do not treat the joint angles as calibrated
  biomechanics.
- **`z` is relative depth, not a calibrated distance.** The pipeline
  documentation classifies MediaPipe as partial 3D for exactly this
  reason. Single-camera monocular capture cannot recover true metric
  depth.
- **Not trained for high-speed sports motion.** The repo's markerless
  notes state plainly that dropouts through the downswing are expected.
- **Frames with no detection are still emitted**, with empty
  `joint_angles` and `confidence` of `0.0`. Filter on confidence
  yourself.
- **Temporal smoothing is always on** and cannot be disabled from this
  GUI: the worker hard-codes `enable_temporal_smoothing=True`. Only the
  two confidence thresholds in the configuration box actually reach the
  estimator; any other key you type is parsed and then ignored.
- **The progress bar is not per-frame.** The worker emits progress once
  at 0 percent and again at 100 percent on completion, so a long video
  looks stalled while it is working.
- **Raw keypoints are not exported** - only a count of them.
- Requires `mediapipe` and `opencv-python`. Without them the run fails
  with an install hint rather than a partial result.
- Single-person only: the estimator reads MediaPipe Pose's single
  `pose_landmarks` result and has no multi-person path.

## See Also
- [Video analysis tutorial](../tutorials/content/04_video_analysis.md)
- [Motion pipeline format matrix](../motion_pipeline/formats.md)
- [Motion pipeline troubleshooting](../motion_pipeline/troubleshooting.md)
- [OpenPose](openpose_analysis.md) - the other estimator tile
- [Motion Capture (FreeMoCap sidecar)](motion_capture.md) - multi-camera markerless 3D
- [C3D Viewer](c3d_viewer.md) - marker-based capture, for contrast
