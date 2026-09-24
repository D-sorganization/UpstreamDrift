---
title: OpenPose
tile_id: openpose_analysis
status: complete
---

# OpenPose

## Purpose

Estimate body pose from a single video file using CMU's OpenPose
BODY_25 model and write the per-frame result out as JSON. Use it when
you want OpenPose's keypoint quality and you are willing to install and
license OpenPose yourself; use [MediaPipe](mediapipe_analysis.md)
instead when you want something that just runs.

## Inputs

| Input | Unit / type | Notes |
| --- | --- | --- |
| Video file | `.mp4`, `.avi`, `.mov` | Chosen through a file dialog. |
| `model_folder` | filesystem path | The only configuration key the worker actually forwards to the estimator. When absent, the estimator resolves a default location itself. |
| Configuration box defaults | text | Pre-filled with `model_pose=BODY_25`, `net_resolution=-1x368`, `number_people_max=1`. These are shown for information; see Limitations. |

The parser accepts both `--key value` and `key=value` lines and treats
`#` as a comment. Frame rate and resolution come from the video itself
(frames, Hz, pixels).

## Outputs

| Output | Unit / type | Notes |
| --- | --- | --- |
| `output/openpose_results.json` | JSON array, one object per processed frame | Written relative to the process working directory. |
| `timestamp` | seconds | Taken from OpenCV's `CAP_PROP_POS_MSEC` divided by 1000. |
| `confidence` | dimensionless, 0.0-1.0 | Per-frame aggregate. `0.0` when no person was detected. |
| `joint_angles` | **radians** | Per-joint mapping, as defined by the shared `PoseEstimationResult` contract. |
| `keypoint_count` | count | Number of raw keypoints retained; present only when raw keypoints exist. |
| In-memory raw keypoints | `[x_px, y_px, score]` per named keypoint | **Pixel** image coordinates plus a 0.0-1.0 detection score. Held during the run, not written to the JSON file. |

The **25 BODY_25 keypoints** are named in the estimator's
`KEYPOINT_MAP`: Nose, Neck, RShoulder, RElbow, RWrist, LShoulder,
LElbow, LWrist, MidHip, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle, REye,
LEye, REar, LEar, LBigToe, LSmallToe, LHeel, RBigToe, RSmallToe, RHeel.

## Method

[`openpose_gui.py`](../../src/shared/python/pose_estimation/openpose_gui.py)
is a PyQt6 front end that runs `OpenPoseEstimator`
([`openpose_estimator.py`](../../src/shared/python/pose_estimation/openpose_estimator.py))
on a `QThread`. The estimator wraps the `pyopenpose` Python bindings,
forces `model_pose = "BODY_25"` and `net_resolution = "-1x368"`, maps
keypoint indices to names, and translates them into joint angles through
the shared `OPENPOSE_TO_CANONICAL` mapping. The result type is
`PoseEstimationResult` in
[`interface.py`](../../src/shared/python/pose_estimation/interface.py).

Background on the integration is in
[OpenPose integration](../engines/openpose.md); OpenPose's role as an
ingestion format (2D only, BODY_25 or COCO_18 JSON) is in the
[format matrix](../motion_pipeline/formats.md).

## Limitations

- **Estimated keypoints are inference, not measurement.** A neural
  network is guessing joint locations from pixels. Treat the angles as
  an indication, not a measurement.
- **2D only.** OpenPose here returns pixel coordinates plus a score.
  There is no depth and no metric scale; the pipeline documentation
  classifies OpenPose output as 2D-only. Any 3D interpretation has to
  come from a separate lifting or multi-camera step.
- **Most of the configuration box does nothing.** The worker reads only
  `model_folder`. `model_pose`, `net_resolution` and
  `number_people_max` are pre-filled text; the estimator hard-codes
  BODY_25 and `-1x368` and does not read a person limit.
- **Single-person in practice**: the estimator takes
  `poseKeypoints[0]`, so in a multi-person frame everyone but the first
  detected person is discarded, with no control over which that is.
- **The progress bar is not per-frame.** Progress is emitted once at 0
  percent and again at 100 percent, so a long video looks stalled.
- **No smoothing.** Unlike the MediaPipe tile there is no temporal
  filter, so per-frame jitter passes straight through.
- **Raw keypoints are not exported** - only a count of them.
- **Requires a separate install and carries licence conditions.**
  `pyopenpose` and the model weights are not shipped with
  UpstreamDrift, and the tile is registered as *Academic License*.
  Without them the run fails with an install hint.

## See Also
- [OpenPose integration](../engines/openpose.md)
- [Video analysis tutorial](../tutorials/content/04_video_analysis.md)
- [Motion pipeline format matrix](../motion_pipeline/formats.md)
- [Motion pipeline troubleshooting](../motion_pipeline/troubleshooting.md)
- [MediaPipe](mediapipe_analysis.md) - locally runnable alternative
- [Motion Capture (FreeMoCap sidecar)](motion_capture.md)
