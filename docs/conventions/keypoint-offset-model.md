# Keypoint Offset Observation Model

> Status: Initial CC-15 contract for calibration-time detector keypoint offsets.
> Scope: engine-agnostic observation calibration consumed by later residual
> assembly.

Detector keypoints and model joint centers are related but not identical. A
video detector usually reports a visible anatomical landmark or surface proxy;
a dynamics model usually exposes a joint center. The CC-15 observation model
represents that systematic displacement as a per-keypoint offset expressed in
the local segment frame.

For a calibrated site:

- `keypoint_name`: detector label, such as `right_hip`.
- `canonical_site`: canonical anatomical site / joint-center identifier.
- `segment_name`: segment whose frame expresses the offset.
- `joint_center_name`: optional model-output key when it differs from the
  detector keypoint label.

## Calibration

The calibration clip must provide aligned per-frame arrays:

- observed 3D keypoints in metres, world frame;
- model joint centers in metres, world frame;
- segment rotations `R_ws`, mapping segment-frame vectors into world frame;
- optional detector confidences in `[0, 1]`.

For each retained frame:

```text
offset_segment_i = R_ws_i.T * (keypoint_world_i - joint_center_world_i)
```

The calibrated offset is the confidence-weighted mean of
`offset_segment_i`. Frames below `min_confidence` are excluded. The uncertainty
fields are reported in the same segment-frame basis:

- `covariance_m2`: weighted 3x3 covariance of retained per-frame offsets.
- `standard_error_m`: square root of the covariance diagonal divided by the
  effective sample count.
- `rms_residual_m`: confidence-weighted world-frame residual after fitting.
- `sample_count` and `mean_confidence`: retained calibration support.

## Prediction And Residuals

At residual time, the observation prediction is:

```text
predicted_keypoint_world = joint_center_world + R_ws * offset_segment
residual_world = observed_keypoint_world - predicted_keypoint_world
```

The `KeypointOffsetModel.residuals_for_clip()` helper returns one `(frames, 3)`
array per calibrated keypoint. This keeps the CC-18 residual builder
engine-agnostic: FK can produce joint centers and segment rotations from any
backend, while the calibrated observation model supplies the detector bias
correction and uncertainty metadata.

## Validation Rules

- All positions are finite metre values with shape `(N, 3)`.
- All segment rotations are finite proper 3x3 rotation matrices.
- At least `min_samples` frames must survive confidence filtering.
- Confidence values must be finite and within `[0, 1]`.
- Offsets are keyed by detector keypoint name, while `joint_center_name` allows
  the model output map to use canonical joint-center labels.
