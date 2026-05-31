# Canonical Observations Schema

`CanonicalObservations` is the markerless-ingestion input contract for
physics-matched estimators. It lives in
`src/shared/python/pose_estimation/observations.py`.

The schema preserves:

- detector layout name and keypoint order
- one calibration record per camera: image size, 3 x 3 intrinsics,
  distortion coefficients, and camera-to-world extrinsics
- per-camera 2D keypoints in pixels
- per-keypoint confidence in `[0, 1]`
- optional triangulated 3D keypoints in metres with optional confidence
- scalar/string provenance metadata

The 2D camera observations remain first-class even when 3D points are
available. Estimators should weight residuals from the original camera
observations instead of collapsing early to a single skeleton.

## JSON Layout

```json
{
  "schema_version": "1.0.0",
  "detector_layout": {
    "name": "coco-3",
    "keypoint_names": ["pelvis", "neck", "head"]
  },
  "cameras": [
    {
      "camera_id": "front",
      "image_size_px": [1280, 720],
      "intrinsics": {
        "matrix": [
          [900.0, 0.0, 640.0],
          [0.0, 900.0, 360.0],
          [0.0, 0.0, 1.0]
        ],
        "distortion": [0.01, -0.02, 0.0, 0.0, 0.0]
      },
      "extrinsics": {
        "rotation_world_from_camera": [
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]
        ],
        "translation_world_from_camera_m": [0.0, -2.0, 1.2]
      }
    }
  ],
  "frames": [
    {
      "camera_id": "front",
      "time_s": 0.0,
      "keypoints_px": [
        [610.0, 540.0],
        [620.0, 400.0],
        [625.0, 310.0]
      ],
      "confidence": [0.96, 0.88, 0.82]
    }
  ],
  "keypoints_3d_m": [
    [
      [0.0, 0.0, 0.95],
      [0.0, 0.0, 1.25],
      [0.0, 0.0, 1.48]
    ]
  ],
  "keypoints_3d_confidence": [[0.92, 0.86, 0.81]],
  "provenance": {
    "source_format": "synthetic-json",
    "detector": "test-fixture"
  }
}
```

## Validation

Construction and loading validate the contract:

- camera IDs are unique, non-empty strings
- each frame references a declared camera
- every frame keypoint count matches `detector_layout.keypoint_names`
- confidence arrays match their keypoint arrays and are bounded to `[0, 1]`
- intrinsics are finite 3 x 3 matrices with positive focal lengths
- extrinsics rotation matrices are orthonormal with determinant `+1`
- optional 3D arrays have shape `(T, K, 3)` and share the detector keypoint count

## Fixture

A minimal multi-camera fixture is available at
`tests/fixtures/pose_estimation/canonical_observations_multi_camera.json`.

## CC-4 Results Writer Bridge

`CanonicalObservations.to_trace()` returns a CC-4 `Trace` envelope. Optional
triangulated 3D keypoints map to `Trace.markers`, while the full camera 2D,
confidence, calibration, and provenance payload is stored in the scalar
`meta_canonical_observations_json` attribute used by
`simulation_backends.trace_io.write_trace`.

Use `CanonicalObservations.from_trace(read_trace(path))` to recover the full
observation schema after an HDF5 write/read round-trip.
