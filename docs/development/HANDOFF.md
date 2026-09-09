# Reference Scene Registration Implementation Handoff

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: C:/Users/diete/Repositories/UpstreamDrift
- Branch: feat/9865-scene-registration
- Baseline commit: 256a0b8c0 (expert reference imports #9870 merged)
- Implementation commit: SELF
- Governing issue/epic: #9865, #9863

## Objective and Status

Calibrated reference scene registration and event synchronization.
Rigid/similarity transform, body-size normalization, coordinate convention conversion
from canonical Z-up to ADR-0041 scene world, event-anchor and offset time warping,
missing-joint gap mask preservation across interpolation, camera projection with distortion
and clipping, and 2D video homography without 3D claims are implemented.
Focused qualification passes.

## Files and Decisions

- `src/motion_capture/reference/registration.py`: ReferenceTransform, EventAnchors,
  TimeMapping, ReferenceRegistration, transform_reference_motion, sample_reference_motion,
  and project_reference_to_camera.
- `src/motion_capture/reconstruct/overlay3d.py`: Extended with reference_track.
- Missing joints are never interpolated across gaps; if either bounding frame is None,
  the interpolated point is masked as invalid (valid_mask=False).
- Distortions use `src.shared.python.estimation.residuals.project_pinhole`.
- Behind-camera and out-of-frame points are clipped with visible=False.
- 2D video reference uses explicit 3x3 homography and declares no 3D claim.

## Validation

- 7 focused tests in `tests/motion_capture/test_reference_registration.py` pass.
- Round-trip JSON serialization yields identical points and pixels.
- Ruff linting passes on all modified and new files.
- File size budget passes cleanly.

## Next Steps

1. Submit PR referencing #9865 with auto-merge enabled.
2. Advance to Subepic #9866 (reference comparison, synchronized projection UI & export).
