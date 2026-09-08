# Markerless Mocap Acceptance Program

Version: 1.0.0

Issues: #9063, #9065; Tools #4706

This program separates software conformance, camera characterization, lab
qualification, and biomechanical use. Passing one level never implies another.

## Qualification Levels

| Level        | Meaning                                                                                                             |
| ------------ | ------------------------------------------------------------------------------------------------------------------- |
| Contract     | Schemas, lifecycle, errors, serialization, units, coordinates, time, and provenance pass deterministic tests.       |
| Algorithm    | A named algorithm/version meets registered synthetic and public benchmark tolerances.                               |
| Camera       | A device/mode/lens/trigger combination meets measured image and timing bounds.                                      |
| Layout       | A versioned camera arrangement meets useful-view, residual, uncertainty, and coverage bounds in its capture volume. |
| Workflow     | Capture through review, reconstruction, export, import, persistence, cancellation, and recovery passes.             |
| Physical Lab | Governed calibration and participant trials meet the approved protocol and acceptance thresholds.                   |

Synthetic evidence may qualify Contract and part of Algorithm. Synthetic evidence
does not qualify a camera, layout, physical lab, participant inference, injury
claim, or coaching recommendation.

## Release Outcomes

- **Supported:** all required evidence and declared bounds pass.
- **Degraded:** usable only inside visible narrower bounds with an actionable
  reason and unavailable claims suppressed.
- **Blocked:** required approval, authentication, hardware, calibration,
  consent, evidence, or dependency is absent.
- **Unavailable:** the system cannot produce the quantity without inventing
  authority.

Failures return typed outcomes. The UI does not replace them with zeros, flat
traces, cached results from another setup, or synthetic observations.

## Camera-Agnostic Acceptance

A provider must declare stable identity, connection transport, sensor modes,
pixel formats, shutter behavior, exposure range, timestamp source, trigger
support, and unsupported or degraded reasons. The application negotiates a
mode and records the effective result.

No fixed camera count is sufficient by itself. For every time sample and
required keypoint, acceptance evaluates:

- number and geometry of calibrated useful views;
- occlusion and image-boundary status;
- timing offset and uncertainty relative to the session clock;
- reprojection residual and outlier decisions;
- 3-D covariance or another versioned uncertainty measure; and
- provenance for every contributing and rejected camera.

Arbitrary placement is supported by learning and versioning the
`T_world_from_camera` transforms. Moving a camera invalidates or degrades the
prior layout until change detection and recalibration pass.

## Synthetic Algorithm Thresholds

The Algorithm level is gated in CI on the synthetic harness
(`motion_capture.reconstruct`, #9629): a rigid 15-joint skeleton on a
swing-like motion rendered through three cameras at 1920x1200 with 1 px
noise, 2 % occlusion and, where stated, gross outliers of up to 120 px. Every
threshold below is asserted by a unit test; the numbers in parentheses are
what the implementation measured on 2026-09-07.

| Quantity                                    | Threshold                                                | Test                 |
| ------------------------------------------- | -------------------------------------------------------- | -------------------- |
| Camera rotation from a 3 deg / 15 cm start  | < 0.5 deg (0.07)                                         | `test_bundle.py`     |
| Camera position from the same start         | < 3 cm (0.4-2.0)                                         | `test_bundle.py`     |
| Bone lengths, subject 6 % taller than prior | < 2 % (1.1)                                              | `test_bundle.py`     |
| Joint position, median                      | < 2 cm (1.4)                                             | `test_bundle.py`     |
| Gross outliers flagged, 3 % injected        | recall >= 0.9, precision >= 0.85                         | `test_bundle.py`     |
| Per-view cleaning, 3 % injected             | recall >= 0.9, precision >= 0.85 (0.94-0.96 / 0.92-0.96) | `test_clean.py`      |
| One-frame spike, smoother                   | rejected; peak within 2 %                                | `test_temporal.py`   |
| Joint-only initialisation, then fit         | axes < 0.5 deg, baselines < 2 %                          | `test_initialize.py` |
| Triangulation, three views, one wrong       | wrong view named                                         | `test_geometry.py`   |
| Turn angles from fitted joints              | within 0.6 deg of the motion                             | `test_analytics.py`  |

Two views cannot assign blame (an error along the epipolar line is absorbed
by depth); the harness states this and it is why a third useful view is
required above. Scale is not observable from images with one camera fixed:
one measured length on the subject is treated as known.

## Physical Lab Hold Points

The following block physical-lab qualification:

1. Shop dimensions, capture volume, subject distance, intended movements,
   lighting, lenses, network topology, storage, and safety boundaries are not
   approved.
2. Camera/lens/mode measurements for blur, exposure, rolling-shutter behavior,
   thermal stability, dropped frames, timestamp skew, and trigger response are
   absent.
3. Intrinsic, extrinsic, time-sync, and change-detection validation does not
   pass its versioned thresholds.
4. Consent, raw video retention, access control, deletion, and no-store policy
   are not configured and tested.
5. Held-out governed movements do not meet point, segment, event, and C3D
   round-trip error/uncertainty requirements.

Procurement remains provisional until these inputs are supplied and a recorded
evaluation approves the camera/lens/network/storage bill of materials.

## UI and Accessibility Acceptance

PyQt6 is the canonical desktop behavior and React/Tauri must have registered
parity. Both surfaces must:

- use UpstreamDrift theme tokens and unit preferences;
- display units beside every measurement and never infer an unstated unit;
- provide hints and hover help for every non-obvious control and status;
- remain keyboard navigable with visible focus and accessible names;
- show supported, degraded, blocked, and unavailable states without relying on
  color alone;
- surface calibration age, camera identity, useful views, dropped frames,
  residuals, uncertainty, model/license provenance, and retention status; and
- provide bounded cancellation, cleanup, recovery, and actionable errors.

Responsive desktop and mobile rendering must be visually inspected. Headless
component tests alone do not qualify UX.

## C3D Compatibility Acceptance

The single canonical export path must round-trip through the characterized
reader without silent change to:

- point labels, point units, frame rate, frame count, and time origin;
- axes/handedness and the recorded source-to-world transform;
- residual, invalid, occluded, and interpolated point semantics;
- events and parameters required by the downstream motion pipeline; and
- analog channel sample ratio, units, labels, and alignment when present.

Golden files cover zero/one/many cameras, mixed models, missing frames,
occlusion, Unicode labels, long sessions, and unsupported analog layouts. C3D
compatibility does not by itself validate pose inference or biomechanics.

## Release Evidence

Every implementation slice records exact source and dependency SHAs, schema
and method versions, deterministic fixtures, focused and full gates, clean
exact-HEAD portability, protected checks, merge SHA, and post-merge behavior.
Handoffs name remaining limitations and the next dependency. An issue closes
only through its merged implementation evidence or an explicitly reviewed
roadmap/duplicate/invalid disposition.
