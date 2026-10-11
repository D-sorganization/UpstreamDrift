# Capture-O Single-View Reconstruction Error Budget and Guidance Summary

Parent epic: #11268 (Capture-O Video Companion).
Governing issue: #11279 ([COV-11] Error-budget receipt and report).
Subject: `subject-O` (single subject, marker-anchored capture `capture-O`).
Schema Version: `error-budget/1.0.0`
Status: Owner-Approved Public Summary

## 1. Executive Summary & Guidance Classification

Single-view historical video reconstruction is evaluated against owner marker ground truth (`capture-O`).
Downstream historical-player programs (Tiger #11226, Hogan #11229, Necromatcher #11232/#11235) cite these empirical bounds.

| Quantity                   | Level | Source                  | Observed p95 | Target Bound | Classification                       | Rationale                                                                                |
| -------------------------- | ----- | ----------------------- | ------------ | ------------ | ------------------------------------ | ---------------------------------------------------------------------------------------- |
| `transverse_2d_landmarks`  | L2    | `mediapipe`             | 0.018 norm   | < 0.020 norm | **trustworthy at p95 < 0.02**        | In-plane landmark tracking agrees with marker projections within 1.8% body height        |
| `pelvis_rotation_top`      | L3    | `necromatcher_anchored` | 6.80 deg     | < 5.00 deg   | **indicative**                       | Observed p95 meets indicative bound (< 10.0 deg) but exceeds tight trustworthy threshold |
| `x_factor`                 | L3    | `necromatcher_anchored` | 8.20 deg     | < 6.00 deg   | **indicative**                       | Torso-pelvis relative angle within indicative bound (< 12.0 deg)                         |
| `kinematic_sequence_order` | L3    | `necromatcher_anchored` | 0.07 phase   | < 0.05 phase | **indicative**                       | Peak angular velocity sequencing order preserved within 7% swing phase                   |
| `hand_path_depth`          | L3    | `hmr2` / `necromatcher` | 52.40 mm     | < 40.00 mm   | **not recoverable from single view** | Monocular depth ambiguity: depth component accounts for >75% of residual variance        |

## 2. Aggregate Error Budget Table

All linear errors are reported in millimeters or body-height normalized units (`norm`), and angular errors in degrees.

| Source                  | Quantity               | Phase     | View | Grade | Level | Metric        | Unit | n Swings | p50   | p95   | Worst | Resolvable |
| ----------------------- | ---------------------- | --------- | ---- | ----- | ----- | ------------- | ---- | -------- | ----- | ----- | ----- | ---------- |
| `mediapipe`             | `lead_wrist`           | all       | dtl  | A     | L2    | residual_norm | norm | 5        | 0.008 | 0.016 | 0.024 | True       |
| `mediapipe`             | `trail_wrist`          | all       | dtl  | A     | L2    | residual_norm | norm | 5        | 0.009 | 0.018 | 0.027 | True       |
| `mediapipe`             | `lead_shoulder`        | all       | dtl  | A     | L2    | residual_norm | norm | 5        | 0.006 | 0.012 | 0.019 | True       |
| `mediapipe`             | `trail_shoulder`       | all       | dtl  | A     | L2    | residual_norm | norm | 5        | 0.007 | 0.014 | 0.021 | True       |
| `mediapipe`             | `lead_hip`             | all       | dtl  | A     | L2    | residual_norm | norm | 5        | 0.005 | 0.011 | 0.016 | True       |
| `mediapipe`             | `trail_hip`            | all       | dtl  | A     | L2    | residual_norm | norm | 5        | 0.006 | 0.012 | 0.017 | True       |
| `hmr2`                  | `full_body_mpjpe`      | all       | dtl  | A     | L3    | mpjpe         | mm   | 5        | 28.50 | 54.20 | 72.00 | True       |
| `hmr2`                  | `full_body_pa_mpjpe`   | all       | dtl  | A     | L3    | pa_mpjpe      | mm   | 5        | 18.20 | 34.60 | 48.00 | True       |
| `hmr2`                  | `depth_axis_residual`  | all       | dtl  | A     | L3    | depth_error   | mm   | 5        | 24.10 | 48.30 | 66.50 | True       |
| `hmr2`                  | `image_plane_residual` | all       | dtl  | A     | L3    | image_plane   | mm   | 5        | 11.20 | 21.40 | 31.00 | True       |
| `necromatcher_anchored` | `pelvis_rotation_top`  | top       | dtl  | A     | L3    | angle_error   | deg  | 5        | 3.40  | 6.80  | 9.50  | True       |
| `necromatcher_anchored` | `x_factor`             | top       | dtl  | A     | L3    | angle_error   | deg  | 5        | 4.10  | 8.20  | 11.00 | True       |
| `necromatcher_anchored` | `hand_path_depth`      | downswing | dtl  | A     | L3    | depth_error   | mm   | 5        | 29.00 | 52.40 | 74.00 | False      |

## 3. Omitted & Unmeasured Channels

| Source      | Quantity           | Metric             | Reason                                                                          |
| ----------- | ------------------ | ------------------ | ------------------------------------------------------------------------------- |
| `mediapipe` | `club_head`        | residual           | High velocity impact blur and occlusion in Grade B/C clips                      |
| `hmr2`      | `velocity_metrics` | angular_velocity   | Physical clock unknown without container slow-motion metadata                   |
| `all`       | `unpaired_swings`  | per_frame_residual | Excluded from L2/L3 by pairing confidence margin threshold $\tau_{\text{pair}}$ |

## 4. Governed Limitations

- **Single Subject:** Measured exclusively on `subject-O`. No population generalization claim is made.
- **Single Camera View:** Monocular depth ambiguity remains unobservable along the camera optical axis without multi-view geometry or dynamic constraints.
- **Unpaired Swings:** Swings without confident marker pairing are held at Level L0/L1 (envelope only) and strictly excluded from per-frame L2/L3 metrics.
- **Clock Qualification:** Streams with unknown frame clocks report phase-normalized time and omit velocity and acceleration metrics.
- **Landmark Conventions:** Marker capture uses skin-surface reflective markers; visual estimators infer internal joint centres, introducing an offset floor.
- **Marker Occlusion:** Known occlusions during high-acceleration swing phases (e.g. lead arm covering chest markers) limit ground-truth visibility.
