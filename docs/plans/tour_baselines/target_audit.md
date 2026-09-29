# Tour Baselines Target Audit, Marker Semantics, Events, and Provenance

Part of the Matched Swing Program (#10363, #10584, #10586).

## Executive Summary

This document establishes the verified baseline data contracts for the two canonical tour-average golf swing captures:

1. **Driver Swing (`C3D_TA_Driver.c3d`)**: 654 frames at 360.0 Hz (1.8139 s), right-handed golfer.
2. **7-Iron Swing (`C3D_TA_Iron.c3d`)**: 657 frames at 359.0 Hz (1.8273 s), right-handed golfer.

Every downstream kinematic reference and dynamic fitting lane consumes these canonical targets directly. Duplicate copies are verified strictly by content SHA-256, never path name alone.

---

## 1. Capture Identity and Hash Contracts

| Attribute                 | Driver Swing (`C3D_TA_Driver.c3d`)                                 | 7-Iron Swing (`C3D_TA_Iron.c3d`)                                   | Validation Contract       |
| :------------------------ | :----------------------------------------------------------------- | :----------------------------------------------------------------- | :------------------------ |
| **SHA-256**               | `545405ccdbae87a297d16951487b501d5d76f5a2ab253cfc6d797744184943ba` | `395deb1f91006586819020fc85180409e716f07e1c680f9fb2ca114759f80845` | Exact match required      |
| **Frame Count**           | 654 frames                                                         | 657 frames                                                         | Exact match required      |
| **Sample Rate**           | 360.0 Hz                                                           | 359.0 Hz                                                           | Native clock isolated     |
| **Duration**              | 1.813889 s                                                         | 1.827298 s                                                         | $\frac{N-1}{f_s}$         |
| **Units**                 | `m` (metres)                                                       | `m` (metres)                                                       | Metres                    |
| **Vertical Axis**         | `y` (Y-up)                                                         | `y` (Y-up)                                                         | Y-up right-handed         |
| **Golfer Handedness**     | Right-handed                                                       | Right-handed                                                       | Declared                  |
| **Total Marker Channels** | 38 channels                                                        | 38 channels                                                        | Exact labels match        |
| **Valid Samples**         | 24,135 / 24,852 (97.11%)                                           | 24,219 / 24,966 (97.01%)                                           | Residual $\ge 0$ & finite |
| **Missing Samples**       | 717 samples                                                        | 747 samples                                                        | Explicitly masked         |

---

## 2. Marker Measurement Semantics (`tour-measurement-map/1.0.0`)

The measurement map classifies each channel into five rigorous categories:

- **`observed_surface`**: Directly observed retroreflective markers on the subject's skin/suit (28 markers).
- **`observed_cluster_centroid`**: Rigid-body triads on the shaft (`Marker_2:2:1..3`) and head (`Marker_3:3:1..3`).
- **`inferred_joint_center`**: Anatomical centers (e.g. `mid_hip`, `neck`) computed from surface proxies with explicit offset assumptions.
- **`calibrated_club_point`**: **Explicitly UNAVAILABLE in raw capture**. Neither capture contains physical clubface normal, groove coordinates, or impact contact sensors.
- **`unassigned_or_sentinel`**: Stuck coordinates (`Marker_0:0:0`), unlabeled channels (`Uname*36`, `Uname*37`), and capture-specific channels (`Uname*38` in Driver, `pelvis` in Iron).

### Driver vs. 7-Iron Label Differences

1. **Channel 38**: Driver has `Uname*38`; Iron has `pelvis`.
2. **Missing Spans**:
   - `RShoulderTop` is occluded during backswing/early downswing in both swings:
     - Driver: frames 0–525 (526 missing samples).
     - Iron: frames 0–561 (562 missing samples).
   - `LShoulderTop`: 100% complete in Driver; occluded in Iron at frames 0–109 (110 missing samples).
   - Grip Cluster (`Marker_2:2:*`): 36 missing in Driver (frames 444, 518–550, 652–653); 17 missing in Iron (frames 544–560).
   - Head Cluster (`Marker_3:3:*`): 21 missing in Driver (frames 545–555, 557–563, 565, 652–653); 0 missing in Iron (100% complete across all 657 frames).

---

## 3. Biomechanical Swing Events and Phase Intervals

All timing events are computed on their respective native clocks without cross-clock distortion.

| Swing Event          | Driver (360.0 Hz)   | 7-Iron (359.0 Hz)   | Detection Method        | Inferred Status   | Confidence |
| :------------------- | :------------------ | :------------------ | :---------------------- | :---------------- | :--------- |
| **Address**          | Frame 0 (0.000 s)   | Frame 0 (0.000 s)   | `capture_boundary`      | Measured boundary | 1.00       |
| **Takeaway**         | Frame 71 (0.197 s)  | Frame 70 (0.195 s)  | `trajectory_speed_peak` | Inferred          | 0.85       |
| **Top of Backswing** | Frame 397 (1.103 s) | Frame 394 (1.097 s) | `trajectory_min_speed`  | Inferred          | 0.90       |
| **Impact**           | Frame 476 (1.322 s) | Frame 480 (1.337 s) | `trajectory_speed_peak` | Inferred          | 0.95       |
| **Finish**           | Frame 653 (1.814 s) | Frame 656 (1.827 s) | `capture_boundary`      | Measured boundary | 1.00       |

### Phase Windows

- **Address**: $[0.000, 0.197]\text{ s}$ (Driver) vs $[0.000, 0.195]\text{ s}$ (Iron)
- **Backswing**: $[0.197, 1.103]\text{ s}$ (Driver) vs $[0.195, 1.097]\text{ s}$ (Iron)
- **Downswing**: $[1.103, 1.322]\text{ s}$ (Driver) vs $[1.097, 1.337]\text{ s}$ (Iron)
- **Impact Window**: $[1.302, 1.342]\text{ s}$ (Driver) vs $[1.317, 1.357]\text{ s}$ (Iron)
- **Follow-through**: $[1.322, 1.814]\text{ s}$ (Driver) vs $[1.337, 1.827]\text{ s}$ (Iron)

---

## 4. Provenance and Subject Anatomy

- **Manufacturer & System**: GearsSports v3 optical motion capture system (`Gears`).
- **Player ID**: `967eac5b-2e78-4207-a99f-d57437296d70`.
  - **Shared Anatomy Principle**: Both swings belong to the same recorded human subject. Anatomical segment lengths, joint limits, and body mass distribution are shared across driver and iron models.
  - **Capture-Specific Geometry**: Club length, clubhead mass, and inertia properties differ between driver and iron.
- **Capture Timestamps**:
  - Driver: Created `2018-04-23T10:04:56Z`, exported `2020-12-12T08:31:05Z` (Capture ID: `22196b66-8e76-41cd-8815-edec0a74312e`).
  - Iron: Created `2018-04-23T10:04:35Z`, exported `2020-12-12T08:35:13Z` (Capture ID: `a805587c-e51b-4cbd-bf87-7a18014e2286`).
- **Averaging & Rights**:
  - `averaging_method`: `"unresolved"` (documented as tour average; exact filtering/averaging pipeline not cited).
  - `usage_rights`: `"internal_fleet_reference_unrestricted_in_workspace"`.
  - `citation_policy`: `"explicit_unresolved_no_fabricated_citation"`.

---

## 5. Acceptance Criteria Verification

- [x] Content-based duplicate file verification.
- [x] Per-marker missing spans and observed coverage masks.
- [x] Versioned measurement maps distinguishing surface markers from cluster centroids.
- [x] Calibrated clubface and impact points explicitly classified as unavailable in raw data.
- [x] Native clocks (360.0 Hz vs 359.0 Hz) isolated; impact events labeled inferred.
- [x] Shared player anatomy separated from capture-specific club geometry.
- [x] Two reproducible audit receipts committed under `docs/plans/tour_baselines/evidence/`.
