# Measurement Uncertainty and Justified Model Floors (MMR-02 #11086)

## 1. Overview and Problem Statement

Optical motion capture (C3D format at 360 Hz for Driver and 359 Hz for 7-Iron) provides retro-reflective marker positions in space. When comparing dynamic multibody simulations (Simscape GS3DX, Pink IK, MuJoCo, OpenSim) against measured tour trajectories, setting arbitrary zero-target optimization thresholds leads to severe overfitting of non-skeletal motion artifacts.

This document establishes the **authoritative measurement uncertainty budget** and **justified model floors** across dual-club (Driver and 7-Iron) observation contracts under MMR-02 (#11086).

---

## 2. Spatial Measurement Uncertainty Budget

Optical capture data is subject to four distinct layers of physical uncertainty:

| Uncertainty Component                   | Physical Source                                                                | Magnitude (RMS / 1-sigma) | Impact on Dynamic Fitting                                   |
| --------------------------------------- | ------------------------------------------------------------------------------ | ------------------------- | ----------------------------------------------------------- |
| **Instrument / Camera Calibration**     | Vicon wand residual calibration, lens distortion, optical ray intersection     | 0.5 – 1.5 mm              | Negligible relative to biomechanical scale                  |
| **Soft Tissue Artifact (STA)**          | Skin displacement over bone, muscle contraction, impact shockwave              | 5.0 – 15.0 mm             | Dominates body marker residuals (shoulders, pelvis, thorax) |
| **Club Marker Flutter / Shaft Bending** | Dynamic shaft deflection, clubhead vibration, high angular acceleration        | 2.0 – 4.0 mm              | Requires elastic shaft model or cluster rigidization        |
| **Joint Center Proxy Invariance**       | Synthetic regression offsets (e.g., hip/shoulder centers from surface markers) | 10.0 – 25.0 mm            | Offsets must be calibrated in static/address, never holdout |

### Key Principle: Anti-Overfitting Threshold

Attempts to optimize a rigid multibody human model to residuals below **15 mm (0.015 m)** inevitably overfit soft tissue deformation rather than underlying skeletal kinematics.

---

## 3. Justified Model Floors

A model floor is the minimum physically achievable residual for a given kinematic and dynamic topology given measurement uncertainty:

### 3.1. Global and Segment Model Floors

| Segment / Target                   | Justified Model Floor (Pooled RMSE) | Justification                                                      |
| ---------------------------------- | ----------------------------------- | ------------------------------------------------------------------ |
| **Full Body (Skeletal)**           | 20.0 mm (0.020 m)                   | Dominated by STA (pelvis, trunk, shoulders) during downswing       |
| **Shaft Cluster**                  | 12.0 mm (0.012 m)                   | Shaft lead/deflection curvature during uncocking                   |
| **Clubhead Cluster**               | 10.0 mm (0.010 m)                   | Rigid cluster tracking with high linear velocity (>45 m/s)         |
| **Overall Full-Swing Model Floor** | **18.5 mm (0.0185 m)**              | Combined justified baseline floor for qualified full-body matching |

### 3.2. Phase-Specific Model Floors

Biomechanical dynamics vary dramatically across swing phases. The model floor scales accordingly:

| Phase                  | Frame Interval (Driver) | Model Floor (RMSE) | Primary Physical Constraint                              |
| ---------------------- | ----------------------- | ------------------ | -------------------------------------------------------- |
| **Address / Takeaway** | 0 – 180                 | 10.0 mm (0.010 m)  | Quasi-static, minimal STA, negligible shaft deflection   |
| **Backswing**          | 180 – 397               | 15.0 mm (0.015 m)  | Moderate joint velocity, elastic torso loading           |
| **Downswing**          | 397 – 472               | 20.0 mm (0.020 m)  | High centripetal acceleration, rapid uncocking, peak STA |
| **Impact Window**      | 465 – 475               | 25.0 mm (0.025 m)  | Maximum clubhead velocity (>45 m/s), impact shockwave    |
| **Follow-Through**     | 472 – 654               | 22.0 mm (0.022 m)  | Deceleration, torso wrap-around, marker occlusion risk   |

---

## 4. Evaluation and Holdout Protection Contracts

Under MMR-02 (#11086) and MMR-02-I (#11105):

1. **Disjoint Calibration vs. Holdout:**

   - Address and backswing frames (frames 0 to Top-of-Backswing, frame 397 driver / 394 iron) may be used for fixed geometry calibration (arm length, club length, sensor offsets).
   - Downswing, impact, and follow-through frames are strictly held out.
   - Holdout observations must **never** update calibrated geometry or offset parameters.

2. **No Gap-Filled / Interpolated Evidence:**

   - Any gap-filled or synthetic observation must be tracked in `interpolated_spans` or marked `is_interpolated=True`.
   - Measured holdout scoring (`measured_only=True`) excludes all interpolated or synthetic samples by construction (`measured_frame_validity()`).
   - Reconstructed frame validity verifies caller evaluation masks bit-for-bit.

3. **Common-Target Comparison with Visible Coverage:**

   - When evaluating competing candidate models or engines, comparisons must be computed on the exact common intersection of valid observations (`compute_common_target_comparison`).
   - Coverage differences (`coverage_ratio_a`, `coverage_ratio_b`, `exclusive_valid_count`) must be reported explicitly alongside error metrics, preventing false superiority claims based on cherry-picked marker subsets.

4. **Multi-Metric Distinctness:**
   - Pooled Euclidean RMSE, median frame RMS, mean frame RMS, 95th percentile error (p95), and maximum error must be computed and reported distinctly.
   - Median frame RMS must never be substituted for pooled RMSE when observation counts vary across frames.
