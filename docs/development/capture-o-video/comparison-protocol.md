# Capture-O Video Comparison Protocol (COV-3)

Parent epic: [#11268](https://github.com/D-sorganization/UpstreamDrift/issues/11268) (Capture-O Video Companion).  
Governing issue: [#11271](https://github.com/D-sorganization/UpstreamDrift/issues/11271) ([COV-3] Protocol and correspondence decisions).  
Related issues: [#11272](https://github.com/D-sorganization/UpstreamDrift/issues/11272) (COV-4), [#11274](https://github.com/D-sorganization/UpstreamDrift/issues/11274) (COV-6), [#11275](https://github.com/D-sorganization/UpstreamDrift/issues/11275) (COV-7), [#11276](https://github.com/D-sorganization/UpstreamDrift/issues/11276) (COV-8), [#11277](https://github.com/D-sorganization/UpstreamDrift/issues/11277) (COV-9), [#11279](https://github.com/D-sorganization/UpstreamDrift/issues/11279) (COV-11).  
Status: Frozen Owner Decision.

---

## 1. Problem Statement & the Null Baseline Principle

The marker capture dataset (`capture-O`, 13 swings at 240 Hz) and the historical video album share the same subject (`subject-O`), day, club, and physical environment. However:

1. **Lack of Hardware Synchronization:** The optical marker system and the video cameras operated independently without timecode or genlock synchronization.
2. **Swing Identity Uncertainty:** Video recordings may capture swings executed during the session that were not among the 13 saved optical marker trials. They are substantially similar, but not necessarily identical physical trials.
3. **Uncalibrated Optics:** Camera intrinsics, position, lens distortion, and optical zoom are unrecorded.
4. **Disparate Observables:** Reflective optical markers sit on the skin and suit; 2D detectors estimate surface visual keypoints; biomechanical models infer internal joint rotation centres. None of these directly coincide.
5. **Clock and Slow-Motion Ambiguity:** Video presentation framerate may not equal real-time acquisition framerate.

### The Null Baseline Principle

A backend estimation error is only meaningful when compared against the subject's own swing-to-swing biomechanical variation. The 13 swings of `capture-O` form an empirical variation envelope. If a markerless estimator's deviation is smaller than or comparable to this inter-swing spread, an unpaired comparison cannot distinguish algorithm error from swing variation. The protocol forbids reporting false precision and uses the inter-swing envelope as the foundational null baseline.

---

## 2. Ratified Comparison Levels

| Level              | Prerequisites                                                             | Claim Allowed                                                                                                    | Example Metrics                                                                                                                             |
| :----------------- | :------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------ |
| **L0 Qualitative** | Graded video swing (A, B, or C)                                           | Visual agreement only                                                                                            | Side-by-side video playback, synchronized overlay inspection, narrative defect notes                                                        |
| **L1 Envelope**    | Virtual camera fit (COV-4); no swing pairing required                     | Video swing trajectory falls inside or outside `capture-O`'s 13-swing variation envelope projected into the view | Phase-normalized 2D/3D trajectory containment fraction (p5–p95); DTW distance relative to median inter-swing DTW spread                     |
| **L2 Paired 2D**   | Pairing confidence $\ge \tau_\text{pair}$ (0.65) and 4-event time mapping | Per-frame 2D pixel and normalized metric agreement with a specific capture trial                                 | Pixel residuals; body-height-normalized residuals per landmark and phase bin; event timing synchronization error in frames                  |
| **L3 Paired 3D**   | L2 prerequisites plus 3D source (HMR2 or Necromatcher fit)                | 3D joint-position and joint-kinematic agreement with scale fixed by marker anthropometry                         | MPJPE and PA-MPJPE (fixed subject scale); segment orientation angles; pelvis and thorax rotation at top; X-factor; kinematic sequence order |

---

## 3. Ratified Decisions and Technical Rationales

### Decision 1: Landmark Correspondence Table

- **Ratification:** All comparisons must follow the authoritative mapping defined in `src/config/cov_landmark_correspondence.json`.
- **Anatomy vs. Markers vs. Detectors:**
  - Optical markers (Capture-34) are mapped to derived joint centres via standard kinematic transforms (`gs3dx_capture_joint_centres` / OpenSim marker bodies).
  - Derived joint centres (`pelvis`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `neck`) are mapped to detector keypoints across MediaPipe-33, COCO-17, OpenPose-25, and SMPL-22.
- **Surface vs. Joint Centre Classification:**
  - Each entry explicitly documents whether the pairing represents an internal joint centre (e.g. glenohumeral, acetabulofemoral), a surface landmark (e.g. metatarsal toe markers), or a proxy cluster centroid.
- **Explicit Exclusions:**
  - `HeadFront` vs `nose`: Excluded. Forehead marker vs nasal tip introduces ~100 mm artificial spatial bias.
  - Scapular markers (`BackLeft`, `BackRight`) and humeral clusters (`LUArmHigh`, `RUArmHigh`): Excluded from 2D comparisons as detectors do not model intermediate skeletal segment markers.
  - Club markers: Excluded from 2D human pose keypoint comparisons.

### Decision 2: Keypoint Offset Calibration & Leakage Prevention

- **Ratification:** Offset fitting via `src.shared.python.pose_estimation.keypoint_offsets.estimate_keypoint_offset` is permitted only on independent, disjoint holdout frames (minimum 15 frames during static address or setup).
- **Leakage Prohibition:** Fitting offsets on evaluation frames (e.g. downswing, impact) is strictly prohibited. If holdout frames are unavailable, unadjusted joint-centre projections must be evaluated and the offset floor reported honestly.

### Decision 3: Virtual Camera Degrees of Freedom (COV-4)

- **Fitted Parameters:** Extrinsics ($R \in \text{SO}(3)$, $T \in \mathbb{R}^3$) and effective focal length $f_x = f_y = f$.
- **Fixed Parameters:** Principal point is fixed at the geometric image center $(W/2, H/2)$; radial lens distortion $k_1$ is constrained to zero during initial camera fit to prevent ill-conditioned non-linear over-parameterization.
- **Fitting Frames:** Calibrated primarily on the address pose (P1) where optical markers and detector keypoints exhibit minimal velocity and motion blur.
- **Reference Backend Holdout:** Camera parameters are fitted against landmark observations from a designated reference backend (MediaPipe Pose). To ensure benchmark fairness, the reference backend's own accuracy evaluations are evaluated with camera uncertainty propagation or against independent manual annotations.

### Decision 4: Swing Pairing Signal and $\tau_\text{pair}$ (COV-6)

- **Pairing Features:**
  1. Hand/wrist 2D path shape evaluated via Dynamic Time Warping (DTW).
  2. Swing tempo ratio (backswing duration divided by downswing duration).
  3. Top-of-backswing 2D position.
  4. Finish pose configuration.
- **Equipment Omission:** Clubhead and shaft trajectories are excluded from pairing due to severe motion blur in single-camera 30/60 fps video.
- **Decision Thresholds:**
  - $\tau_\text{pair} = 0.65$: Minimum confidence required to elevate a video swing to Level L2 or L3.
  - Abstention Margin $\epsilon = 0.05$: If the top two candidate capture swings have confidence scores within $\epsilon$, the system must abstain from claiming a specific match, logging an ambiguous pairing and holding the evaluation at Level L1.

### Decision 5: Time Mapping and Clock Qualification

- **Event Anchors:** Four primary kinematic events are detected: Address (P1), Top of Backswing (P4), Impact (P7), and Finish (P10).
- **Interpolation:** Piecewise-linear time mapping via `src.motion_capture.reference.registration.TimeMapping`.
- **Unknown Clock Policy:** When video metadata lacks explicit slow-motion container flags or confirmed physical sample rates (`physical_clock == unknown`), Level L2 and L3 analyses report errors exclusively in phase-normalized time ($0.0 \dots 1.0$). Velocity, acceleration, and jerk channels are strictly omitted.

### Decision 6: Swing Phase Stratification (P1–P10)

- **Phase Authority:** Swing evaluation is stratified across the 10 canonical golf swing phase bins (P1–P10) established in #11226:
  - P1: Address ($0.00 - 0.10$)
  - P2: Takeaway ($0.10 - 0.25$)
  - P3: Mid-Backswing ($0.25 - 0.45$)
  - P4: Top of Backswing ($0.45 - 0.55$)
  - P5: Early Downswing ($0.55 - 0.70$)
  - P6: Shaft Horizontal Down ($0.70 - 0.80$)
  - P7: Impact ($0.80 - 0.85$)
  - P8: Shaft Horizontal Follow-Through ($0.85 - 0.90$)
  - P9: Mid-Follow-Through ($0.90 - 0.95$)
  - P10: Finish ($0.95 - 1.00$)
- No secondary or ad-hoc phase detector is permitted.

### Decision 7: Frozen Comparison Profile

- **Ratification:** The metric thresholds, units, percentiles, and reporting classifications are frozen in `src/config/cov_comparison_profile.v1.json`.
- **Classification vs. Discard:** Thresholds classify observational agreement:
  - **Trustworthy:** Sufficient precision to support downstream historical-player inverse kinematics and dynamic optimization.
  - **Indicative:** Agreement demonstrates general timing and trajectory trends, but quantitative values carry high variance.
  - **Unresolvable / Outside Envelope:** Physical limitations (e.g. monocular depth ambiguity) prevent quantitative resolution from a single camera view.
- Imperfect data is graded and retained at its supported level; it is never discarded.

### Decision 8: Club Handling

- **Scope Exclusion:** Clubhead and shaft kinematics are excluded from markerless 2D per-frame residual metrics.
- **Physical Rationale:** Impact clubhead speeds exceeding 45 m/s (100 mph) produce 150–300 mm of motion blur per exposure in standard video, rendering centroid detection unreliable.
- **High-Speed Boundary:** Club comparisons are valid only when evaluating high-speed cameras ($\ge 240$ Hz) or direct impact-sensor telemetries.

---

## 4. Machine-Readable Configuration Assets

The formal specifications corresponding to this protocol are:

1. **Landmark Correspondence:** [`src/config/cov_landmark_correspondence.json`](file:///C:/Users/diete/Repositories/UpstreamDrift-worktrees/agy-cov3-11271/src/config/cov_landmark_correspondence.json)
2. **Comparison Profile:** [`src/config/cov_comparison_profile.v1.json`](file:///C:/Users/diete/Repositories/UpstreamDrift-worktrees/agy-cov3-11271/src/config/cov_comparison_profile.v1.json)
3. **Verification Suite:** [`tests/unit/motion_capture/test_cov3_protocol_and_correspondence.py`](file:///C:/Users/diete/Repositories/UpstreamDrift-worktrees/agy-cov3-11271/tests/unit/motion_capture/test_cov3_protocol_and_correspondence.py)
