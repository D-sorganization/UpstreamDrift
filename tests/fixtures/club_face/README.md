# Club Face Fixtures

This directory contains standardized swing fixtures and model specifications for kinematic tracking, club face orientation calibration, and range-of-motion verification across driver and iron-7 swing models.

## Fixture Inventory

- `swing_q_driver.npz`: 908-frame joint trajectory for the driver swing (A2 grip configuration, 1 kHz sample rate downsampled / resampled per pipeline specification).
- `swing_q_iron7.npz`: Joint trajectory for the iron-7 swing model.
- `tracked_q_iron7.npz`: Ground-truth tracked marker trajectory for iron-7 swings.
- `spec_driver.json` / `spec_iron7.json`: Kinematic specifications corresponding to each club model.
- `provenance.json`: Traceable generation receipts, commit hashes, marker offsets, and club face orientation errors.

## SHOW-A18 Range-of-Motion Diagnostic (`swing_q_driver.npz`)

Under `HUMAN_RANGES_DEG` (`src/shared/python/motion_matching/range_of_motion.py`), `swing_q_driver.npz` exhibits excursions beyond anatomical limits resulting from the A2 copied grip offset:

| Coordinate                                   | Human Range (`HUMAN_RANGES_DEG`) | Violation Frames | Max Excess | Fraction |
| -------------------------------------------- | -------------------------------- | ---------------- | ---------- | -------- |
| **RWInputY** (trail wrist flexion/extension) | `[-70.0, 70.0]°`                 | **691 / 908**    | **60.18°** | 76.10%   |
| **LWInputY** (lead wrist flexion/extension)  | `[-70.0, 70.0]°`                 | **216 / 908**    | **22.52°** | 23.79%   |
| **LWInputX** (lead wrist radial/ulnar dev.)  | `[-40.0, 25.0]°`                 | **130 / 908**    | **22.81°** | 14.32%   |
| **RWInputX** (trail wrist radial/ulnar dev.) | `[-40.0, 25.0]°`                 | **66 / 908**     | **37.24°** | 7.27%    |
| **RScapInputX** (trail scapula elevation)    | `[-10.0, 30.0]°`                 | **378 / 908**    | **27.36°** | 41.63%   |
| **subtalar_angle_l**                         | `[-20.0, 20.0]°`                 | **122 / 908**    | **5.66°**  | 13.44%   |
| **mtp_angle_l**                              | `[-30.0, 30.0]°`                 | **47 / 908**     | **32.65°** | 5.18%    |

### Cause Decomposition

1. **Underlying Capture Trajectory**: The raw capture-fitted trajectory (`1b514e12f~1`, recorded in `docs/development/full_body_models/full_body_spec_anthro_driver_capture.json`) had **0 / 908** frames out of range for trail wrist coordinates (`RWInputY` in `[-61.92, 63.85]°`, `RWInputX` in `[-29.06, 23.76]°`).
2. **A2 Static Grip Offset**: SHOW-A2 (#14551, PR #14644) assigned the lead-hand Euler triple `[-75.3894, -9.7161, 4.5795]` directly to both hands (`L == R`), collapsing the dorsal follower separation to 0.00°. When the kinematic solver maintained closure and club path, it forced the trail wrist into severe unanatomical hyperextension (reaching +130.18°).
3. **G2 Candidate Resolution**: Option (b15) (#14751, PR #14819) applies an explicit trail alpha of 15.0° (dorsal separation 115.89°), reducing `RWInputY` violation frames from 691 to 420 (-39.2% / -271 frames) and clearing `LWInputY` (216 -> 0 frames). Option (c) is unreachable at address inside current limits (26.7 mm / 5.1° residual; #14795).
4. **Ranges Preserved**: In accordance with SHOW-A18 acceptance criteria, `HUMAN_RANGES_DEG` is **not widened**.
