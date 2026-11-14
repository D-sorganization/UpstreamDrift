# Best Match Candidate Refit: MuJoCo Driver (A7 Grip + A8 Trunk)

SHOW-A9 (#14734), part of epic #14549 (SHOW-A). Successor to SHOW-A6 (#14555).
Evaluated against 11-04 baseline (`fa7a9b7075`) using driver capture (`capture-A`).

## Overview & Registry Decision
- **Model id:** `mujoco_driver_candidate_a9` (`mujoco`)
- **Spec:** `docs/development/full_body_models/full_body_spec_anthro_driver_v2.json`
- **Verdict:** `REJECTED (G3)` (closure residual 76.8 mm > 5.0 mm; weight fraction 0.00-2.82 BW).
- **Registry Action:** Recorded as candidate. Best-match registry entry `mujoco_driver` remains unchanged per Acceptance Criterion 3 (candidate refit did not beat baseline on marker RMS and closure).

## Acceptance Comparison vs 11-04 Baseline (`fa7a9b7075`)

| Metric | 11-04 Baseline (`fa7a9b7075`) | A9 Candidate Refit | Delta | G3 Horizon | Status |
|---|---|---|---|---|---|
| Marker RMS | 50.9 mm | 55.3 mm | +4.4 mm | <= 60.0 mm | PASSED |
| Joint RMS | 2.4 deg | 2.4 deg | 0.0 deg | n/a (vs own IK) | PASSED |
| Closure Residual | 62.2 mm | 76.8 mm | +14.6 mm | <= 5.0 mm (#13109) | FAILED |
| Weight Fraction | 0.04–2.82 BW | 0.00–2.82 BW | min -0.04 BW | [0.20, 3.00] BW | FAILED |
| Peak Joint Torque | 537 N·m | 537 N·m | 0 N·m | <= 200 N·m | EXCEEDED |
| In-Support Fraction | 0.637 | 0.637 | 0.000 | >= 0.850 | FAILED |

## Grip Convention & the A6 Lesson
- **Address Grip Roll:** 15.0° (lead hand holds target 15° ± 1° on A7 clock; trail at -100.73°).
- **Modelling Choice:** Capture carries no hand markers; grip is a modelling choice, not a fitted output.
- **A6 Lesson:** With bounded wrist ranges (`coordinate_ranges_deg`), the neutral grip cannot close on the club without wrist overruns; the fit does not absorb wrist error into grip roll.
- **Wrist Violations:** `RWInputX` (408 frames outside [-40, 25]), `RWInputY` (654 frames outside [-70, 70]), `LWInputX` (239 frames outside [-40, 25]), `LWInputY` (192 frames outside [-70, 70]), `RScapInputX` (576 frames outside [-10, 30]), `RScapInputY` (281 frames outside [-40, 40]).

## Pinned Hashes
- `v1_spec_sha256`: `fd72d8be7a6c5c557a10886a7d2717a4eb4619805a24cc1be644215eea21dcb3`
- `v2_spec_sha256`: `323c2182ed1592552de700c02ab0f297e620660f2ac5a9c7f0f01e6af8f25b8c`
- `capture_sha256`: `cbcb4d84aca0a1f558073f0c2328c3f501c4726aaf699e382cf20c5861a0971d`
- `baseline_commit`: `fa7a9b7075`
