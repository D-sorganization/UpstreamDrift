# Canonical Ground-Support Reference Runs and Bisect Analysis

**Author:** Dieter Olson (`agent:local`)  
**Date:** 2026-09-17 (canonical designation revised 2026-09-27, #11044)  
**Issue:** MS-03 (#10324), Epic #10363 (Matched Swing Program)  
**Related Issues:** #10108 (HO-8), #10250 (HO-11), #10258, #10271, #10322 (MS-01), #10374 (MS-100)

---

## 1. Overview and Executive Summary

In `docs/development/full_body_models/HANDOFF.md` and related program documentation, the headline kinematic and dynamic accuracy metrics for the MuJoCo full-body model have been quoted as:

- **Tour Driver:** Address calibrated marker RMS 5.1 mm, full-swing IK marker RMS 27.3 mm, whole-run forward dynamics marker RMS 74.6 mm.
- **Tour 7-Iron:** Address calibrated marker RMS 4.4 mm, full-swing IK marker RMS 28.6 mm, forward dynamics marker RMS 112.3 mm (historical) / 144.0 mm (with ZMP filter).

However, following HO-8 (#10108 / #10258) and HO-11 (#10250), the primary benchmark receipts at `evidence/ground_support/anthro_driver/receipt.json` and `evidence/ground_support/anthro_iron/receipt.json` recorded:

- `anthro_driver`: address 42.2 mm, IK 52.3 mm, dynamics 89.1 mm.
- `anthro_iron`: address 70.2 mm, IK 72.0 mm, dynamics 116.0 mm.

This document establishes the canonical reference designation for both tour captures, presents the full bisect analysis explaining the quantitative shift between the runs, and documents the resolution of the provenance hash chain (#10271).

---

## 2. Canonical Run Designations

Per Owner-Authorized Contract Revision on #10324 and #10363:

> "A canonical benchmark selection is not threshold relaxation. Preserve historical receipts; document calibration differences and regenerate evidence with exact source/model/capture/runtime hashes."

We establish explicit categories of ground-support evidence:

### A. Canonical Calibrated Reference Runs (Tour Matched Accuracy)

These runs employ full subject-specific static trial calibration (`--static-seeds`), which solves for individual upper-body marker attachments against the neutral static trial pose before solving the address frame and full-capture IK.

| Capture    | Canonical Run Directory | Receipt Path                                                                                   | Address RMS              | IK RMS                    | Dynamics RMS              | Configuration                                                                                                                                                                                                                                                       |
| ---------- | ----------------------- | ---------------------------------------------------------------------------------------------- | ------------------------ | ------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Driver** | `anthro_driver_seeds`   | `docs/development/full_body_models/evidence/ground_support/anthro_driver_seeds/receipt.json`   | **6.4 mm** (`0.00635 m`) | **29.5 mm** (`0.02954 m`) | **80.8 mm** (`0.08076 m`) | Static seeds calibration, anatomical leg bounds (`BOUND_WIDENING = 1.0`), hip zero-twist, mirrored left hip (OSV-6), flexion-negative right knee (#12057), hips rewritten with the spec's own pelvis alignment (#12109), bounded wrists; 0 IK range-of-motion flags |
| **7-Iron** | `anthro_iron_seeds_zmp` | `docs/development/full_body_models/evidence/ground_support/anthro_iron_seeds_zmp/receipt.json` | **6.6 mm** (`0.00660 m`) | **26.8 mm** (`0.02677 m`) | **54.5 mm** (`0.05448 m`) | Static seeds calibration, dynamics ZMP filter, anatomical leg bounds, hip zero-twist, mirrored left hip (OSV-6), flexion-negative right knee (#12057), own pelvis alignment (#12109), bounded wrists; 0 IK range-of-motion flags                                    |

The two calibrated receipts above were then regenerated on the consolidated code of #12145 (ControlTower `ud-sim`, head `9a1052d779`). That code adds the GCV-20 ball impact, tracking weld projection and impact-split reference ZMP (#11767, #12117), plus the turn and phase-split reporting blocks. Address and IK are unchanged. The dynamics replay changes from 75.4 to 80.8 mm on the driver and from 59.6 to 54.5 mm on the 7-iron. Address-to-impact / after-impact FD is 34.2 / 145.0 mm (driver) and 30.1 / 92.1 mm (7-iron).

All four receipts in §2A and §2B were regenerated for #12109 on ControlTower (`ud-sim`). The hip rewrite had taken its pelvis alignment from `build_receipt_v2.json`, which turned the anthropometric hips 24 to 66 deg off the pelvis at zero coordinates (see `docs/research/hip_axis_mirroring/`, Hip Rewrite Pelvis Alignment). Against the #12057 receipts:

- Calibrated driver 5.6 / 31.0 / 77.0 mm becomes 6.4 / 29.5 / 75.4 mm, and the 7-iron 5.0 / 28.8 / 71.2 mm becomes 6.6 / 26.8 / 59.6 mm. IK and dynamics improve. The address marker RMS is worse by 0.8 and 1.6 mm; with foot progression on, the 7-iron address improves from 7.3 to 4.9 mm (`evidence/foot_progression/pelvis_alignment_12109/`).
- In the IK, the hips no longer sit on non-physiological bounds. Before, the driver had `hip_flexion_r` at -30 deg, `hip_adduction_r` at -50 deg and `hip_rotation_r` at -40 deg. `hip_adduction` still reaches +30 deg and the lead `hip_rotation_l` still reaches +40 deg during the swing (#12042).
- Nominal driver 43.7 / 56.3 / 90.4 mm becomes 42.6 / 52.7 / 79.3 mm. The nominal 7-iron keeps its trajectory-IK failure (70.5 / 264.8 / 1355.0 mm, #12030).

These receipts were previously regenerated for #12057 after the spec's right knee was mirrored to the flexion-negative convention of the left knee and of the declared [-120, 10] deg range. Before the fix the trail knee could not flex past 10 deg and the IK sat on that bound. All four runs were made on ControlTower (`ud-sim`). A control run of the driver `--static-seeds` command at the parent commit `80450f267a` reproduced the previous receipt exactly (7.7 / 33.0 / 61.6 mm). The changes below are therefore caused by the fix:

- Address and IK improve on both calibrated runs (driver 7.7 / 33.0 to 5.6 / 31.0 mm, 7-iron 6.5 / 31.1 to 5.0 / 28.8 mm).
- The calibrated dynamics replay is worse: 61.6 to 77.0 mm (driver) and 47.0 to 71.2 mm (7-iron). The backswing root error grows (driver maximum 5.7 to 20.3 mm), while the simulated finish foot slide falls from 411 to 169 mm. The tracking controller was tuned on the capped trail knee; this regression is tracked in #12110.
- The trail knee at the calibrated address is now -26.9 deg (driver) and -20.8 deg (7-iron), against -0.1 and -0.6 deg before.

Before #12057, both receipts were regenerated by OSV-6 (#11737) after the left hip adduction and rotation axes were mirrored like OpenSim, with `--static-seeds` (driver) and `--static-seeds --zmp-filter` (7-iron). The same commands on `main` @ `011d6141e6` without the mirror gave 7.2 / 33.6 / 65.0 mm (driver) and 6.3 / 31.7 / 53.0 mm (7-iron); the earlier 7.9 / 34.1 / 84.5 mm and 6.6 / 31.6 / 88.6 mm receipts came from `main` @ `3a11b5c07` (after #11047) and are kept in git history. The final scaled specification is committed beside each receipt so the provenance chain (§4) is checked by `tests/unit/motion_matching/pipeline/test_receipt_provenance_chain.py`.

The driver's per-frame dynamics record, `anthro_driver_seeds/dynamics_record.npz`, is committed beside its receipt for the GCV-4 no-clamp check (#11710). It holds 606 frames at 3 ms and is the record of the exact run behind that receipt. The peak vertical ground-reaction force is 3.12 BW (`dynamics.weight_fraction.max`, 2432 N at 79.4 kg, frame 405, t = 1.215 s); the peak net-GRF arrow the native export builds, which includes friction, is 3.18 BW. Both sit under the 6 BW ceiling of `default_glyph_style` (0.5 m per BW, 3 m cap), and zero force arrows are clamped over all frames. The earlier 1.2 BW figure is only the address-to-1 s window (`backswing_to_1s`), not the whole swing. Net free-moment torque arcs do clamp at their 0.6 m cap in 8 frames (peak 575 N m); that is torque scaling, outside GCV-4. The proof is `tests/unit/force_overlay/test_canonical_driver_grf_no_clamp.py`.

### C. Historical Pre-HO-8 Calibrated Runs (Not Reproducible by Current Code)

These were the canonical calibrated runs until 2026-09-27. They were produced with widened leg bounds (`BOUND_WIDENING = 2.0`), before hip zero-twist and on the pre-HO-11 base specification. Current code does not reproduce them (#11044), and their own receipts record IK range-of-motion violations, so they are kept as history only and must not be cited as current accuracy.

| Capture    | Run Directory              | Receipt Path                                                                                      | Address RMS          | IK RMS                | Dynamics RMS           | IK range-of-motion flags                                        |
| ---------- | -------------------------- | ------------------------------------------------------------------------------------------------- | -------------------- | --------------------- | ---------------------- | --------------------------------------------------------------- |
| **Driver** | `anthro_driver_shoot_g025` | `docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json` | 5.1 mm (`0.00507 m`) | 27.3 mm (`0.02729 m`) | 74.6 mm (`0.07458 m`)  | 10 coordinates (e.g. right knee outside range on 47% of frames) |
| **7-Iron** | `anthro_iron_zmp`          | `docs/development/full_body_models/evidence/ground_support/anthro_iron_zmp/receipt.json`          | 4.4 mm (`0.00443 m`) | 28.6 mm (`0.02858 m`) | 144.0 mm (`0.14400 m`) | 9 coordinates                                                   |

### B. Primary Uncalibrated Baselines (Nominal Geometry)

These runs evaluate model performance starting strictly from the nominal anthropometric document attachments without static-trial marker placement (`--static-seeds` omitted).

| Capture                     | Run Directory            | Receipt Path                                                                                    | Address RMS               | IK RMS                     | Dynamics RMS                | Configuration                                                                                                                                              |
| --------------------------- | ------------------------ | ----------------------------------------------------------------------------------------------- | ------------------------- | -------------------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Driver**                  | `anthro_driver`          | `docs/development/full_body_models/evidence/ground_support/anthro_driver/receipt.json`          | **42.6 mm** (`0.04260 m`) | **52.7 mm** (`0.05270 m`)  | **79.3 mm** (`0.07928 m`)   | Nominal marker attachments, functional hip zero-twist ($R_z(\theta)$), mirrored left hip (OSV-6), unwidened anatomical leg bounds (`BOUND_WIDENING = 1.0`) |
| **7-Iron**                  | `anthro_iron`            | `docs/development/full_body_models/evidence/ground_support/anthro_iron/receipt.json`            | **70.5 mm** (`0.07045 m`) | **264.8 mm** (`0.26479 m`) | **1355.0 mm** (`1.35495 m`) | Same configuration; the trajectory IK is about 250 mm off from frame 0 (#12030)                                                                            |
| Driver (pre-OSV-6, history) | `anthro_driver_pre_osv6` | `docs/development/full_body_models/evidence/ground_support/anthro_driver_pre_osv6/receipt.json` | 42.2 mm (`0.04220 m`)     | 52.3 mm (`0.05227 m`)      | 89.1 mm (`0.08913 m`)       | Unmirrored left hip; not reproduced by `main` @ `011d6141e6` (44.9 / 62.0 / 115.6 mm)                                                                      |
| 7-Iron (pre-OSV-6, history) | `anthro_iron_pre_osv6`   | `docs/development/full_body_models/evidence/ground_support/anthro_iron_pre_osv6/receipt.json`   | 70.2 mm (`0.07021 m`)     | 72.0 mm (`0.07198 m`)      | 116.0 mm (`0.11596 m`)      | Unmirrored left hip; not reproduced by `main` @ `011d6141e6` (71.9 / 269.5 / 2042.1 mm)                                                                    |

The nominal 7-iron failure is pre-existing: a control run of the same command on `main` @ `011d6141e6` (no mirror) also gives a trajectory IK of 269.5 mm and dynamics of 2042.1 mm, while the address solve stays at about 70 mm. It is tracked in #12030. Do not cite either nominal 7-iron receipt as model accuracy.

---

## 3. Bisect Analysis: Explaining the 27.3 mm -> 52.3 mm Shift

Detailed factor isolation reveals three distinct factors between the canonical calibrated runs and the primary baseline runs:

### Measured Bisect on Current Code (#11044)

Each row is a full pipeline regeneration of the driver with `--static-seeds` on `main` @ `3a11b5c07`, one factor reverted at a time in a scratch checkout. Raw numbers: [`bisect_11044_receipt.json`](bisect_11044_receipt.json).

| Permutation                           | Address RMS | IK RMS  | Dynamics RMS | IK range-of-motion flags |
| ------------------------------------- | ----------- | ------- | ------------ | ------------------------ |
| Current code                          | 7.9 mm      | 34.1 mm | 84.5 mm      | 0                        |
| Pre-HO-11 base specification          | 7.9 mm      | 34.1 mm | 84.5 mm      | 0                        |
| `BOUND_WIDENING = 2.0`                | 6.3 mm      | 28.1 mm | 53.8 mm      | 10                       |
| Hip zero-twist off                    | 6.0 mm      | 36.0 mm | 82.0 mm      | 0                        |
| All three reverted                    | 4.9 mm      | 25.7 mm | 55.1 mm      | 9                        |
| Historical `anthro_driver_shoot_g025` | 5.1 mm      | 27.3 mm | 74.6 mm      | 10                       |

The widened leg bounds explain the gap: they account for 6.0 mm of IK RMS and 30.7 mm of dynamics RMS on the calibrated configuration, not the +0.5 mm estimated below (the 2026-09-17 bisect compared four existing receipts and had no isolated widening run). The base specification change has no effect because the anthropometric candidate rescales the document. Hip zero-twist costs 1.9 mm of IK RMS. With all three reverted, current code reaches 4.9 / 25.7 mm, within 1.6 mm of the historical receipt; dynamics is 19.5 mm lower than the historical run, so later dynamics fixes (including #11047) are not attributed here. The 7-iron shows the same pattern: `--static-seeds --zmp-filter` gives 6.6 / 31.6 / 88.6 mm with 0 IK range-of-motion flags, and 4.1 / 23.5 / 53.4 mm with 9 flags at `BOUND_WIDENING = 2.0`.

### Factor 1: Static-Seeds Trial Marker Calibration (Dominant Factor: $\Delta \approx +24.9$ mm IK RMS)

- **Mechanism:** In `src/shared/python/motion_matching/pipeline/address.py`, when `static_seeds=True`, `lane.static_trial` solves for upper-body marker offsets using the subject's 24-frame neutral posture static trial.
- **Data Evidence:**
  - With static seeds: upper body marker residuals at address are ~3 mm to 7 mm per segment (head: 2.9 mm, trunk: 2.8 mm, pelvis: 4.0 mm, left arm: 7.2 mm, right arm: 3.3 mm).
  - Without static seeds: nominal offsets from the generic anthropometric document exhibit large subject-morphology offsets: head RMS is 64.9 mm, trunk is 33.3 mm, left arm is 61.9 mm, and right arm is 64.1 mm.
  - This initial 42.2 mm address offset propagates across all 654 frames, raising whole-swing IK RMS from 27.3 mm to 52.3 mm.
  - In HO-8 (`d05edafb0`), the regeneration command for `anthro_driver` inadvertently omitted `--static-seeds`.

### Factor 2: Leg Bound Widening (`BOUND_WIDENING`: $2.0 \to 1.0$) (Estimated +0.5 mm IK RMS on 2026-09-17; Measured +6.0 mm With Static Seeds, #11044)

- **Mechanism:** In pre-HO-8 runs, `pipeline/constants.py` had `BOUND_WIDENING = 2.0`, effectively doubling joint range margins on leg degrees of freedom (hip rotation, knee flexion, ankle subtalar). This yielded lower mathematical residuals (27.3 mm) at the expense of non-physiological joint angles (e.g. 71 frames of right hip rotation excess up to 24.3 deg).
- **HO-8 Correction:** Clamping `BOUND_WIDENING = 1.0` restored strict anatomical joint bounds, eliminating all leg RoM flags on the IK reference trajectory.

### Factor 3: Functional Hip Zero-Twist ($R_z(\theta)$)

- **Mechanism:** HO-8 introduced `hip_rotation_zero` to rotate coordinate frame zero based on medial knee markers or shank-thigh flexion normal, avoiding an artificial internal/external hip twist offset.
- **Data Evidence:** Zero twist applies $R_z(-25.7^\circ)$ (right) and $R_z(45.1^\circ)$ (left) to the hip frame `parent_to_base` transform. This improves biomechanical realism while having minimal independent effect on marker RMS (< 0.3 mm) when static seeds are enabled.

---

## 4. Hash Chain and Provenance Audit (#10271)

Issue #10271 reported that `base_spec_sha256` and `de_leva_table_sha256` were broken in `anthro_driver/receipt.json` and `anthro_iron/receipt.json`.

### Root Cause:

HO-8 (`d05edafb0`) branched from an earlier commit prior to the merge of HO-11 (`81ea27bfb` / #10250). When HO-8 regenerated receipts on its branch:

1. It used the pre-HO-11 base specification files, which carried canonical hash `a73d9e0623e79c5becd2b304cbc72940d16451097c4da0474a7e17d15fe3717a` (driver) instead of the post-HO-9/HO-11 canonical hash `174a6cb8dfd7f9347606789c7e6b77f602643e3139590126ac3af94e1f588b42`.
2. It dropped the `de_leva_table_sha256` field from the emitted JSON.

### Canonical Hashes:

- De Leva Male (1996) verified table SHA-256: `7e930d2248251ae345af1b3c889d47b50bd3a032dbe22345752a0e18178aa1c1`
- `full_body_spec_anthro_driver.json` canonical SHA-256: `174a6cb8dfd7f9347606789c7e6b77f602643e3139590126ac3af94e1f588b42`
- `full_body_spec_anthro_iron7.json` canonical SHA-256: `aba8196843e66c897770795e04cc0d2fca7c9ee4ba2472b2aa380d62a817ff31`

### Resolution (Software Contract):

Shared validator `pipeline.receipt_provenance.validate_receipt_provenance_chain`
(`receipt-provenance-chain/1`) is wired through unit/CI tests on the two
designated baselines. Contract:

- `base_spec_sha256` = canonical document digest
- `spec_sha256` = raw file bytes of the final scaled specification
- `spec_canonical_sha256` = canonical digest of that final document (formatting-only
  raw drift may pass when this field matches; numeric edits fail closed)
- `de_leva_table_sha256` required and must match table, base, and final docs

Intermediate `full_body_spec_hipcal.json` files are **not** fabricated; retention
policy is final-scaled only (`require_hipcal_document=False`).

Baseline receipts were re-anchored to the committed current bases and scaled
specs after verifying scaled anthropometry already embeds the current de Leva
table (including shank). Physical RMS numbers were not altered and remain
kinematic milestone evidence, not physical acceptance. Full native MuJoCo
re-execution was deferred on this workstation due to ~2 GB free disk.

## 5. Summary Policy for Program Documentation

1. **Path-Anchored Metrics:** Every metric cited in documentation must use explicit path-anchored references: `path/to/receipt.json#field.path`.
2. **Honest Distinctions:** Cite `anthro_driver_seeds/receipt.json` (6.4 mm / 29.5 mm / 80.8 mm) and `anthro_iron_seeds_zmp/receipt.json` (6.6 mm / 26.8 mm / 54.5 mm) as the calibrated tour match. 5.1 mm / 27.3 mm / 74.6 mm (`anthro_driver_shoot_g025`) and 4.4 mm / 28.6 mm / 144.0 mm (`anthro_iron_zmp`) are pre-HO-8 numbers produced with widened leg bounds; cite them only as history. When citing nominal baseline performance without static-trial calibration, cite `anthro_driver/receipt.json` (42.6 mm / 52.7 mm / 79.3 mm); the nominal 7-iron receipt records a known trajectory-IK failure (#12030).
3. **Automated Enforcement:** Continuous validation is enforced by `tests/unit/motion_matching/test_handoff_numbers_match_receipts.py`.
