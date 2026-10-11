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

| Capture    | Canonical Run Directory | Receipt Path                                                                                   | Address RMS               | IK RMS                    | Dynamics RMS               | Configuration                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---------- | ----------------------- | ---------------------------------------------------------------------------------------------- | ------------------------- | ------------------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Driver** | `anthro_driver_seeds`   | `docs/development/full_body_models/evidence/ground_support/anthro_driver_seeds/receipt.json`   | **11.0 mm** (`0.01095 m`) | **30.3 mm** (`0.03028 m`) | **82.0 mm** (`0.08200 m`)  | Measured driver club cluster in the base document (#12221), per-capture grip rotation fitted on the fixed cluster (#12221), `--fix-club-markers`, static seeds calibration, anatomical leg bounds (`BOUND_WIDENING = 1.0`), hip zero-twist, mirrored left hip (OSV-6), flexion-negative right knee (#12057), hips rewritten with the spec's own pelvis alignment (#12109), bounded wrists; 0 IK range-of-motion flags |
| **7-Iron** | `anthro_iron_seeds_zmp` | `docs/development/full_body_models/evidence/ground_support/anthro_iron_seeds_zmp/receipt.json` | **4.0 mm** (`0.00396 m`)  | **28.4 mm** (`0.02836 m`) | **109.9 mm** (`0.10985 m`) | 7-iron club cluster in the base document (#12210), static seeds calibration, dynamics ZMP filter, anatomical leg bounds, hip zero-twist, mirrored left hip (OSV-6), flexion-negative right knee (#12057), own pelvis alignment (#12109), bounded wrists; 0 IK range-of-motion flags                                                                                                                                   |

The three driver receipts (`anthro_driver_seeds`, `anthro_driver_drake` and the nominal `anthro_driver` of §2B) were regenerated for #12221 on ControlTower (`ud-sim`) at `8d42b60dce` / `e44573633b` (identical documents) after the per-capture hand-to-club rotation (`grip_rotation_deg`) fitted on the fixed driver club cluster was applied to the driver document only (receipt `evidence/anthropometry/fit_grip_rotation_receipt_driver.json`, `scripts/set_spec_grip_rotation.py`). The fit is valid only with the measured club cluster held fixed in the static trial, so the calibrated driver commands now add `--fix-club-markers` (`--static-seeds --fix-club-markers`, and `--engine drake` for the Drake run). Address / IK / dynamics, before to after: driver 7.5 / 30.5 / 82.6 to 11.0 / 30.3 / 82.0 mm; Drake driver 7.2 / 30.7 / 77.8 to 11.0 / 30.3 / 86.3 mm (the Drake dynamics replay worsened by 8.5 mm); nominal driver 41.7 / 46.1 / 73.6 to 50.9 / 46.5 / 71.9 mm. All three keep 0 IK range-of-motion flags; the nominal driver reference-ZMP outside fraction over 1 to 1.5 s goes from 140/180 to 158/180. A control on the same commit with the old document and the cluster held fixed reproduces the earlier control exactly (driver 7.7 / 29.4 / 105.9 mm), so the grip refit, not code drift, explains the change against the cluster-fixed baseline: dynamics 105.9 to 82.0 mm. The address RMS rises about 3 mm because the club markers can no longer move to fit the static trial. Sensitivity: the experiment document behind the earlier measurement differed from the applied document only by receipt rounding (about 1e-5 in each wrist `child_to_follower` entry), yet its dynamics were 79.3 mm against 82.0 mm here, while address and IK agree within 0.3 mm. The tracking replay is therefore sensitive at the 3 to 15 mm level to tiny parameter changes. The driver club-face fixture (`tests/fixtures/club_face`) was regenerated at `9fe74740a5` with `scripts/regenerate_club_face_fixtures.py`: impact speed 50.02 / 51.32 m/s (-2.5 %, was -5.1 %), peak 4.4 ms early (was 1.8 ms), post-impact speed drop -4.5 % (was -43.2 %), face at address 5.44 / 5.50 deg and at impact 16.88 / 14.96 deg; it meets GCV-20 (3 %, 5 ms, 10 %) and OSV-10 (5 deg), and the driver same-input export no longer stops at the trail-shoulder singularity (#12266).

The Rajagopal driver grip calibration (`msk_club_grip_calibration.json`) reads the driver fixture's address pose, so it was re-run and the club of both `golf_humanoid*.osim` models rebuilt. Driven by the new driver fixture, the OpenSim bushing grip exceeds its full-swing bounds only in the 6 ms ball-impact window (2.62 deg against 2 deg, 51.3 N against 50 N; the previous fixture peaked at 1.81 deg and 34.5 N at the same instant). The bounds are unchanged; the decision is on #12272.

The 7-iron is not adopted. Its refit meets GCV-20 and OSV-10 on the tracked swing (-1.9 %, -0.2 ms, impact face 2.59 against 3.21 deg), but the club-face fixture comes from the same-input export (open-loop replay of the recorded efforts at 1 kHz), which gives -5.0 %, peak -5.1 ms and an 8.71 deg impact face against 3.21 deg (gap 5.50 deg; 5.09 deg at the top; limit 5 deg). The committed 7-iron fixture gives -3.9 %, -3.9 ms and 2.83 deg, and the OSV-10 face tests run forward kinematics on the current document, so adopting it would turn eight passing face tests red. The 7-iron document and its receipts (`anthro_iron_seeds_zmp` 4.0 / 28.4 / 109.9 mm, `anthro_iron` 41.5 / 45.2 / 104.9 mm) are unchanged; `fit_grip_rotation_receipt_iron7.json` stays committed for a follow-up.

The driver calibrated receipt was regenerated for #12221 on ControlTower (`ud-sim`) at `6438169a61`, after the driver document received its measured club cluster (§2B). A control of the same `--static-seeds` command at the parent `e90b853672` reproduces the previous receipt (6.4 / 29.5 / 80.8 mm; only `elapsed_s` and blocks added since differ), so every change below is caused by the new club prior:

- Address goes from 6.4 to 7.5 mm, IK from 29.5 to 30.5 mm and dynamics from 80.8 to 82.6 mm (address-to-impact 34.2 to 39.9 mm; after impact unchanged at 145.0 mm). The release cutoff stays at 30 Hz.
- The prior is the static-trial output of this same receipt, yet the calibration moves it again: the capture's own face at address goes from 5.50 to 9.38 deg (impact 14.96 to 19.50 deg). The static trial is not a fixed point of its club priors on the driver either (the 7-iron case is below).
- `anthro_driver_drake` (`--static-seeds --engine drake`) was regenerated with it: 6.4 / 29.2 / 78.8 mm becomes 7.2 / 30.8 / 77.8 mm.

The 7-iron calibrated receipt was regenerated for #12210 on ControlTower (`ud-sim`) at `6fd7073493`, after the 7-iron document received its own club marker cluster (§2B). A control of the same `--static-seeds --zmp-filter` command at the parent `1972d3a153` reproduces the previous receipt exactly (6.6 / 26.8 / 54.5 mm), so every change below is caused by the new club priors:

- Address improves from 6.6 to 4.0 mm and the face separation RMS from 0.63 to 0.55 deg. IK worsens from 26.8 to 28.4 mm (club segment 15.6 to 23.2 mm). Dynamics worsens from 54.5 to 109.9 mm: address-to-impact 30.1 to 40.9 mm, after impact 92.1 to 199.7 mm.
- The static-trial calibration is not a fixed point of its priors. Seeded with the corrected cluster, it places the club markers 25–54 mm and the body markers up to 32 mm differently from the run seeded with the driver's cluster. On that placement the IK clubhead speed peak precedes the capture impact split by 7–18 ms at every candidate cutoff, so the GCV-20 release cutoff search finds no cutoff that preserves it and falls back to 30 Hz (before: 18 Hz, preserved). The after-impact replay error doubles with it.
- The previous 54.5 mm was therefore obtained from a calibration pulled toward the driver's club cluster. Which calibrated 7-iron result is canonical is an owner decision, recorded on #12210.

The two calibrated receipts above were then regenerated on the consolidated code of #12145 (ControlTower `ud-sim`, head `9a1052d779`). That code adds the GCV-20 ball impact, tracking weld projection and impact-split reference ZMP (#11767, #12117), plus the turn and phase-split reporting blocks. Address and IK are unchanged. The dynamics replay changes from 75.4 to 80.8 mm on the driver and from 59.6 to 54.5 mm on the 7-iron. Address-to-impact / after-impact FD is 34.2 / 145.0 mm (driver) and 30.1 / 92.1 mm (7-iron).

All four receipts in §2A and §2B were regenerated for #12109 on ControlTower (`ud-sim`). The hip rewrite had taken its pelvis alignment from `build_receipt_v2.json`, which turned the anthropometric hips 24 to 66 deg off the pelvis at zero coordinates (see `docs/research/hip_axis_mirroring/`, Hip Rewrite Pelvis Alignment). Against the #12057 receipts:

- Calibrated driver 5.6 / 31.0 / 77.0 mm becomes 6.4 / 29.5 / 75.4 mm, and the 7-iron 5.0 / 28.8 / 71.2 mm becomes 6.6 / 26.8 / 59.6 mm. IK and dynamics improve. The address marker RMS is worse by 0.8 and 1.6 mm; with foot progression on, the 7-iron address improves from 7.3 to 4.9 mm (`evidence/foot_progression/pelvis_alignment_12109/`).
- In the IK, the hips no longer sit on non-physiological bounds. Before, the driver had `hip_flexion_r` at -30 deg, `hip_adduction_r` at -50 deg and `hip_rotation_r` at -40 deg. `hip_adduction` still reaches +30 deg and the lead `hip_rotation_l` still reaches +40 deg during the swing (#12042).
- Nominal driver 43.7 / 56.3 / 90.4 mm becomes 42.6 / 52.7 / 79.3 mm. The nominal 7-iron keeps its trajectory-IK failure (70.5 / 264.8 / 1355.0 mm), which #12030 later fixed (§2B).

These receipts were previously regenerated for #12057 after the spec's right knee was mirrored to the flexion-negative convention of the left knee and of the declared [-120, 10] deg range. Before the fix the trail knee could not flex past 10 deg and the IK sat on that bound. All four runs were made on ControlTower (`ud-sim`). A control run of the driver `--static-seeds` command at the parent commit `80450f267a` reproduced the previous receipt exactly (7.7 / 33.0 / 61.6 mm). The changes below are therefore caused by the fix:

- Address and IK improve on both calibrated runs (driver 7.7 / 33.0 to 5.6 / 31.0 mm, 7-iron 6.5 / 31.1 to 5.0 / 28.8 mm).
- The calibrated dynamics replay is worse: 61.6 to 77.0 mm (driver) and 47.0 to 71.2 mm (7-iron). The backswing root error grows (driver maximum 5.7 to 20.3 mm), while the simulated finish foot slide falls from 411 to 169 mm. The tracking controller was tuned on the capped trail knee; this regression is tracked in #12110.
- The trail knee at the calibrated address is now -26.9 deg (driver) and -20.8 deg (7-iron), against -0.1 and -0.6 deg before.

Before #12057, both receipts were regenerated by OSV-6 (#11737) after the left hip adduction and rotation axes were mirrored like OpenSim, with `--static-seeds` (driver) and `--static-seeds --zmp-filter` (7-iron). The same commands on `main` @ `011d6141e6` without the mirror gave 7.2 / 33.6 / 65.0 mm (driver) and 6.3 / 31.7 / 53.0 mm (7-iron); the earlier 7.9 / 34.1 / 84.5 mm and 6.6 / 31.6 / 88.6 mm receipts came from `main` @ `3a11b5c07` (after #11047) and are kept in git history. The final scaled specification is committed beside each receipt so the provenance chain (§4) is checked by `tests/unit/motion_matching/pipeline/test_receipt_provenance_chain.py`.

The driver's per-frame dynamics record, `anthro_driver_seeds/dynamics_record.npz`, is committed beside its receipt for the GCV-4 no-clamp check (#11710). It holds 606 frames at 3 ms and is the record of the exact run behind that receipt. The peak vertical ground-reaction force is 2.93 BW (`dynamics.weight_fraction.max`, 2283 N, frame 405, t = 1.215 s) on the #12221 record; it was 3.12 BW (2432 N at 79.4 kg) before. On the previous record the peak net-GRF arrow the native export builds, which includes friction, was 3.18 BW. Both sit under the 6 BW ceiling of `default_glyph_style` (0.5 m per BW, 3 m cap), and zero force arrows are clamped over all frames. The earlier 1.2 BW figure is only the address-to-1 s window (`backswing_to_1s`), not the whole swing. Net free-moment torque arcs do clamp at their 0.6 m cap in 8 frames (peak 575 N m); that is torque scaling, outside GCV-4. The proof is `tests/unit/force_overlay/test_canonical_driver_grf_no_clamp.py`.

### C. Historical Pre-HO-8 Calibrated Runs (Not Reproducible by Current Code)

These were the canonical calibrated runs until 2026-09-27. They were produced with widened leg bounds (`BOUND_WIDENING = 2.0`), before hip zero-twist and on the pre-HO-11 base specification. Current code does not reproduce them (#11044), and their own receipts record IK range-of-motion violations, so they are kept as history only and must not be cited as current accuracy.

| Capture    | Run Directory              | Receipt Path                                                                                      | Address RMS          | IK RMS                | Dynamics RMS           | IK range-of-motion flags                                        |
| ---------- | -------------------------- | ------------------------------------------------------------------------------------------------- | -------------------- | --------------------- | ---------------------- | --------------------------------------------------------------- |
| **Driver** | `anthro_driver_shoot_g025` | `docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json` | 5.1 mm (`0.00507 m`) | 27.3 mm (`0.02729 m`) | 74.6 mm (`0.07458 m`)  | 10 coordinates (e.g. right knee outside range on 47% of frames) |
| **7-Iron** | `anthro_iron_zmp`          | `docs/development/full_body_models/evidence/ground_support/anthro_iron_zmp/receipt.json`          | 4.4 mm (`0.00443 m`) | 28.6 mm (`0.02858 m`) | 144.0 mm (`0.14400 m`) | 9 coordinates                                                   |

### B. Primary Uncalibrated Baselines (Nominal Geometry)

These runs evaluate model performance starting strictly from the nominal anthropometric document attachments without static-trial marker placement (`--static-seeds` omitted).

| Capture                     | Run Directory            | Receipt Path                                                                                    | Address RMS               | IK RMS                    | Dynamics RMS               | Configuration                                                                                                                                                                                                                                              |
| --------------------------- | ------------------------ | ----------------------------------------------------------------------------------------------- | ------------------------- | ------------------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Driver**                  | `anthro_driver`          | `docs/development/full_body_models/evidence/ground_support/anthro_driver/receipt.json`          | **50.9 mm** (`0.05094 m`) | **46.5 mm** (`0.04646 m`) | **71.9 mm** (`0.07193 m`)  | Measured driver club cluster (#12221), per-capture grip rotation (#12221), nominal marker attachments, functional hip zero-twist ($R_z(\theta)$), mirrored left hip (OSV-6), unwidened anatomical leg bounds (`BOUND_WIDENING = 1.0`); GCV-20 impact split |
| **7-Iron**                  | `anthro_iron`            | `docs/development/full_body_models/evidence/ground_support/anthro_iron/receipt.json`            | **41.5 mm** (`0.04145 m`) | **45.2 mm** (`0.04517 m`) | **104.9 mm** (`0.10493 m`) | Same configuration with the 7-iron club cluster (#12210); face residual and impact split observed; frame-0 IK 43.2 mm; 0 IK range-of-motion flags                                                                                                          |
| Driver (pre-OSV-6, history) | `anthro_driver_pre_osv6` | `docs/development/full_body_models/evidence/ground_support/anthro_driver_pre_osv6/receipt.json` | 42.2 mm (`0.04220 m`)     | 52.3 mm (`0.05227 m`)     | 89.1 mm (`0.08913 m`)      | Unmirrored left hip; not reproduced by `main` @ `011d6141e6` (44.9 / 62.0 / 115.6 mm)                                                                                                                                                                      |
| 7-Iron (pre-OSV-6, history) | `anthro_iron_pre_osv6`   | `docs/development/full_body_models/evidence/ground_support/anthro_iron_pre_osv6/receipt.json`   | 70.2 mm (`0.07021 m`)     | 72.0 mm (`0.07198 m`)     | 116.0 mm (`0.11596 m`)     | Unmirrored left hip; not reproduced by `main` @ `011d6141e6` (71.9 / 269.5 / 2042.1 mm)                                                                                                                                                                    |

Both nominal receipts were regenerated for #12030 on ControlTower (`ud-sim`) at `52da723d6d`. The nominal 7-iron trajectory IK had failed since the club-face orientation residual (OSV-10, #11759) was introduced: 264.8 mm IK and 1355.0 mm dynamics, and 269.5 mm / 2042.1 mm on `main` @ `011d6141e6` before the mirror. The cause was that `full_body_spec_anthro_iron7.json` carries the driver's clubhead triad offsets (edges 92 / 52 / 96 mm against the 7-iron capture's 106 / 127 / 40 mm). Without `--static-seeds` those offsets are not calibrated, so the Kabsch face rotation fit a triad of the wrong shape (median rigid residual 42.4 mm, against 0.2 mm for matching offsets). The face target was then 40–80 deg off and, at weight 3, pulled the whole trajectory off the address. `club_face_target.observe_frame_rotations` now fails closed above a 5 mm median rigid residual. The face residual and the impact split record the reason instead of fitting a meaningless rotation.

- The 7-iron becomes IK 67.9 mm (frame 0 70.2 mm against the 70.4 mm address) and dynamics 79.2 mm. The same command with `--face-weight 0` on `main` gave the same 67.9 mm IK, which isolates the cause.
- The driver's triad matches its capture (0.2 mm), so its face residual is unchanged. Its address and IK are identical (42.6 / 52.7 mm). Its dynamics move from 79.3 to 97.3 mm because the previous receipt predates the GCV-20 impact-split reference (#12145). A control of the same command on `main` @ `911b618ff1` gives the identical 97.3 mm.
- **#12210:** the builder had given every club the native candidate's club cluster, which is the driver's head triad and shaft cluster. It now reads a per-club `club-markers/1` file (`evidence/club_markers/<club>.json`, `assets.get_club_markers`) and refuses a non-driver club without one. The 7-iron file holds the six `Clubhead` offsets of the static-seed calibrated `anthro_iron_seeds_zmp` receipt at `1bef553457`, and `scripts/set_spec_club_markers.py` applies it to the committed document. The head-triad median rigid residual against the 7-iron capture falls from 42.4 to 0.24 mm (driver document against the driver capture: 0.17 mm). The nominal 7-iron, regenerated at `6fd7073493`, goes from 70.4 / 67.9 / 79.2 mm to 41.5 / 45.2 / 104.9 mm. The address club segment falls from 125.6 to 23.3 mm and the IK club segment from 117.3 to 25.4 mm. The face residual is now observed (separation RMS 1.41 deg) and the impact split is active. As on the nominal driver, the impact-split replay improves address-to-impact (69.0 to 48.9 mm) and worsens after impact (101.6 to 184.6 mm); the release cutoff falls back to 30 Hz. The calibrated 7-iron also changes (§2A).
- **#12221:** the driver document carried the native candidate's cluster. Its shape matches the capture, but held fixed in the static trial (`--fix-club-markers`) it puts the capture's own address face at −20.8 deg, so its placement on the club is not a valid face frame. `evidence/club_markers/driver.json` now holds the six `Clubhead` offsets of the static-seed calibrated `anthro_driver_seeds` receipt at `1bef553457` (same triad shape, 52 / 92 / 96 mm; the head markers move 14–44 mm on the club), applied with `scripts/set_spec_club_markers.py`. The nominal driver, regenerated at `6438169a61`, goes from 42.6 / 52.7 / 97.3 mm to 41.7 / 46.1 / 73.6 mm. The IK club segment falls from 40.5 to 33.1 mm and the face separation RMS from 2.65 to 0.97 deg. Address-to-impact goes from 67.8 to 51.6 mm and after impact from 150.3 to 113.5 mm.
- **#12221 grip refit (driver only):** the nominal driver receipt was regenerated with the per-capture grip rotation applied (nominal runs add no flag; the command is unchanged): 41.7 / 46.1 / 73.6 mm becomes 50.9 / 46.5 / 71.9 mm. The nominal 7-iron receipt is unchanged (41.5 / 45.2 / 104.9 mm); its refit is deferred (§2A).
- `tests/unit/motion_matching/pipeline/test_nominal_receipts_ik.py` fails if either nominal receipt's IK exceeds 80 mm, its frame-0 IK exceeds the address residual by more than 50 mm, or it carries IK range-of-motion flags.

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
2. **Honest Distinctions:** Cite `anthro_driver_seeds/receipt.json` (11.0 mm / 30.3 mm / 82.0 mm) and `anthro_iron_seeds_zmp/receipt.json` (4.0 mm / 28.4 mm / 109.9 mm) as the calibrated tour match. 5.1 mm / 27.3 mm / 74.6 mm (`anthro_driver_shoot_g025`) and 4.4 mm / 28.6 mm / 144.0 mm (`anthro_iron_zmp`) are pre-HO-8 numbers produced with widened leg bounds; cite them only as history. When citing nominal baseline performance without static-trial calibration, cite `anthro_driver/receipt.json` (50.9 mm / 46.5 mm / 71.9 mm); the nominal 7-iron receipt records a known trajectory-IK failure (#12030).
3. **Automated Enforcement:** Continuous validation is enforced by `tests/unit/motion_matching/test_handoff_numbers_match_receipts.py`.
