# Canonical Ground-Support Reference Runs and Bisect Analysis

**Author:** Dieter Olson (`agent:local`)  
**Date:** 2026-09-17  
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

We establish two explicit categories of ground-support evidence:

### A. Canonical Calibrated Reference Runs (Tour Matched Accuracy)

These runs employ full subject-specific static trial calibration (`--static-seeds`), which solves for individual upper-body marker attachments against the neutral static trial pose before solving the address frame and full-capture IK.

| Capture    | Canonical Run Directory    | Receipt Path                                                                                      | Address RMS              | IK RMS                    | Dynamics RMS               | Configuration                                                                             |
| ---------- | -------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------ | ------------------------- | -------------------------- | ----------------------------------------------------------------------------------------- |
| **Driver** | `anthro_driver_shoot_g025` | `docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json` | **5.1 mm** (`0.00507 m`) | **27.3 mm** (`0.02729 m`) | **74.6 mm** (`0.07458 m`)  | Static seeds calibration, contact-aware shooting fit ($g=0.25$), bounded wrists           |
| **7-Iron** | `anthro_iron_zmp`          | `docs/development/full_body_models/evidence/ground_support/anthro_iron_zmp/receipt.json`          | **4.4 mm** (`0.00443 m`) | **28.6 mm** (`0.02858 m`) | **144.0 mm** (`0.14400 m`) | Static seeds calibration, dynamics ZMP filter inside foot support polygon, bounded wrists |

### B. Primary Uncalibrated Baselines (Nominal Geometry)

These runs evaluate model performance starting strictly from the nominal anthropometric document attachments without static-trial marker placement (`--static-seeds` omitted).

| Capture    | Run Directory   | Receipt Path                                                                           | Address RMS               | IK RMS                    | Dynamics RMS               | Configuration                                                                                                                   |
| ---------- | --------------- | -------------------------------------------------------------------------------------- | ------------------------- | ------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Driver** | `anthro_driver` | `docs/development/full_body_models/evidence/ground_support/anthro_driver/receipt.json` | **42.2 mm** (`0.04220 m`) | **52.3 mm** (`0.05227 m`) | **89.1 mm** (`0.08913 m`)  | Nominal marker attachments, functional hip zero-twist ($R_z(\theta)$), unwidened anatomical leg bounds (`BOUND_WIDENING = 1.0`) |
| **7-Iron** | `anthro_iron`   | `docs/development/full_body_models/evidence/ground_support/anthro_iron/receipt.json`   | **70.2 mm** (`0.07021 m`) | **72.0 mm** (`0.07198 m`) | **116.0 mm** (`0.11596 m`) | Nominal marker attachments, functional hip zero-twist ($R_z(\theta)$), unwidened anatomical leg bounds (`BOUND_WIDENING = 1.0`) |

---

## 3. Bisect Analysis: Explaining the 27.3 mm -> 52.3 mm Shift

Detailed factor isolation reveals three distinct factors between the canonical calibrated runs and the primary baseline runs:

### Factor 1: Static-Seeds Trial Marker Calibration (Dominant Factor: $\Delta \approx +24.9$ mm IK RMS)

- **Mechanism:** In `src/shared/python/motion_matching/pipeline/address.py`, when `static_seeds=True`, `lane.static_trial` solves for upper-body marker offsets using the subject's 24-frame neutral posture static trial.
- **Data Evidence:**
  - With static seeds: upper body marker residuals at address are ~3 mm to 7 mm per segment (head: 2.9 mm, trunk: 2.8 mm, pelvis: 4.0 mm, left arm: 7.2 mm, right arm: 3.3 mm).
  - Without static seeds: nominal offsets from the generic anthropometric document exhibit large subject-morphology offsets: head RMS is 64.9 mm, trunk is 33.3 mm, left arm is 61.9 mm, and right arm is 64.1 mm.
  - This initial 42.2 mm address offset propagates across all 654 frames, raising whole-swing IK RMS from 27.3 mm to 52.3 mm.
  - In HO-8 (`d05edafb0`), the regeneration command for `anthro_driver` inadvertently omitted `--static-seeds`.

### Factor 2: Leg Bound Widening (`BOUND_WIDENING`: $2.0 \to 1.0$) ($\Delta \approx +0.5$ mm IK RMS, but Eliminates Joint Limit Violations)

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

---

## 5. Summary Policy for Program Documentation

1. **Path-Anchored Metrics:** Every metric cited in documentation must use explicit path-anchored references: `path/to/receipt.json#field.path`.
2. **Honest Distinctions:** When citing 5.1 mm / 27.3 mm / 74.6 mm, cite `anthro_driver_shoot_g025/receipt.json` as the calibrated tour match. When citing nominal baseline performance without static-trial calibration, cite `anthro_driver/receipt.json` (42.2 mm / 52.3 mm / 89.1 mm).
3. **Automated Enforcement:** Continuous validation is enforced by `tests/unit/motion_matching/test_handoff_numbers_match_receipts.py`.
