# Knee Flexion at Address: Owner Prior, Capture Override, No Hyperextension (R4, #12684)

Parent epic: #12678. The engine is MuJoCo, and the stage is the calibrated address
(inverse kinematics only). This is not a dynamics or replay result.

## Decision

Owner ruling on #12684 (2026-10-15):

- The address knee flexion is a **prior of about 15 deg with a ±5 deg band**, for
  both knees and especially on the iron. These are owner-set values, not
  literature numbers. They live once, in `address_knee.KNEE_FLEX_PRIOR_DEG` and
  `address_knee.KNEE_FLEX_BAND_DEG`.
- **Capture wins** wherever the markers determine the knee flexion reliably.
- **Hyperextension is forbidden** as a hard bound, never as a soft penalty.
- Each knee's target source goes into the receipt, using R3's `target_source`
  vocabulary.

## Implementation

- `src/shared/python/motion_matching/address_knee.py` is the single module that
  defines, measures and resolves the address knee flexion.
  - Geometric flexion is the angle between the hip→knee and knee→ankle
    joint-centre vectors, with 0 meaning a straight leg. The model joint centres
    are the `femur_*`, `tibia_*` and `talus_*` body origins.
  - **Capture value.** The hip centre comes from the functional (sphere-fit) hip
    calibration, carried per frame by the rigid waist-marker pelvis pose.
    `*KneeOut` and `*AnkleOut` are moved medially onto the joint centres along
    the knee flexion axis (`hip_calibration.knee_flexion_axis`, using the leg-seed
    lateral offsets of 0.06 m and 0.056 m). The value is the median over R3's
    address window.
  - **Reliability.** A knee is reliable when its interquartile range is at most
    `foot_progression.MAX_SPREAD_DEG` and at least `MIN_VALID_FRACTION` of the
    window frames are usable. These are R3's address-window criteria, reused
    rather than restated.
- `resolve_knee_target(measured, mode=...)`:
  - `prior` mode: 15 deg for both knees, source `prior`.
  - `capture` mode: the reliable capture value (source `capture`), otherwise the
    prior (source `prior`).
  - A negative (hyperextended) target raises.
- **Source vocabulary.** `foot_progression.TARGET_SOURCES` is now the shared
  receipt vocabulary: `capture`, `default`, `style` and `prior`.
  `FOOT_TARGET_SOURCES` keeps the feet strict, so `prior` is not a valid foot
  source. `KNEE_TARGET_SOURCES` is `capture` and `prior`, and an import-time check
  keeps it a subset of the shared vocabulary.
- **Address solve** (`pipeline/address.py`, `best_address`). With `lane.knees`
  set:
  - The knee ranges stop at the anatomical zero. This is a hard bound on the
    coordinate; it tightens the swing-wide `(-120, +10)` range to `(-120, 0)`.
    The flexion direction is found from forward kinematics, so the sign is safe.
  - A solve-pose `posture_prior` pulls each knee coordinate to the value whose
    forward-kinematics flexion equals the target. The weight is
    `KNEE_FLEXION_PRIOR_WEIGHT` = 1 per rad², re-exported by `pipeline/constants.py`.
  - `assert_no_hyperextension` is the postcondition on the result.
  - The static trial re-places the leg markers at this pose, so the knee is
    anchored through the marker calibration.
- **CLI.** `--knee-flexion {off,prior,capture}` defaults to `capture`
  (`CANONICAL_KNEE_FLEXION`). A programmatic `Namespace` without the attribute
  resolves to `off`.
- **Receipt.** The address report gains `knee_flexion_deg`. It records the mode,
  prior, band and weight, and for each knee:
  - role, target, `target_source` and model flexion;
  - the knee coordinate, the error, `within_band` and `hyperextended`;
  - the capture measurement.
- **`off` is unchanged.** `_knee_terms` returns the input bounds and no extra
  keyword, so the solve arguments are identical. A unit test pins this. The
  `off` runs below reproduce R3's published numbers exactly: driver 19.80 deg /
  10.54 mm, 7-iron 13.53 deg / 12.19 mm.

## Capture-Derived Knee Flexion (Diagnose First)

| Capture          | Trail (R) | Lead (L) | Window IQR      |
| ---------------- | --------- | -------- | --------------- |
| capture-A driver | 26.2 deg  | 28.6 deg | 0.37 / 0.06 deg |
| capture-B 7-iron | 24.6 deg  | 26.3 deg | 0.70 / 0.15 deg |

All four knees are reliable. Without the medial correction (lateral markers used
directly), the values are 1.5 to 2 deg higher. The capture therefore says the
tour players stand with about 25 to 29 deg of knee flexion. That is **above** the
owner's 15 ± 5 deg prior. Under the ruling ("capture wins"), the canonical
`capture` mode uses these values.

## Measured Calibrated Address

All runs use the canonical `style` foot progression (R3). Angles are geometric
knee flexion in degrees. Full numbers are in
[`address_knee_summary.json`](address_knee_summary.json).

| Run                             | Trail knee              | Lead knee               | Trail toe-out | `hip_rotation_r` | Address RMS |
| ------------------------------- | ----------------------- | ----------------------- | ------------- | ---------------- | ----------- |
| driver, `off` (before)          | 29.49                   | 35.97                   | 19.80         | -34.38           | 10.54 mm    |
| driver, `prior`                 | **14.93**               | **15.12**               | 19.64         | -37.87           | 11.14 mm    |
| driver, `capture` (new default) | **26.10** (target 26.2) | **28.57** (target 28.6) | 20.00         | -36.06           | **9.08 mm** |
| 7-iron, `off` (before)          | 16.34                   | 32.23                   | 13.53         | -40.00           | 12.19 mm    |
| 7-iron, `prior`                 | **15.13**               | **15.21**               | 13.46         | -40.00           | 11.14 mm    |
| 7-iron, `capture` (new default) | **24.43** (target 24.6) | **26.36** (target 26.3) | 13.40         | -40.00           | 11.40 mm    |

- **Band.** Every knee in every opted-in run is within 0.3 deg of its target.
  That is well inside the ±5 deg band. No knee is hyperextended.
- **Address RMS.** No tolerance was changed and no joint range was widened.
  - `capture` improves the address RMS on both clubs: by 1.46 mm on the driver
    and 0.79 mm on the 7-iron.
  - `prior` costs 0.60 mm on the driver and improves the 7-iron by 1.05 mm.
- **Trail-foot toe-out re-measure (R3 follow-up).** Knee flexion does not change
  the 7-iron trail toe-out: 13.53 → 13.46 (`prior`) and 13.40 deg (`capture`).
  `hip_rotation_r` stays on its -40 deg limit in every mode. The shortfall is not
  a knee-coupling effect. The R3 ruling's next step applies: a static
  tibial-torsion / foot-yaw offset in the shared spec (R8, #12685), with a cited
  anthropometric value.
- **Hyperextension evidence.** In the frame-0 single-seed unit solve with the
  knee option off, the trail knee lands on the legacy +10 deg coordinate bound,
  which is 8.7 deg of geometric hyperextension, on both captures. With either
  mode on, it cannot.

## Not Done Here (Left for R10, #12692)

- The canonical ground-support receipts (`anthro_driver_seeds`,
  `anthro_iron_seeds_zmp`) are not regenerated. Regenerate them once, after R3
  and R4, on ControlTower (`ud-sim`), and report address, IK and FD RMS against
  these numbers.
- **Through the swing.** The knee residual acts at the address stage only. The
  trajectory IK, forward-dynamics frame 0 and the knee flexion through the first
  static window are not yet measured. That is issue acceptance item 4.
- **Cross-engine.** Drake, Pinocchio, OpenSim and MyoSuite replays of the new
  address have not been run. That is issue acceptance item 5.
- **Swing-wide ROM.** `range_of_motion.LOWER_LIMB_RANGES_DEG["knee_angle"]` is
  unchanged at `(-120, 10)`. Hyperextension is forbidden at the address only. A
  swing-wide change needs capture evidence through impact and the finish.

## Design Manual

`manuals/upstreamdrift` has no address-stage chapter, and
`calculation-registry.json` registers no address calculation. The
calculation-level reference is therefore `DESIGN_DECISIONS_2.md` §22 together
with this README.

## Reproduce

```bash
# PYTHONPATH: vendor/ud-tools/src/shared/python, ., src, src/shared/python
python - <<'PY'
import sys
from src.shared.python.motion_matching import address_knee
from src.shared.python.motion_matching.pipeline import cli
args = cli.build_parser().parse_args(sys.argv[1:] or [
    "--static-seeds", "--fix-club-markers", "--capture", "driver",
    "--spec", "docs/development/full_body_models/full_body_spec_anthro_driver.json",
    "--knee-flexion", "capture", "--out", "runs/r4_driver"])
res = cli.calibrate_run(args).cal_res
print(address_knee.model_knee_flexion_deg(res.kin, res.address2.q))
print(res.address_report["knee_flexion_deg"])
PY
```

For the 7-iron, add `--zmp-filter` and use `--capture iron --spec
docs/development/full_body_models/full_body_spec_anthro_iron7.json`. Each run takes
about 4 to 5 minutes on one Windows host.
