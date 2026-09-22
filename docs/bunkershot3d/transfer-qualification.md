# BunkerShot3D Sand-to-Ball Transfer Qualification

**Issue #9543 (epic #9541). Depends on #9286 evidence admission. Metric: ASME V&V 20-2009.**

The [validation roadmap](validation-roadmap.md) says which _sand_ measurement buys which
credibility level. This document is the launch-side half: the program by which the
`MomentumTransfer` parameters of `bunkershot3d.ball.splash` — every one of them a
stated placeholder — are fitted on measured strokes, frozen, and tested on held-out
sessions, and the only path by which the unconditional `BEYOND_VALIDATION` floor on a
launch verdict can be lifted.

## Read This First

**Nothing has been measured.** The shipped register of measured strokes is empty and
every launch verdict this package produces today is floored at `BEYOND_VALIDATION`. What
shipped under #9543 is the software half of the program: the contract a measured stroke
must satisfy, the registered intended-use matrix and protocol, the predeclared
tolerances, the bounded fit with identifiability and sensitivity checks, the held-out
comparison under V&V 20, the versioned evidence object, and the report. Physical
qualification stays blocked until suitable strokes are measured, and the issue is not
closed by this software.

## Where It Lives

| Thing                                                             | Where                                                                                            |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Contract: stroke, dataset, matrix, protocol, tolerances, evidence | [`bunkershot3d/ball/qualification.py`](../../src/bunkershot3d/ball/qualification.py)             |
| Fit, held-out validation, `qualify()`, report                     | [`bunkershot3d/ball/qualification_fit.py`](../../src/bunkershot3d/ball/qualification_fit.py)     |
| Verdict floor and provenance wiring                               | [`bunkershot3d/ball/splash.py`](../../src/bunkershot3d/ball/splash.py)                           |
| What the three-camera rig can and cannot measure                  | [`bunkershot3d/ball/rig_capability.py`](../../src/bunkershot3d/ball/rig_capability.py)           |
| On-file measurement admitted to a V&V 20 comparison               | [`bunkershot3d/vandv/validation.py`](../../src/bunkershot3d/vandv/validation.py)                 |
| Tests, including the shipped-state pin                            | [`test_transfer_qualification.py`](../../tests/bunkershot3d/ball/test_transfer_qualification.py) |

## The Pipeline

```text
MeasuredStroke ... -> QualificationDataset -> fit_transfer -> validate_holdout -> TransferQualification
   (intake)            (split by session)     (calibration)    (held-out)          (versioned evidence)
```

`qualify(dataset)` runs the three steps and is a pure function of the dataset and the
tolerances: the version string is `transfer-qualification/1+<digest>` where the digest
covers every stroke id, session, batch and raw-data digest, so re-running on the same
evidence gives the same version and a changed stroke gives a different one.

### Intake

A `MeasuredStroke` carries the solver's `SandDelivery` for the measured delivery, the
lie, the loft, and the measured launch as `MeasurementRecord` values against three
explicit reference keys that extend the ledger's seven sand specs:

| Key                     | Unit    | Required |
| ----------------------- | ------- | -------- |
| `ball_launch_speed_m_s` | `m/s`   | yes      |
| `ball_launch_angle_rad` | `rad`   | no       |
| `ball_spin_rate_rad_s`  | `rad/s` | no       |

A synthetic fixture is refused at construction; a record in another unit or against
another key is refused; a stroke needs an ISO date and a SHA-256 digest of its raw
capture. Missing quantities stay `None`. Every sand batch must be characterised in the
dataset's register under the ledger's own acceptance criteria (#9286): the registered
protocol requires `bunker_sand_bulk_density_kg_m3` and
`bunker_sand_drained_friction_angle_deg`.

### No Leakage

The calibration subset is designated by **session** in the dataset, before any fit. Two
strokes with one raw-data digest are refused as one capture filed twice. A dataset with
no held-out session, or with a designated session that has no strokes, is refused. The
evidence object refuses overlapping calibration and held-out stroke sets.

### Calibration

`fit_transfer` fits `efficiency` and `packing_sensitivity` by default, bounded to
`[0, 1]` so the momentum budget of #8657 survives the fit, weighting each residual by the
stroke's expanded uncertainty. It refuses, and records the refusal, when:

- fewer than ten calibration strokes are offered;
- `packing_sensitivity` is declared but the calibration beds span less than 0.15 in
  relative density — at one packing the two parameters are one number;
- `sand_ball_friction` is declared without a spin record on every stroke;
- the column-scaled Jacobian's condition number exceeds `1e6`.

`spin_lever_arm_fraction` is never fitted: spin identifies only its product with the
friction share. A converged fit reports standard errors from the Jacobian, the parameter
covariance, the mean normalised local sensitivity of ball speed to each parameter, and
whether any parameter sits at a bound.

### Held-Out Validation

With the parameters frozen, each held-out stroke is compared under V&V 20 through
`ValidationComparison(..., measured_record=stroke.ball_speed)`: the on-file instrument
record is what lifts `require_measurable`'s literature refusal, and nothing else does.
`u_num` comes from the stroke's grid study, `u_exp` from the record, and `u_input` is the
sand-state input uncertainty in quadrature with the fit's parameter uncertainty
propagated through the prediction gradient. Each stroke therefore says whether its
discrepancy is noise-limited or a detected model-form error, and each regime reports
which of the three uncertainties dominates.

Per regime of the intended-use matrix the verdict compares relative bias, relative RMS
error, prediction-interval coverage and launch-angle bias against
`PRACTICAL_TOLERANCES`, which are fixed in the source with their rationale (±5 % bias
and 10 % RMS from the ±10 % carry window; 0.80 coverage; 3° angle; five held-out strokes
per regime). A regime that fails is recorded as rejected with its reasons; a regime with
no held-out stroke is rejected, not skipped.

### Lifting the Floor

`compute_ball_launch_from_splash(..., qualification=...)` uses the frozen parameters and
refuses any other explicit parameters beside them. `launch_verdict` reads `WITHIN` for
the launch model's own statement only inside a qualified regime and keeps
`BEYOND_VALIDATION` — with the uncalibrated reasons — outside the matrix, in a rejected
regime, or on a failed fit. The result is still combined with the solver's verdict, so a
carry never reads better than the shot behind it. Fitted parameters carry the new
`ProvenanceBasis.CALIBRATED`, ranked with `SPECIFICATION` and below `MEASURED`;
`measured_constants()` stays empty.

## Disposition of Issue #9239

`objective_disposition` answers the degenerate-objective question with data rather than
policy: ranking on carry is `unavailable-uncalibrated` without evidence,
`unavailable-outside-qualified-regime` when the strike is not covered, `degenerate-target`
when the qualified model's nominal carry falls outside the window at the requested target,
and `supported` otherwise. Until a qualification exists every target is unavailable, which
is the scientifically supported disposition today.

## What the Three-Camera Rig Can and Cannot Measure

`THREE_CAMERA_RIG_CAPABILITY` records, from the measured constraints in
[`usb_camera_rig_bringup.md`](../motion_capture/usb_camera_rig_bringup.md), that the lab's
three global-shutter cameras at 1920x1200 and 60 fps can resolve ball launch speed and
direction over a calibrated volume and the divot geometry afterwards, and cannot resolve
club pose during the five-millisecond engagement, ball spin, ejecta motion, or head force.
The report renders this table beside the verdicts. No camera purchase and no experiment
is assumed complete.

## What This Does Not Do

It does not validate anything today. It does not move the solver's own verdict, the 17×
speed exceedance, or `MAX_VALIDATED_SPEED_M_S`. It does not enter a fitted parameter
uncertainty into the design-ranking budget of #9243; that wiring follows once a
qualification exists. It does not load strokes from files: a stroke needs the solver's
verdict for its delivery, so intake is programmatic and the raw-data digest is what ties
each stroke back to its capture.
