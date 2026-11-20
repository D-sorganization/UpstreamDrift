# Preset Fit Path for the Tour-Average Driver (#15296)

This note records how the repository fits the full-body model to the driver
capture from the `tour_average_driver` subject preset, the exact commands, and
what one run of those commands gave. It is software-path evidence only: the
numbers below do not meet the address target of the epic and nothing here is a
physical acceptance. Machine-readable values are in
[`tour_average_driver_fit_summary.json`](tour_average_driver_fit_summary.json).

## Entry Points

1. `execution/spec_builder.py` (`main`, `compose_anthropometric_document`)
   builds the document. With `--subject-preset` it writes the preset's marker
   seeds into `marker_attachments` and its dropped labels into
   `subject.dropped_markers`.
2. `execution/driver.py` (`main`) or `pipeline/cli.py` (`main`) runs the fit.
   Both stop after the calibrated address under `--address-only` and write
   `address_report.json`.
3. `pipeline/cli.py::calibrate_run` loads the capture and calls
   `pipeline/marker_selection.py::select_markers`, which decides the fitted
   labels, the fixed upper-body attachments and the leg seeds.
4. `pipeline/address.py` solves the address pose (`solve_address_stage`, with
   the static trial under `--static-seeds`) and calibrates the eight leg marker
   offsets (`calibrate_legs`), starting from `Lane.leg_seeds()`.
5. `pipeline/cli.py` then searches the femur and tibia scales, solves the
   per-frame inverse kinematics, writes `ik_trajectory.npz`, tracks the motion
   in forward dynamics and writes `receipt.json`.

## What the Fit Reads From the Document

- **Leg seeds.** A document that carries offsets for all eight leg markers
  starts the leg calibration from them (`--leg-seeds document`, the default).
  `--leg-seeds builtin` uses the built-in `LEG_SEEDS` instead. A document
  seeding only some of the eight is refused.
- **Dropped markers.** Labels in `subject.dropped_markers` are left out of the
  attachments and of the fitted labels. A label that is not in the capture, or
  that the leg calibration needs, is refused.
- **Club sole.** `club.sole_point_m` is the club-body point the address solve
  rests on the ground: the lowest point of the head ellipsoid in the preset's
  address lie (`club_models.sole_point_m` from `club_shaft_elevation_deg` and
  `club_head_roll_deg`). Without it the pin stays the club-body origin.
- **Unchanged default.** The committed documents carry none of these, so their
  selection and receipts are unchanged; any other selection is recorded as
  `address.marker_selection` in the receipt.

## Reproduction Commands

Run from the repository root with the project interpreter, on `main` at
`669e075e8c` plus this change. `OUT` is any scratch folder.

```bash
python3 -m src.shared.python.motion_matching.execution.spec_builder \
  --native docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json \
  --osim src/engines/physics_engines/opensim/models/golf_humanoid.osim \
  --native-candidate docs/development/full_body_models/evidence/native_candidates/returned81_candidate.json \
  --subject-preset tour_average_driver --output "$OUT/build"

MUJOCO_GL=egl python3 -m src.shared.python.motion_matching.execution.driver \
  --spec "$OUT/build/full_body_spec_anthro_driver.json" --capture driver \
  --skip-hip-calibration --static-seeds --fix-club-markers --out "$OUT/fit"
```

The run used all 654 frames (no subsampling) and one process. It took 3168 s
on a heavily loaded four-core host; the same command before this change took
1192 s on the same host.

## Results

Marker RMS in millimetres from `ik_trajectory.npz` (`errors_m` where `valid`),
over frame 0, frames 0 to 388 (top), 0 to 476 (impact) and all 654 frames.
"Experiment" is the multi-frame bundle-adjustment fit made outside the
repository; no repository command reproduces it.

| Region           | Address | To Top | To Impact | Whole Swing | Experiment (Same Four) |
| ---------------- | ------- | ------ | --------- | ----------- | ---------------------- |
| All 34 markers   | 15.7    | 24.4   | 26.6      | 29.7        | 7.7 / 7.9 / 9.4 / 11.5 |
| Pelvis           | 16.3    | 16.3   | 16.8      | 20.2        | 3.2 / 5.4 / 6.0 / 7.5  |
| Trunk            | 5.8     | 25.4   | 27.8      | 32.6        | 7.6 / 9.0 / 10.6 / 12.4 |
| Head             | 1.0     | 18.8   | 19.8      | 26.0        | 16.1 / 12.2 / 11.4 / 18.0 |
| Arms             | 15.9    | 30.9   | 33.1      | 37.2        | 8.9 / 8.5 / 10.6 / 12.8 |
| Club             | 22.8    | 26.6   | 31.4      | 32.6        | 5.9 / 7.2 / 10.6 / 11.4 |
| Thigh            | 14.6    | 25.1   | 26.1      | 23.7        | 3.0 / 6.7 / 8.0 / 8.9  |
| Shank            | 6.6     | 9.5    | 10.4      | 13.4        | 1.6 / 3.4 / 4.2 / 5.2  |
| Foot             | 16.2    | 17.4   | 17.9      | 22.4        | 2.4 / 6.5 / 7.0 / 7.6  |

The address stage itself reports 34.2 mm (club segment 68.7 mm); forward
dynamics tracks at 109.5 mm. With the built-in leg seeds (the code before this
change, same document and flags) the same four numbers were 15.0, 22.1, 24.3
and 27.6 mm, and the address stage 31.8 mm: the preset's leg seeds are now
honoured, but they do not improve this pipeline's fit.

Address joint angles at frame 0 (degrees): knee −18.4 left, −20.5 right; hip
flexion 31.1, 36.9; elbow −18.8 lead, −15.5 trail; wrist X −35.9, −32.3; wrist
Y 30.6, −28.1. The club head body origin is 44.6 mm above the ground at frame
0 and 86.9 mm at impact. A coordinate sits within 0.1 degrees of a limit in
65.7 % of frames (53.0 % to impact, none at address); with the built-in seeds
it was 100 %.

## Address Stage With the Sole Pin

Two `pipeline.cli --address-only` runs with the flags above, on this change
merged with `main` at `c2161d36e0`: the preset document, and the same document
without `club.sole_point_m` (the origin pin). Marker RMS in millimetres from
`address_report.json`; the whole-swing table above was not rerun.

| Stage or Segment        | Origin Pin | Sole Pin   |
| ----------------------- | ---------- | ---------- |
| Static trial, all       | 30.8       | 13.0       |
| Calibrated address, all | 31.6       | 16.0       |
| Club                    | 67.8       | 23.5       |
| Pelvis                  | 6.8        | 14.6       |
| Trunk                   | 3.7        | 9.3        |
| Head                    | 3.7        | 7.1        |
| Left arm, right arm     | 19.4, 22.4 | 15.7, 18.0 |
| Left leg, right leg     | 6.9, 16.2  | 9.2, 16.2  |

With the sole pin the sole is 0.04 mm above the ground, the head centre 43.4 mm
(45.4 mm in the registration lie) and the club-body origin 74.2 mm. The club is
still 23.5 mm off and the pelvis, trunk and head fit worse: target not met.

## Why the Repository Fit Differs From the Experiment

- The experiment frees all 28 body-marker offsets in a 35-frame bundle
  adjustment with pose priors. The pipeline fixes the upper-body offsets from
  one address frame (static trial) and calibrates only the eight leg offsets.
- The experiment used the coupled-trunk model with the skinned-hand wrist; the
  preset document builds the earlier trunk.
- The whole-swing run above pinned the club-body origin to the ground as the
  sole. It is 64 mm from the head centre (the `Clubhead` frame origin) across
  the shaft and, in the preset's club registration, above it at address, which
  forced the club about 68 mm off its markers; the sole point is pinned now.
- The experiment's final leg lengths are not in the preset on `main`.

## Bundle Adjustment Stage

`pipeline/bundle_adjustment.py` holds the multi-frame marker-offset adjustment
as an engine-free core (`solve_bundle`: frame-wise elimination of the pose
blocks, exact offset derivatives, loop-closure and ground residuals, an inner
pose settle per frame). `pipeline/bundle_stage.py` builds its problem from a
pipeline lane. The fit commands take it as an opt-in stage after the leg
calibration; a run without `--bundle-adjust` is unchanged. Its receipt block is
`address.bundle_adjustment`.

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--bundle-adjust` | off | Run the stage; the swing solve uses its attachments |
| `--bundle-stride`, `--bundle-anchor-frames` | 20, none | Every n-th frame plus named frames |
| `--bundle-address-repeats` | 1 | Times the address frame enters the problem |
| `--bundle-hold` | none | Label fragments of markers kept at their calibrated seeds |
| `--bundle-held-out` | none | Markers left out of the start fit and the adjustment, then scored |
| `--bundle-start` | `strided` | `consecutive` starts from a full consecutive solve |
| `--bundle-pose-prior W SPIN` | lane prior | Pose prior weight per radian, and for the spin coordinates |
| `--bundle-offset-reach-mm` | three sigma | Hard bound on an offset's shift |
| `--bundle-lane-weights` | off | Use the lane's marker weights |
| `--bundle-plant-mm` | off | Sigma of a stance-sphere slide penalty inside the adjustment |
| `--ik-iterations` | solver's 60 | Iteration cap of each full-capture pose solve |
| `--ik-spin-prior-weight` | 0.1 | Previous-frame tie of the six arm spin coordinates |
| `--ik-free-stance-slide` | off | Stance spheres stay on the ground but are not anchored |
| `--ik-girdle-range-deg` | document ranges | Symmetric bound of the four girdle coordinates |

The coupled-trunk `full-body-v2` document with skinned-hand wrists is derived
from the preset build with `derive_showpiece_v2` and
`apply_skinned_hand_wrists` (set `subject.hand_visual_backend` to
`skinned_human` between the two) and passed as `--spec`.

## Bundle Adjustment Results

One run each on one host, all 654 frames, software-path evidence only. Marker
RMS in millimetres over the same four spans as above; "BA" is the adjustment's
own RMS over its frames; "Bound" counts offset components on their bound;
"Shift" is the largest marker shift from its calibrated seed; "Dyn" is the
forward-dynamics tracking RMS. Every flagged run adds
`--bundle-adjust --bundle-anchor-frames 391 476`; every adjustment converged.

| Model | Extra Flags | Address | To Top | To Impact | Whole | BA | Bound | Shift | Dyn |
| ----- | ----------- | ------- | ------ | --------- | ----- | --- | ----- | ----- | --- |
| Preset | No `--bundle-adjust` | 14.8 | 23.1 | 25.5 | 28.4 | – | – | – | 106 |
| Preset | None | 15.6 | 14.9 | 16.6 | 19.2 | 14.5 | 9 | 86 | 90 |
| Preset | `--bundle-pose-prior 0.1 3` | 15.2 | 14.3 | 16.0 | 18.8 | 14.7 | 9 | 83 | 90 |
| Preset | `--bundle-start consecutive` | 15.6 | 14.9 | 16.6 | 19.2 | 14.5 | 9 | 86 | 90 |
| Preset | `--bundle-offset-reach-mm 35` | 16.2 | 16.0 | 17.7 | 20.6 | 16.0 | 10 | 56 | 91 |
| Preset | `--bundle-hold Knee Ankle Toe` | 13.9 | 14.4 | 16.9 | 19.0 | 15.5 | 8 | 83 | 117 |
| Preset | The same and `--bundle-lane-weights` | 13.9 | 14.4 | 16.9 | 19.0 | 15.5 | 8 | 83 | 117 |
| Preset | The same hold and `--bundle-address-repeats 8` | 12.6 | 14.3 | 16.9 | 19.1 | 14.9 | 9 | 85 | 116 |
| Preset | `--bundle-hold Ankle Toe --bundle-address-repeats 8` (H8) | 12.8 | 13.9 | 16.5 | 18.9 | 14.7 | 9 | 86 | 105 |
| v2 | No `--bundle-adjust` | 7.0 | 14.8 | 17.5 | 20.8 | – | – | – | 104 |
| v2 | No `--bundle-adjust`, `--ik-free-stance-slide` | 5.9 | 12.4 | 14.9 | 18.1 | – | – | – | 56 |
| v2 | H8 | 9.5 | 10.6 | 13.1 | 15.0 | 10.0 | 0 | 75 | 97 |
| v2 | H8, `--ik-iterations 200` | 9.5 | 10.4 | 12.8 | 14.9 | 10.0 | 0 | 75 | 98 |
| v2 | H8, `--ik-iterations 200 --ik-spin-prior-weight 0.001` | 9.1 | 10.4 | 12.7 | 14.5 | 10.0 | 0 | 75 | 70 |
| v2 | Hold, sixteen repeats, `--bundle-pose-prior 0.1 3` | 9.3 | 10.3 | 12.8 | 15.0 | 9.7 | 0 | 69 | 91 |
| v2 | H8, `--ik-free-stance-slide` | 8.6 | 9.8 | 11.3 | 13.1 | 10.0 | 0 | 75 | 63 |
| v2 | H8, slide, `--ik-spin-prior-weight 0.001` (S) | 8.2 | 9.7 | 11.0 | 12.8 | 10.0 | 0 | 75 | 67 |
| v2 | S, `--ik-girdle-range-deg 15` | 8.2 | 9.8 | 11.6 | 14.2 | 10.4 | 0 | 66 | 70 |
| v2 | S, `--ik-girdle-range-deg 12` | 8.2 | 10.0 | 11.7 | 14.6 | 10.6 | 0 | 62 | 70 |
| v2 | S, `--ik-girdle-range-deg 10` | 8.3 | 10.2 | 12.0 | 15.0 | 10.8 | 0 | 65 | 71 |
| v2 | S, `--ik-girdle-range-deg 8` | 8.3 | 10.5 | 12.2 | 15.2 | 11.0 | 0 | 67 | 71 |
| v2 | H8, `--ik-girdle-range-deg 12` | 9.6 | 10.8 | 13.2 | 16.3 | 10.6 | 0 | 62 | 81 |

Held-out markers (`BackTop`, `RKneeOut`, `LUArmHigh`, left out of the start
fit too; anchored feet, hold and eight repeats). "Seed" scores the marker at
its seed attachment; "Scatter" is the seed-free spread of the marker in its
body frame over the adjustment's frames. Start to adjusted, millimetres:

| Model | Marker | Seed | Scatter |
| ----- | ------ | ---- | ------- |
| Preset | `BackTop` | 28.6 to 79.0 | 24.4 to 16.6 |
| Preset | `RKneeOut` | 30.9 to 76.4 | 29.5 to 44.7 |
| Preset | `LUArmHigh` | 46.1 to 59.0 | 27.7 to 23.7 |
| v2 | `BackTop` | 30.3 to 62.5 | 24.5 to 13.7 |
| v2 | `RKneeOut` | 38.5 to 112.2 | 38.2 to 70.5 |
| v2 | `LUArmHigh` | 48.0 to 49.3 | 32.8 to 18.9 |

The seed score worsens for every held-out marker: its neighbours move in
common mode and its seed does not. Seed-free, the trunk and arm markers
improve; `RKneeOut` is the only marker on its thigh, so holding it out leaves
that segment unobserved. The held-out criterion is therefore not met as
stated, and the stage stays off by default.

What the runs show:

- The model is the largest term (18.9 to 15.0 mm), then the stance anchor
  (15.0 to 12.8 mm). The pose prior, the start poses, the iteration cap and
  the spin prior are each under 0.5 mm.
- A 35 mm offset reach made the fit worse and put more offsets on the bound.
- Free ankle and toe markers cost the shank about 10 mm in the swing solve:
  the feet hardly move, so a foot marker's offset and the foot's position are
  not separable. They are held (`--bundle-hold Ankle Toe`).
- With the stance anchor off, the largest slide of a sphere while in stance
  is 5 to 25 mm to the top, 12 to 28 mm from the top to impact and 22 to 47 mm
  after impact (trail toe largest); the lowest sphere stays between -4.1 and
  0.0 mm. The anchored fit is the one to use where planted feet matter.
- Address with the adjustment (8.2 mm) is above the same model without it
  (5.9 mm): the static trial fits the address frame alone.
- The experiment (7.7 / 7.9 / 9.4 / 11.5 mm) is approached, not reached; the
  remaining gap is largest in the arms (15.7 against 12.8 mm) and the feet.
- Forward-dynamics tracking stays at 56 to 117 mm in every run: these tables
  are the kinematic fit.

## Shoulder-Girdle Excursion

Displacement of the shoulder joint from where the neutral girdle puts it
(180 mm lever), from the fitted girdle coordinates; v2 model. S is the
lowest-RMS flag set of the table above.

| Run | Left: Frame 0, Range, Mean (mm) | Right: Frame 0, Range, Mean (mm) | `RScapInputY` Range (deg) | Whole Swing (mm) |
| --- | ------------------------------- | -------------------------------- | ------------------------- | ---------------- |
| No `--bundle-adjust` | 19, 3 to 81, 23 | 4, 3 to 70, 24 | -13.6 to 5.9 | 20.8 |
| H8 | 37, 35 to 91, 56 | 110, 94 to 136, 106 | -44.0 to -29.9 | 15.0 |
| S | 36, 35 to 94, 57 | 107, 83 to 138, 105 | -44.7 to -26.5 | 12.8 |
| S, range 15 | 47, 27 to 66, 55 | 42, 27 to 66, 46 | -15.0 to -7.0 | 14.2 |
| S, range 12 | 38, 19 to 53, 44 | 36, 21 to 53, 38 | -12.0 to -5.5 | 14.6 |
| S, range 10 | 31, 23 to 44, 37 | 30, 15 to 44, 31 | -10.0 to -4.0 | 15.0 |
| S, range 8 | 25, 19 to 35, 29 | 25, 9 to 35, 24 | -8.0 to -2.4 | 15.2 |

With the adjustment and the document girdle ranges the trail girdle sits near
-35 degrees for the whole swing: it absorbs the static gap between the skeleton
and the shoulder markers, which the address stage (girdle locked at neutral)
does not allow. `--ik-girdle-range-deg` bounds it. The girdle then rides the
bound (at 8 degrees: `LScapInputX` 205, `LScapInputY` 494, `RScapInputX` 88
and `RScapInputY` 166 of 654 frames), the offsets do not grow, and the cost
shows in the trunk, head and arm markers after impact (to impact 11.0 to
12.2 mm, whole swing 12.8 to 15.2 mm at 8 degrees). The capture asks for more
protraction than a bounded girdle gives; whether that is the subject's posture
or the model's shoulder geometry is open.

## Bundle Adjustment Against the Experiment, Term by Term

The experiment's 7.7 / 9.4 / 11.5 mm came from a 35-frame bundle adjustment
followed by a per-frame marker fit of the whole swing. The same problem as the
repository stage (`pipeline/bundle_stage.py`) sets it up:

| Term | Experiment | Repository Stage | Effect Measured Here |
| ---- | ---------- | ---------------- | -------------------- |
| Model | Coupled-trunk `full-body-v2` with skinned-hand wrists, derived from the preset document | Whatever `--spec` names; the preset document is the earlier trunk | Largest single term: 18.9 to 15.0 mm whole swing with the derived document |
| Prior centre of an offset | Seeds computed from the model's own joint positions | Attachments after the static trial and the leg calibration, already 50 to 127 mm from the document seeds | Not isolated |
| Offset sigmas, bound | Same table, three sigma | Same table (head 40 mm instead of 25 / 20 / 25), three sigma | A 35 mm reach made the fit worse (`--bundle-offset-reach-mm 35`) |
| Marker weights | Head 0.1, one shoulder marker 0.3, others 1 | All 1; `--bundle-lane-weights` applies the lane's, which are all 1 in this lane | None |
| Frames | Every 20th, top and impact, the address frame eight times | Every 20th plus `--bundle-anchor-frames`; `--bundle-address-repeats` | Eight repeats: address 13.9 to 12.6 mm; sixteen with the pose prior: no further gain |
| Address pose prior | Elbow flexion and humeral rotation targets on the address frame | None | Not ported |
| Pose prior | 0.1 per radian (spin 3.0) about the current iterate: zero at the solution | The lane's joint prior about the start poses; `--bundle-pose-prior 0.1 3` | 19.2 to 18.8 mm whole swing |
| Start poses | An earlier whole-swing fit | The lane's strided inverse kinematics; `--bundle-start consecutive` | No change (identical to 0.1 mm) |
| Solver | Inner pose fit to convergence per frame, Gauss-Newton on the offsets, stop at a relative gain of 1e-4 | Schur-complement Levenberg-Marquardt with the same inner pose settling, stop at 1e-6 | Converges in about a dozen outer steps; before the settling it had not in 200 |
| Residual scales | Marker 10 mm, weld 0.5 mm and 5 mrad, ground 5 mm | The same | None |
| Leg markers | Free, with the stance spheres on the ground | Free by default; `--bundle-hold` keeps named ones | Freeing ankle and toe markers cost the shank about 10 mm in the swing solve |
| Club markers | Fixed to the club registration | Fixed (`--fix-club-markers`), sole point on the ground at address | None |
| Whole-swing fit after it | Raw per-frame marker fit with the experiment's own solver and weights | The pipeline's trajectory solve: planted feet, face-orientation target, joint prior, then smoothing and dynamics | The stance anchor (100 times stiffer than a marker) costs 4 to 13 mm from the top onward: `--ik-free-stance-slide` gives 15.0 to 12.8 mm whole swing. The iteration cap (`--ik-iterations 200`) is worth 0.1 mm, the spin prior (`--ik-spin-prior-weight 0.001`) 0.4 mm |

The experiment's own offsets ended 21 to 98 mm from its seeds with three
components on a bound, so offsets of that size are not a defect of the port:
they are whole clusters moving together, which is how either fit absorbs the
gap between the skeleton and the subject.

## Open Items

- Rerun the whole pipeline with the sole pin; the swing table predates it.
- The sole point is exact only in the registration lie, whose roll about the
  shaft is a setting the club markers cannot identify.
- The bundle-adjusted fit below does not reach the experiment; see the two
  sections that follow for what is measured and what is left.
- `LATERAL_SEEDS_M` in the hip calibration is a literal that does not follow
  the preset's leg seeds.
