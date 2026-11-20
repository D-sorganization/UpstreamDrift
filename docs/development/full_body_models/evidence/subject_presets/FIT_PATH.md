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
   The driver always runs the whole pipeline; only `pipeline.cli` honours
   `--address-only`.
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
- **Unchanged default.** The committed documents carry neither, so their
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

## Why the Repository Fit Differs From the Experiment

- The experiment frees all 28 body-marker offsets in a 35-frame bundle
  adjustment with pose priors. The pipeline fixes the upper-body offsets from
  one address frame (static trial) and calibrates only the eight leg offsets.
- The experiment used the coupled-trunk model with the skinned-hand wrist; the
  preset document builds the earlier trunk.
- The pipeline's address solve pins the `Clubhead` body origin to the ground as
  the club sole. The preset's club registration treats that origin as the head
  centre, about 50 mm above the ground, so the address club segment is forced
  about 68 mm off its markers.
- The experiment's final leg lengths are not in the preset on `main`.

## Open Items

- Decide the `Clubhead` origin convention (sole or head centre) for the address
  pin; it is the largest single address error in this run.
- Bring the multi-frame marker-offset adjustment into the repository if its
  accuracy is to be claimed from a repository command.
- `LATERAL_SEEDS_M` in the hip calibration is a literal that does not follow
  the preset's leg seeds, and `execution.driver` ignores `--address-only`.
