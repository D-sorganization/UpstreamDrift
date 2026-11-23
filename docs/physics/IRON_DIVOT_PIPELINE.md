# Iron Divot Pipeline

The optional ground-contact input to `SwingBallFlightPipeline.run` connects
the heuristic divot model from PR 15545 to the existing impact solver and
flight simulator. It does not qualify engine-level turf contact or add a
second flight model. Refs #15523.

## Input and State Contract

Supply `divot_arc` and `turf_material` together. The swing's clubhead velocity
is the velocity before turf entry in global x-forward, y-left, z-up axes,
in metres per second. Do not supply a state already slowed by engine contact.
The arc uses metres from the ball toward the target and metres above the turf;
negative height cuts turf. Effective mass is `SwingState.clubhead_mass` in kg,
matching the subsequent impact solve. Omitting both inputs preserves the
existing pipeline. The caller's swing and velocity array are unchanged.

## Calculation and Limits

`compute_divot` retains its default 1.5 m arc radius, 0.07 m cut width,
401-point trapezoidal integration and hardness-scaled 200 kPa shear heuristic.
For turf work W and entry kinetic energy K = m v² / 2, speed retention is
sqrt(max(0, 1 - W/K)). The impact solver receives the original velocity vector
scaled by that retention, preserving direction. Its derived launch spin is
multiplied by the divot spin retention and the caller's existing spin factor
before the existing spin-envelope clamp. This models additional face
contamination heuristically; it is not a calibrated spin prediction.

The result's optional `divot` field records geometry, work and retentions;
`swing_state` remains the original input. With no pre-ball turf work, the
retentions are one. A club stopped by turf raises `ValueError` before impact
and flight; no fabricated trajectory is returned. This scalar model does not
predict changes in club direction, face rotation or angular velocity.

## Reproduction and Acceptance

Run `py -3.12 -m pytest -o addopts='' tests/unit/physics/test_divot_pipeline_15523.py
tests/unit/physics/test_divot_model_15523.py
tests/unit/physics/test_swing_ball_flight_pipeline_5337.py -q` on one line.
Tests verify solver input speed, composed spin retention, original-state
preservation, ball-first/no-contact identity, incomplete-context rejection
and stopped-club refusal. These are software contracts, not measured physical
validation. The supplementary LaTeX source is `iron_divot_pipeline.tex`.
