# Motor Solution Bundle Research

This research checkpoint connects the merged JSON/NPZ Solution Bundle codec to
an existing MuJoCo fixed-gain motor-stage replay. The no-impact representation
uses explicit schema version 2; version 1 events can be explicitly migrated
without inventing an event. Readers reject unsupported versions, duplicate
JSON/NPZ keys, and non-finite or nonnumeric extra arrays.

The bridge accepts one fixed ZOH hold per step, a rigid scalar-coordinate
plant, optional fixed-step closure projection, and either no impact or an explicitly recorded latched force. Flight is unsupported. The bundle carries
model JSON, motor XML, controller JSON and the typed motor-stage NPZ as byte
arrays. A separately trusted manifest digest binds the NPZ digest and asset
hashes. The bridge checks the hash-bound configuration, model, coordinate
order, timing, controller references, gains, spline coefficients and recorded
boundary states/forces before accepting fresh numerical re-simulation.
Boundary torque arrays repeat the last held value at the final state; the
stage archive is the evidence for all numerical stage inputs.

The measured runtime fingerprint covers the bridge, codec and an existing
runtime subset; it is not a complete environment lock. The verifier assumes a
stable, trusted directory during loading. It provides no independent-runtime,
continuous-time, first-contact, model-origin, clinical, actuator-capacity or
physical-feasibility claim. Synthetic limits and exploratory reference data
are not admitted baselines. This oversized research branch is not a PR or a
complete Slice 2.

Focused headless validation (with the pinned `vendor/ud-tools` initialized):

```powershell
$env:QT_QPA_PLATFORM='offscreen'; $env:MPLBACKEND='Agg'; $env:MUJOCO_GL='disable'
python -m pytest -q tests/unit/optimization/test_solution_bundle.py tests/unit/motion_matching/same_input/test_mujoco_actuator_probe.py tests/unit/motion_matching/same_input/test_stage_numeric_archive.py tests/unit/motion_matching/same_input/test_solution_bundle_prototype.py
```

The editable calculation-level reference is [motor_solution_bundle.tex](motor_solution_bundle.tex).


The prescribed-force replay is not contact detection: its event is labeled
`prescribed_latched_force` with `contact_verified: false`. The force is mapped
through the existing point Jacobian and applied separately from post-saturation
motor force, so the external force is never clipped as motor control. The
actual split event state is re-simulated and compared with the bundle. The
projection uses the existing weld-closure projection at each completed step;
the archive binds both pre- and post-projection states. These checks establish
numerical replay only, not an on-face root or physical validity.
