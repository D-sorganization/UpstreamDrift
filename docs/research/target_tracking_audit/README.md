# Controller Target Tracking Audit

The same-input closed-loop receipt reports parity against its recorded engine
rollout separately from the controller's actual position target. The target
is the linearly sampled `q_track` at each rollout time on the reference clock.
For each coordinate, the report gives sampled RMS and maximum absolute error
in that coordinate's native units, in the recorded coordinate order. It does
not combine rotational and translational units or infer anatomical angles.

The tracking record must match the hash in the bundle provenance. This binds
the selected data to the bundle; a hash does not establish its source,
calibration, model compatibility or an accepted fit. The report does not
qualify motion between samples, actuator torque, contact, or physical limits.
The calculation reference is [target_tracking_audit.tex](target_tracking_audit.tex).
Run `python -m pytest -q tests/unit/motion_matching/same_input/test_bundle_cli.py`
with the repository headless settings to check the report contract.

The fixed-step rollout rounds its duration to the nearest 1 ms. When that
rounding places the final state at most half a step beyond the source target,
the receipt samples the controller's clamped endpoint and reports the
clamped duration separately. Longer uncovered spans remain invalid.
