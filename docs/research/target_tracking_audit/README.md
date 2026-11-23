# Controller Target Tracking Audit

The same-input closed-loop receipt reports parity against its recorded engine
rollout separately from the controller's actual position target. The target
is the linearly sampled `q_track` at each rollout time on the reference clock.
For each coordinate, the report gives sampled RMS and maximum absolute error
in that coordinate's native units, in the recorded coordinate order. It does
not combine rotational and translational units or infer anatomical angles.
The receipt separately compares velocity against the controller-derived rate
table: finite differences of the position target, split at the recorded
impact when specified and sampled with the controller's same side clamp.
These rates are not measured velocities or derivatives of the piecewise
linear position interpolant. Velocity units are native coordinate units per
second. Hash binding applies to the source position record for both reports.

The tracking record must match the hash in the bundle provenance. The run's
model, with the recorded shaft configuration applied, must also match the
bundle model before replay; JSON formatting does not affect this comparison.
These checks do not establish that the record was generated from that model,
its calibration, a valid model migration or an accepted fit. The report does not
qualify motion between samples, actuator torque, contact, or physical limits.
The calculation reference is [target_tracking_audit.tex](target_tracking_audit.tex).
Run `python -m pytest -q tests/unit/motion_matching/same_input/test_bundle_cli.py`
with the repository headless settings to check the report contract.

The fixed-step rollout rounds its duration to the nearest 1 ms. When that
rounding places the final state at most half a step beyond the source target,
the receipt samples the controller's clamped endpoint and reports the
clamped duration separately. Longer uncovered spans remain invalid.

For a recorded settled lead-in, the audit repeats the lead-in before scoring
the bundle window and evaluates both targets at their original clock offset.
Only the existing no-impact fixed-step lead-in contract is supported; invalid
or inconsistent metadata is rejected rather than silently scoring the wrong
segment. This does not independently establish the lead-in state provenance.
