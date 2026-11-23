# Sampled Optimization Kinematic Audit

Refs #15493, Slice 3. This research diagnostic reads the existing Solution
Bundle codec and audits actual recorded angular positions and velocities
against explicit ROM, speed and tracking limits. It does not run an optimizer
or qualify a swing. The editable calculation reference is
[`kinematic_audit.tex`](kinematic_audit.tex); this is separate from the governed
engineering design manual.

## Inputs and Conventions

Run `python -m scripts.audit_optimization_kinematics BUNDLE LIMITS.json` from
the repository root. The codec verifies the bundle archive integrity. The
limits JSON must contain these fields (each vector has one entry per joint):

| Field | Meaning |
| --- | --- |
| joint_order | Exact bundle joint names and order |
| position_lower_rad | Minimum actual angular position |
| position_upper_rad | Maximum actual angular position |
| speed_limit_rad_s | Maximum absolute actual angular velocity |
| position_error_limit_rad | Maximum absolute position tracking error |
| velocity_error_limit_rad_s | Maximum absolute velocity tracking error |

Limits must be finite; speed/error budgets must be nonnegative. The caller
must supply scientifically justified bounds and ensure all included
coordinates are angular, in the same frame, sign and unwrapped radian
convention. Mixed translation/quaternion coordinates require a separate
coordinate-aware analysis and must not be passed to this angular-only tool.
No anatomical bounds or periodic wrapping are inferred from joint names.

## Equations and Interpretation

For each joint, the ROM margin is the minimum of `q - lower` and `upper - q`
over samples; speed margin is `speed_limit - max(abs(qd))`. Tracking margins
are the corresponding absolute-error budget minus the maximum absolute
actual-minus-reference difference. The audit uses actual rollout q/qd, so a
feasible reference cannot hide actual ROM/speed violations. Signed negative
margins indicate violations; zero is inclusive. Every margin must be
nonnegative for sampled_feasible to be true. No epsilon is added.

Output is strict JSON, with solution identity, joint order, sample count and
four per-joint margin vectors. Exit codes: 0 sampled feasible, 1 violation,
2 invalid input. At least two strictly increasing finite sample times and
finite shape-aligned trajectories are required. Overflow fails closed.

## Evidence and Limits

Reproduce with `py -3.12 -m pytest -o addopts='' tests/unit/scripts/test_audit_optimization_kinematics.py -q`.
The RED run failed collection because the audit module did not exist; GREEN
tests include hand-calculated asymmetric ROM and speed violations, inclusive
boundaries, independent tracking budgets, malformed data, overflow and CLI
round trips through the real bundle codec.

This is a sampled diagnostic, not a continuous-time or biomechanical safety
certificate. Inter-sample excursions, frame/joint semantics, controller
stability, applied effort/work and independent forward replay need their own
evidence. This audit does not change feasibility fields stored in the bundle.
