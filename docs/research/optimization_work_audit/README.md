# Optimization Joint Work Audit

This standalone research reference documents the sampled mechanical-work
audit for optimization epic #15493. Editable equations and assumptions are in
[work_audit.tex](work_audit.tex); this is separate from the governed design manual.

Run `python3 -m scripts.audit_optimization_work /path/to/solution-bundle` from
the repository root. The existing Solution Bundle codec verifies the manifest
and array integrity. The JSON report preserves solution identity and joint order,
and reports signed, positive and negative work per joint in joules, together
with the sum of positive work across joints.

Inputs are total **applied** torque (including feedback after saturation) and
recorded joint velocity in matching generalized coordinates, SI units and order.
Commanded torque and feedforward residuals are not substitutes. Endpoint samples
must be physical power observations; omit terminal padding before using the
array API. This command audits the recorded bundle arrays without re-simulating.

Power is reconstructed linearly between samples. Exact zero crossings of that
reconstruction separate positive and negative work. Neither braking later in a
trajectory nor simultaneous absorption at another joint cancels positive work.
This is a numerical sampling convention, not a statement that torque and velocity
are both linear or that an actuator uses first-order hold. Unobserved transients,
impact impulses and work between samples remain unqualified. No physiological
safety, feasible motion or forward-replay acceptance is implied by this audit.

Reproduce analytic acceptance fixtures:
`python3 -m pytest tests/unit/scripts/test_audit_optimization_work.py -m unit --no-cov -q`.
They cover crossing power, nonuniform time, joint cancellation, zero endpoints,
refinement consistency and malformed or overflowing inputs. No recorded golfer
experiment or LaTeX compilation is claimed.
