# Manufactured Nonlinear-Controller Solver Qualification

## Purpose

This qualification tests one prospective solver kernel on a deterministic
nonlinear fixture without executing the registered double-pendulum evaluation
grid. It establishes limited numerical-mechanics evidence, not golf
performance.

## Qualified Scope

The committed
[`nonlinear_controller_solver_qualification.json`](data/nonlinear_controller_solver_qualification.json)
binds the result to the frozen comparison registration, implementation sources,
and `requirements.lock`. The fixture is a damped nonlinear pendulum with a
40 ms explicit step, 24 control intervals, one input bounded to
`[-0.40, 0.40]`, and a declared quadratic tracking objective.

One kernel is exercised:

- projected first-order iLQR using reset-safe central derivatives and box
  projection inside each rollout.

The kernel reduces the declared cost through nonincreasing accepted iterations,
enforces the bound, reproduces states, controls, cost, and status on
exact replay, and passes the registered cold/warm sensitivity limit. An
independent directional difference checks the local dynamics derivatives. A
nonfinite dynamics result returns `dynamics_failure` without a state or control
trajectory.

The `bounded_nmpc_collocation` identity fails closed as unimplemented. A
box-bounded shooting optimizer is not relabeled as collocation NMPC.

## Falsification and Remaining Gates

The validator fails if the committed report differs from recomputation,
derivatives disagree, accepted costs increase, a bound is violated, replay
changes, initialization sensitivity exceeds its threshold, a failed rollout
fabricates a trajectory, or the kernel becomes ranking-eligible. The report
records zero double-pendulum evaluations and zero eligible methods.

The kernels still require plant-specific derivative checks, typed event and
integration outcomes, matched replay, outcome-blind tuning, held-out execution,
and adequate optimality evidence. This result neither compares clubhead speed
nor identifies human control, anatomy, fatigue, safety, passive torque, or a
coaching strategy.

## Reproduction

```powershell
python -m scripts.research.proximal_distal_energy.nonlinear_controller_qualification validate
python -m pytest -q -n 0 tests/research/test_nonlinear_controller_solver_qualification.py
```
