# Nonlinear-Controller Plant Transport Qualification

## Purpose

This qualification defines the numerical boundary between prospective
controller solvers and the canonical analytical golf double pendulum. It tests
state order, control order, units, integration, replay, and input handling. It
does not optimize a controller or report a golf-performance outcome.

## Registered Map

The controller-facing state is shoulder angle, wrist-relative angle, shoulder
rate, and wrist-relative rate. The inputs are shoulder and wrist joint torques.
Constant-control RK4 steps are checked independently at 0.5, 1, and 2 ms.

The implementation in
[`nonlinear_controller_plant_transport.py`](../../../scripts/research/proximal_distal_energy/nonlinear_controller_plant_transport.py)
uses the canonical parameter authority and analytical control-affine equations.
The committed
[`nonlinear_controller_plant_transport.json`](data/nonlinear_controller_plant_transport.json)
binds the exact parameters, source files, environment lock, and comparison
registration by SHA-256 digest.

## Parity and Negative Controls

Four states and controls span the top-of-backswing state, zero command,
positive and opposing torques, high forward rates, and adverse reverse rates.
Across all three step sizes, the 12 controller-facing steps match the canonical
ODE backend within the `1e-12` gate. Repeated calls are exact, and neither input
array is mutated. Wrong-sized or nonfinite inputs raise `ValueError` without a
trajectory.

This is code-path parity, not independent physics validation: both maps use the
same equations and parameter authority. An independently implemented plant is
still required before broader robustness is claimed.

## Remaining Falsification Gates

The milestone records zero controller evaluations and zero ranking-eligible
methods. Outcome-blind tuning, typed plant outcomes, matched solver replay,
frozen held-out execution, event-retention accounting, failure-region mapping,
and adequate optimality evidence remain open. Any failed gate suppresses
ranking.

The result cannot identify human control, intention, anatomy, fatigue,
strength, safety, passive biological torque, or technique. It does not show
that either solver increases clubhead speed.

## Reproduction

```powershell
python -m scripts.research.proximal_distal_energy.nonlinear_controller_plant_transport validate
python -m pytest -q -n 0 tests/research/test_nonlinear_controller_plant_transport.py
```
