# Prospective Nonlinear-Controller Comparison Registration

## Purpose

This registration freezes the next matched controller-comparison protocol
before nonlinear solver choices can be tuned against evaluation outcomes. It
creates no controller-performance result and supports no ranking, human-control
inference, or coaching recommendation.

## Current Parent Authorities

The committed
[`nonlinear_controller_comparison_registration.json`](data/nonlinear_controller_comparison_registration.json)
digest-binds three current authorities: trajectory-varying control authority,
bounded event reachability, and channel-specific event-topology robustness.
The prospective comparison separately fixes the analytical ODE plant, four
state coordinates, two torque coordinates, scaling, a 1 ms integration step,
a 400-step horizon, actuator bounds, the positive transverse delivery guard,
and a common random-stream rule.

The 24 evaluation trials are outcome-blind scaled-state perturbations. Eight
diagonal tuning trials are disjoint from them. Solver, horizon, scaling, and
objective choices freeze before the evaluation set is executed. Delivery
speed is an outcome rather than the tuning objective.

## Controller Families and Ranking Suppression

Nine families are named so absence cannot be mistaken for evidence. Open-loop,
LTV, adverse-observation LTV, and zero-command controls are reference families
pending a matched current-parent run. Projected first-order iLQR is prospective
pending current-parent qualification. Bounded collocation NMPC, second-order
DDP, risk-sensitive control, and scenario-stochastic MPC remain unimplemented;
their names are protocol placeholders, not solver evidence.

Every family has `eligible_for_ranking: false`. A ranking remains prohibited if
any comparability, adequacy, replay, convergence, optimality, event, or held-out
gate fails. Solver, integration, event-loss, and gate failures stay typed; none
may be converted to a terminal score.

## Execution Boundary

Any later run is limited to one worker and must checkpoint after every
controller--trial pair. Resume requires an exact identity match over source,
registration, parent evidence, environment lock, solver, objective, and trial.
The registration command launches no campaign. Execution requires a separate
operator decision after the protected parent stack is verified on remote
`main`.

Even a qualified local comparison would apply only to the declared analytical
model and finite grid. It could not establish global optimality, participant
control, motor intent, anatomical feasibility, passive biological torque,
injury risk, fatigue response, or a preferred technique.

## Reproduction

```powershell
python -m scripts.research.proximal_distal_energy.nonlinear_controller_registration validate
python -m pytest -q -n 0 tests/research/test_nonlinear_controller_comparison_registration.py
```
