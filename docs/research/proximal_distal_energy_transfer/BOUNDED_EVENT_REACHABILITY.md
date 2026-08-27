# Bounded Nonlinear Event Reachability

This evidence package tests whether small, finite event-surface targets can be
reached under declared torque and slew limits on the registered analytical
double-pendulum trajectory. It is the nonlinear follow-up to the local
first-order authority audit; it is not a human performance or coaching study.

## Registered Problem

- Initial state: $(-2.2,-1.57,0,0)$ in radians and radians per second.
- Event guard: first positive crossing of
  $\theta_s+\theta_{w,\mathrm{relative}}=0$.
- Horizon and base integration step: 0.40 s and 2 ms.
- State scales: $(\pi,\pi,10,10)$.
- Control scales: $(100,100)$ N m.
- Enabled-channel amplitude bound: $\pm20$ N m.
- Enabled-channel slew bound: 10,000 N m s$^{-1}$.
- Event-tangent residual tolerance: $2\times10^{-6}$.
- Solver: four-interval SLSQP multiple shooting with exact RK4 continuity and
  independent exact-RK4 replay.

The amplitude and slew limits are model-scenario bounds, not measured human
strength, activation, fatigue, safety, or torque-rate limits.

## Continuation and Killswitch Matrix

Seven symmetric event-tangent offsets ($0$, $\pm0.5$, $\pm1.0$, and
$\pm2.0$ mrad) are crossed with four matched channel masks: both channels,
shoulder only, wrist only, and zero authority. The resulting 28-case matrix is
retained in full, including failures.

- Both channels: 7/7 feasible.
- Shoulder only: 7/7 feasible.
- Wrist only: 7/7 feasible.
- Zero authority: the nominal target is feasible; all six displaced targets
  are infeasible.

Across continuation and the ten registered falsification controls, all 38
events are unique and transverse. Thirty-two replays are feasible and six are
infeasible. No numerical failure or replay rejection occurs. The largest
event-tangent residual among feasible replays is $8.83\times10^{-11}$.

## Falsification Controls

The study retains two multistarts, three shooting meshes, three integration
steps, and two adverse initial states. Three-, four-, and five-interval meshes;
1, 2, and 4 ms steps; and both adverse states converge and replay feasible for
the registered $+1$ mrad target.

The multistart objective comparison fails its optimality gate. The two
converged objectives are $2.05356\times10^{-8}$ and
$2.56595\times10^{-8}$, a 24.9517% relative spread against a 5% gate. This
does not invalidate the independently replayed feasibility outcomes, but it
does prevent minimum-effort, channel-superiority, or controller rankings.

## Evidence Boundary

The available result is bounded local model-scenario feasibility for one
synthetic trajectory and one same-bracket event family. The following remain
unavailable:

- global nonlinear reachability or crossing-topology maps;
- human torque, torque-rate, fatigue, safety, or activation interpretation;
- passive negative-torque or biological allocation inference;
- shoulder-versus-wrist controller ranking;
- delay/noise robustness outside the registered adverse states;
- bilateral contact, three-dimensional, or participant-held-out validation;
- coaching or universal strategy recommendations.

## Reproduction

```powershell
python -m scripts.research.proximal_distal_energy.run_bounded_event_reachability write
python -m scripts.research.proximal_distal_energy.run_bounded_event_reachability validate
python -m scripts.research.proximal_distal_energy.make_bounded_event_reachability_figure
python -m pytest -n 0 -q tests/research/test_bounded_event_reachability.py tests/research/test_bounded_event_multiple_shooting.py tests/research/test_bounded_event_reachability_study.py tests/research/test_bounded_event_reachability_evidence.py
```

The JSON report is the reviewer-facing record. The NPZ archive retains the
full-precision segment perturbations, multiple-shooting nodes, event states,
availability masks, and typed solver outcomes used by the evidence tests.
