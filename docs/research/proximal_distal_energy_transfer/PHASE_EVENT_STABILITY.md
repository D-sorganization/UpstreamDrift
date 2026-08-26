# Phase/Event Sensitivity and Finite-Time Stability

## Purpose and Scope

This milestone separates three questions that are often conflated:

1. how a local state perturbation is amplified over a finite downswing window;
2. how the time of a transverse geometric event changes with initial state; and
3. whether a trajectory is periodic enough to authorize Floquet analysis.

The registered model is the exact analytical double pendulum under one
synthetic open-loop torque history. It is not a participant, anatomical,
neuromuscular, or coaching model.

## Registered Computation

The paper's 1 ms reference program is retained as provenance. The derivative
audit uses the same RK4 model and torque switch on a 0.125 ms grid. Every
one-step Jacobian central-differences the actual discrete RK4 map, and the
state-transition matrix is the ordered product of those step maps. Positive
state scales make gains dimensionless.

The machine-readable JSON reports the step-refinement residual to two
significant digits, the resolution supported across the registered Python
runtimes. The convergence decision is evaluated from the unrounded computation,
and the full-precision transition arrays remain available in the NPZ evidence;
the reporting projection therefore limits claimed numerical precision without
relaxing the registered $10^{-5}$ convergence gate.

The first positive crossing of
$h(\mathbf x)=\theta_s+\theta_w=0$ is interpolated on the common grid. Its
implicit event-time derivative is accepted only when
$|\nabla h^\mathsf T\mathbf f|>10^{-6}$ s$^{-1}$. Complete symmetric
perturbation rollouts provide a separate implementation control. A constructed
orthogonal guard supplies the near-grazing killswitch.

## Interpretation Boundary

Finite-time singular gains can exceed and fall below one simultaneously.
They describe direction-dependent local amplification over the registered
window; they do not establish asymptotic stability or a basin. Event-time
sensitivity is a model derivative, not neural timing demand. Because the
event state does not close onto the initial state, Floquet multipliers are
unavailable and intentionally absent from the evidence record.

## Reproduction

```powershell
python -m scripts.research.proximal_distal_energy.run_phase_event_stability validate
python -m scripts.research.proximal_distal_energy.make_phase_event_stability_figure
python -m pytest -n 0 -q tests/research/test_phase_event_stability.py tests/research/test_phase_event_stability_evidence.py
```

The machine-readable summary is `data/phase_event_stability.json`; full
transition, spectrum, and perturbation arrays are in
`data/phase_event_stability.npz`.
