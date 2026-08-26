# Hybrid-System Contract and Qualification Boundary

The proximal-distal model ladder is registered as a hybrid constrained system
in [`data/hybrid_system_contract_v1.json`](data/hybrid_system_contract_v1.json).
The contract names the continuous state, controls, algebraic constraints,
contact modes, guards, resets, impacts, actuator dynamics, uncertain event
surfaces, and cross-tier observables for every rung of the eight-tier research
program.

Registration is not scientific promotion. The validator proves that the
declared topology is internally consistent and that implemented or partial
tiers point to existing source authorities. It does not prove observability,
controllability, parameter identifiability, stability, controller superiority,
participant validity, or a coaching recommendation.

## Current Authority Classification

| Model Tier                               | Authority Status | Permitted Interpretation                                     |
| ---------------------------------------- | ---------------- | ------------------------------------------------------------ |
| Analytical Double Pendulum               | Implemented      | Declared pointwise mechanics and analytical diagnostics only |
| Forward Two-Arm Constrained Planar Model | Implemented      | Declared synthetic forward comparisons                       |
| Moving Base With Compliant Grip and Club | Implemented      | Declared synthetic forward comparisons                       |
| Articulated Spatial Whole-Body Model     | Partial          | Registered spatial diagnostics within stated contact limits  |
| Neuromusculoskeletal Model               | Partial          | Surrogate redundancy and transmission diagnostics only       |
| Club-Ball Impact and Ball Flight         | Partial          | Separately qualified impact and flight diagnostics           |
| Participant-Calibrated Digital Twin      | Unavailable      | No comparison or participant inference                       |
| Governed Human Validation                | Unavailable      | No human-mechanism or coaching inference                     |

An unavailable tier remains a first-class record. Its modes, guards, reset,
observables, and decisive falsifier are named, but every absent component must
carry a specific reason and no source path may be claimed. This prevents a
synthetic or motion-only artifact from silently filling a human-data boundary.

## Referential and Numerical Gates

The executable validator fails when:

- a source path escapes the repository or does not exist;
- the eight tiers are missing, duplicated, or reordered;
- a mode, state block, guard, reset, impact, actuator, uncertainty surface, or
  observable identifier is duplicated;
- a guard names an unknown mode or uncertainty surface;
- a reset or impact names an unknown guard;
- a control names an unknown actuator, or an actuator names an unknown state;
- an unavailable or not-applicable component omits its reason;
- an unavailable tier claims an implementation path or comparison eligibility;
- a numerical tolerance or uncertainty bound is nonfinite or invalid.

The registered tolerances govern contract validation only. Individual models
retain their stricter equation, constraint, convergence, energy, and
cross-engine tolerances in their own experiment manifests.

## Reproducible Validation

From a clean UpstreamDrift checkout:

```powershell
python -m scripts.research.proximal_distal_energy.hybrid_system_contract validate
python -m pytest tests/research/test_hybrid_system_contract.py -q
```

The command reports the tier count and authority-status census. Tests also
apply semantic tampering to references, identifiers, reasons, and source paths
and require fail-closed rejection.

## Remaining Nonlinear-Control Work

This contract is the dependency for, not the result of, the remaining #9027
analyses. Subsequent protected slices must add:

1. model-tier-specific observability, controllability, constraint-rank,
   structural-identifiability, and practical-identifiability reports;
2. event-time sensitivity and finite-time stability methods only where their
   assumptions are satisfied;
3. matched open-loop, state-triggered, impedance, robust-control, stochastic,
   and optimal-control comparisons when each method is actually available;
4. separate speed, face/path, strike, balance, load, effort-proxy, variability,
   and robustness objectives with machine-readable dominance rules;
5. manufactured solutions, killswitches, adverse cases, cross-engine parity,
   held-out evaluation, and explicit ranking suppression.

Until those gates pass, the contract must not be cited as evidence that one
policy, timing pattern, torque allocation, or human strategy is preferable.
