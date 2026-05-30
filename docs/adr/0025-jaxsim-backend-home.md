# ADR-0025: JaxSim backend home and bridge contract

- Status: Proposed
- Date: 2026-05-30
- Decision Makers: @D-sorganization/maintainers
- Related Issues/PRs: #6647, #6650, #6651, ADR-0002, ADR-0023, ADR-0024

## Context

The JaxSim roadmap needs one explicit architecture decision before M1 work can
start: JaxSim can play two roles that currently live behind different suite
interfaces.

1. `src/shared/python/engine_core/PhysicsEngine` is the canonical humanoid
   engine layer from ADR-0002. It serves biomechanical engines and exposes
   dynamics terms such as `M(q)`, bias/drift terms, Jacobians, contact forces,
   and checkpointable engine state.
2. `src/shared/python/simulation_backends/SimulationBackend` is the reduced
   golf-model rollout layer from ADR-0023. It serves the `GolfModelParams`
   double-pendulum model through `make_backend()`, `Trace`/`BatchTrace`, optional
   batched execution, and `BackendCapabilities.is_differentiable`.

ADR-0024 already reserves differentiable rollouts for a future backend behind
the existing `SimulationBackend` Protocol. It also states that the differentiable
path must reuse the existing capability flag rather than create a parallel
gradient architecture.

JaxSim's dynamics-terms role fits `engine_core`: it works on the canonical
humanoid model and can provide mass matrix, bias, gravity, Jacobians, contacts,
and convention-sensitive floating-base quantities. Its differentiable rollout
role overlaps with `simulation_backends`, but that package is intentionally
scoped to the reduced golf model and its `Trace` schema.

## Decision

Use a documented bridge, not a third abstraction:

- **Primary home:** implement JaxSim as an `engine_core` backend for the
  canonical humanoid model.
- **No new top-level backend abstraction:** do not add `jaxsim_core`,
  `differentiable_engines`, or another registry next to `engine_core` and
  `simulation_backends`.
- **Bridge only when needed:** if a reduced golf-model JaxSim rollout is later
  required, expose it as a `simulation_backends` adapter that delegates to the
  JaxSim `engine_core` implementation and returns the existing `Trace` /
  `BatchTrace` schema.
- **Capability reuse:** M3 gradient support must extend the existing capability
  declarations instead of duplicating `BackendCapabilities.is_differentiable`.
  The bridge advertises differentiability through `simulation_backends` only
  when it truly returns differentiable rollouts. The `engine_core` side reports
  parameter/state-control gradient support through the ADR-0002 capability
  taxonomy follow-up tracked by #6651.

The bridge boundary is therefore:

```text
canonical humanoid model, M/h/g/J/contact terms
    -> engine_core JaxSim backend

reduced golf rollout, Trace/BatchTrace, batched/differentiable rollout flag
    -> simulation_backends adapter, optionally delegating to engine_core
```

## Required Contracts

### Engine-core JaxSim backend

The canonical implementation must:

- implement the `PhysicsEngine` contract and existing checkpoint/state
  expectations from `engine_core`;
- load the ADR-0020 canonical URDF/SDF model, not an arbitrary duplicate model;
- normalize frame, unit, base-frame, and floating-base velocity representation
  conventions before returning any dynamics term;
- preserve `SPATIAL_JACOBIAN_ORDER = ("angular", "linear")` for combined
  spatial Jacobians;
- report capabilities through `EngineCapabilities` once #6651 adds the gradient
  and forward-simulation tiers.

### Simulation-backends bridge

A bridge is allowed only if it satisfies all existing `simulation_backends`
rules:

- constructed through `make_backend(...)`;
- returns the current `Trace` / `BatchTrace` schema;
- uses `BackendCapabilities` for `supports_batched`, `is_differentiable`, and
  `provides_dynamics`;
- degrades gracefully when optional JAX/JaxSim dependencies are absent;
- compares results to the `ode`/`mujoco` reference backends with documented
  tolerances, never equality.

The bridge may not expose humanoid-only concepts through the reduced golf-model
Protocol. If a caller needs humanoid `M(q)`, `h(q, v)`, gravity, contacts, or
floating-base convention control, the caller must use `engine_core` directly.

## Consequences

### Positive

- Keeps ADR-0002 as the single canonical home for humanoid physics engines.
- Keeps ADR-0023/0024 as the single home for reduced golf rollouts and the
  existing differentiable-rollout capability flag.
- Avoids a third abstraction and a second gradient capability system.
- Makes the JaxSim M1 work sequence explicit: model/convention/dynamics terms
  land in `engine_core`; rollout adapters are later and narrower.
- Lets UI/API capability-aware selectors distinguish "humanoid dynamics terms"
  from "reduced golf rollout backend" without `hasattr` probing.

### Negative

- A future differentiable golf rollout may require a thin adapter even after the
  canonical JaxSim backend exists.
- Two capability surfaces must stay aligned: `EngineCapabilities` for humanoid
  dynamics and `BackendCapabilities` for reduced-rollout behavior.
- The bridge must be tested carefully so it does not silently reinterpret
  canonical humanoid coordinates as reduced golf-model coordinates.

## Alternatives Considered

1. **Put JaxSim only in `engine_core`.** Rejected as incomplete: it handles M1
   dynamics terms cleanly, but it leaves no documented path for ADR-0024-style
   differentiable rollouts if the golf-model use case materializes.
2. **Put JaxSim only in `simulation_backends`.** Rejected: JaxSim's immediate
   value is canonical humanoid dynamics terms and convention-sensitive
   floating-base quantities, which do not fit the reduced golf `Trace` contract.
3. **Create a new differentiable-engine abstraction.** Rejected: it duplicates
   both ADR-0002 capability discovery and ADR-0024's differentiability flag.
4. **Duplicate model loaders in both layers.** Rejected: it violates the
   ADR-0020 canonical-model boundary and would let two JaxSim paths drift.

## Follow-ups

- #6651: extend `EngineCapabilities` with gradient and forward-simulation tiers,
  then fill the verified capability matrix.
- #6652: define the floating-base velocity-representation convention contract
  before returning JaxSim `h`, `g`, or Jacobian terms.
- M1 implementation: add the JaxSim `engine_core` adapter only after the
  ADR-0020 URDF/SDF gate and convention contract are accepted.
- If a reduced golf JaxSim rollout becomes necessary, add a
  `simulation_backends` adapter and update ADR-0024 with the chosen
  differentiable backend status.

## Validation

This ADR ships no runtime code. Validation for the decision is review-based:

- `engine_core` remains the only canonical humanoid physics-engine registry;
- `simulation_backends` remains the only reduced golf-model rollout factory;
- no new top-level backend abstraction is introduced;
- the bridge path reuses existing capability flags and schemas.

Implementation PRs that follow this ADR must add tests at the layer they touch:
`engine_core` capability/convention tests for the canonical JaxSim backend, and
`simulation_backends` Protocol/schema tests only for an actual rollout bridge.
