# ADR-0025: JaxSim Backend Home

- Status: Accepted
- Date: 2026-05-30
- Decision Makers: @D-sorganization/maintainers
- Related Issues/PRs: #6647, #6650, #6651, ADR-0023, ADR-0024

## Context

JaxSim spans two existing backend concerns:

1. `engine_core.PhysicsEngine` owns full humanoid and biomechanical engine
   adapters. Its current contract is the right place for canonical humanoid
   dynamics terms: `M(q)`, bias forces, gravity, Jacobians, inverse dynamics,
   and drift/control acceleration terms.
2. `simulation_backends.SimulationBackend` owns rollout-oriented reduced-model
   backends. ADR-0023 made it the home for ODE, MuJoCo CPU, and MuJoCo Warp
   rollouts, with `BackendCapabilities.is_differentiable` already describing
   whether gradients can flow through a rollout.

JaxSim can provide both dynamics terms and JAX-native differentiability. Putting
it entirely in either layer would either overload `simulation_backends` with
humanoid engine responsibilities or duplicate the rollout/differentiability
contract inside `engine_core`.

The design constraint from #6650 is explicit: do not create a third backend
abstraction.

## Decision

JaxSim uses a documented bridge between the two existing abstractions:

- The canonical humanoid JaxSim adapter lives under `engine_core` as a
  `PhysicsEngine` implementation when it exposes suite-level dynamics terms.
- Differentiable rollout workloads live under `simulation_backends` only when
  they satisfy the `SimulationBackend` protocol from ADR-0023.
- Shared JaxSim installation probes, model-loading helpers, and convention
  utilities may live in small common modules, but they are not a new backend
  interface. They are support code imported by one or both adapters.

This means the first production JaxSim adapter should be named and reviewed by
role:

- `JaxSimPhysicsEngine`: full humanoid dynamics terms and convention
  normalization behind `engine_core.PhysicsEngine`.
- `jaxsim` `SimulationBackend`: optional future reduced-model rollout backend
  only if it returns the existing `Trace`/`BatchTrace` schema and advertises
  `BackendCapabilities.is_differentiable=True`.

No caller should need to branch on a special JaxSim-only protocol. Code that
needs dynamics terms depends on `PhysicsEngine` or one of its segregated
sub-protocols. Code that needs differentiable rollouts depends on
`SimulationBackend` and its existing capability flag.

## Gradient Flow

M3 gradient work flows through `simulation_backends` when the workload is a
rollout optimization problem. The signal is the existing
`BackendCapabilities.is_differentiable` flag, not a duplicate
`EngineCapabilities.is_differentiable` field.

The extended `EngineCapabilities` taxonomy from #6651 describes which gradient
surfaces an engine can provide:

- `parameter_gradients`
- `state_control_gradients`
- `forward_sim`
- `contact_step`
- `trajectory_opt`

Those fields are capability reporting for the engine layer. They do not replace
`simulation_backends.BackendCapabilities.is_differentiable`, which remains the
rollout-layer contract for automatic differentiation through `rollout`.

If a future feature needs both full humanoid dynamics terms and differentiable
rollout traces, it should compose the two adapters through explicit data
conversion at the boundary. It should not add a combined mega-interface.

## Consequences

### Positive

- Preserves ADR-0002's plugin architecture and ADR-0023's rollout protocol.
- Avoids a third abstraction and keeps LOD/interface-segregation intact.
- Lets #6652 velocity-representation normalization live in engine-core
  convention utilities and be reused by any JaxSim adapter.
- Keeps `is_differentiable` single-sourced in `simulation_backends` for rollout
  code.

### Negative

- Some JaxSim support modules may be imported by both layers, so review must
  keep those modules free of adapter-specific policy.
- A JaxSim feature request must state which role it needs: humanoid dynamics
  terms, differentiable rollouts, or both composed explicitly.

## Validation

- ADR-0023 remains the source for `SimulationBackend` and
  `BackendCapabilities.is_differentiable`.
- ADR-0024 remains the differentiable rollout planning reference.
- #6651 extends `EngineCapabilities` without adding an engine-layer
  `is_differentiable` duplicate.
- #6652 should implement velocity representation conventions in `engine_core`
  so the eventual `JaxSimPhysicsEngine` normalizes mixed/body-fixed/inertial
  velocity terms before exposing suite-level dynamics.
