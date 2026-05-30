# Engine Capability Taxonomy

This matrix extends ADR-0002's engine capability manifest for the JaxSim
integration gates in #6651. The levels use
`src.shared.python.engine_core.capabilities.CapabilityLevel`:

- `FULL`: implemented and exposed through the suite's current adapter or a
  stable upstream API that the adapter can call directly.
- `PARTIAL`: available only for some model classes, through a side API, behind
  a convention gap, or not yet normalized by the suite adapter.
- `NONE`: no verified support in the suite or upstream library path inspected
  for this matrix.

## Verified Matrix

| Engine    | Parameter gradients | State/control gradients | Forward simulation | Contact step | Trajectory optimization | Verification note                                                                                                                                                  |
| --------- | ------------------- | ----------------------- | ------------------ | ------------ | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| MuJoCo    | `PARTIAL`           | `PARTIAL`               | `FULL`             | `FULL`       | `PARTIAL`               | MuJoCo exposes forward/contact stepping and finite-difference/derivative-adjacent APIs; gradient optimization remains external to the current adapter.             |
| Drake     | `PARTIAL`           | `FULL`                  | `FULL`             | `FULL`       | `FULL`                  | Drake's AutoDiff and MathematicalProgram stack cover state/control gradients and trajectory optimization; parameter gradients are model-scope dependent.           |
| Pinocchio | `PARTIAL`           | `PARTIAL`               | `FULL`             | `PARTIAL`    | `PARTIAL`               | Pinocchio exposes rigid-body dynamics derivatives and contact algorithms; optimization is composed through companion stacks rather than this adapter alone.        |
| OpenSim   | `PARTIAL`           | `PARTIAL`               | `FULL`             | `PARTIAL`    | `PARTIAL`               | OpenSim/Moco support optimization and sensitivity workflows for supported models; contact and gradient availability depend on model components.                    |
| MyoSuite  | `NONE`              | `NONE`                  | `FULL`             | `FULL`       | `NONE`                  | MyoSuite runs on MuJoCo tasks but does not expose suite-normalized gradients or trajectory optimization through the current engine adapter.                        |
| JaxSim    | `FULL`              | `FULL`                  | `FULL`             | `PARTIAL`    | `PARTIAL`               | JaxSim's JAX-native model supports differentiable dynamics; contact and trajectory optimization need suite-level policy and validation before promotion to `FULL`. |

## Contract

`EngineCapabilities` is the source of truth surfaced to the API and UI. New
gradient/rollout capabilities must be added there first, with:

- an immutable dataclass field defaulting to `CapabilityLevel.NONE`;
- a `has_*` accessor;
- `to_dict()` and `from_dict()` round-trip support;
- a documented profile update when an engine advertises non-`NONE` support.

The `simulation_backends.BackendCapabilities.is_differentiable` flag remains the
rollout-backend flag from ADR-0023/ADR-0024. It must not be duplicated into
`EngineCapabilities`; the engine-core taxonomy instead records which gradient
surfaces are available.
