# JaxSim Backend — Differentiable Rigid-Body Dynamics

JaxSim is a JAX-based, hardware-accelerated, fully differentiable rigid-body
dynamics and contact simulator. This directory holds the engine-core adapter
that maps JaxSim's functional API onto the suite's `PhysicsEngine` protocols.

> JaxSim and JAX are **Linux-only** in this suite. The adapter keeps every
> JaxSim import lazy, so core installs (and Windows/macOS dev boxes) can import
> this package without the optional extra. JaxSim-dependent tests carry the
> `requires_jaxsim` marker and run on the Linux CI fleet via
> `.github/workflows/cross-engine-equivalence.yml`.

## Installation

```bash
python -m pip install "upstream-drift[jaxsim]"
```

The extra pins `jaxsim==0.9.0` (CPU JAX; CUDA is intentionally not selected).
The pin is enforced by `tests/unit/test_jaxsim_optional_dependency.py` and the
upgrade-guard `scripts/jaxsim/check_jaxsim_pin.py` (issue #6660).

## Files

| File                     | Purpose                                                   |
| ------------------------ | --------------------------------------------------------- |
| `jaxsim_backend.py`      | `JaxSimBackend` adapter: load/query/dynamics/rollout.     |
| `parameter_gradients.py` | ZTCF parameter-gradient sensitivity (issue #6656, gated). |
| `__init__.py`            | Public exports.                                           |

## Conventions (Design by Contract)

The adapter normalizes JaxSim's native `Mixed` representation to the suite
**canonical inertial** convention at the boundary (see
`src/shared/python/engine_core/velocity_conventions.py`):

- Generalized velocity is ordered `[base angular; base linear; joint]`,
  matching `SPATIAL_JACOBIAN_ORDER = ("angular", "linear")`.
- `JaxSimBackend.__init__` asserts the requested convention is the canonical
  inertial representation and raises `ValueError` otherwise.
- JaxSim's free-floating `[linear; angular]` block is permuted to canonical
  order before any term reaches a caller.

## Capabilities

`JaxSimBackend.get_capabilities()` returns an `EngineCapabilities` report. The
exercise-dashboard selector (`src/launchers/jaxsim_dashboard.py`) reads this
report to grey out features that are not `FULL` (e.g. `contact_forces` and
`drift_acceleration` are `PARTIAL`). The parameter-sensitivity panel entry is
stubbed and gated on issue #6656.

## Validation gates

- **Dynamics parity** — `tests/cross_engine/test_jaxsim_vs_pinocchio.py`
  compares `M, h, g, C` against Pinocchio (issue #6654).
- **Forward-sim rollout** — `tests/cross_engine/test_jaxsim_forward_sim.py`
  integrates a contact-free free body and asserts the JaxSim rollout matches
  the analytic torque-free trajectory and conforms to the canonical `Trace`
  schema (issue #6655).
- **Adapter contracts** — `tests/unit/engines/test_jaxsim_backend.py` exercises
  the adapter against an injectable mock JaxSim API (no JAX required).

## See also

- [`docs/engines/jaxsim.md`](../../../../docs/engines/jaxsim.md) — full engine notes.
- [`docs/adr/0025-jaxsim-backend-home.md`](../../../../docs/adr/0025-jaxsim-backend-home.md) — ADR.
- [`docs/tutorials/same_swing_three_engines.md`](../../../../docs/tutorials/same_swing_three_engines.md) — cross-engine tutorial.
