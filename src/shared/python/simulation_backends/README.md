# `simulation_backends` — backend-agnostic simulation layer

A clean abstraction over multiple physics backends for the golf
double-pendulum / club model. The model is described **once** by
`GolfModelParams` and rendered to every backend; rollouts come back in one
shared `Trace` / `BatchTrace` schema so the analysis layer never depends on a
concrete engine.

See the design rationale in
[ADR-0023](../../../../docs/adr/0023-mujoco-warp-backend.md) and the user guide
in [`docs/simulation_backends/README.md`](../../../../docs/simulation_backends/README.md).

## Module map

| Module              | Responsibility                                                                                                                                                         |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `protocol.py`       | **Frozen interface.** `SimulationBackend`, `DynamicsProvider`, `BatchedBackend` Protocols; `SimState`, `Trace`, `BatchTrace`, `BackendCapabilities`, `SCHEMA_VERSION`. |
| `model_params.py`   | `GolfModelParams` — the single source of truth. `to_double_pendulum_parameters()`, `projected_gravity`, `default()`.                                                   |
| `mjcf.py`           | `params_to_mjcf()` — renders `GolfModelParams` to MuJoCo MJCF XML (the second renderer of the one model).                                                              |
| `factory.py`        | `make_backend(name, params, **kwargs)` and `available_backends()`. Imports backends **lazily**.                                                                        |
| `comparison.py`     | CC-27 cross-engine report service: `compare()`, `compare_traces()`, divergence annotations, and Markdown/JSON rendering.                                               |
| `compare_cli.py`    | One-command report entry point: `python -m src.shared.python.simulation_backends.compare_cli --engines ode,mujoco ...`.                                                |
| `capabilities.py`   | Guarded optional-dependency probes: `has_mujoco`, `has_mjx`, `has_warp`, `warp_device_available`, `require_*`.                                                         |
| `exceptions.py`     | Typed hierarchy: `BackendError`, `UnknownBackendError`, `BackendNotAvailableError`, `BackendCapabilityError`.                                                          |
| `ode_backend.py`    | `ode` — CPU reference backend wrapping the analytical RK4 dynamics; provides `M(q)` / bias forces.                                                                     |
| `mujoco_backend.py` | `mujoco` — CPU MuJoCo backend; independent dynamics primitives for cross-validation.                                                                                   |
| `mjwarp_backend.py` | `mjwarp` — GPU MuJoCo Warp backend for batched rollouts (optional `[warp]` extra).                                                                                     |
| `mjx_backend.py`    | `mjx` — MJX/JAX backend for batched differentiable rollouts (optional `[mjx]` extra).                                                                                  |
| `trace.py`          | HDF5 (de)serialisation of `Trace` / `BatchTrace`.                                                                                                                      |

## Choosing a backend

| Backend  | Device | Batched | Differentiable | Dynamics primitives | Use it for                                                        |
| -------- | ------ | ------- | -------------- | ------------------- | ----------------------------------------------------------------- |
| `ode`    | CPU    | no      | no             | **yes** (`M`, bias) | Single rollouts; the ground-truth reference; `M(q)` gate          |
| `mujoco` | CPU    | no      | no             | **yes** (`M`, bias) | Independent dynamics cross-validation; single rollouts            |
| `mjwarp` | CUDA   | **yes** | no             | no                  | **Hundreds-to-thousands** of parallel rollouts (sweeps, MPPI/CEM) |
| `mjx`    | JAX    | **yes** | **yes**        | no                  | Batched rollout gradients and direct-shooting/MPC-style studies   |

> **Honest performance note.** A _single_ 2-DoF rollout is **slower** on the GPU
> than on the CPU — kernel-launch latency and host↔device transfer dwarf the
> arithmetic. The GPU only pays off when one launch amortises across many
> parallel environments (`rollout_batch` with a large `num_envs`). For one
> rollout, use `ode`. See the value assessment in
> [ADR-0023](../../../../docs/adr/0023-mujoco-warp-backend.md).

MJX is the differentiable rollout path. Its host-facing `rollout` and
`rollout_batch` methods still return NumPy-backed `Trace` / `BatchTrace`
objects, while `rollout_batch_arrays()` and `final_state_control_jacobian()`
keep the JAX-native path available for gradient workloads. MJWarp remains the
non-differentiable throughput backend. See
[ADR-0024](../../../../docs/adr/0024-differentiable-backend.md).

## Usage

```python
from src.shared.python.simulation_backends import GolfModelParams, make_backend

params = GolfModelParams.default()
backend = make_backend("ode", params)  # CPU reference; no GPU deps

# rollout(controls, horizon, dt) -> Trace with horizon + 1 samples (t[0] == 0).
trace = backend.rollout(controls=None, horizon=200, dt=0.005)  # passive swing
trace.num_steps        # 201
trace.final_state().v  # final joint velocities
```

Backends are interchangeable behind the Protocol; swap `"ode"` for `"mujoco"`
(CPU), `"mjwarp"` (GPU, requires the `[warp]` extra), or `"mjx"` (JAX,
requires the `[mjx]` extra with `mujoco-mjx`) without touching the analysis code.
Requesting a backend whose optional dependencies are missing raises a
`BackendNotAvailableError` with an install hint — importing this package itself
never pulls in any GPU or JAX dependency.

## Cross-engine reports

The CC-27 comparison service produces user-facing reports from the same
Protocol surface:

```bash
python -m src.shared.python.simulation_backends.compare_cli \
  --engines ode,mujoco \
  --horizon 200 \
  --dt 0.005 \
  --output reports/ode_vs_mujoco.md
```

Reports include kinematics, kinetics, ZTCF/ZVCF when the selected backend
implements `DynamicsProvider`, optional wrench comparison, divergence registry
links, and provenance for every panel.
