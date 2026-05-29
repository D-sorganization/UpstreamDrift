# Simulation backends — user guide

The `simulation_backends` package lets you drive the golf double-pendulum model
through several interchangeable physics engines behind one Protocol. This guide
covers installation (including the optional GPU stack), CUDA setup, **when the
GPU is worth it**, the cross-validation tolerance rules, and how to run each
backend from Python.

- Architecture & rationale: [ADR-0023](../adr/0023-mujoco-warp-backend.md)
- Differentiable backend (future): [ADR-0024](../adr/0024-differentiable-backend.md)
- Package reference & module map:
  [`src/shared/python/simulation_backends/README.md`](../../src/shared/python/simulation_backends/README.md)

## The three backends at a glance

| Name     | Device | Batched | Dynamics primitives | Optional deps               |
| -------- | ------ | ------- | ------------------- | --------------------------- |
| `ode`    | CPU    | no      | yes (`M`, bias)     | none (always available)     |
| `mujoco` | CPU    | no      | yes (`M`, bias)     | `mujoco`                    |
| `mjwarp` | CUDA   | **yes** | no                  | `[warp]` extra + NVIDIA GPU |

`ode` is the analytical ground-truth reference. `mujoco` is an independent
dynamics derivation used to cross-validate it. `mjwarp` is the GPU engine for
_batched_ throughput.

## Installation

The base install gives you the `ode` backend with **no third-party physics
dependency** — it runs everywhere, including CPU-only CI.

```bash
pip install upstream-drift            # ode backend only
pip install 'upstream-drift[mujoco]'  # adds the mujoco CPU backend
pip install 'upstream-drift[warp]'    # adds the mjwarp GPU backend (CUDA)
```

The `[warp]` extra pulls in pinned versions of `mujoco-warp` (MJWarp) and
`warp-lang` (NVIDIA Warp). MJWarp is an **alpha, actively-developed** project, so
the extra pins specific known-good versions; upgrading the pin is a deliberate,
re-validated change (see ADR-0023). If you only ever run single rollouts, do
**not** install `[warp]` — the GPU is pure overhead for one 2-DoF rollout (see
below).

### Checking what is available at runtime

Importing the package never pulls in a GPU dependency. Probe capabilities with
the guarded helpers — they never raise:

```python
from src.shared.python.simulation_backends import (
    has_mujoco,
    has_warp,
    warp_device_available,
)

has_mujoco()             # True if the mujoco CPU bindings import
has_warp()               # True if both warp and mujoco_warp import
warp_device_available()  # True only if a usable CUDA device is also visible
```

`warp_device_available()` is stricter than `has_warp()`: a machine can have the
wheels installed yet expose no CUDA device. Gate GPU work on
`warp_device_available()`, not merely `has_warp()`.

## GPU / CUDA setup

To run the `mjwarp` backend you need:

- An NVIDIA GPU (any recent GeForce, Quadro, or Tesla).
- NVIDIA drivers installed on the host.
- A CUDA-capable runtime compatible with the pinned `warp-lang` version.
- The `[warp]` extra installed.

Verify the stack end-to-end:

```python
from src.shared.python.simulation_backends import has_warp, warp_device_available

assert has_warp(), "install: pip install 'upstream-drift[warp]'"
assert warp_device_available(), "no usable CUDA device visible to Warp"
```

If you run inside Docker, the container images already wire up the NVIDIA
Container Toolkit — see [`docs/docker-gpu.md`](../docker-gpu.md) for
`docker compose` GPU invocation, `NVIDIA_VISIBLE_DEVICES`, and `MUJOCO_GL`
settings. Requesting `make_backend('mjwarp', ...)` on a CPU-only box raises a
`BackendNotAvailableError` with an install hint rather than a bare `ImportError`.

## Batched vs single — when does the GPU pay off?

This is the single most important operational guidance, taken directly from the
ADR-0023 value assessment:

> **A single 2-DoF rollout is _slower_ on the GPU than on the CPU.**

For one rollout of a two-joint pendulum the dominant costs are kernel-launch
latency and host↔device memory transfer; the actual arithmetic (a 2×2 mass
matrix) is trivial. The CPU analytical stepper finishes before the GPU has even
staged the work. The GPU only wins when a single launch amortises across many
environments running in parallel.

| Workload                                             | Use            | Why                                                  |
| ---------------------------------------------------- | -------------- | ---------------------------------------------------- |
| One rollout, or a handful                            | `ode`          | GPU launch + transfer overhead dwarfs the work       |
| Need `M(q)` / bias forces                            | `ode`/`mujoco` | only the CPU backends expose dynamics primitives     |
| Independent dynamics cross-check                     | `mujoco`       | second derivation of the EOM                         |
| Hundreds-to-thousands of rollouts (sweeps, MPPI/CEM) | `mjwarp`       | parallelism dominates the fixed launch/transfer cost |

We do **not** claim the GPU accelerates the golf model in general — only that it
accelerates _batched_ workloads. The capability matrix exposes this via the
`supports_batched` flag rather than hiding it behind a "fast backend" label.

## Cross-validation tolerance rationale

The CPU backends compute in **float64**; MJWarp computes in **float32** on the
GPU. Consequently:

> Cross-backend agreement is asserted with `numpy.allclose` and an explicit,
> documented tolerance — **never** with `==` (or `array_equal`).

- **CPU ↔ CPU** (analytical `ode` vs `mujoco` `M(q)` and bias): both float64,
  so they match tightly. The verified `M(q)` / bias gate agrees to
  ~**`1e-9`–`1e-11`**; use a tight `atol`/`rtol` there.
- **Anything crossing into MJWarp** (float32): use a _loose_ tolerance — a few
  `1e-3`–`1e-4` on state trajectories after many steps, because float32 error
  accumulates over a rollout. Size the tolerance to the float32 epsilon and the
  horizon length.

Any test that asserts bitwise equality across the CPU/GPU boundary is wrong by
construction and will flake.

## Running each backend from Python

The model is described once and rendered to whichever backend you pick:

```python
from src.shared.python.simulation_backends import (
    GolfModelParams,
    has_mujoco,
    make_backend,
    warp_device_available,
)

params = GolfModelParams.default()

# --- CPU reference (always available) -----------------------------------
ode = make_backend("ode", params)
# rollout(controls, horizon, dt) -> Trace with horizon + 1 samples; t[0] == 0.
trace = ode.rollout(controls=None, horizon=200, dt=0.005)  # passive swing
print(trace.backend, trace.num_steps, trace.final_state().v)

# --- CPU MuJoCo (needs the `mujoco` package) ----------------------------
if has_mujoco():
    mj = make_backend("mujoco", params)
    mj_trace = mj.rollout(controls=None, horizon=200, dt=0.005)

# --- GPU MuJoCo Warp (needs the [warp] extra + a CUDA device) -----------
if warp_device_available():
    gpu = make_backend("mjwarp", params)
    # batched: many parallel envs amortise the single GPU launch.
    batch = gpu.rollout_batch(controls=None, horizon=200, dt=0.005, num_envs=1024)
    print(batch.num_envs, batch.num_steps)
```

### Dynamics primitives (CPU backends only)

```python
import numpy as np

from src.shared.python.simulation_backends import DynamicsProvider, make_backend

ode = make_backend("ode", GolfModelParams.default())
q = np.array([0.3, -0.2])
if isinstance(ode, DynamicsProvider):  # ode/mujoco satisfy this; mjwarp does not
    mass = ode.mass_matrix(q)          # (2, 2)
    bias = ode.bias_forces(q, np.zeros(2))  # (2,)
```

Asking `mjwarp` for `mass_matrix` raises `BackendCapabilityError` — its
`provides_dynamics` flag is `False`. Branch on the `DynamicsProvider` Protocol
(or `backend.capabilities.provides_dynamics`) rather than assuming every backend
offers every service.

## A runnable example

A CPU-only Monte-Carlo parameter sweep that needs no GPU lives at
[`examples/simulation_backends/parameter_sweep.py`](../../examples/simulation_backends/parameter_sweep.py).
It sweeps a model parameter across ~64 samples, runs a rollout per sample through
the `ode` backend, computes a clubhead-speed proxy, prints a summary table, and
writes one HDF5 trace. Run it with:

```bash
python3 examples/simulation_backends/parameter_sweep.py
```
