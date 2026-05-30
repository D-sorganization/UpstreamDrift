# Simulation Backends — user guide

Welcome! The **Simulation Backends** suite lets you drive the golf
double-pendulum / club model through several interchangeable physics engines
that all share one interface. You describe the model once, then run it on an
analytical CPU reference, on MuJoCo's CPU solver, or — when you have an NVIDIA
GPU — on MuJoCo Warp for massively parallel batched runs.

This guide is the friendly, step-by-step companion. It covers:

- [What the suite is and when to use each backend](#what-it-is-and-when-to-use-each-backend)
- [Opening the launcher tile (GUI)](#opening-the-launcher-tile-gui)
- [A tour of every GUI control](#a-tour-of-every-gui-control)
- [Python API quickstart](#python-api-quickstart)
- [Installing the optional GPU stack](#installing-the-optional-gpu-stack)
- [Troubleshooting](#troubleshooting)

Looking for the terse reference, tolerance rules, and CUDA details instead? See
the [package user guide](README.md), the
[package reference](../../src/shared/python/simulation_backends/README.md), and
the design rationale in
[ADR-0023](../adr/0023-mujoco-warp-backend.md) /
[ADR-0024](../adr/0024-differentiable-backend.md).

## What it is and when to use each backend

There are three backends behind one `SimulationBackend` interface. They are
fully interchangeable: swap the name, keep the rest of your code.

| Backend  | Device   | Batched sweeps | Dynamics primitives (`M`, bias) | When to reach for it                                         |
| -------- | -------- | -------------- | ------------------------------- | ------------------------------------------------------------ |
| `ode`    | CPU      | no             | **yes**                         | Your default. Single rollouts and the ground-truth reference |
| `mujoco` | CPU      | no             | **yes**                         | An independent dynamics check; cross-validation              |
| `mjwarp` | CUDA GPU | **yes**        | no                              | Hundreds-to-thousands of rollouts at once (sweeps, MPPI/CEM) |

A few rules of thumb so you pick the right tool:

- **One rollout, or a handful? Use `ode`.** It is always installed (no
  third-party physics dependency), it computes in float64, and for a single
  2-DoF swing it is the ground truth everything else is measured against.
- **Want a second opinion on the physics? Use `mujoco`.** It is an entirely
  independent derivation of the equations of motion, which is exactly what makes
  it useful for cross-validation. Cross-validation needs this **CPU `mujoco`
  backend** — it cannot be done against the GPU alone (more on that below).
- **Running a big sweep or sampling-based optimizer? Use `mjwarp`.** The GPU
  only pays off at **batch scale**. A single 2-DoF rollout is actually _slower_
  on the GPU than on the CPU, because kernel-launch latency and host↔device
  memory transfer dwarf the tiny 2×2 arithmetic. The GPU wins only when one
  launch is amortised across many parallel environments. Treat it as a
  throughput engine for `rollout_batch`, not a faster way to do one run.

Only the two CPU backends (`ode`, `mujoco`) expose the dynamics primitives —
the mass matrix `M(q)` and the bias (Coriolis + gravity) forces. `mjwarp` is a
batched rollout engine and does not offer them.

## Opening the launcher tile (GUI)

The suite ships as a launcher tile so you do not have to write any code to
explore it.

**From the main launcher:** look for the **Simulation Backends** tile (it lives
under the **simulation** category) and click it. The tile is registered in
[`src/config/launcher_manifest.json`](../../src/config/launcher_manifest.json)
under the id `simulation_backends`, so it appears consistently in both the PyQt
and the Tauri/React launcher UIs.

**Standalone, from a terminal:** you can launch the same window directly without
opening the full launcher:

```bash
python -m src.tools.simulation_backends_launcher
```

The window opens on the always-available `ode` backend, so it works on a
CPU-only machine out of the box. MuJoCo (CPU) and MuJoCo Warp (GPU) light up in
the backend picker only when their optional dependencies are installed.

## A tour of every GUI control

The window is organised top-to-bottom: pick a backend, edit the model, then run
one of the actions. Here is what each control does and what output to expect.

### 1. Backend picker (with availability)

A dropdown lets you choose `ode`, `mujoco`, or `mjwarp`. Each entry shows its
availability so you are never guessing:

- `ode` — always available.
- `mujoco` — available when the `mujoco` package is installed.
- `mjwarp` — available only when the `[warp]` extra is installed **and** a usable
  CUDA device is visible. On a CPU-only box it appears disabled (or greyed) with
  a hint, rather than letting you pick it and then fail.

Switching the backend re-renders the same model parameters onto the newly
selected engine; you do not re-enter anything.

### 2. Model parameter spinners (with units)

A panel of numeric spinners edits the golf double-pendulum model. Every field is
in **SI units**, labelled with its unit, and seeded with the model defaults so
you can start from a sensible swing and nudge from there. The fields map
directly to `GolfModelParams`:

| Spinner                    | Field                        | Unit    | Default   |
| -------------------------- | ---------------------------- | ------- | --------- |
| Upper segment length       | `upper.length_m`             | m       | `0.75`    |
| Upper segment mass         | `upper.mass_kg`              | kg      | `7.5`     |
| Upper centre-of-mass ratio | `upper.center_of_mass_ratio` | (0–1)   | `0.45`    |
| Lower (club) length        | `lower.length_m`             | m       | `1.0`     |
| Shaft mass                 | `lower.shaft_mass_kg`        | kg      | `0.15`    |
| Clubhead mass              | `lower.clubhead_mass_kg`     | kg      | `0.2`     |
| Shaft centre-of-mass ratio | `lower.shaft_com_ratio`      | (0–1)   | `0.43`    |
| Swing-plane inclination    | `plane_inclination_deg`      | degrees | `35.0`    |
| Shoulder damping           | `damping_shoulder`           | N·m·s   | `0.4`     |
| Wrist damping              | `damping_wrist`              | N·m·s   | `0.25`    |
| Gravity                    | `gravity_m_s2`               | m/s²    | `9.80665` |

There are also controls for the **rollout length** (`horizon`, number of steps)
and the **timestep** (`dt`, in seconds). A rollout of `horizon` steps returns
`horizon + 1` samples, with the first sample at `t = 0`.

> Tip: the parameter model is immutable under the hood. The GUI rebuilds it for
> you on each run, so editing a spinner and re-running always reflects your
> latest values.

### 3. Run Rollout

Runs a **single** rollout on the selected backend and plots the result.

- **What it does:** integrates the model for `horizon` steps at the chosen `dt`.
  With no controls supplied this is a passive swing under gravity and damping.
- **What you get:** a **trajectory plot** of the joint angles (and/or velocities)
  versus time. This is the quickest way to sanity-check a parameter change.
- **Which backend:** any. On `ode`/`mujoco` it runs on the CPU; if you select
  `mjwarp` it still works but, as noted, a single rollout is not where the GPU
  shines.

### 4. Run Parameter Sweep

Runs **many** rollouts while varying a model parameter across a range — this is
the batched workload.

- **What it does:** sweeps one parameter across a set of samples, runs a rollout
  per sample, and reduces each to a summary metric (for example a clubhead-speed
  proxy).
- **What you get:** a **sweep plot** of the metric versus the swept parameter, so
  you can see how the outcome responds across the range.
- **Which backend:** this is the workload where `mjwarp` pays off — a single GPU
  launch is amortised across all the parallel environments. On a CPU-only
  machine the sweep still runs on `ode`/`mujoco`; it is simply serial.

### 5. Cross-validate vs ODE

Checks a second backend against the analytical `ode` reference and reports the
agreement.

- **What it does:** compares the selected backend's dynamics (mass matrix `M(q)`,
  bias forces, and/or a trajectory) against `ode` and measures the difference.
- **What you get:** a **report text** panel with a pass/fail verdict and the
  maximum absolute error.
- **Important:** meaningful cross-validation needs the **CPU `mujoco` backend**.
  Because both `ode` and `mujoco` compute in float64, they agree extremely
  tightly — to roughly `1e-9`. You cannot cross-validate the dynamics primitives
  against `mjwarp`: the GPU backend does not expose `M(q)` / bias, and it
  computes in float32, so it is never bit-exact against the CPU (see
  [Troubleshooting](#troubleshooting)).

### 6. Export HDF5

Saves the most recent run to a portable HDF5 trace file.

- **What it does:** writes the trajectory (`t`, `q`, `v`, controls, `dt`, and the
  backend name) to an `.h5` file using the shared trace schema.
- **What you get:** a file you can reload later — in the GUI, from Python via
  `trace_io.read_trace`, or in any HDF5-aware tool — for offline analysis or
  comparison.

## Python API quickstart

Prefer code? The same capabilities are a few lines away. This mirrors the
[package README](../../src/shared/python/simulation_backends/README.md).

```python
from src.shared.python.simulation_backends import (
    GolfModelParams,
    has_mujoco,
    make_backend,
)

# 1. Describe the model once (immutable; derive variants with model_copy).
params = GolfModelParams.default()

# 2. Make a backend and run a single rollout. `ode` needs no extra deps.
#    rollout(controls, horizon, dt) -> Trace with horizon + 1 samples; t[0] == 0.
ode = make_backend("ode", params)
trace = ode.rollout(controls=None, horizon=200, dt=0.005)  # passive swing
print(trace.backend, trace.num_steps, trace.final_state().v)
```

### Cross-validate against the CPU MuJoCo backend

```python
import numpy as np

from src.shared.python.simulation_backends import make_backend
from src.shared.python.simulation_backends.validation import (
    cross_validate_mass_matrix,
)

if has_mujoco():
    ode = make_backend("ode", params)
    mj = make_backend("mujoco", params)
    q_samples = [np.array([0.3, -0.2]), np.array([-0.1, 0.4])]
    report = cross_validate_mass_matrix(ode, mj, q_samples)
    print(report.passed, report.max_abs_error)  # passed=True, error ~1e-9
```

### Save and reload a trace

```python
from src.shared.python.simulation_backends.trace_io import read_trace, write_trace

write_trace(trace, "swing.h5")
reloaded = read_trace("swing.h5")
print(reloaded.num_steps, reloaded.backend)
```

### A full runnable example

A CPU-only Monte-Carlo parameter sweep (no GPU required) ships at
[`examples/simulation_backends/parameter_sweep.py`](../../examples/simulation_backends/parameter_sweep.py).
It sweeps a parameter across ~64 samples, runs a rollout per sample through
`ode`, computes a clubhead-speed proxy, prints a summary table, and writes one
HDF5 trace:

```bash
python3 examples/simulation_backends/parameter_sweep.py
```

## Installing the optional GPU stack

The base install gives you the `ode` backend with **no third-party physics
dependency** — it runs everywhere, including CPU-only machines and CI.

```bash
pip install upstream-drift            # ode backend only
pip install 'upstream-drift[mujoco]'  # adds the mujoco CPU backend
pip install 'upstream-drift[warp]'    # adds the mjwarp GPU backend (needs CUDA)
```

The `[warp]` extra pulls in pinned, known-good versions of MuJoCo Warp and
NVIDIA Warp. To actually run `mjwarp` you need an NVIDIA GPU, the NVIDIA drivers,
and a CUDA-capable runtime compatible with the pinned Warp version. Verify the
stack end-to-end before relying on it:

```python
from src.shared.python.simulation_backends import has_warp, warp_device_available

assert has_warp(), "install: pip install 'upstream-drift[warp]'"
assert warp_device_available(), "no usable CUDA device visible to Warp"
```

**Graceful CPU-only fallback.** You do not need the GPU stack to use the suite.
Importing the package never pulls in a GPU dependency, the GUI opens on `ode`,
and everything except batched GPU rollouts works on a plain CPU. If you only
ever run single rollouts, skip `[warp]` entirely — the GPU would be pure
overhead. For Docker GPU invocation and CUDA environment variables, see
[`docs/docker-gpu.md`](../docker-gpu.md).

## Troubleshooting

### "mjwarp not available" (this is expected without a GPU)

Selecting `mjwarp` — or calling `make_backend("mjwarp", ...)` — on a machine with
no GPU raises a `BackendNotAvailableError` with an install/availability hint.
**This is the designed behaviour, not a bug.** The `mjwarp` backend requires both
the `[warp]` extra and a usable CUDA device. Confirm what your machine can do:

```python
from src.shared.python.simulation_backends import has_warp, warp_device_available

has_warp()               # True only if the warp wheels import
warp_device_available()  # True only if a usable CUDA device is ALSO visible
```

`warp_device_available()` is stricter than `has_warp()`: a machine can have the
wheels installed yet expose no CUDA device. Gate GPU work on
`warp_device_available()`. On a CPU-only box, use `ode` for single rollouts and
`mujoco` for cross-validation — both are fully featured without a GPU.

### Cross-validation tolerances — why it is never bit-exact CPU vs GPU

Cross-backend agreement is checked with a documented numerical tolerance, never
with `==`. The reason is precision:

- **CPU ↔ CPU** (`ode` vs `mujoco`): both compute in **float64**, so they agree
  very tightly — the verified mass-matrix / bias gate matches to roughly
  `1e-9`–`1e-11`. Use a tight tolerance here.
- **Anything crossing into `mjwarp`**: MuJoCo Warp computes in **float32** on the
  GPU. float32 rounding accumulates over a rollout, so CPU↔GPU comparisons need a
  _loose_ tolerance (a few `1e-3`–`1e-4` on state trajectories after many steps),
  sized to the float32 epsilon and the horizon length.

A check that demands bitwise equality across the CPU/GPU boundary is wrong by
construction and will flake. That is also why the **Cross-validate vs ODE**
action targets the CPU `mujoco` backend: it is the apples-to-apples float64
comparison.

### "BackendCapabilityError" when asking mjwarp for the mass matrix

`mjwarp` is a batched rollout engine; its `provides_dynamics` flag is `False`.
Asking it for `mass_matrix` / `bias_forces` raises `BackendCapabilityError`. Use
`ode` or `mujoco` for dynamics primitives, or branch on
`backend.capabilities.provides_dynamics` first.

### "UnknownBackendError"

The backend name must be one of `available_backends()` —
`("mjwarp", "mujoco", "ode")`. A typo (or a different name) raises
`UnknownBackendError`. All of these exceptions subclass `BackendError`, so you
can catch the whole family with a single `except BackendError`.
