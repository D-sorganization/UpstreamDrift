# ADR-0023: MuJoCo Warp GPU backend and MuJoCo CPU backend behind the SimulationBackend Protocol

- Status: Accepted
- Date: 2026-05-29
- Decision Makers: @D-sorganization/maintainers
- Related Issues/PRs: GPU-accelerated simulation-backend epic

## Context

The golf double-pendulum model (`GolfModelParams`) was historically driven by a
single analytical RK4 integrator
(`DoublePendulumDynamics` /
`PendulumPhysicsEngine`). That code is correct and fast for one rollout, but two
workloads have outgrown it:

1. **Massively parallel rollouts.** Monte-Carlo parameter sweeps, sampling-based
   trajectory optimisation (MPPI / CEM), and policy-evaluation batches want to
   integrate hundreds-to-thousands of slightly different rollouts at once. On a
   CPU these run serially; on a GPU they run in lock-step.
2. **An independent derivation of the equations of motion.** A single
   hand-derived EOM has no cross-check. A second, independently-implemented
   dynamics engine that agrees to numerical tolerance is strong evidence the
   physics is right.

[MuJoCo](https://mujoco.org/) answers both. The CPU bindings give a mature,
independently-derived rigid-body engine with first-class access to the
joint-space inertia matrix `M(q)` and bias forces. [MuJoCo
Warp](https://github.com/google-deepmind/mujoco_warp) ("MJWarp") compiles the
same model to NVIDIA [Warp](https://github.com/NVIDIA/warp) kernels and runs
many environments in parallel on a CUDA device.

The constraint is that the suite **must import and run on a machine with no
GPU** — the ODE and MuJoCo-CPU backends are the default, and the GPU stack is an
optional extra. The architectural question is how to add two new engines without
(a) leaking optional GPU dependencies into the import path, (b) duplicating the
model definition three ways, or (c) letting the three engines silently disagree.

## Decision

Add a MuJoCo Warp **GPU** backend and a MuJoCo **CPU** backend behind the
existing `SimulationBackend`
structural `Protocol` (ADR-0002's plugin philosophy, applied to integrators).
All three backends are constructed through one factory,
`make_backend(name, params, **kwargs)`, and emit the same `Trace` / `BatchTrace`
schema so the analysis layer never depends on a concrete engine.

### Backend capability matrix

Capabilities are declared statically via the frozen `BackendCapabilities`
dataclass so callers branch _without_ `hasattr` probing or importing optional
GPU modules. Interface segregation (LOD) is enforced by splitting the optional
services into their own Protocols: `DynamicsProvider` (`mass_matrix`,
`bias_forces`) and `BatchedBackend` (`rollout_batch`).

| Backend | `name`     | Device | `supports_batched` | `is_differentiable` | `provides_dynamics` | Role                                                            |
| ------- | ---------- | ------ | ------------------ | ------------------- | ------------------- | --------------------------------------------------------------- |
| ODE     | `"ode"`    | `cpu`  | no                 | no                  | **yes**             | CPU **reference** — wraps the analytical RK4 EOM + `M(q)`/bias  |
| MuJoCo  | `"mujoco"` | `cpu`  | no                 | no                  | **yes**             | CPU MuJoCo — independent dynamics **primitives** for the gate   |
| MJWarp  | `"mjwarp"` | `cuda` | **yes**            | no                  | no                  | GPU **batched** rollouts; no per-call dynamics primitives       |
| MJX     | `"mjx"`    | `jax`  | **yes**            | **yes**             | no                  | JAX-native differentiable batched rollouts; no dense primitives |

Read across the matrix:

- **`ode`** is the ground truth. It wraps `DoublePendulumDynamics` and exposes
  `mass_matrix` / `bias_forces` directly from the analytical expressions
  (`coriolis + gravity + damping`). Single-rollout, CPU-only.
- **`mujoco`** is the _cross-validation_ engine. It compiles the MJCF model and
  exposes `M(q)` (via `mj_fullM`) and bias (`qfrc_bias - qfrc_passive`) so the
  analytical derivation can be checked against an independent one. CPU-only and
  intentionally **not** batched — the CPU MuJoCo path is for correctness and
  single rollouts, not throughput.
- **`mjwarp`** is the _throughput_ engine. It is batched (`rollout_batch` over
  `num_envs`) and GPU-resident. It deliberately does **not** advertise
  `provides_dynamics`: MJWarp's data model does not expose a per-call dense
  `M(q)` the way the CPU engine does, so rather than fake it we mark the
  capability `False` and route any `mass_matrix` request to a
  `BackendCapabilityError`. Asking for a capability a backend doesn't have is a
  loud failure, not a silent wrong answer.
- **`mjx`** is the differentiable rollout engine. It also reuses the generated
  MJCF, returns the same host-side `Trace` / `BatchTrace` schema, and exposes a
  JAX-native array path for rollout-control gradients. Dense `M(q)` / bias
  primitive requests still belong to `ode` or `mujoco`.

### Honest value assessment — GPU pays off for batches, not single 2-DoF rollouts

This is the most important and most counter-intuitive consequence, so it is
stated plainly here and repeated in the user guide:

> **A single 2-DoF rollout is _slower_ on the GPU than on the CPU.**

For one rollout of a two-joint pendulum, the dominant costs are kernel-launch
latency and host↔device memory transfer, both of which swamp the trivial
arithmetic of a 2×2 mass matrix. The CPU analytical stepper finishes before the
GPU has finished staging the work. The GPU only wins when the _same_ launch
amortises across many environments running in parallel — i.e. `rollout_batch`
with a large `num_envs`. As a rule of thumb for this model:

- **1 rollout**, or a handful: use `ode` (or `mujoco`). The GPU is pure
  overhead.
- **Hundreds-to-thousands of rollouts** of the same model with varied
  controls/parameters: use `mjwarp`. This is where the GPU's parallelism
  dominates the fixed launch/transfer cost.

We are **not** claiming the GPU accelerates the golf model in general. We are
claiming it accelerates _batched_ workloads, and we expose that distinction in
the capability matrix (`supports_batched`) rather than hiding it behind a
"faster backend" label. For users who only ever run single rollouts, the GPU
backend should never be installed.

### float32-vs-float64 and tolerance-based cross-validation

The CPU backends compute in float64 (NumPy / MuJoCo CPU). MJWarp computes in
**float32** on the GPU. The two will therefore **never** be bit-identical, and
any cross-validation that asserts `==` (or `np.array_equal`) across the CPU/GPU
boundary is wrong by construction and will flake.

The decision is to cross-validate with **tolerances**, never equality:

- CPU↔CPU comparisons (ODE analytical vs MuJoCo CPU `M(q)` and bias) are both
  float64 and match to ~`1e-9`–`1e-11` — see the verified result below. These
  use a tight `atol`/`rtol`.
- Any comparison that crosses into MJWarp's float32 world uses a _loose_
  tolerance (a few `1e-3` to `1e-4` on state trajectories after many steps,
  because float32 error accumulates over a rollout). Trajectories are compared
  with `np.allclose(..., rtol=..., atol=...)` sized to the float32 epsilon and
  the horizon length, not with equality.

This is captured as a project invariant: **cross-backend agreement is asserted
with `np.allclose` and an explicit, documented tolerance; never with `==`.**

### MJWarp alpha/active status, version pinning, and upgrade path

MJWarp is an **alpha, actively-developed** project. Its API and kernel set
change between releases, and not every MuJoCo feature is implemented yet. We
therefore:

- **Pin** `mujoco-warp` and `warp-lang` to known-good versions in the optional
  `[warp]` extra rather than tracking `main`. The pin is the _only_ version of
  the GPU stack the cross-validation suite is verified against.
- Treat a version bump as a **deliberate, tested change**: bump the pin,
  re-run the GPU cross-validation suite on CUDA hardware, and only then update
  the pin in `pyproject.toml`. The `capabilities.warp_device_available()` probe
  guards against a wheel that imports but has no usable device.
- Isolate all MJWarp-specific code inside `mjwarp_backend` so an upstream API
  change touches exactly one module behind the Protocol; the ODE and MuJoCo-CPU
  backends and every caller are insulated.

### Optional `[warp]` extra and graceful CPU-only degradation

The GPU stack is an **optional dependency**, installed via
`pip install 'upstream-drift[warp]'`. The package import path has **zero** GPU
dependency:

- `capabilities.has_warp()` / `has_mujoco()` perform guarded imports and never
  raise; `warp_device_available()` additionally probes for a live CUDA device.
- `make_backend('mjwarp', ...)` on a CPU-only box raises a
  `BackendNotAvailableError` with an actionable install message — it does not
  fail with a bare `ImportError` at package-load time, because the factory
  imports backend modules **lazily** only when requested.
- The ODE and MuJoCo-CPU backends are fully functional with no extra installed,
  so CI, laptops, and CPU-only CI runners get the complete reference + dynamics
  story without CUDA.

### "One model, many renderers" invariant and the `M(q)` cross-validation gate

`GolfModelParams` is the single source of truth. It is rendered into **two**
downstream representations, never hand-maintained in parallel:

1. `GolfModelParams.to_double_pendulum_parameters()`
   → the analytical EOM parameters consumed by the ODE backend.
2. `simulation_backends.mjcf.params_to_mjcf()`
   → the MJCF XML consumed by both MuJoCo backends.

Because both renderers consume the _same_ immutable instance — and the default
constants are imported from the analytical module rather than duplicated — the
analytical model and the MuJoCo model cannot silently drift. A regression test
asserts that perturbing any parameter changes _both_ outputs.

The keystone of the decision is the **`M(q)` cross-validation gate**: the dense
joint-space inertia matrix and bias forces computed by the analytical ODE
backend are compared, across a grid of configurations, against those computed by
the independent MuJoCo CPU engine. Agreement to numerical tolerance is treated
as the acceptance criterion for "the MuJoCo model is the same physics as the
analytical model".

#### Verified result

Running the gate (both float64): the MuJoCo CPU `M(q)` and bias forces match the
analytical double-pendulum model to approximately **`1e-9` to `1e-11`** across
the tested configuration grid. This confirms the two independent derivations
agree to near machine precision, validating the "one model, many renderers"
invariant for the dynamics primitives.

## Alternatives Considered

1. **Keep only the analytical ODE backend.** Rejected: no parallel throughput
   for sweeps/sampling-based optimisation, and no independent EOM to
   cross-check the hand-derived dynamics against.
2. **Make MuJoCo a hard dependency and drop the analytical engine.** Rejected:
   the analytical engine is the human-readable ground truth and runs with zero
   third-party physics deps; losing it would remove the very reference the
   MuJoCo engine is validated against.
3. **Expose `mass_matrix` from the MJWarp backend by reconstructing it on the
   host.** Rejected: it would either be a float32 round-trip of dubious value or
   a CPU recomputation masquerading as a GPU capability. Marking
   `provides_dynamics=False` and raising `BackendCapabilityError` is honest.
4. **Track `mujoco-warp` `main` instead of pinning.** Rejected: MJWarp is alpha;
   an unpinned dependency would break the cross-validation suite unpredictably.
   A tested pin with a deliberate upgrade path is safer.
5. **Assert bitwise equality across backends.** Rejected as physically wrong:
   float32 (GPU) vs float64 (CPU) cannot be equal; tolerance-based comparison is
   the only correct cross-validation.

## Consequences

- **Positive:**

  - Three interchangeable engines behind one Protocol and one factory; the
    analysis layer is backend-agnostic.
  - An independent dynamics derivation (MuJoCo CPU) cross-validates the
    analytical model to ~`1e-9`–`1e-11`, giving high confidence in the physics.
  - GPU throughput is available for the workloads that actually benefit
    (batched rollouts) without forcing GPU deps on anyone else.
  - Optional-dependency discipline: the suite imports and runs CPU-only;
    missing GPU stack degrades to a clear, actionable error.
  - Capability flags + segregated Protocols mean callers ask for exactly the
    service they need and get a loud failure otherwise.

- **Negative:**

  - Three integrators to maintain, two of which (MuJoCo CPU/GPU) carry external
    dependencies with their own release cadence.
  - MJWarp's alpha status means version bumps require manual GPU re-validation.
  - The float32/float64 split forces every cross-backend test author to choose
    and justify a tolerance — there is no "just use `==`" escape hatch.

- **Follow-ups:**
  - A differentiable backend (MJX or hand-written Warp kernels) is **out of
    scope** here and tracked separately in **ADR-0024** — MJWarp is _not_
    differentiable via Warp autodiff today.
  - Periodic re-validation of the MJWarp pin against newer MuJoCo releases as
    the project matures past alpha.

## Validation

- The `M(q)` cross-validation gate compares analytical (`ode`) vs MuJoCo CPU
  (`mujoco`) `mass_matrix` and `bias_forces` over a configuration grid and
  asserts agreement to ~`1e-9`–`1e-11` (float64 both sides). Verified.
- A "one model, two renderers" regression test asserts that perturbing a
  `GolfModelParams` field changes both `to_double_pendulum_parameters()` and
  `params_to_mjcf()` outputs.
- GPU-path tests are marked `@pytest.mark.requires_gpu` and skipped via
  `skipif(not warp_device_available(), ...)`; CPU-only CI exercises `ode` and
  (when installed) `mujoco`, both guarded by `has_mujoco()`.
- Cross-backend trajectory comparisons that cross the float32/float64 boundary
  use `np.allclose` with an explicit, documented tolerance — never equality.
- `make_backend('mjwarp', ...)` without the `[warp]` extra raises
  `BackendNotAvailableError` (tested), proving graceful CPU-only degradation.
