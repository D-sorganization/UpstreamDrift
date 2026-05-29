# ADR-0024: Differentiable backend — MJX (JAX) vs custom Warp kernels

- Status: Proposed
- Date: 2026-05-29
- Decision Makers: @D-sorganization/maintainers
- Related Issues/PRs: GPU-accelerated simulation-backend epic; follows ADR-0023

## Context

ADR-0023 added three integrators behind the `SimulationBackend` Protocol: the
analytical `ode` reference, the `mujoco` CPU dynamics-primitives engine, and the
`mjwarp` GPU **batched** engine. None of them is **differentiable** — gradients
cannot flow through `rollout`, so the `is_differentiable` capability flag is
`False` on all three.

Some trajectory-optimisation methods want gradients of a rollout's outcome with
respect to the controls (or model parameters): direct-shooting with a
gradient-based optimiser (L-BFGS, Adam), differentiable MPC, or learning model
parameters by backprop through the simulator. For the golf swing this would mean
"optimise the shoulder/wrist torque schedule to maximise clubhead speed" via
gradients rather than the sampling-based methods (MPPI/CEM) that the existing
batched `mjwarp` backend already serves.

The key technical fact that forces this to be its **own** decision:

> **MuJoCo Warp (MJWarp) is _not_ differentiable via Warp autodiff today.**

Although NVIDIA Warp has an autodiff facility in general, the MJWarp physics
kernels are not written to be traced/differentiated through Warp's tape in the
current alpha. So the GPU backend we already have cannot simply be flipped to
`is_differentiable=True`. A differentiable backend is a genuinely new
implementation, and there are two credible ways to build one.

## Decision

**Recommend MJX (JAX) as the differentiable backend, but defer building it until
a concrete gradient-based use-case exists.** Whichever option is eventually
chosen, it implements the **same** `SimulationBackend` Protocol from ADR-0023 —
the differentiable engine is another interchangeable backend (e.g.
`make_backend('mjx', params)`), not a parallel architecture. It would advertise
`is_differentiable=True` and, like `mjwarp`, would be an optional extra that
degrades gracefully when its stack is absent.

Concretely:

- Do **not** implement a differentiable backend now. No current workload in the
  epic needs rollout gradients; the sampling-based optimisation served by
  `mjwarp` covers the present need.
- When a concrete gradient use-case lands, **prefer MJX** (see rationale below)
  and revisit this ADR (promote to Accepted, fill in the chosen pin and the
  gradient cross-validation plan).
- Keep the door open structurally: the Protocol and capability flag
  (`is_differentiable`) already exist precisely so a differentiable backend
  slots in without disturbing callers.

## Alternatives Considered

1. **MJX (MuJoCo on JAX) — recommended.** MJX re-expresses MuJoCo's dynamics in
   JAX, so a rollout is differentiable through `jax.grad`/`jax.jacobian` and
   JIT-compilable to CPU/GPU/TPU. It is the same MuJoCo physics family as the
   `mujoco`/`mjwarp` backends, which keeps cross-validation conceptually
   simple (compare MJX float32 trajectories against the CPU reference with a
   tolerance, exactly as in ADR-0023). It is more mature for autodiff than
   MJWarp and `vmap` gives batching for free. Cost: adds a JAX dependency and a
   second GPU/accelerator stack; JAX's tracing/JIT model is a different
   programming style from the imperative backends.

2. **Hand-written NVIDIA Warp kernels with autodiff.** Write the 2-DoF EOM as
   custom Warp kernels authored to be differentiable on Warp's tape, reusing the
   GPU stack the `[warp]` extra already pulls in (no JAX dependency). Cost: we
   own and maintain the analytical gradients/kernels by hand for every model
   change; far more implementation and validation effort than reusing MJX; and
   it duplicates dynamics that already exist analytically. Justified only if a
   JAX dependency is unacceptable or the model stays permanently 2-DoF.

3. **Finite-difference gradients over the existing backends.** Wrap `ode` or
   `mjwarp` and estimate `d(outcome)/d(control)` by finite differences. Cheap to
   build and needs no new dependency, but it is `O(n_controls)` rollouts per
   gradient, noisy near float32 precision, and not a true differentiable
   backend. Acceptable as a _stopgap_ for a one-off study, not as the
   architecture.

4. **Make `mjwarp` differentiable.** Rejected as infeasible today: MJWarp's
   kernels are not autodiff-traceable in the current alpha (the motivating fact
   above). Reconsider only if upstream MJWarp ships differentiable kernels.

## Consequences

- **Positive:**

  - Records the non-obvious fact (MJWarp ≠ differentiable) so a future
    contributor does not waste time trying to backprop through `mjwarp`.
  - Names the recommended path (MJX) and the same-Protocol constraint, so the
    eventual implementation is a drop-in backend, not a redesign.
  - Defers cost until a real gradient use-case justifies the extra dependency
    and validation burden (YAGNI).

- **Negative:**

  - The capability `is_differentiable=True` remains unsatisfied by any backend
    until this is implemented; gradient-based optimisation is unavailable in the
    interim (sampling-based via `mjwarp` is the workaround).
  - When built, MJX adds a second accelerator stack (JAX) alongside Warp, with
    its own version-pinning and CUDA-compatibility surface.

- **Follow-ups:**
  - On the first concrete gradient-based trajectory-optimisation requirement:
    promote this ADR to Accepted, pin the MJX/JAX versions in an optional
    extra, define the gradient cross-validation (compare MJX gradients to
    finite-difference gradients of the `ode` reference within tolerance), and
    add `make_backend('mjx', ...)` to the factory registry.

## Validation

- This ADR is **Proposed**; no code ships with it. Validation is deferred to the
  implementation ADR.
- When implemented, validation will mirror ADR-0023's discipline: the MJX
  backend honours the `SimulationBackend` Protocol (structural `isinstance`
  check), advertises `is_differentiable=True`, degrades gracefully without its
  optional extra (`BackendNotAvailableError`), and its gradients are
  cross-validated against finite-difference gradients of the analytical `ode`
  reference using `np.allclose` with a documented tolerance (never `==`, per the
  float32/float64 rule in ADR-0023).
