# Assumptions — Option 4

Everything Option 4 takes as given. If any of these change, the adapter is no longer guaranteed to work and the relevant tests must be rewritten.

Cross-cutting assumptions are also in the shared specs and are linked back here. Option 4 does not relax any of them.

## 1. MATLAB and Simulink licenses are required at runtime

- The host that runs `SimscapeAdapter` **must** have a MATLAB license, a Simulink license, and a Simscape Multibody license that can all be checked out at engine startup.
- If a license check fails, the adapter raises `LicenseError` (a subclass of `SimulationError`) with a clear message. It does **not** silently fall back to a different engine.
- CI hosts without licenses must mark Option 4 tests with `pytest.mark.skipif(not has_matlab_license(), reason="...")`. See [TESTING.md § skip policy](TESTING.md#skip-policy).
- A floating-license deployment is supported; per-host concurrency is bounded by the license count, which is the practical upper bound on `SimscapeAdapterPool.pool_size`.

## 2. MATLAB Engine API for Python is installed and importable

- The Python environment has `matlabengine` installed via `python -m pip install matlabengine`. **The package version must match the host MATLAB release** (e.g. `matlabengine==24.1.*` for MATLAB R2024a). Mismatches surface as `ImportError` at engine startup.
- See [INSTALLATION.md](INSTALLATION.md) for the version-compatibility matrix and the install procedure on Windows.
- The adapter does a one-time `import matlab.engine` inside `__init__` and raises `EngineStartupError` (with the install hint) if it fails. It does **not** do this at module import time — module import remains side-effect-free so unit tests on machines without MATLAB still load `simscape_adapter.py`.

## 3. The .slx model is callable headlessly (no GUI)

- `GolfSwing3D_Kinetic.slx` runs without opening the Simulink editor. The adapter starts MATLAB with `-nodesktop -nosplash` (or via `matlab.engine.start_matlab("-nodesktop -nosplash")`).
- The model must compile and run with `sim()` from a fresh MATLAB session. Any GUI-only callbacks, missing toolbox dependencies, or interactive dialogs that would block headless execution are bugs and must be fixed in the .slx (out of scope for this option — file an upstream issue).
- The model is treated as **fixed**. Option 4 does not modify `.slx`, body parameters, or integrator settings. See [shared/README — Hard constraints](../README.md#hard-constraints-assumptions-all-four-options-must-respect).

## 4. Per-call latency is acceptable for inverse-problem fits but **not** for RL inner loops

- One `simulate_with_coefficients(theta)` round trip costs **~50–200 ms** end-to-end on a warm engine, dominated by:
  - Python ↔ MATLAB Engine marshalling: ~1–5 ms.
  - `set_param`/`setVariable` of the coefficient struct: ~5–20 ms.
  - Simscape `sim()` execution at 1 kHz × 0.3 s: ~30–150 ms.
  - `logsout → numpy` extraction: ~5–20 ms.
- This is fine for `system_identification` (typically < 10⁴ evaluations per fit) and for round-trip validation of Options 2 and 3.
- It is **not** fine for RL training (10⁵–10⁷ steps). Hybrid recommendation: use Option 2's surrogate inside the inner loop, use the adapter for periodic eval rollouts only. See [APPROACH.md § Hybrid with Options 1/2/3](APPROACH.md#hybrid-with-options-123).

## 5. Engine startup is amortized across many calls

- The MATLAB Engine takes **~10–30 s** to start (process spawn + license checkout + path setup + first model load).
- The adapter is designed for **one engine per process, lazy-started, kept alive for the process lifetime**. Re-creating an engine per call would be ~1000× slower than the per-call latency in assumption 4.
- `SimscapeAdapterPool` (issue #038) reuses a fixed number of engines across calls. It does not start/stop engines per call.

## 6. Determinism

- Within a single MATLAB session, `simulate_with_coefficients(theta)` returns **bit-identical** output for identical input (same `theta`, same model params, same rng seed propagated through `setVariable` to the model's `Simulink.SimulationInput`).
- **Across** MATLAB sessions, output is RMSE-equivalent but not bit-identical (BLAS/MKL differences). The cache (assumption 9) is keyed by MATLAB version to avoid stale-cache issues.
- The `rng_seed` field on `SimscapeAdapter` is propagated to the model on every call; callers that need full reproducibility must set it before any `simulate_*` call. See [APPROACH.md § Determinism](APPROACH.md#determinism).

## 7. Decision variables: polynomial coefficients (same as Options 1–3)

- `theta` is a flat numpy array of length `n_joints × 7`, joint-major / coefficient-minor (the same convention as [option1/ASSUMPTIONS § 2](../option1_direct_optimization/ASSUMPTIONS.md#2-decision-variables-polynomial-coefficients)).
- Bounds are unchanged: A,B ∈ ±1000; C,D ∈ ±500; E,F ∈ ±100; G ∈ ±25.
- The adapter does **not** validate bounds — that is the optimizer's job. The adapter passes whatever it receives to MATLAB and lets the model error if the coefficients diverge the integrator.

## 8. State-management strategy

- The adapter sets coefficients and other run-time inputs by building a `Simulink.SimulationInput` object on the MATLAB side via `setVariable`. It does **not** mutate the .slx file or the base workspace. See [APPROACH.md § State management](APPROACH.md#state-management).
- Long-lived model parameters (link masses, joint damping, friction) are set with `set_param` once when the model is loaded, then re-used across calls. Per-call inputs (coefficients, rng seed) go through `Simulink.SimulationInput.setVariable`.
- Two parallel engines never share a `SimulationInput` object. State isolation is per-engine; the pool guarantees no cross-talk (`test_concurrent_engines_isolated`).

## 9. Caching

- The adapter caches `(sha256(coefficients), sha256(model_params), matlab_version) → SimscapeOutput`.
- Cache lookup is in-process and never crosses worker boundaries (consistent with [option1/ASSUMPTIONS § 9](../option1_direct_optimization/ASSUMPTIONS.md#9-parallelism)).
- `cache_max_entries` defaults to 1024 (LRU eviction). For a typical `system_identification` fit (~10³ evaluations) this means full coverage; for larger sweeps the user is expected to bump the limit.
- The cache is opt-in via `SimscapeAdapter(cache_enabled=True)`; it defaults to **on**. Disabling it is supported for benchmarking and for tests that want to measure raw throughput.

## 10. Failure mode: MATLAB exceptions

- Any exception raised in MATLAB (integrator divergence, license loss mid-call, model-compile error) is wrapped in `SimulationError` with the original MATLAB error text preserved on `.matlab_traceback`.
- The adapter does **not** auto-restart the engine on failure. The optimizer is responsible for retry policy. (Auto-restart is a v2 feature — see [APPROACH.md § Open future work](APPROACH.md#open-future-work).)
- If the engine process dies (segfault, SIGKILL), the adapter detects the broken pipe on the next call and raises `EngineStartupError("engine process died, restart required")`.

## 11. Coordinate convention (same as the other options)

- Position in metres, time in seconds, orientation as unit quaternion `[w x y z]`.
- The adapter exposes the same canonical schema as the other options (`SimscapeOutput.butt`, `.clubhead`, `.club_quat`).

## 12. Protocol compliance

- `SimscapeAdapter` satisfies the **full** `PhysicsEngine` protocol from [`src/shared/python/engine_core/interfaces.py`](../../../../../../shared/python/engine_core/interfaces.py): Loadable, Steppable, Queryable, DynamicsComputable, CounterfactualComputable, Recordable.
- Methods that are not natural for a Simscape model (e.g. `compute_jacobian` on a body that doesn't exist in the .slx) raise `NotImplementedError("not supported by Simscape adapter")` rather than returning bogus values. The protocol allows this — `compute_jacobian` is permitted to return `None`. See [INTERFACES.md § Protocol-method coverage matrix](INTERFACES.md#protocol-method-coverage-matrix).

## 13. Thread / process model

- One `SimscapeAdapter` instance owns one MATLAB Engine process. Sharing a single instance across threads is **not** safe — the underlying `matlab.engine` is not thread-safe on Windows. Use `SimscapeAdapterPool` for concurrency.
- The pool uses Python `concurrent.futures.ProcessPoolExecutor` semantics: each pool worker has its own engine, its own cache, and its own state.

## 14. Windows-first

- The host platform is Windows 11 (the user's dev box). The install instructions in [INSTALLATION.md](INSTALLATION.md) are Windows-specific.
- Linux and macOS are best-effort: the same `pip install matlabengine` command works, but the version-compatibility matrix and the path conventions differ. CI does not exercise Option 4 on non-Windows hosts.

---

### Cross-cutting links (do not duplicate)

- Cost function spec: [shared/COST_FUNCTION_SPEC.md](../shared/COST_FUNCTION_SPEC.md).
- Club IK / target schema: [shared/CLUB_IK_SPEC.md](../shared/CLUB_IK_SPEC.md).
- Coding standards (TDD, DbC, DRY, LOD, file size): [shared/CODING_STANDARDS.md](../shared/CODING_STANDARDS.md).
- Visualization spec: [shared/VISUALIZATION_SPEC.md](../shared/VISUALIZATION_SPEC.md) and [VISUALIZATION.md](VISUALIZATION.md).
- Repository-wide standards (ruff, file size, pytest markers): [`CLAUDE.md`](../../../../../../../CLAUDE.md).
