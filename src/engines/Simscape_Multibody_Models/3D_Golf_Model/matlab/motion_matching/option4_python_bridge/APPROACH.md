# Approach — Option 4

How `SimscapeAdapter` wraps MATLAB's Simscape Multibody simulator, what state it owns, how it routes errors, and how it interoperates with the other three options.

## Architecture

```
+----------------------------------+
|  Python process                  |
|  +----------------------------+  |
|  |  Caller                    |  |
|  |  (system_identification,   |  |
|  |   dataset_generator,       |  |
|  |   RL env, retargeter, ...) |  |
|  +-------------+--------------+  |
|                | PhysicsEngine   |
|                | protocol calls  |
|  +-------------v--------------+  |
|  |  SimscapeAdapter           |  |
|  |  (this option)             |  |
|  |  - cache                   |  |
|  |  - logsout -> numpy        |  |
|  |  - error wrapper           |  |
|  +-------------+--------------+  |
|                | matlab.engine   |
|                | RPC (stdio)     |
+----------------|-----------------+
                 |
+----------------v-----------------+
|  MATLAB Engine process           |
|  +----------------------------+  |
|  |  matlab.engine session     |  |
|  |  - simulate_with_coeffi-   |  |
|  |    cients(theta) (#018)    |  |
|  |  - load_system(.slx)       |  |
|  |  - sim() -> logsout        |  |
|  +-------------+--------------+  |
|                |                 |
|  +-------------v--------------+  |
|  |  Simscape Multibody solver |  |
|  |  GolfSwing3D_Kinetic.slx   |  |
|  +----------------------------+  |
+----------------------------------+
```

A pool deployment runs N of these MATLAB Engine processes side-by-side, each owned by one `SimscapeAdapter` in a pool worker. See [§ Pooling](#pooling).

## Lifecycle

### Lazy startup

```
SimscapeAdapter.__init__()      # cheap — no MATLAB yet
SimscapeAdapter.load_from_path(slx_path)
    |
    +-- if engine is None:
    |     import matlab.engine                   # raises EngineStartupError on failure
    |     engine = matlab.engine.start_matlab(   # 10–30 s, license checkout
    |              "-nodesktop -nosplash")
    |     engine.addpath(motion_matching_root, nargout=0)
    |
    +-- engine.load_system(slx_path, nargout=0)  # loads .slx into MATLAB workspace
```

The first `load_from_path` pays the engine-startup cost. Every subsequent call (including `step`, `simulate_with_coefficients`, `reset`) reuses the live engine.

### Single shared engine per process

Each `SimscapeAdapter` owns one `matlab.engine` instance for the lifetime of the Python process or until `close()` is called explicitly. The protocol is:

- `SimscapeAdapter.__init__(...)` → no MATLAB started yet.
- First call that needs MATLAB → engine starts.
- Engine remains live across all subsequent calls.
- `SimscapeAdapter.close()` → `engine.quit()`. Idempotent.
- `__del__` calls `close()` defensively, but the test suite enforces explicit `close()` in fixtures so engines do not leak. See `test_engine_starts_and_stops_cleanly` in [TESTING.md](TESTING.md).

### Pooling

`SimscapeAdapterPool(pool_size=N, model_path=...)` starts N engines. Each engine lives in its own pool-worker process (using `concurrent.futures.ProcessPoolExecutor`). The pool's public surface:

```python
pool = SimscapeAdapterPool(pool_size=4, model_path=slx_path)
results = pool.map_simulate(thetas)   # list[np.ndarray] -> list[SimscapeOutput]
pool.close()
```

The pool is opt-in. The default deployment is a single shared engine. Pool size is bounded above by the host's MATLAB license count (assumption #1 in [ASSUMPTIONS.md](ASSUMPTIONS.md)).

## State management

Three categories of state, three different mechanisms:

| Category                      | Examples                                                  | Mechanism                                                           | Frequency                                      |
| ----------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------- |
| Long-lived model parameters   | link masses, joint damping, friction, integrator settings | `engine.set_param(model_name, "Mass", "5.0", nargout=0)`            | Once at load, then on `set_link_masses` etc.   |
| Per-call run inputs           | polynomial coefficients, rng seed                         | `Simulink.SimulationInput` + `setVariable`                          | Every `simulate_with_coefficients` call        |
| Live state during stepped sim | q, v at time `t`                                          | Block parameter writes + `set_param("SimulationCommand", "update")` | Per `step()` (slow path; only used by RL eval) |

The first two are the hot paths and what the existing MATLAB code (`simulate_with_coefficients.m` from issue #018) already supports. The third is the awkward path; see [§ Stepped vs whole-simulation execution](#stepped-vs-whole-simulation-execution).

### Why `Simulink.SimulationInput` instead of base-workspace assignment

`set_param`/`assignin('base', ...)` mutates global state and would race between concurrent calls in a pool. `Simulink.SimulationInput` is per-call, scoped to a single `sim()` invocation, and is the MathWorks-recommended pattern for parsim/parallel runs. It also keeps the model file unmodified, which means git diffs stay clean.

```matlab
% MATLAB side, called from Python via engine.feval()
function out = simulate_with_coefficients(model_name, theta, opts)
    in = Simulink.SimulationInput(model_name);
    in = in.setVariable("PolynomialCoefficients", reshape(theta, [], 7));
    in = in.setVariable("RngSeed", opts.rng_seed);
    in = in.setVariable("SimulationDuration", opts.T);
    out = sim(in);
end
```

The Python side passes `theta` as a `matlab.double` (no copy on the MATLAB side):

```python
theta_m = matlab.double(theta.flatten().tolist())
matlab_out = self._engine.simulate_with_coefficients(
    self._model_name, theta_m, self._opts_struct, nargout=1
)
```

### Stepped vs whole-simulation execution

The protocol's `step(dt)` semantics fit a per-step solver loop. Simscape's natural unit is `sim(model, T_end)` — a whole-simulation call. The adapter handles this by:

- **Whole-simulation path (recommended).** `simulate_with_coefficients(theta)` runs a 0.3 s sim end-to-end and returns the full `SimscapeOutput`. This is the path `system_identification`, `dataset_generator`, and the validation-oracle role of Options 2 and 3 use. **~50–200 ms per call.**
- **Stepped path (slow).** `step(dt)` calls `sim(model, get_time() + dt)` with the current state injected via `Simulink.SimulationInput.setInitialState`. Each call pays the dispatch + setup overhead. This is the path RL evaluation rollouts use; it is **~10–20 ms per step** at best, prohibitively slow for inner-loop training.

Both paths use the same underlying `simulate_with_coefficients.m`. The protocol coverage table in [INTERFACES.md](INTERFACES.md#protocol-method-coverage-matrix) marks `step` as supported but slow.

## Output extraction

Simscape emits `logsout`, a hierarchical `Simulink.SimulationData.Dataset`. The adapter converts it to a flat `SimscapeOutput` numpy dataclass:

```python
@dataclass(frozen=True)
class SimscapeOutput:
    time:       np.ndarray   # (N,)   float64 — simulation timegrid (seconds)
    butt:       np.ndarray   # (N, 3) float64 — butt position (metres)
    clubhead:   np.ndarray   # (N, 3) float64 — clubhead position (metres)
    club_quat:  np.ndarray   # (N, 4) float64 — [w, x, y, z], unit-norm
    q:          np.ndarray   # (N, n_q)
    v:          np.ndarray   # (N, n_v)
    tau:        np.ndarray   # (N, n_joints) — applied joint torques
    omega:      np.ndarray   # (N, n_joints) — joint angular velocities
    impact_idx: int          # index of max clubhead speed
```

Conversion uses MATLAB-side helpers (already part of issue #018) so we avoid round-tripping the full `Dataset` through the engine — only flat double arrays cross the bridge.

The schema mirrors `ClubTarget` from [shared/CLUB_IK_SPEC.md](../shared/CLUB_IK_SPEC.md) so the cost function can subtract a `SimscapeOutput` from a `ClubTarget` directly.

## Error handling

```
MATLAB exception (any kind)
    -> matlab.engine.MatlabExecutionError on the Python side
    -> wrapped by SimscapeAdapter into SimulationError(...)
        - .matlab_traceback : str  (full MATLAB stack)
        - .matlab_error_id  : str  (e.g. "Simulink:Compile:...")
        - .__cause__        : original MatlabExecutionError
```

Sub-types:

| Python exception     | When raised                                                                                                                    |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `EngineStartupError` | `import matlab.engine` failed, `start_matlab` failed, license checkout failed at startup, or the engine process died mid-call. |
| `LicenseError`       | License checkout failure with a recognizable error id (`MATLAB:license:...`). Subclass of `EngineStartupError`.                |
| `SimulationError`    | Any error during a `sim()` call (integrator divergence, missing block, etc.).                                                  |
| `ModelLoadError`     | `load_system` failed (file missing, .slx corrupt). Subclass of `SimulationError`.                                              |

Errors are **not** caught and converted to dummy outputs. The optimizer is responsible for the penalty-on-failure policy; see [option1/ASSUMPTIONS § 12](../option1_direct_optimization/ASSUMPTIONS.md#12-failure-mode-simulator-side-errors).

## Caching

```python
key = sha256(theta.tobytes()) + sha256(json(model_params)) + matlab_version
if key in self._cache:
    return self._cache[key]
out = self._run_simulation(theta)
self._cache[key] = out
return out
```

- LRU bounded at `cache_max_entries` (default 1024).
- Cache is opt-in (default on); disable via `SimscapeAdapter(cache_enabled=False)`.
- Cache is in-process; pool workers have separate caches.
- Cache invalidation: `set_link_masses`, `set_joint_damping`, `load_from_path`, and any other `set_*` call clears the cache.

See [TESTING.md](TESTING.md) for `test_cache_hit_skips_simulation`.

## Determinism

- The adapter takes an `rng_seed: int = 42` constructor argument.
- On every `simulate_with_coefficients(theta)` call, the seed is propagated to MATLAB via `Simulink.SimulationInput.setVariable("RngSeed", seed)` and consumed inside the .slx (or by a wrapper script) before any random block fires.
- Within a single MATLAB session, repeated calls with the same `theta` and the same seed return bit-identical output. **Verified by `test_simulate_with_known_coefficients_matches_matlab_direct`**.
- Across MATLAB sessions, output is RMSE-equivalent but not bit-identical. The cache key includes the MATLAB version to prevent stale-cache reads after a MATLAB upgrade.

## Decision-variable layout

Identical to Options 1–3 (joint-major, coefficient-minor):

```
theta_flat[7*j + k] = coefficient k of joint j
```

where `j ∈ [0, n_joints)` and `k ∈ [0, 7)` (with `k=0` ↔ A ↔ t^6, ..., `k=6` ↔ G ↔ t^0). The adapter reshapes flat → `(n_joints, 7)` on the MATLAB side.

## Hybrid with Options 1, 2, 3

The bridge is the **glue** between the four options.

### As Option 1's Python sibling

Option 1's `fit_swing_*.m` runs the optimizer entirely in MATLAB. With the bridge, a Python `system_identification.SystemIdentifier` can run the **same** optimization (or a different one — `scipy.optimize`, Optuna, BayesianOptimization) against the **same** simulator. Useful for:

- Cross-validating that the same target converges to the same coefficients regardless of optimizer environment.
- Reusing existing Python tooling (Optuna dashboard, mlflow, wandb) without rewriting in MATLAB.

### As Options 2 and 3's round-trip oracle

```
target  --> Option 2 surrogate --> theta_hat
                                    |
                                    v
                              SimscapeAdapter.simulate_with_coefficients(theta_hat)
                                    |
                                    v
                              compare(SimscapeOutput, target)
                                    |
                                    +-- |residual| < budget   -> accept
                                    \-- |residual| >= budget  -> reject (out-of-distribution)
```

This is the validation step Options 2 and 3 cannot perform without a Python-callable Simscape. Today it would require `pyrunfile` boilerplate per call; Option 4 collapses it to one method invocation.

### As Option 2's training-data generator

`dataset_generator/core.py` (Python) currently has to shell out to MATLAB or call `pyrunfile` to generate the parquet sweep. With the adapter:

```python
adapter = load_matlab_3d_engine(suite_root)
for theta in random_coefficient_iterator(n=10_000):
    out = adapter.simulate_with_coefficients(theta)
    write_to_parquet(theta, out)
```

This is single-process; for a 10⁴-trial sweep on a 4-engine pool, throughput is ~20–80 trials/s vs. the surrogate's training requirement of ~10⁴ trials per epoch.

## Integration list

The following Python modules **work against the Simscape model** the moment Option 4 lands. None of them require any change beyond passing a `SimscapeAdapter` instance.

| Module                                                                                                           | What it does                                                  | What changes with Option 4                                                                                                 |
| ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `src/learning/sim2real/system_identification.py`                                                                 | Fits link masses, damping, friction to measured trajectories. | Today: works against MuJoCo/Drake/Pinocchio. Tomorrow: works against Simscape — the highest-fidelity option in the suite.  |
| `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/functions/dataset_generator/` (Python `core.py`) | Generates the random-sweep parquet dataset.                   | Today: shells out to MATLAB or uses `pyrunfile`. Tomorrow: clean Python loop calling `adapter.simulate_with_coefficients`. |
| `src/shared/python/data_io/swing_capture_import.py`                                                              | Imports C3D / CSV / JSON swing demos as `Demonstration`s.     | Demos can now be replayed and re-simulated through Simscape for ground-truth comparison.                                   |
| `src/learning/rl/humanoid_envs.py` (and `manipulation_envs.py`)                                                  | RL gym environments.                                          | Eval rollouts run through the adapter; training inner loop still uses Option 2's surrogate.                                |
| `src/learning/imitation/` (retargeter, BC, GAIL)                                                                 | Imitation learning over `Demonstration`s.                     | Retargeted demos can be validated against Simscape.                                                                        |
| `src/learning/sim2real/domain_randomization.py`                                                                  | Randomizes engine params for sim-to-real.                     | Now includes Simscape link masses / damping / friction in the randomization domain.                                        |

## Open future work

- **Engine auto-restart on segfault.** v1 surfaces `EngineStartupError`; v2 should auto-restart and replay the last call.
- **Cross-process cache.** v1 caches in-process; a shared on-disk cache (sqlite or parquet) would let pool workers and re-runs share results.
- **Stream-mode simulation.** v1 only supports whole-sim and slow stepped. A streaming mode that emits state at every solver step without restarting the sim would close the RL inner-loop gap (still 10× slower than the surrogate, but usable for short rollouts).
- **Linux/macOS CI.** v1 is Windows-only in CI. Adding Ubuntu CI requires a MATLAB-on-Linux license and a self-hosted runner.
