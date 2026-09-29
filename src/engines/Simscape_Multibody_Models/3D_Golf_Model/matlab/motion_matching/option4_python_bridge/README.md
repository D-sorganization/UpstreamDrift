# Option 4 — Python ↔ Simscape Bridge (`SimscapeAdapter`)

> **Read first**: [PROJECT_SPEC.md](../../../PROJECT_SPEC.md), [MATLAB_GOLF_MODEL_GUIDE.md](../../MATLAB_GOLF_MODEL_GUIDE.md), [GRIP_FIT_PLAYBOOK.md](../shared/GRIP_FIT_PLAYBOOK.md).

> **What.** A Python adapter class `SimscapeAdapter` that satisfies the repo-wide `PhysicsEngineProtocol` and wraps `GolfSwing3D_Kinetic.slx` through the **MATLAB Engine API for Python**. Once this lands, the existing learning stack — `system_identification`, `dataset_generator/core`, `swing_capture_import`, RL envs, retargeter — works against the Simscape model with no further plumbing.
>
> **Why.** The other three options each unlock one workflow (fitting, fast inference, surrogate). This one unlocks **the entire learning stack at once** by making Simscape a first-class `PhysicsEngine` to Python.
>
> **Why not first.** Setup cost is the highest of the four options: MATLAB Engine for Python plumbing, license management, dispatch latency (~50–200 ms per call), engine-process lifecycle, and Windows-specific install gotchas. RL inner loops are infeasible without combining with Option 2's surrogate.

## Status

**Phase 2 — partial implementation landed for issue #4077.** The motion-matching headline surface is shipping:

- `simscape_adapter.py` — `SimscapeAdapter` lifecycle (`start()` / `close()` / `__enter__`/`__exit__`), `simulate_with_coefficients(theta) -> SimOut`, `target_from_xlsx(path, sheet) -> ClubTarget`, `compute_cost(theta, target, opts) -> float`, `get_polynomial_bounds(n_joints)`, `get_n_joints(default=...)`. Errors wrapped as `SimulationError` / `EngineStartupError`.
- `fit_swing_python.py` — `fit_swing_scipy(target, adapter, options) -> FitResult` driving `scipy.optimize.minimize(method="SLSQP")` over the bound-constrained polynomial coefficients. `fit_swing_jax` raises `NotImplementedError` referencing issue #4075 (Option-2 surrogate) — a JAX path needs the differentiable surrogate.
- `tests/` — three `test_simscape_adapter.py` round-trip tests and one `test_fit_swing_python.py` recovery test, all marked `@pytest.mark.requires_matlab_engine`. The conftest auto-skips the marked tests with a loud "matlab.engine not importable" reason on hosts without MATLAB Engine for Python; tests that need Simscape Multibody additionally skip with a license-missing reason.

The full PhysicsEngine protocol coverage (step / reset / compute_mass_matrix / SimscapeAdapterPool / loader.py wiring) is deliberately deferred to issues **#036–#040** (see [GitHub issues](#github-issues-for-option-4)) and tracked separately. Issue #4077's scope was the headline motion-matching surface only.

## When to use this option

- You want to run **`system_identification.py`** against the Simscape model, not just MuJoCo/Drake/Pinocchio.
- You want **`dataset_generator/core.py`** (Python) to drive the canonical Simscape forward sim end-to-end without `pyrunfile` boilerplate.
- You want to round-trip **Option 2 / Option 3** results against the canonical simulator from a Python notebook, without a MATLAB editor open.
- You want the **`SwingCaptureImporter`** demonstrations to feed an RL env that uses Simscape forward dynamics for evaluation (training still needs Option 2's surrogate).
- You are building a hybrid pipeline: surrogate for inner-loop speed, Simscape via this adapter for periodic ground-truth checks.

When **not** to use it:

- Per-step RL training inside a tight loop (10⁵–10⁷ steps). The MATLAB Engine dispatch latency makes this infeasible. Use Option 2's surrogate; come back here only for evaluation rollouts.
- A single one-off fit on a single swing. Option 1 is faster end-to-end because it avoids the engine-startup tax.
- Any environment where MATLAB and Simulink licenses are not available at runtime.

## What it ships

| File                       | Purpose                                                                                                       |
| -------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `simscape_adapter.py`      | `SimscapeAdapter(PhysicsEngine)` — implements the full protocol and adds `simulate_with_coefficients`.        |
| `simscape_adapter_pool.py` | `SimscapeAdapterPool` — pool of N engines for parallel inference.                                             |
| `simscape_output.py`       | `SimscapeOutput` dataclass and the `logsout → numpy` converter.                                               |
| `simscape_errors.py`       | `SimulationError`, `EngineStartupError`, `LicenseError` — Python wrappers around MATLAB exceptions.           |
| `cache.py`                 | Hash-keyed cache of `(coefficients, model_params) → SimscapeOutput`.                                          |
| `loader.py`                | `load_matlab_3d_engine(suite_root)` — to be wired into `src/engines/loaders.py` under `EngineType.MATLAB_3D`. |
| `tests/`                   | `pytest` suite per [TESTING.md](TESTING.md).                                                                  |

The contracts are in [INTERFACES.md](INTERFACES.md). The architecture and lifecycle are in [APPROACH.md](APPROACH.md). The assumptions are in [ASSUMPTIONS.md](ASSUMPTIONS.md). Install steps are in [INSTALLATION.md](INSTALLATION.md). How to run it is in [RUNBOOK.md](RUNBOOK.md). What to test is in [TESTING.md](TESTING.md).

## Immediate consumers (the value justification)

These are the modules that "just work" the moment `SimscapeAdapter` lands. Every one of them is in `src/` today and currently has no path to Simscape.

- **[src/learning/sim2real/system_identification.py](../../../../../../learning/sim2real/system_identification.py)** — fits real-world parameters (link masses, joint damping, friction) against measured trajectories. The optimizer treats the engine as a black box behind `PhysicsEngineProtocol`. Today: works against MuJoCo, Drake, Pinocchio. With Option 4: works against Simscape too.
- **[src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/functions/dataset_generator/](../../src/functions/dataset_generator/)** (Python `core.py` consumer) — Python-side dataset generation that today must shell out to MATLAB or use `pyrunfile`. With Option 4 it calls a clean `engine.simulate_with_coefficients(theta)`.
- **[src/shared/python/data_io/swing_capture_import.py](../../../../../../shared/python/data_io/swing_capture_import.py)** — produces `Demonstration` records that feed imitation learning. Currently engine-agnostic; with Option 4 those demonstrations can be replayed and re-simulated through Simscape for ground-truth comparison.
- **[src/learning/rl/humanoid_envs.py](../../../../../../learning/rl/humanoid_envs.py)** — RL gym environments. Inner-loop training requires the surrogate (Option 2); evaluation rollouts and demo replay run through the adapter.
- **Retargeter** (`src/learning/imitation/`) — re-uses any `PhysicsEngineProtocol` implementation. Adding Option 4 means retargeted demos can be validated against Simscape.

## Dependencies

### Hard dependencies (runtime)

- **MATLAB R2021b+** (R2024a or later strongly preferred — Engine API for Python parity with Python 3.11).
- **Simulink** and **Simscape Multibody** licenses checked out at engine startup.
- **MATLAB Engine API for Python** installed in the active Python env (`python -m pip install matlabengine`). The pinned version **must** match the host MATLAB release; see [INSTALLATION.md](INSTALLATION.md#version-pinning).
- **Python 3.10+** (per repo [`CLAUDE.md`](../../../../../../../CLAUDE.md)) — and the Python minor version must be on MathWorks' compatibility matrix for the installed MATLAB release.

### Soft dependencies (developer-only)

- `pytest` ≥ 7 (for the test suite — already in `requirements.lock`).
- `numpy` ≥ 1.24 (already pinned).

### Shared dependencies (issues #013–#023)

Option 4 consumes — does not re-implement:

- `simulate_with_coefficients.m` (issue #018) — the MATLAB-side forward callback. Option 4's `simulate_with_coefficients(theta)` Python method delegates to it via the engine.
- `compute_cost.m` / `cost.py` (issue #015) — for in-Python verification harnesses; see [shared/COST_FUNCTION_SPEC.md](../shared/COST_FUNCTION_SPEC.md).
- `synthesize_target_from_coefficients.m` (issue #014) — used by `test_simulate_with_known_coefficients_matches_matlab_direct`.

## GitHub issues for Option 4

| #        | Title                                                                                | Notes                                                                                                                                                                              |
| -------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **#036** | `SimscapeAdapter(PhysicsEngine)` core                                                | Engine lifecycle, protocol methods, `simulate_with_coefficients`. Acceptance: passes `test_protocol_compliance` and `test_simulate_with_known_coefficients_matches_matlab_direct`. |
| **#037** | `load_matlab_3d_engine` + registration in `src/engines/loaders.py`                   | Wires `EngineType.MATLAB_3D` to a working factory. Acceptance: `loaders.LOADER_MAP[EngineType.MATLAB_3D]` returns a working adapter on a license-equipped host.                    |
| **#038** | `SimscapeAdapterPool`                                                                | Pool of N engines for parallel inference. Acceptance: `test_concurrent_engines_isolated` passes.                                                                                   |
| **#039** | Output extraction + cache                                                            | `logsout → numpy` converter and `(coeffs, model_params) → SimscapeOutput` cache. Acceptance: `test_cache_hit_skips_simulation` passes.                                             |
| **#040** | Integration tests against `system_identification.py` and `dataset_generator/core.py` | End-to-end fit using the adapter. Acceptance: `test_system_identification_works_against_adapter` passes.                                                                           |

Shared infrastructure consumed by Option 4: issues #014 (synthetic target oracle), #015 (`compute_cost`), #018 (`simulate_with_coefficients`).

## How Option 4 relates to Options 1, 2, 3

- **Option 1 (direct MATLAB fmincon)** — Option 4 calls the _same_ `simulate_with_coefficients.m` that Option 1 calls. Useful for: cross-validating that a Python-driven fit and a MATLAB-driven fit on the same target converge to the same coefficients.
- **Option 2 (NN surrogate)** — Option 4 is the round-trip oracle. After the surrogate produces `θ̂`, the adapter runs `simulate_with_coefficients(θ̂)` and the result is compared to the surrogate's prediction. Mismatches above a configurable budget reject the fit as out-of-distribution.
- **Option 3 (inverse NN)** — same role as for Option 2: validation oracle. The inverse network proposes `θ̂` from `q_meas`; the adapter checks the round-trip.

## Open questions for the human

- See [ASSUMPTIONS.md](ASSUMPTIONS.md) for the full list. Headlines:
  - **Phase 2 timing.** Confirm Phase 2 starts only after Options 1 and 2 have shipped a working fit. The issues are filed but should be left in the backlog until then.
  - **License pool.** On a CI host with one MATLAB license, the pool degenerates to size 1. Confirm the deployment target's license count before we set the default `pool_size`.
  - **MATLAB version pinning.** [INSTALLATION.md](INSTALLATION.md) recommends R2024a + Python 3.11. Confirm the deployment target's MATLAB release before agents start work — a mismatch is a multi-hour debugging trap.
  - **Determinism across MATLAB sessions.** [APPROACH.md § Determinism](APPROACH.md#determinism) assumes bit-identical output for identical input within a session, but only RMSE-equivalent across sessions. Confirm this is acceptable for `system_identification` round-trips.
