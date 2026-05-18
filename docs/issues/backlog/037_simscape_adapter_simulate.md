# Issue: Implement SimscapeAdapter.simulate_with_coefficients via MATLAB Engine (Option 4)

## Summary

Fill in the headline `simulate_with_coefficients` method on `SimscapeAdapter`:
start the MATLAB Engine for Python, build a `Simulink.SimulationInput`, call
`simulate_with_coefficients.m` from #018, marshal the result into a
`SimscapeOutput`, and respect the LRU cache from #036. Lazy engine startup,
full error wrapping per `simscape_errors.py`.

## Motivation

See `motion_matching/option4_python_bridge/INTERFACES.md` §"The motion-matching
headline method" and `RUNBOOK.md`. This is the actual Python ↔ Simscape bridge.
Once it works, the whole system_identification stack (#040) can use Simscape as
just another engine.

## Dependencies

- #018 (`simulate_with_coefficients.m`) — Python side calls this MATLAB
  function via the Engine.
- #036 (skeleton) — adapter class, `SimscapeOutput`, error classes, cache.

## File targets

- Modify: `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option4_python_bridge\simscape_adapter.py` (replace stubs with real implementations)
- New: `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option4_python_bridge\engine_startup.py` (lazy `matlab.engine.start_matlab` with timeout and license error wrapping)
- New: `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option4_python_bridge\logsout_to_simscape_output.py` (turns the flat-double struct returned by `simulate_with_coefficients.m` into a `SimscapeOutput` numpy view)
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option4\test_simscape_adapter_simulate.py`
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option4\test_engine_startup.py`
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option4\test_logsout_conversion.py`
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option4\fixtures\sample_logsout.json` (captured from a real MATLAB run for offline testing)

## Public API

The method signature is fixed by `INTERFACES.md`:

```python
@precondition(
    lambda self, coeffs:
        isinstance(coeffs, np.ndarray) and coeffs.ndim == 1
        and coeffs.size > 0 and coeffs.size % 7 == 0
        and np.all(np.isfinite(coeffs)),
    "coeffs must be a finite 1-D numpy array of length n_joints*7",
)
@postcondition(
    lambda self, coeffs, result:
        isinstance(result, SimscapeOutput)
        and result.time.shape[0] == result.butt.shape[0]
        == result.clubhead.shape[0] == result.club_quat.shape[0],
    "result is a SimscapeOutput with consistent N across all arrays",
)
def simulate_with_coefficients(self, coeffs: np.ndarray) -> SimscapeOutput:
    """Run one Simscape simulation with the given polynomial coefficients.

    Behaviour:
        1. Hash (coeffs, model_params) and check cache.
        2. On cache hit, return the cached SimscapeOutput.
        3. On miss, build a Simulink.SimulationInput, call sim() via the
           MATLAB Engine, extract logsout into a SimscapeOutput, cache it.

    Latency: ~50-200 ms warm, ~10-30 s on the first call (engine startup).

    Raises:
        SimulationError: any MATLAB-side failure.
    """
```

Plus the supporting helpers:

```python
def start_matlab_engine_with_timeout(timeout_s: float) -> "matlab.engine.MatlabEngine":
    """Start matlab.engine, raise EngineStartupError on timeout, raise LicenseError
    on MATLAB:license:* errors."""


def logsout_to_simscape_output(logsout_struct: dict, joint_names: list[str]) -> SimscapeOutput:
    """Convert the flat-double struct returned by simulate_with_coefficients.m
    into a SimscapeOutput numpy view."""
```

## Required tests (TDD)

- `test_simulate_first_call_starts_matlab_engine_lazily`
- `test_simulate_returns_simscape_output_with_consistent_N_across_arrays`
- `test_simulate_calls_matlab_simulate_with_coefficients_not_a_separate_simscape_call`
- `test_simulate_cache_hit_skips_matlab_call_and_returns_identical_output`
- `test_simulate_cache_key_uses_sha256_of_coeffs_and_model_params`
- `test_simulate_cache_eviction_when_max_entries_exceeded`
- `test_simulate_set_link_masses_clears_cache`
- `test_simulate_set_joint_damping_clears_cache`
- `test_simulate_wraps_matlab_exception_in_simulation_error`
- `test_simulate_recognizes_license_error_id_and_raises_LicenseError`
- `test_simulate_engine_died_mid_call_raises_EngineStartupError`
- `test_simulate_engine_startup_timeout_raises_EngineStartupError`
- `test_simulate_results_match_matlab_direct_call_for_fixed_coeffs_within_1e_minus_8`
- `test_logsout_conversion_extracts_time_butt_clubhead_quat_q_v_tau_omega`
- `test_logsout_conversion_normalizes_quaternion_sign_w_nonnegative`
- `test_logsout_conversion_detects_impact_idx_at_max_clubhead_speed`
- `test_logsout_conversion_uses_offline_fixture_for_ci_without_matlab`
- `test_simulate_marked_live_simulation_for_real_matlab_engine_path`

Tests that touch the live MATLAB Engine should be marked
`@pytest.mark.live_simulation` so CI without MATLAB can skip them; offline
tests use `sample_logsout.json` fixture.

## DbC contract

All preconditions and postconditions inherited verbatim from
`INTERFACES.md`. The implementation must NOT loosen them.

Additional postcondition:

- After `simulate_with_coefficients` returns, the cache contains the new entry
  unless `cache_enabled is False`.

## Acceptance Criteria

- [ ] `simulate_with_coefficients` works end-to-end against the real MATLAB
      Engine for a known synthetic theta.
- [ ] Live tests pass under `pytest -m live_simulation`.
- [ ] Offline tests pass under `pytest -m "not live_simulation"` (use fixture).
- [ ] Errors wrap MATLAB exceptions per `simscape_errors.py`.
- [ ] LRU cache hits verified to skip the MATLAB call and return identical
      `SimscapeOutput`.
- [ ] `set_link_masses` and `set_joint_damping` invalidate the cache.
- [ ] `INSTALLATION.md` (under `option4_python_bridge/`) updated with the
      pinned `matlabengine` version and license-checkout sanity check.
- [ ] DbC decorators present and unchanged from `INTERFACES.md`.
- [ ] `ruff check` and `ruff format --check` clean.
- [ ] No file exceeds 1200 lines.
- [ ] No `print()`; use `get_logger`.
- [ ] No TODO/FIXME without a tracked issue link.

## Labels

`motion-matching`, `option4`, `python`, `matlab`, `tdd`, `dbc`

## Effort estimate

L (3-7 days). MATLAB Engine plumbing has many real-world footguns: license
checkout, R2024b vs R2025a API drift, pyenv vs venv issues, and slow startup.
