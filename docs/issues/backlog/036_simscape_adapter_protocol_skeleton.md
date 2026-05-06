# Issue: Implement SimscapeAdapter Protocol Skeleton (Option 4)

## Summary

Implement the `SimscapeAdapter` class scaffold that satisfies the
`PhysicsEngineProtocol` (`PhysicsEngine` ABC), without yet implementing the
actual MATLAB Engine call. Every method must either return a stub or raise
`NotImplementedError` with a clear message; `test_protocol_compliance` must pass.

## Motivation

See `motion_matching/option4_python_bridge/INTERFACES.md` and
`motion_matching/option4_python_bridge/RUNBOOK.md`. Wiring the protocol
skeleton first lets `loaders.py` integrate (#038) and lets `system_identification`
import the adapter without blocking on the heavy MATLAB Engine plumbing
(deferred to #037).

## Dependencies

None — but #037 will fill in the `simulate_with_coefficients` body, and #038
wires this into the loader registry.

## File targets

- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option4_python_bridge\__init__.py`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option4_python_bridge\simscape_adapter.py` (skeleton)
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option4_python_bridge\simscape_output.py` (`SimscapeOutput` dataclass)
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option4_python_bridge\simscape_errors.py` (`SimulationError`, `EngineStartupError`, `LicenseError`, `ModelLoadError`)
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option4_python_bridge\cache.py` (`_ResultCache`)
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option4\test_protocol_compliance.py`
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option4\test_simscape_output.py`
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option4\test_simscape_errors.py`
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option4\test_cache.py`

## Public API

Verbatim from `option4_python_bridge/INTERFACES.md`:

```python
@dataclass(frozen=True)
class SimscapeOutput:
    time:       np.ndarray   # (N,) float64, monotonic, dt=1/sample_rate
    butt:       np.ndarray   # (N, 3) float64, metres, world frame
    clubhead:   np.ndarray   # (N, 3) float64, metres, world frame
    club_quat:  np.ndarray   # (N, 4) float64, [w, x, y, z], unit-norm
    q:          np.ndarray   # (N, n_q) float64
    v:          np.ndarray   # (N, n_v) float64
    tau:        np.ndarray   # (N, n_joints) float64, N*m
    omega:      np.ndarray   # (N, n_joints) float64, rad/s
    impact_idx: int


class SimulationError(GolfModelingError): ...
class EngineStartupError(SimulationError): ...
class LicenseError(EngineStartupError): ...
class ModelLoadError(SimulationError): ...


@invariant(lambda self: self._engine is None or self._model_name != "",
           "if engine is started, a model name must be set")
@invariant(lambda self: self._cache_max_entries >= 0,
           "cache_max_entries must be non-negative (0 = disabled)")
class SimscapeAdapter(PhysicsEngine):
    def __init__(self, rng_seed: int = 42, cache_enabled: bool = True,
                 cache_max_entries: int = 1024,
                 startup_timeout_s: float = 60.0) -> None: ...

    @property
    def model_name(self) -> str: ...
    def load_from_path(self, path: str) -> None: ...
    def load_from_string(self, content: str, extension: str | None = None) -> None: ...
    def reset(self) -> None: ...
    def step(self, dt: float | None = None) -> None: ...
    def forward(self) -> None: ...
    def get_state(self) -> tuple[np.ndarray, np.ndarray]: ...
    def set_state(self, q: np.ndarray, v: np.ndarray) -> None: ...
    def set_control(self, u: np.ndarray) -> None: ...
    def get_time(self) -> float: ...
    def compute_mass_matrix(self) -> np.ndarray: ...
    def compute_bias_forces(self) -> np.ndarray: ...
    def compute_gravity_forces(self) -> np.ndarray: ...
    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray: ...
    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None: ...
    def compute_drift_acceleration(self) -> np.ndarray: ...
    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray: ...
    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray: ...
    def compute_zvcf(self, q: np.ndarray) -> np.ndarray: ...
    def get_time_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray | list]: ...
    def get_induced_acceleration_series(self, source_name: str | int) -> tuple[np.ndarray, np.ndarray]: ...
    def set_analysis_config(self, config: dict) -> None: ...
    def get_link_masses(self) -> np.ndarray: ...
    def set_link_masses(self, masses: np.ndarray) -> None: ...
    def get_joint_damping(self) -> np.ndarray: ...
    def set_joint_damping(self, damping: np.ndarray) -> None: ...
    def simulate_with_coefficients(self, coeffs: np.ndarray) -> SimscapeOutput: ...
    def close(self) -> None: ...
    def __enter__(self) -> "SimscapeAdapter": ...
    def __exit__(self, *exc_info: object) -> None: ...
```

## Required tests (TDD)

- `test_protocol_compliance_iterates_every_method_on_PhysicsEngine_protocol`
- `test_simscape_output_dataclass_is_frozen`
- `test_simscape_output_arrays_share_n_along_axis_0_invariant_violation_raises`
- `test_simscape_output_time_strictly_increasing_starts_at_zero`
- `test_simscape_output_quaternion_unit_norm_to_1e_minus_6`
- `test_simscape_output_impact_idx_within_bounds`
- `test_simulation_error_carries_matlab_error_id_and_traceback`
- `test_engine_startup_error_subclass_of_simulation_error`
- `test_license_error_subclass_of_engine_startup_error`
- `test_model_load_error_subclass_of_simulation_error`
- `test_simulation_error_inherits_from_GolfModelingError`
- `test_adapter_init_does_not_start_matlab_engine`
- `test_adapter_init_rejects_negative_rng_seed`
- `test_adapter_init_rejects_negative_cache_max_entries`
- `test_adapter_invariant_engine_is_None_or_model_name_nonempty`
- `test_adapter_load_from_string_raises_NotImplementedError_with_clear_message`
- `test_adapter_load_from_path_rejects_non_slx_extension`
- `test_adapter_close_is_idempotent_on_unstarted_engine`
- `test_adapter_context_manager_calls_close_on_exit`
- `test_cache_max_entries_zero_disables_caching`
- `test_cache_lru_eviction_when_full`
- `test_cache_keys_use_sha256_of_coefficients_and_model_params`

## DbC contract

Every public method has a `@precondition` and/or `@postcondition` decorator
exactly as documented in `INTERFACES.md`. `@invariant` on the class enforces
the engine/model-name and cache-size invariants.

## Acceptance Criteria

- [ ] `SimscapeAdapter` is a subclass of `PhysicsEngine` and passes
      `test_protocol_compliance` (every method either implemented as a stub
      that raises `NotImplementedError("not yet — see #037")` or implemented
      as a no-op stub returning a typed default).
- [ ] All four error classes implemented and their inheritance verified.
- [ ] `SimscapeOutput` dataclass implemented with all documented fields and
      invariants checked at construction time.
- [ ] `_ResultCache` is a private LRU cache keyed by sha256(coeffs, model_params).
- [ ] All listed tests pass.
- [ ] DbC decorators applied verbatim from `INTERFACES.md`.
- [ ] `ruff check` and `ruff format --check` clean.
- [ ] No file exceeds 1200 lines.
- [ ] No `print()`; use `get_logger`.
- [ ] No TODO/FIXME without a tracked issue link (use `# placeholder until #037`
      for stub bodies).

## Labels

`motion-matching`, `option4`, `python`, `tdd`, `dbc`

## Effort estimate

M (1-3 days). The protocol surface is wide; the value is the test bed for #037.
