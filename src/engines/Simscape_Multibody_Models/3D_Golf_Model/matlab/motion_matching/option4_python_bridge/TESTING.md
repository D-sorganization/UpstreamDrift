# Testing — Option 4

TDD plan for the bridge. Tests live under `option4_python_bridge/tests/` and run via `pytest`. Per [shared/CODING_STANDARDS.md § TDD](../shared/CODING_STANDARDS.md#tdd-test-driven-development), tests are written **before** the implementation and committed in the same PR.

## Markers and skip policy

Every test that needs MATLAB is marked:

```python
pytest.mark.integration            # most tests in this folder
pytest.mark.slow                   # any test that starts or restarts an engine
pytest.mark.requires_matlab        # custom marker; defined in conftest.py
```

`requires_matlab` registration in `option4_python_bridge/tests/conftest.py`:

```python
import importlib.util
import pytest


def _matlab_available() -> bool:
    if importlib.util.find_spec("matlab.engine") is None:
        return False
    try:
        import matlab.engine  # type: ignore[import-not-found]
        eng = matlab.engine.start_matlab("-nodesktop -nosplash")
        eng.quit()
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    skip_no_matlab = pytest.mark.skip(reason="MATLAB Engine not available")
    if not _matlab_available():
        for item in items:
            if "requires_matlab" in item.keywords:
                item.add_marker(skip_no_matlab)
```

CI hosts without MATLAB skip the suite cleanly. Local devs with MATLAB run the full suite. CI with a self-hosted runner that has MATLAB installed runs everything.

## Required tests (issue #036–#040 acceptance criteria)

Each test below has a one-line description, the file it lives in, and the issue whose acceptance it gates.

### Lifecycle tests

#### `test_engine_starts_and_stops_cleanly`

**File:** `tests/test_lifecycle.py` — **gates:** #036

```python
@pytest.mark.requires_matlab
@pytest.mark.integration
@pytest.mark.slow
def test_engine_starts_and_stops_cleanly(slx_path: str) -> None:
    """No zombie matlab.exe processes after explicit close()."""
    pre = count_matlab_processes()
    adapter = SimscapeAdapter()
    adapter.load_from_path(slx_path)
    mid = count_matlab_processes()
    adapter.close()
    post = count_matlab_processes()

    assert mid == pre + 1, "engine startup must spawn exactly one matlab process"
    assert post == pre, "close() must reap the engine process"
```

`count_matlab_processes()` uses `psutil` to count processes named `matlab.exe`. Tolerates a 5 s grace period for the OS to reap.

#### `test_load_simscape_model_succeeds`

**File:** `tests/test_lifecycle.py` — **gates:** #036

```python
@pytest.mark.requires_matlab
@pytest.mark.integration
def test_load_simscape_model_succeeds(slx_path: str) -> None:
    with SimscapeAdapter() as adapter:
        adapter.load_from_path(slx_path)
        assert adapter.model_name == "GolfSwing3D_Kinetic"
```

#### `test_load_invalid_path_raises`

**File:** `tests/test_lifecycle.py` — **gates:** #036

```python
@pytest.mark.unit
def test_load_invalid_path_raises() -> None:
    with SimscapeAdapter() as adapter:
        with pytest.raises(FileNotFoundError):
            adapter.load_from_path("/does/not/exist.slx")
```

This test does **not** require MATLAB — `load_from_path` validates the path before starting the engine.

### Behavioural tests

#### `test_simulate_with_zero_coefficients_produces_static_pose`

**File:** `tests/test_behaviour.py` — **gates:** #036

```python
@pytest.mark.requires_matlab
@pytest.mark.integration
def test_simulate_with_zero_coefficients_produces_static_pose(
    adapter: SimscapeAdapter,
) -> None:
    """With all torques zero, the model only falls under gravity from address.

    Clubhead motion should be small and nearly vertical (gravity-driven sag),
    not a swing.
    """
    n_joints = adapter.n_joints  # MATLAB-side query
    theta = np.zeros(n_joints * 7)
    out = adapter.simulate_with_coefficients(theta)

    horizontal_displacement = np.linalg.norm(
        out.clubhead[-1, :2] - out.clubhead[0, :2]
    )
    assert horizontal_displacement < 0.10, (
        "Zero-torque sim should not produce horizontal swing motion"
    )
```

#### `test_simulate_with_known_coefficients_matches_matlab_direct`

**File:** `tests/test_regression.py` — **gates:** #036

```python
@pytest.mark.requires_matlab
@pytest.mark.integration
@pytest.mark.slow
def test_simulate_with_known_coefficients_matches_matlab_direct(
    adapter: SimscapeAdapter, known_theta: np.ndarray
) -> None:
    """Round-trip: Python adapter must produce the same SimscapeOutput
    as a direct MATLAB call to simulate_with_coefficients(theta).
    """
    out_py = adapter.simulate_with_coefficients(known_theta)

    eng = adapter._engine  # noqa: SLF001 — test-only access
    theta_m = matlab.double(known_theta.tolist())
    out_matlab = eng.simulate_with_coefficients(
        adapter.model_name, theta_m, default_opts(), nargout=1,
    )

    np.testing.assert_allclose(
        out_py.clubhead, np.asarray(out_matlab["clubhead"]),
        rtol=0, atol=1e-10,
        err_msg="Python adapter and direct MATLAB call must match bit-by-bit "
                "within a single MATLAB session",
    )
```

`known_theta` is a fixture: a fixed coefficient vector with seed 42, loaded from the test fixture file or generated by `generateRandomCoefficients(seed=42)`.

### Protocol-compliance test

#### `test_protocol_compliance`

**File:** `tests/test_protocol_compliance.py` — **gates:** #036

```python
import pytest
from src.shared.python.engine_core.interfaces import PhysicsEngine


@pytest.mark.unit
def test_protocol_compliance() -> None:
    """Every PhysicsEngine method is present on SimscapeAdapter and either:
       - implemented (callable), or
       - raises NotImplementedError with a clear message.
    """
    from src.engines.Simscape_Multibody_Models._3D_Golf_Model.matlab.\
        motion_matching.option4_python_bridge.simscape_adapter import (
        SimscapeAdapter,
    )

    # @runtime_checkable Protocol; this asserts ABC compatibility
    assert isinstance(SimscapeAdapter, type)

    # Every protocol method exists
    expected = {
        "model_name", "load_from_path", "load_from_string",
        "reset", "step", "forward",
        "get_state", "set_state", "set_control", "get_time",
        "compute_mass_matrix", "compute_bias_forces",
        "compute_gravity_forces", "compute_inverse_dynamics",
        "compute_jacobian", "compute_drift_acceleration",
        "compute_control_acceleration",
        "compute_ztcf", "compute_zvcf",
        "get_time_series", "get_induced_acceleration_series",
        "set_analysis_config",
    }
    actual = {name for name in dir(SimscapeAdapter) if not name.startswith("_")}
    missing = expected - actual
    assert not missing, f"SimscapeAdapter missing protocol methods: {missing}"
```

This test does **not** require MATLAB — it inspects the class only.

### Concurrency test

#### `test_concurrent_engines_isolated`

**File:** `tests/test_pool.py` — **gates:** #038

```python
@pytest.mark.requires_matlab
@pytest.mark.integration
@pytest.mark.slow
def test_concurrent_engines_isolated(slx_path: str) -> None:
    """Two engines run in parallel without state cross-talk.

    Engine A simulates theta_A; Engine B simulates theta_B at the same time.
    Each must produce the same output as it would alone.
    """
    theta_a = make_random_theta(seed=1)
    theta_b = make_random_theta(seed=2)

    # Reference outputs (sequential)
    with SimscapeAdapter() as ref:
        ref.load_from_path(slx_path)
        ref_a = ref.simulate_with_coefficients(theta_a)
        ref_b = ref.simulate_with_coefficients(theta_b)

    # Parallel outputs
    with SimscapeAdapterPool(pool_size=2, model_path=slx_path) as pool:
        out_a, out_b = pool.map_simulate([theta_a, theta_b])

    np.testing.assert_allclose(out_a.clubhead, ref_a.clubhead, atol=1e-9)
    np.testing.assert_allclose(out_b.clubhead, ref_b.clubhead, atol=1e-9)
```

### Cache test

#### `test_cache_hit_skips_simulation`

**File:** `tests/test_cache.py` — **gates:** #039

```python
@pytest.mark.requires_matlab
@pytest.mark.integration
def test_cache_hit_skips_simulation(adapter: SimscapeAdapter) -> None:
    """Calling simulate_with_coefficients twice with the same theta
    runs MATLAB exactly once.
    """
    theta = make_random_theta(seed=99)

    t0 = time.perf_counter()
    out1 = adapter.simulate_with_coefficients(theta)
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    out2 = adapter.simulate_with_coefficients(theta)
    t_second = time.perf_counter() - t0

    np.testing.assert_array_equal(out1.clubhead, out2.clubhead)
    assert t_second < 0.5 * t_first, (
        f"Cache hit should be substantially faster: {t_second:.3f}s vs "
        f"{t_first:.3f}s"
    )
    assert adapter.cache_stats().hits == 1
    assert adapter.cache_stats().misses == 1
```

#### `test_cache_invalidates_on_param_change`

**File:** `tests/test_cache.py` — **gates:** #039

```python
@pytest.mark.requires_matlab
@pytest.mark.integration
def test_cache_invalidates_on_param_change(adapter: SimscapeAdapter) -> None:
    """Changing link masses must clear the cache."""
    theta = make_random_theta(seed=99)
    _ = adapter.simulate_with_coefficients(theta)
    assert adapter.cache_stats().misses == 1

    masses = adapter.get_link_masses()
    adapter.set_link_masses(masses * 1.1)

    _ = adapter.simulate_with_coefficients(theta)
    assert adapter.cache_stats().misses == 2  # forced recomputation
```

### Determinism test

#### `test_simulation_is_deterministic_within_session`

**File:** `tests/test_determinism.py` — **gates:** #036

```python
@pytest.mark.requires_matlab
@pytest.mark.integration
def test_simulation_is_deterministic_within_session(
    adapter: SimscapeAdapter,
) -> None:
    """Same theta + same seed -> bit-identical output (within session)."""
    theta = make_random_theta(seed=7)
    adapter.cache_enabled = False  # force recomputation

    out1 = adapter.simulate_with_coefficients(theta)
    out2 = adapter.simulate_with_coefficients(theta)

    np.testing.assert_array_equal(out1.clubhead, out2.clubhead)
    np.testing.assert_array_equal(out1.club_quat, out2.club_quat)
```

### Integration tests against existing consumers

#### `test_system_identification_works_against_adapter`

**File:** `tests/test_integration_sysid.py` — **gates:** #040

```python
@pytest.mark.requires_matlab
@pytest.mark.integration
@pytest.mark.slow
def test_system_identification_works_against_adapter(slx_path: str) -> None:
    """End-to-end: SystemIdentifier consumes a SimscapeAdapter and
    recovers a known link-mass perturbation.
    """
    from src.learning.sim2real.system_identification import SystemIdentifier

    with SimscapeAdapter() as truth:
        truth.load_from_path(slx_path)
        truth_masses = truth.get_link_masses()
        # Generate ground-truth trajectories with perturbed masses
        truth.set_link_masses(truth_masses * 1.10)
        target_traj = truth.simulate_with_coefficients(reference_theta())

    with SimscapeAdapter() as adapter:
        adapter.load_from_path(slx_path)
        identifier = SystemIdentifier(model=adapter)
        result = identifier.identify_from_trajectory(
            target_traj, max_iters=20, tol=1e-3,
        )

    assert result.converged
    np.testing.assert_allclose(
        result.identified_params["masses"],
        truth_masses * 1.10,
        rtol=0.05,  # 5% mass-recovery tolerance
    )
```

#### `test_dataset_generator_core_works_against_adapter`

**File:** `tests/test_integration_datagen.py` — **gates:** #040

```python
@pytest.mark.requires_matlab
@pytest.mark.integration
@pytest.mark.slow
def test_dataset_generator_core_works_against_adapter(
    adapter: SimscapeAdapter, tmp_path: Path,
) -> None:
    """The Python dataset_generator/core.py path can drive Simscape
    end-to-end via the adapter and produce a valid parquet shard.
    """
    from src.engines.Simscape_Multibody_Models._3D_Golf_Model.matlab.\
        src.functions.dataset_generator.core import generate_shard

    shard = generate_shard(
        engine=adapter,
        n_trials=4,
        seed=42,
        output_dir=tmp_path,
    )
    assert shard.exists()
    df = pd.read_parquet(shard)
    assert len(df) == 4
    assert "clubhead_x" in df.columns
```

(If `dataset_generator/core.py` does not yet accept an engine argument, this test gates a small refactor of its signature; flag in the issue.)

## Fixtures

`tests/conftest.py`:

```python
import os
from pathlib import Path
import pytest
import numpy as np
from src.engines.Simscape_Multibody_Models._3D_Golf_Model.matlab.\
    motion_matching.option4_python_bridge.simscape_adapter import (
    SimscapeAdapter,
)


@pytest.fixture(scope="session")
def slx_path() -> str:
    return os.environ.get(
        "GOLF_SWING_3D_SLX_PATH",
        str(Path(__file__).parents[5] / "src" / "model"
            / "GolfSwing3D_Kinetic.slx"),
    )


@pytest.fixture(scope="module")
def adapter(slx_path: str):
    """Module-scoped adapter; one engine startup cost amortized over all
    tests in the module."""
    a = SimscapeAdapter()
    a.load_from_path(slx_path)
    yield a
    a.close()


@pytest.fixture(scope="session")
def known_theta() -> np.ndarray:
    """Fixed coefficient vector for regression tests."""
    rng = np.random.default_rng(42)
    n_joints = 16  # placeholder; query model in real impl
    theta = rng.uniform(-25, 25, size=n_joints * 7)
    return theta
```

## Coverage targets

- Every public method of `SimscapeAdapter` is exercised by at least one test.
- Every error class (`SimulationError`, `EngineStartupError`, `LicenseError`, `ModelLoadError`) is raised in at least one test.
- Pool concurrency tested with `pool_size ∈ {1, 2, 4}`.
- Cache hit, cache miss, cache invalidation all covered.

CI fails if coverage of `option4_python_bridge/` drops below the project-wide threshold from `pyproject.toml [tool.coverage.report] fail_under`.

## Order of test authorship (TDD playbook)

1. Write `test_protocol_compliance` first — it doesn't need MATLAB and tells you immediately whether the class skeleton is complete.
2. Write `test_load_invalid_path_raises` second — also doesn't need MATLAB, verifies precondition handling.
3. With MATLAB available, write `test_engine_starts_and_stops_cleanly` and `test_load_simscape_model_succeeds`. Implement `__init__`, `load_from_path`, `close` to make them pass.
4. Write `test_simulate_with_zero_coefficients_produces_static_pose`. Implement `simulate_with_coefficients` (no cache yet).
5. Write `test_simulate_with_known_coefficients_matches_matlab_direct`. This is the regression oracle; if it passes, the bridge is faithful.
6. Add `test_simulation_is_deterministic_within_session`.
7. Add caching: `test_cache_hit_skips_simulation`, `test_cache_invalidates_on_param_change`.
8. Add the pool: `test_concurrent_engines_isolated`.
9. Finally, the integration tests: `test_system_identification_works_against_adapter`, `test_dataset_generator_core_works_against_adapter`.

Each step ships in its own PR per [shared/CODING_STANDARDS.md](../shared/CODING_STANDARDS.md).
