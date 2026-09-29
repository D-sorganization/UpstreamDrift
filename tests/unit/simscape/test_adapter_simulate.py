"""Unit tests for the #4006 SimscapeAdapter MATLAB-Engine plumbing.

These tests cover the additions made on top of the #4005 skeleton:

- Lazy MATLAB engine startup behaviour (offline path).
- The :class:`SimscapeOutput` dataclass invariants.
- The :class:`_ResultCache` LRU semantics and cache-key stability.
- The adapter's cache-hit short-circuit (mocking ``_simulate_uncached``).

Tests that need a live MATLAB engine are marked ``requires_matlab``;
they live in ``tests/integration/simscape/test_adapter_simulate.py``
and skip automatically when ``matlab.engine`` is not importable.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from src.engines.simscape import (
    SimscapeAdapter,
    SimscapeNotInstalledError,
    SimscapeOutput,
)
from src.engines.simscape._cache import _ResultCache, make_cache_key

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def slx_path(tmp_path: Path) -> str:
    slx = tmp_path / "GolfSwing3D_Kinetic.slx"
    slx.write_bytes(b"FAKE_SLX_FOR_TESTS")
    metadata = tmp_path / "PolynomialInputValues.mat"
    metadata.write_bytes(b"FAKE_MAT_FOR_TESTS")
    return str(slx)


@pytest.fixture
def loaded_adapter(slx_path: str) -> SimscapeAdapter:
    a = SimscapeAdapter()
    a.load_from_path(slx_path)
    return a


def _fake_simscape_output(n: int = 4, n_joints: int = 16) -> SimscapeOutput:
    """Return a valid :class:`SimscapeOutput` for cache tests."""
    time = np.linspace(0.0, 0.1, n, dtype=np.float64)
    zeros_j = np.zeros((n, n_joints), dtype=np.float64)
    zeros_3 = np.zeros((n, 3), dtype=np.float64)
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    return SimscapeOutput(
        time=time,
        q=zeros_j,
        qd=zeros_j.copy(),
        qdd=zeros_j.copy(),
        tau=zeros_j.copy(),
        omega=zeros_j.copy(),
        r_butt=zeros_3,
        r_clubhead=zeros_3.copy(),
        q_club=quat,
        v_clubhead=zeros_3.copy(),
    )


# ---------------------------------------------------------------------------
# SimscapeOutput dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_simscape_output_validates_consistent_n() -> None:
    out = _fake_simscape_output(n=5)
    assert out.n_samples == 5
    assert out.n_joints == 16


@pytest.mark.unit
def test_simscape_output_rejects_non_unit_quat() -> None:
    n, j = 3, 16
    bad_quat = np.tile(np.array([2.0, 0.0, 0.0, 0.0]), (n, 1))
    with pytest.raises(ValueError, match="unit-norm"):
        SimscapeOutput(
            time=np.linspace(0.0, 0.1, n),
            q=np.zeros((n, j)),
            qd=np.zeros((n, j)),
            qdd=np.zeros((n, j)),
            tau=np.zeros((n, j)),
            omega=np.zeros((n, j)),
            r_butt=np.zeros((n, 3)),
            r_clubhead=np.zeros((n, 3)),
            q_club=bad_quat,
            v_clubhead=np.zeros((n, 3)),
        )


@pytest.mark.unit
def test_simscape_output_rejects_non_monotonic_time() -> None:
    n, j = 3, 16
    bad_time = np.array([0.0, 0.2, 0.1])
    with pytest.raises(ValueError, match="strictly increasing"):
        SimscapeOutput(
            time=bad_time,
            q=np.zeros((n, j)),
            qd=np.zeros((n, j)),
            qdd=np.zeros((n, j)),
            tau=np.zeros((n, j)),
            omega=np.zeros((n, j)),
            r_butt=np.zeros((n, 3)),
            r_clubhead=np.zeros((n, 3)),
            q_club=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)),
            v_clubhead=np.zeros((n, 3)),
        )


@pytest.mark.unit
def test_simscape_output_rejects_nonzero_t0() -> None:
    n, j = 2, 16
    with pytest.raises(ValueError, match=r"time\[0\]"):
        SimscapeOutput(
            time=np.array([0.5, 0.6]),
            q=np.zeros((n, j)),
            qd=np.zeros((n, j)),
            qdd=np.zeros((n, j)),
            tau=np.zeros((n, j)),
            omega=np.zeros((n, j)),
            r_butt=np.zeros((n, 3)),
            r_clubhead=np.zeros((n, 3)),
            q_club=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)),
            v_clubhead=np.zeros((n, 3)),
        )


# ---------------------------------------------------------------------------
# _ResultCache LRU semantics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_result_cache_basic_get_put() -> None:
    cache: _ResultCache[int] = _ResultCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    assert cache.get("b") == 2
    assert cache.hits == 2
    assert cache.misses == 0


@pytest.mark.unit
def test_result_cache_evicts_lru() -> None:
    cache: _ResultCache[int] = _ResultCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # bump a to MRU
    cache.put("c", 3)  # should evict b
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


@pytest.mark.unit
def test_result_cache_capacity_zero_disables() -> None:
    cache: _ResultCache[int] = _ResultCache(capacity=0)
    cache.put("a", 1)
    assert cache.get("a") is None
    assert len(cache) == 0


@pytest.mark.unit
def test_result_cache_negative_capacity_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _ResultCache(capacity=-1)


@pytest.mark.unit
def test_make_cache_key_stable_for_identical_inputs() -> None:
    coeffs = np.arange(7, dtype=np.float64)
    k1 = make_cache_key(coeffs, model_params=b"{}", matlab_version="R2024b")
    k2 = make_cache_key(coeffs.copy(), model_params=b"{}", matlab_version="R2024b")
    assert k1 == k2


@pytest.mark.unit
def test_make_cache_key_differs_on_different_coeffs() -> None:
    a = np.zeros(7, dtype=np.float64)
    b = np.ones(7, dtype=np.float64)
    ka = make_cache_key(a, model_params=b"{}", matlab_version="R2024b")
    kb = make_cache_key(b, model_params=b"{}", matlab_version="R2024b")
    assert ka != kb


@pytest.mark.unit
def test_make_cache_key_differs_on_different_matlab_version() -> None:
    coeffs = np.zeros(7, dtype=np.float64)
    k1 = make_cache_key(coeffs, model_params=b"{}", matlab_version="R2024b")
    k2 = make_cache_key(coeffs, model_params=b"{}", matlab_version="R2025a")
    assert k1 != k2


# ---------------------------------------------------------------------------
# Adapter cache hit/miss behaviour (mocks _simulate_uncached, no MATLAB)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cache_hit_skips_simulation(loaded_adapter: SimscapeAdapter) -> None:
    """Second call with identical coeffs must not re-enter ``_simulate_uncached``."""
    coeffs = np.zeros(16 * 7, dtype=np.float64)
    fake = _fake_simscape_output()
    with patch.object(
        loaded_adapter, "_simulate_uncached", return_value=fake
    ) as mock_inner:
        out1 = loaded_adapter.simulate_with_coefficients(coeffs)
        out2 = loaded_adapter.simulate_with_coefficients(coeffs)
    assert mock_inner.call_count == 1
    assert out1 is out2
    assert out1 is fake


@pytest.mark.unit
def test_cache_miss_for_distinct_coeffs(loaded_adapter: SimscapeAdapter) -> None:
    fake_a = _fake_simscape_output()
    fake_b = _fake_simscape_output(n=5)
    with patch.object(
        loaded_adapter,
        "_simulate_uncached",
        side_effect=[fake_a, fake_b],
    ) as mock_inner:
        out_a = loaded_adapter.simulate_with_coefficients(
            np.zeros(16 * 7, dtype=np.float64)
        )
        out_b = loaded_adapter.simulate_with_coefficients(
            np.ones(16 * 7, dtype=np.float64)
        )
    assert mock_inner.call_count == 2
    assert out_a is fake_a
    assert out_b is fake_b


@pytest.mark.unit
def test_cache_disabled_always_calls_simulate(slx_path: str) -> None:
    a = SimscapeAdapter(cache_enabled=False)
    a.load_from_path(slx_path)
    fake = _fake_simscape_output()
    with patch.object(a, "_simulate_uncached", return_value=fake) as mock_inner:
        a.simulate_with_coefficients(np.zeros(16 * 7))
        a.simulate_with_coefficients(np.zeros(16 * 7))
    assert mock_inner.call_count == 2


# ---------------------------------------------------------------------------
# Lazy-engine error paths (no MATLAB present)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_from_path_does_not_require_matlab(slx_path: str) -> None:
    """Without MATLAB, ``load_from_path`` must still succeed in skeleton mode."""
    a = SimscapeAdapter()
    a.load_from_path(slx_path)
    assert a.model_loaded
    assert a.dof == 16


@pytest.mark.unit
def test_simulate_without_matlab_raises_not_installed(
    loaded_adapter: SimscapeAdapter,
) -> None:
    coeffs = np.zeros(16 * 7, dtype=np.float64)
    with pytest.raises(SimscapeNotInstalledError):
        loaded_adapter.simulate_with_coefficients(coeffs)


@pytest.mark.unit
def test_simulate_with_invalid_coeffs_length_raises(
    loaded_adapter: SimscapeAdapter,
) -> None:
    """Coeffs whose size is not a multiple of 7 must violate the precondition."""
    bad = np.zeros(15, dtype=np.float64)  # 15 is not divisible by 7
    with pytest.raises(ValueError):  # precondition wraps into a contract error
        loaded_adapter.simulate_with_coefficients(bad)


@pytest.mark.unit
def test_simulate_with_non_finite_coeffs_raises(
    loaded_adapter: SimscapeAdapter,
) -> None:
    bad = np.full(16 * 7, np.nan, dtype=np.float64)
    with pytest.raises(ValueError):
        loaded_adapter.simulate_with_coefficients(bad)


@pytest.mark.unit
def test_close_is_idempotent_no_matlab(slx_path: str) -> None:
    a = SimscapeAdapter()
    a.load_from_path(slx_path)
    a.close()
    a.close()  # must not raise
    assert not a.model_loaded


@pytest.mark.unit
def test_protocol_compliance_after_implementation(
    loaded_adapter: SimscapeAdapter,
) -> None:
    """Re-confirm the headline protocol still composes after #4006 changes."""
    from src.shared.python.engine_core.interfaces import PhysicsEngine

    assert isinstance(loaded_adapter, PhysicsEngine)
