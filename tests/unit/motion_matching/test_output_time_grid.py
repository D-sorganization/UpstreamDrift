"""Tests for the shared canonical output time grid (issue #7740).

Drake and MuJoCo previously built their output grids with different
expressions (``arange * dt`` vs ``linspace``), which only agreed when
``(T, rate)`` divided evenly. This module pins:

* the shared builder is bit-identical to the legacy ``arange(N) * (1/rate)``
  for integer-divisible cases (so existing trajectories are unchanged), and
* the endpoint is aligned to ``T`` for non-divisible cases (the bug fix).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.shared.python.motion_matching.output_time_grid import (
    build_output_grid,
    canonical_sample_count,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("t_s", "rate"),
    [(0.3, 1000.0), (0.2, 500.0), (0.25, 400.0), (1.0, 240.0)],
)
def test_divisible_matches_legacy_arange(t_s: float, rate: float) -> None:
    """For integer-divisible ``(T, rate)`` the grid equals ``arange(N) * dt``."""
    n = int(round(t_s * rate)) + 1
    legacy = np.arange(n, dtype=np.float64) * (1.0 / rate)
    grid = build_output_grid(t_s, rate)
    np.testing.assert_array_equal(grid, legacy)


def test_grid_is_inclusive_and_monotonic() -> None:
    """Grid starts at 0, ends exactly at T, and strictly increases."""
    grid = build_output_grid(0.3, 1000.0)
    assert grid[0] == 0.0
    assert grid[-1] == pytest.approx(0.3, abs=1e-12)
    assert np.all(np.diff(grid) > 0.0)


@pytest.mark.parametrize(("t_s", "rate"), [(0.123, 777.0), (0.3, 333.0)])
def test_non_divisible_endpoint_is_aligned(t_s: float, rate: float) -> None:
    """Non-divisible ``(T, rate)`` still pins the endpoint to exactly ``T``.

    The legacy ``arange * dt`` overshot ``T`` in this regime, leaving the
    engines' endpoints misaligned. The shared builder fixes that.
    """
    grid = build_output_grid(t_s, rate)
    assert grid[-1] == pytest.approx(t_s, abs=1e-12)
    assert grid[0] == 0.0
    assert np.all(np.diff(grid) > 0.0)
    # Sanity: the legacy construction did NOT land on the endpoint here.
    n = canonical_sample_count(t_s, rate)
    legacy_endpoint = (n - 1) * (1.0 / rate)
    assert not np.isclose(legacy_endpoint, t_s, atol=1e-9)


def test_drake_and_mujoco_wrappers_delegate_to_shared_grid() -> None:
    """Engine wrappers should share the pinned endpoint builder (#7740)."""
    from src.engines.physics_engines.drake.python.motion_matching.simulate import (
        _sample_grid as drake_grid,
    )
    from src.engines.physics_engines.mujoco.python.motion_matching.simulate import (
        _output_grid as mujoco_grid,
    )

    t_s = 0.123
    rate = 777.0
    expected = build_output_grid(t_s, rate)
    np.testing.assert_array_equal(drake_grid(t_s, rate), expected)
    np.testing.assert_array_equal(mujoco_grid(t_s, rate), expected)


def test_pinocchio_integration_clock_exception_is_explicit() -> None:
    """Pinocchio keeps a fixed-step integration clock, not an output grid."""
    from src.engines.physics_engines.pinocchio.python.motion_matching import simulate

    source = inspect.getsource(simulate.simulate_with_coefficients)
    assert "fixed-step *integration clock*" in source
    assert "np.arange(n_samples" in source


def test_sample_count_formula() -> None:
    """N == round(T * rate) + 1."""
    assert canonical_sample_count(0.3, 1000.0) == 301
    assert canonical_sample_count(0.25, 400.0) == 101


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_rejects_non_positive_inputs(bad: float) -> None:
    with pytest.raises(ValueError):
        build_output_grid(bad, 1000.0)
    with pytest.raises(ValueError):
        build_output_grid(0.3, bad)
