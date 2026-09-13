"""Shared native closure chart wrapper reused by shooting drivers."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python.native_node_chart import (
    NativeNodeChart,
)

pytestmark = pytest.mark.unit
NAMES = tuple(f"c{i}" for i in range(27))


class _FakeEngine:
    """Linear weld oracle: pose rows q[:6]-0.3, rate rows qd[:6]+q[6:12]."""

    def closure_residuals(self, coordinates, rates=None):
        q = np.array([coordinates[n] for n in NAMES])
        v = np.zeros(27) if rates is None else np.array([rates[n] for n in NAMES])
        return q[:6] - 0.3, v[:6] + q[6:12]

    def closure_trajectory_linearization(
        self, coordinates, rates, accelerations, *, finite_difference_step
    ):
        assert set(accelerations.values()) == {0.0}
        dq, dv = np.zeros((12, 27)), np.zeros((12, 27))
        dq[:6, :6] = np.eye(6)
        dq[6:, 6:12] = np.eye(6)
        dv[6:, :6] = np.eye(6)
        return SimpleNamespace(dq=dq, dv=dv)


def _chart() -> NativeNodeChart:
    return NativeNodeChart(
        NAMES,
        _FakeEngine(),
        state_scales=np.r_[np.full(27, 0.1), np.ones(27)],
        residual_scales=np.r_[np.full(6, 0.01), np.full(6, 0.1)],
        radius=0.5,
        tolerance=1e-8,
        jacobian_step=1e-6,
    )


def test_closure_and_jacobian_follow_the_engine_oracle() -> None:
    chart = _chart()
    state = np.zeros(54)
    state[:6] = 0.3
    np.testing.assert_allclose(chart.closure(state), 0.0)
    expected = np.zeros((12, 54))
    expected[:6, :6] = np.eye(6)
    expected[6:, 6:12] = np.eye(6)
    expected[6:, 27:33] = np.eye(6)
    np.testing.assert_array_equal(chart.jacobian(state), expected)


def test_zero_retraction_reproduces_reference_and_basis_is_tangent() -> None:
    chart = _chart()
    reference = np.zeros(54)
    reference[:6] = 0.3
    basis = chart.basis(reference)
    assert basis.shape == (54, 42)
    node = chart.retract(reference, basis, np.zeros(42))
    np.testing.assert_allclose(node.state, reference, atol=1e-12)
    assert node.state_jacobian.shape == (54, 42)
    moved = chart.retract(reference, basis, 1e-3 * np.eye(42)[3])
    assert np.max(abs(chart.closure(moved.state))) <= 1e-8
    assert chart.describe()["state_scales"][0] == 0.1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"state_scales": np.ones(3)},
        {"residual_scales": np.ones(3)},
        {"radius": 0.0},
        {"tolerance": -1.0},
        {"jacobian_step": 0.0},
    ],
)
def test_invalid_chart_settings_are_rejected(kwargs: dict) -> None:
    base = {
        "state_scales": np.ones(54),
        "residual_scales": np.ones(12),
        "radius": 0.5,
        "tolerance": 1e-8,
        "jacobian_step": 1e-6,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        NativeNodeChart(NAMES, _FakeEngine(), **base)


def test_state_shape_is_enforced() -> None:
    with pytest.raises(ValueError):
        _chart().closure(np.zeros(53))
