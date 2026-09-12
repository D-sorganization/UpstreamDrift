"""Node charts preserve closure and explicit scaled tangent coordinates."""

import numpy as np
import pytest
from src.shared.python.motion_matching.node_retraction import retract_node

pytestmark = pytest.mark.unit


def closure(x):
    return np.array([x[0] ** 2 / 4 + x[1] ** 2 / 9 - 1])


def jacobian(x):
    return np.array([[x[0] / 2, 2 * x[1] / 9]])


def run(z, **kwargs):
    return retract_node(
        np.array([2.0, 0.0]),
        np.array([[0.0], [1.0]]),
        np.array([z]),
        closure,
        jacobian,
        state_scales=np.array([2.0, 3.0]),
        residual_scales=np.ones(1),
        radius=0.5,
        **kwargs,
    )


def test_scaled_ellipse_retraction_and_derivative() -> None:
    r = run(0.2)
    np.testing.assert_allclose(r.state, [2 * np.sqrt(0.96), 0.6], atol=1e-10)
    np.testing.assert_allclose(
        r.state_jacobian[:, 0], [-0.4 / np.sqrt(0.96), 3.0], atol=1e-9
    )
    assert r.closure_max_abs < 1e-10
    assert not r.state.flags.writeable
    assert not r.state_jacobian.flags.writeable


def test_derivative_matches_separate_retractions() -> None:
    h = 1e-5
    np.testing.assert_allclose(
        run(0.2).state_jacobian[:, 0],
        (run(0.2 + h).state - run(0.2 - h).state) / (2 * h),
        atol=1e-8,
    )


def test_outside_chart_rejected() -> None:
    with pytest.raises(ValueError, match="radius"):
        run(0.6)


def test_nontangent_basis_rejected() -> None:
    with pytest.raises(ValueError, match="tangent"):
        retract_node(
            np.array([2.0, 0.0]),
            np.array([[1.0], [0.0]]),
            np.array([0.1]),
            closure,
            jacobian,
            state_scales=np.array([2.0, 3.0]),
            residual_scales=np.ones(1),
            radius=0.5,
        )


def test_rank_deficient_chart_rejected() -> None:
    with pytest.raises(ValueError, match="rank"):
        retract_node(
            np.zeros(2),
            np.array([[0.0], [1.0]]),
            np.array([0.0]),
            lambda x: np.zeros(1),
            lambda x: np.zeros((1, 2)),
            state_scales=np.ones(2),
            residual_scales=np.ones(1),
            radius=0.5,
        )


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan")])
def test_invalid_scale_rejected(scale) -> None:
    with pytest.raises(ValueError):
        retract_node(
            np.zeros(2),
            np.array([[0.0], [1.0]]),
            np.array([0.0]),
            closure,
            jacobian,
            state_scales=np.array([scale, 1.0]),
            residual_scales=np.ones(1),
            radius=0.5,
        )
