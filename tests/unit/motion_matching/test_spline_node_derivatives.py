import numpy as np

from src.shared.python.motion_matching.constrained_trajectory import (
    compose_chart_residual_jacobian,
    spline_chart_derivative_jacobians,
    spline_node_derivative_maps,
)


def test_spline_node_derivative_maps_reproduce_cubic_values() -> None:
    time = np.array([0.0, 1.0, 2.0, 3.0])
    first, second = spline_node_derivative_maps(time)
    values = time**3
    np.testing.assert_allclose(first @ values, 3 * time**2)
    np.testing.assert_allclose(second @ values, 6 * time)


def test_spline_chart_maps_chain_each_node_jacobian() -> None:
    first = np.array([[1.0, 2.0], [3.0, 4.0]])
    velocity, acceleration = spline_chart_derivative_jacobians(
        first, 2.0 * first, np.array([[[5.0]], [[7.0]]])
    )
    assert velocity.shape == (2, 1, 2, 1)
    np.testing.assert_allclose(velocity[:, 0, :, 0], [[5.0, 14.0], [15.0, 28.0]])
    np.testing.assert_allclose(acceleration[:, 0, :, 0], [[10.0, 28.0], [30.0, 56.0]])


def test_chart_residual_jacobian_combines_local_and_spline_partials() -> None:
    """Each residual sees its own q node and all qd/qdd source-node charts."""
    local_q = np.array([[[2.0]], [[3.0]]])
    local_v = np.array([[[5.0]], [[7.0]]])
    local_a = np.array([[[11.0]], [[13.0]]])
    node = np.array([[[17.0]], [[19.0]]])
    qd = np.array([[[[23.0], [29.0]]], [[[31.0], [37.0]]]])
    qdd = np.array([[[[41.0], [43.0]]], [[[47.0], [53.0]]]])

    result = compose_chart_residual_jacobian(local_q, local_v, local_a, node, qd, qdd)

    np.testing.assert_allclose(
        result[:, 0, :, 0],
        [
            [2 * 17 + 5 * 23 + 11 * 41, 5 * 29 + 11 * 43],
            [7 * 31 + 13 * 47, 3 * 19 + 7 * 37 + 13 * 53],
        ],
    )
