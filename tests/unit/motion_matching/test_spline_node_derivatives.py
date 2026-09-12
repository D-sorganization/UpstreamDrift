import numpy as np

from src.shared.python.motion_matching.constrained_trajectory import (
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
