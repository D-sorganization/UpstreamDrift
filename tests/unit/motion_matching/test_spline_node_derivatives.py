import numpy as np

from src.shared.python.motion_matching.constrained_trajectory import (
    spline_node_derivative_maps,
)


def test_spline_node_derivative_maps_reproduce_cubic_values() -> None:
    time = np.array([0.0, 1.0, 2.0, 3.0])
    first, second = spline_node_derivative_maps(time)
    values = time**3
    np.testing.assert_allclose(first @ values, 3 * time**2)
    np.testing.assert_allclose(second @ values, 6 * time)
