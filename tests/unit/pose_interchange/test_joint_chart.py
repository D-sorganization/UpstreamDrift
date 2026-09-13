"""Contract checks independent of engine installation."""

import itertools
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from src.shared.python.pose_interchange.joint_chart import (
    SerialRotationChart,
    SingularChartError,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("axes", ["".join(p) for p in itertools.permutations("XYZ")])
def test_pose_rate_acceleration_and_power(axes: str) -> None:
    chart = SerialRotationChart(axes)
    q = np.array([0.3, -0.7, 0.8])
    v = np.array([0.7, -0.3, 0.4])
    a = np.array([-0.2, 0.9, 0.1])
    torque = np.array([1.2, -2.0, 0.5])
    rotation = chart.rotation(q)
    np.testing.assert_allclose(
        rotation, Rotation.from_euler(axes, q).as_matrix(), atol=1e-14
    )
    h = 1e-6
    derivative = (chart.rotation(q + h * v) - chart.rotation(q - h * v)) / (2 * h)
    skew = derivative @ rotation.T
    omega = chart.angular_velocity(q, v)
    np.testing.assert_allclose(omega, [skew[2, 1], skew[0, 2], skew[1, 0]], atol=2e-10)
    alpha_fd = (
        chart.angular_velocity(q + h * v, v + h * a)
        - chart.angular_velocity(q - h * v, v - h * a)
    ) / (2 * h)
    np.testing.assert_allclose(
        chart.angular_acceleration(q, v, a), alpha_fd, atol=2e-10
    )
    np.testing.assert_allclose(chart.coordinate_rate(q, omega), v, atol=1e-14)
    moment = chart.parent_moment(q, torque)
    assert moment @ omega == pytest.approx(torque @ v)
    np.testing.assert_allclose(chart.coordinate_effort(q, moment), torque, atol=1e-14)
    np.testing.assert_allclose(chart.coordinates(chart.quaternion(q), q), q, atol=1e-14)


@pytest.mark.parametrize("q", [[6.7, -2.0, -7.2], [-8.0, 4.0, 9.0]])
def test_inverse_retains_reference_branch(q: list[float]) -> None:
    chart = SerialRotationChart("ZYX")
    np.testing.assert_allclose(
        chart.coordinates(-chart.quaternion(q), q), q, atol=1e-14
    )


def test_singularity_preserves_forward_pose_but_refuses_inverse() -> None:
    chart = SerialRotationChart("XYZ")
    q = [0.2, np.pi / 2, 0.4]
    assert np.isfinite(chart.quaternion(q)).all()
    assert chart.condition_number(q) > 1e12
    for call in [
        lambda: chart.coordinate_rate(q, [1, 2, 3]),
        lambda: chart.parent_moment(q, [1, 2, 3]),
        lambda: chart.coordinates(chart.quaternion(q), q),
    ]:
        with pytest.raises(SingularChartError):
            call()


@pytest.mark.parametrize("axes", ["XXZ", "xyz", "XY", "ABC"])
def test_rejects_unsupported_axis_contract(axes: str) -> None:
    with pytest.raises(ValueError):
        SerialRotationChart(axes)


def test_rejects_nonfinite_inputs_and_nonunit_quaternion() -> None:
    chart = SerialRotationChart("XYZ")
    with pytest.raises(ValueError):
        chart.rotation([0, np.nan, 1])
    with pytest.raises(ValueError):
        chart.coordinates([2, 0, 0, 0], [0, 0, 0])


def test_inverse_acceleration_includes_convective_term() -> None:
    chart = SerialRotationChart("YZX")
    q, v, a = (
        np.array([0.4, -0.2, 0.8]),
        np.array([1.0, 2.0, -3.0]),
        np.array([0.1, 0.7, -0.4]),
    )
    alpha = chart.angular_acceleration(q, v, a)
    np.testing.assert_allclose(
        chart.coordinate_acceleration(q, v, alpha), a, atol=1e-14
    )
