"""Tests for the shared ground-support contracts (frame map, ground height, support)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching import ground_support as module
from src.shared.python.motion_matching.contact_law import ContactSample, GroundPlane

pytestmark = pytest.mark.unit


def test_capture_to_native_world_is_x_minus_z_y_and_keeps_nan() -> None:
    points = np.array([[1.0, 2.0, 3.0], [np.nan, 1.0, 1.0]])
    mapped = module.capture_to_native_world(points)
    np.testing.assert_allclose(mapped[0], [1.0, -3.0, 2.0])
    assert np.isnan(mapped[1, 0]) and mapped[1, 1] == -1.0 and mapped[1, 2] == 1.0
    rotation = module.y_up_to_z_up_rotation()
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        module.capture_to_native_world(np.zeros((2, 2)))


def test_ground_height_from_lowest_valid_toe_marker() -> None:
    labels = ("LToeIn", "LKneeOut", "RToeOut")
    points = np.array(
        [
            [[0, 0, 0.05], [0, 0, 0.5], [0, 0, 0.041]],
            [[0, 0, 0.02], [0, 0, 0.5], [0, 0, 0.046]],  # LToeIn invalid here
        ]
    )
    valid = np.array([[True, True, True], [False, True, True]])
    cal = module.calibrate_ground_height(
        points, valid, labels, ("LToeIn", "RToeOut"), standoff_m=0.03
    )
    assert cal.lowest_marker_height_m == pytest.approx(0.041)
    assert cal.height_m == pytest.approx(0.011)
    assert cal.labels == ("LToeIn", "RToeOut") and cal.frames == 2
    with pytest.raises(ValueError):
        module.calibrate_ground_height(points, valid, labels, ("Nope",), standoff_m=0.0)
    with pytest.raises(ValueError):
        module.calibrate_ground_height(
            points, valid, labels, ("LToeIn",), standoff_m=-1
        )


def test_convex_hull_containment() -> None:
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    assert module.convex_hull_contains(np.array([0.5, 0.5]), square)
    assert not module.convex_hull_contains(np.array([1.2, 0.5]), square)
    assert module.convex_hull_contains(np.array([1.01, 0.5]), square, tolerance_m=0.02)
    assert not module.convex_hull_contains(np.array([0.0, 0.0]), square[:2])


def _sample(point: list[float], normal: float) -> ContactSample:
    return ContactSample(
        penetration_m=0.001,
        penetration_rate_m_s=0.0,
        contact_point_m=np.array(point),
        normal_force_n=np.array([0.0, 0.0, normal]),
        friction_force_n=np.zeros(3),
    )


def test_support_report_weight_fraction_and_cop() -> None:
    plane = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)
    samples = {
        "heel_l": _sample([0.0, 0.1, 0.0], 300.0),
        "heel_r": _sample([0.0, -0.1, 0.0], 300.0),
        "toe_l": _sample([0.2, 0.1, 0.0], 0.0),
        "toe_r": _sample([0.2, -0.1, 0.0], 381.0),
    }
    polygon = {k: np.array(v.contact_point_m) for k, v in samples.items()}
    report = module.support_report(
        samples, polygon, plane, mass_kg=100.0, gravity_m_s2=(0, 0, -9.81)
    )
    assert report.total_normal_force_n == pytest.approx(981.0)
    assert report.weight_fraction == pytest.approx(1.0)
    assert report.active_spheres == ("heel_l", "heel_r", "toe_r")
    assert report.inside_support_polygon
    cop = np.array(report.centre_of_pressure_m)
    np.testing.assert_allclose(
        cop, [0.2 * 381 / 981, (300 - 300 - 381) * 0.1 / 981, 0.0]
    )
    empty = module.support_report({}, {}, plane, 100.0, (0, 0, -9.81))
    assert empty.centre_of_pressure_m is None and not empty.inside_support_polygon
    with pytest.raises(ValueError):
        module.support_report(samples, {}, plane, 100.0, (0, 0, -9.81))
