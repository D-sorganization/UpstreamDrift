"""Contract tests for the planar two-hand wrench analysis."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.two_hand_wrench import (
    find_zero_crossings,
    resolve_grip_components,
    transport_planar_moment,
    two_contact_power,
    two_contact_wrench,
)

pytestmark = pytest.mark.scientific


def test_opposed_normal_forces_form_a_pure_couple() -> None:
    positions = np.array([[-0.1, 0.0], [0.1, 0.0]])
    forces = np.array([[0.0, 100.0], [0.0, -100.0]])

    resultant, moment = two_contact_wrench(positions, forces, np.zeros(2))

    np.testing.assert_allclose(resultant, np.zeros(2), atol=1e-12)
    assert moment == pytest.approx(-20.0)


def test_wrench_transport_preserves_the_physical_force_system() -> None:
    positions = np.array([[-0.12, 0.04], [0.08, -0.02]])
    forces = np.array([[40.0, 120.0], [-10.0, -65.0]])
    point_a = np.array([0.0, 0.0])
    point_b = np.array([0.3, -0.2])

    resultant_a, moment_a = two_contact_wrench(positions, forces, point_a)
    resultant_b, moment_b = two_contact_wrench(positions, forces, point_b)

    np.testing.assert_allclose(resultant_a, resultant_b)
    assert transport_planar_moment(moment_a, resultant_a, point_a, point_b) == (
        pytest.approx(moment_b)
    )


def test_rigid_translation_and_rotation_leave_resultant_norm_and_moment() -> None:
    positions = np.array([[-0.1, 0.0], [0.1, 0.0]])
    forces = np.array([[20.0, 80.0], [-5.0, -30.0]])
    reference = np.array([0.02, -0.03])
    angle = 0.73
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    offset = np.array([1.7, -0.4])

    resultant, moment = two_contact_wrench(positions, forces, reference)
    transformed_resultant, transformed_moment = two_contact_wrench(
        positions @ rotation.T + offset,
        forces @ rotation.T,
        reference @ rotation.T + offset,
    )

    assert np.linalg.norm(transformed_resultant) == pytest.approx(
        np.linalg.norm(resultant)
    )
    assert transformed_moment == pytest.approx(moment)


def test_grip_component_resolution_round_trips() -> None:
    forces = np.array([[5.0, -2.0], [-3.0, 7.0]])
    grip_axis = np.array([3.0, 4.0])

    axial, normal, axial_axis, normal_axis = resolve_grip_components(forces, grip_axis)

    reconstructed = axial[:, None] * axial_axis + normal[:, None] * normal_axis
    np.testing.assert_allclose(reconstructed, forces, atol=1e-12)
    assert np.dot(axial_axis, normal_axis) == pytest.approx(0.0)


def test_contact_power_includes_force_and_free_torque_terms() -> None:
    forces = np.array([[10.0, 0.0], [0.0, -4.0]])
    velocities = np.array([[2.0, 3.0], [-1.0, 5.0]])
    torques = np.array([3.0, -1.0])
    angular_velocity = 4.0

    force_power, torque_power, total = two_contact_power(
        forces, velocities, torques, angular_velocity
    )

    assert force_power == pytest.approx(0.0)
    assert torque_power == pytest.approx(8.0)
    assert total == pytest.approx(8.0)


def test_zero_crossing_is_linearly_interpolated_and_direction_labeled() -> None:
    time = np.array([0.0, 0.1, 0.2, 0.3])
    values = np.array([2.0, 1.0, -1.0, 3.0])

    crossings = find_zero_crossings(time, values)

    assert crossings == [
        (pytest.approx(0.15), "positive_to_negative"),
        (pytest.approx(0.225), "negative_to_positive"),
    ]


@pytest.mark.parametrize(
    ("positions", "forces", "reference"),
    [
        (np.zeros((2, 3)), np.zeros((2, 2)), np.zeros(2)),
        (np.zeros((2, 2)), np.zeros((3, 2)), np.zeros(2)),
        (np.zeros((2, 2)), np.zeros((2, 2)), np.zeros(3)),
    ],
)
def test_two_contact_wrench_rejects_invalid_shapes(
    positions: np.ndarray, forces: np.ndarray, reference: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        two_contact_wrench(positions, forces, reference)
