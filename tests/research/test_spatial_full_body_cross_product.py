"""Equivalence contract for the fast 3-vector cross product (#9231).

``spatial_full_body`` replaces ``numpy.cross`` with ``_cross3`` on its hot
Jacobian-assembly paths.  The substitution is only admissible if it is
bit-identical, because the module's outputs are registered research evidence.
These tests pin that equivalence so a future edit to ``_cross3`` cannot
silently move the numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.spatial_full_body import (
    _cross3,
    build_spatial_model,
    forward_kinematics,
    point_contact_jacobians,
    prescribed_state,
)

pytestmark = pytest.mark.unit


def test_cross3_matches_numpy_cross_bit_for_bit_over_random_vectors() -> None:
    generator = np.random.default_rng(20260829)
    for scale in (1.0e-8, 1.0, 1.0e6):
        first = generator.standard_normal((2000, 3)) * scale
        second = generator.standard_normal((2000, 3)) * scale
        for row in range(first.shape[0]):
            expected = np.cross(first[row], second[row])
            actual = _cross3(first[row], second[row])
            assert actual.dtype == expected.dtype
            assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((-2.5, 7.25, 0.125), (0.5, -0.25, 3.0)),
        ((1.0e-300, 1.0e-300, 1.0e-300), (1.0e-300, -1.0e-300, 1.0e-300)),
    ],
)
def test_cross3_matches_numpy_cross_on_declared_edge_cases(
    first: tuple[float, float, float], second: tuple[float, float, float]
) -> None:
    left = np.array(first, dtype=np.float64)
    right = np.array(second, dtype=np.float64)
    assert np.array_equal(_cross3(left, right), np.cross(left, right))


def test_jacobian_assembly_reproduces_numpy_cross_reference() -> None:
    model = build_spatial_model()
    q, _, _ = prescribed_state(model, 0.17)
    kinematics = forward_kinematics(model, q)

    # Independent reference assembly for one body, built with ``numpy.cross``.
    body_index = 5
    body = model.bodies[body_index]
    rotation = kinematics.joint_rotation[body.joint]
    com = kinematics.joint_position_m[body.joint] + rotation @ body.com_offset_m
    reference = np.zeros((3, model.nq))
    cursor = body.joint
    while cursor >= 0:
        joint = model.joints[cursor]
        if joint.kind == "prismatic":
            reference[:, cursor] = kinematics.joint_axis_world[cursor]
        else:
            reference[:, cursor] = np.cross(
                kinematics.joint_axis_world[cursor],
                com - kinematics.joint_position_m[cursor],
            )
        cursor = joint.parent
    assert np.array_equal(kinematics.body_linear_jacobian[body_index], reference)

    point, linear, angular = point_contact_jacobians(
        model, kinematics, model.lead_hand_joint, np.array([0.055, 0.0, 0.0])
    )
    assert point.shape == (3,)
    assert linear.shape == angular.shape == (3, model.nq)
    assert np.all(np.isfinite(linear)) and np.all(np.isfinite(angular))
