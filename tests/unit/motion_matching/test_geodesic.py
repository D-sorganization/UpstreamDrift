"""Unit tests for ``_geodesic.quaternion_geodesic_angles``."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.motion_matching._geodesic import (
    quaternion_geodesic_angles,
)


def test_identical_quaternions_yield_zero_angle() -> None:
    q = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    angles = quaternion_geodesic_angles(q, q)
    assert np.allclose(angles, 0.0, atol=1e-15)


def test_sign_flip_yields_zero_angle() -> None:
    q = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
    angles = quaternion_geodesic_angles(q, -q)
    assert np.allclose(angles, 0.0, atol=1e-12)


def test_known_90_degree_rotation_about_z() -> None:
    # qz(90deg) = [cos(45), 0, 0, sin(45)] -> angle vs identity = pi/2
    s = np.sqrt(0.5)
    q1 = np.array([[1.0, 0.0, 0.0, 0.0]])
    q2 = np.array([[s, 0.0, 0.0, s]])
    angles = quaternion_geodesic_angles(q1, q2)
    assert angles.shape == (1,)
    assert abs(angles[0] - np.pi / 2) < 1e-12


def test_geodesic_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        quaternion_geodesic_angles(np.zeros((3, 4)), np.zeros((2, 4)))


def test_bad_inner_dim_raises() -> None:
    with pytest.raises(ValueError):
        quaternion_geodesic_angles(np.zeros((3, 3)), np.zeros((3, 3)))
