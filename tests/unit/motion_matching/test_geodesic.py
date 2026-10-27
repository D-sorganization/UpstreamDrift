"""Unit tests for ``_geodesic`` quaternion and geodesic operations (UD-24, #13868)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.motion_matching._geodesic import (
    normalise_quaternions,
    quaternion_geodesic_angles,
    quaternion_multiply,
    quaternion_to_rotmat_series,
    rotmat_frobenius_distances,
    rotmat_to_quaternion,
)

pytestmark = pytest.mark.unit


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


def test_geodesic_normalises_non_unit_quaternions() -> None:
    """DbC Postcondition: Non-unit sim quaternions produce exact geodesic angles (UD-24)."""
    s = np.sqrt(0.5)
    # Non-unit scaled quaternions (scale factors 3.0 and 5.0)
    q1_non_unit = np.array([[3.0, 0.0, 0.0, 0.0]])
    q2_non_unit = np.array([[5.0 * s, 0.0, 0.0, 5.0 * s]])
    angles = quaternion_geodesic_angles(q1_non_unit, q2_non_unit)
    assert angles.shape == (1,)
    assert abs(angles[0] - np.pi / 2) < 1e-12


def test_rotmat_frobenius_distances() -> None:
    s = np.sqrt(0.5)
    q1 = np.array([[1.0, 0.0, 0.0, 0.0]])
    q2 = np.array([[s, 0.0, 0.0, s]])
    dist = rotmat_frobenius_distances(q1, q2)
    assert dist.shape == (1,)
    # For 90 degree rotation: ||R - I||_F = sqrt(2 * (1 - cos(90))) * 2 = 2.0
    assert abs(dist[0] - 2.0) < 1e-12


def test_normalise_quaternions() -> None:
    # 2D batch
    q = np.array([[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 4.0, 0.0]])
    normed = normalise_quaternions(q)
    assert np.allclose(normed[0], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(normed[1], [0.0, 0.6, 0.8, 0.0])
    # Zero vector falls back to canonical identity [1, 0, 0, 0]
    zero_q = np.array([[0.0, 0.0, 0.0, 0.0]])
    normed_zero = normalise_quaternions(zero_q)
    assert np.allclose(normed_zero, [[1.0, 0.0, 0.0, 0.0]])


def test_quaternion_to_rotmat_series() -> None:
    # Identity
    q_id = np.array([[1.0, 0.0, 0.0, 0.0]])
    r_id = quaternion_to_rotmat_series(q_id)
    assert np.allclose(r_id[0], np.eye(3))

    # 90 deg around Z
    s = np.sqrt(0.5)
    q_z90 = np.array([[s, 0.0, 0.0, s]])
    r_z90 = quaternion_to_rotmat_series(q_z90)
    expected = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(r_z90[0], expected, atol=1e-12)


def test_quaternion_multiply() -> None:
    # i * j = k
    # [w, x, y, z]: i = [0, 1, 0, 0], j = [0, 0, 1, 0], k = [0, 0, 0, 1]
    qi = np.array([0.0, 1.0, 0.0, 0.0])
    qj = np.array([0.0, 0.0, 1.0, 0.0])
    qk = quaternion_multiply(qi, qj)
    assert np.allclose(qk, [0.0, 0.0, 0.0, 1.0])


def test_rotmat_to_quaternion_roundtrip() -> None:
    s = np.sqrt(0.5)
    q_orig = np.array([[s, 0.0, 0.0, s], [s, s, 0.0, 0.0]])
    rotmats = quaternion_to_rotmat_series(q_orig)
    q_recovered = rotmat_to_quaternion(rotmats)
    # Check that each recovered quaternion represents the same rotation (|q . q_orig| == 1)
    dots = np.abs(np.sum(q_orig * q_recovered, axis=-1))
    assert np.allclose(dots, 1.0, atol=1e-12)


def test_geodesic_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        quaternion_geodesic_angles(np.zeros((3, 4)), np.zeros((2, 4)))


def test_bad_inner_dim_raises() -> None:
    with pytest.raises(ValueError):
        quaternion_geodesic_angles(np.zeros((3, 3)), np.zeros((3, 3)))
