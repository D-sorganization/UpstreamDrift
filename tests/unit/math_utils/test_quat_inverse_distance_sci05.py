"""SO(3) rotation angle from ``quat_inverse_distance`` (issue #15044, SCI-05)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.math_utils.quaternion import quat_inverse_distance

pytestmark = pytest.mark.unit


def _axis_angle_quat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = 0.5 * angle_rad
    return np.array(
        [math.cos(half), *(math.sin(half) * axis)],
        dtype=np.float64,
    )


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def test_quat_inverse_distance_small_angle_matches_axis_angle() -> None:
    theta = 1e-10
    q_ident = np.array([1.0, 0.0, 0.0, 0.0])
    q_rot = _axis_angle_quat(np.array([1.0, 0.0, 0.0]), theta)
    assert quat_inverse_distance(q_ident, q_rot) == pytest.approx(
        theta, rel=0.0, abs=1e-20
    )


def test_quat_inverse_distance_pi_rotation_about_z() -> None:
    q_ident = np.array([1.0, 0.0, 0.0, 0.0])
    q_pi = _axis_angle_quat(np.array([0.0, 0.0, 1.0]), math.pi)
    assert quat_inverse_distance(q_ident, q_pi) == pytest.approx(
        math.pi, rel=0.0, abs=1e-12
    )


def test_quat_inverse_distance_renormalizes_inputs() -> None:
    theta = 0.25
    q0 = 2.0 * _axis_angle_quat(np.array([0.0, 1.0, 0.0]), 0.0)
    q1 = 3.0 * _axis_angle_quat(np.array([0.0, 1.0, 0.0]), theta)
    assert quat_inverse_distance(q0, q1) == pytest.approx(theta, rel=0.0, abs=1e-12)


def test_quat_inverse_distance_small_angle_near_pi_base_x() -> None:
    """Regression: vector-part difference fails when base orientation is ~180° (#15044)."""
    eps = 1e-6
    half = 0.5 * eps
    a = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    b = np.array([-math.sin(half), math.cos(half), 0.0, 0.0], dtype=np.float64)
    assert quat_inverse_distance(a, b) == pytest.approx(eps, rel=0.0, abs=1e-9)


@pytest.mark.parametrize("base_angle", [0.1, 0.75 * math.pi, math.pi, 1.25 * math.pi])
def test_quat_inverse_distance_invariant_to_global_rotation(base_angle: float) -> None:
    """Relative angle unchanged under a common left rotation."""
    delta = 0.03
    axis = np.array([0.3, 0.7, 0.2], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    q_left = _axis_angle_quat(axis, base_angle)
    q_a = _axis_angle_quat(np.array([1.0, 0.0, 0.0]), 0.0)
    q_b = _axis_angle_quat(np.array([1.0, 0.0, 0.0]), delta)
    expected = quat_inverse_distance(q_a, q_b)
    q_a_g = _quat_mul(q_left, q_a)
    q_b_g = _quat_mul(q_left, q_b)
    assert quat_inverse_distance(q_a_g, q_b_g) == pytest.approx(expected, abs=1e-12)
