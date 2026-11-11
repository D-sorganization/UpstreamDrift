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
