"""SO(3) geodesic angle from ``quat_inverse_distance`` (issue #15044)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.math_utils.quaternion import quat_inverse_distance

pytestmark = pytest.mark.unit


def test_quat_inverse_distance_zero_for_identity_and_antipode() -> None:
    q = np.array([0.6, 0.0, 0.8, 0.0], dtype=np.float64)
    q = q / np.linalg.norm(q)
    assert quat_inverse_distance(q, q) == pytest.approx(0.0, abs=1e-12)
    assert quat_inverse_distance(q, -q) == pytest.approx(0.0, abs=1e-12)


def test_quat_inverse_distance_pi_rotation_about_z() -> None:
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = np.array([0.0, 0.0, 0.0, 1.0])
    assert quat_inverse_distance(q0, q1) == pytest.approx(math.pi, rel=0.0, abs=1e-12)


def test_quat_inverse_distance_tiny_angle() -> None:
    theta = 1e-10
    half = 0.5 * theta
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    imaginary = math.sin(half)
    q1 = np.array([math.sqrt(1.0 - imaginary * imaginary), imaginary, 0.0, 0.0])
    observed = quat_inverse_distance(q0, q1)
    assert observed == pytest.approx(theta, rel=0.0, abs=1e-14)


def test_quat_inverse_distance_normalizes_non_unit_inputs() -> None:
    q0 = np.array([2.0, 0.0, 0.0, 0.0])
    q1 = np.array([0.0, 0.0, 0.0, 2.0])
    assert quat_inverse_distance(q0, q1) == pytest.approx(math.pi, rel=0.0, abs=1e-12)
