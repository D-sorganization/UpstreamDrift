"""Parity tests for the consolidated quaternion SLERP (issue #7707).

The four historical SLERP copies (canonical ``math_utils.quaternion.slerp`` plus
three siblings) now delegate to a single implementation. These tests pin the
canonical behaviour and verify the siblings produce identical results, including
across the ``0.9995`` nlerp-fallback threshold boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from src.research.multi_robot.coordination import CooperativeManipulation
from src.shared.python.math_utils.quaternion import (
    SLERP_LERP_FALLBACK_THRESHOLD,
    slerp,
)
from src.shared.python.spatial_algebra.pose6dof.rotations import (
    slerp as rotations_slerp,
)
from src.unreal_integration.skeleton_mapper import SkeletonMapper

pytestmark = pytest.mark.unit


def _angle_quat(angle: float) -> np.ndarray:
    """Unit quaternion of a rotation by ``angle`` about the X axis (w-first)."""
    return np.array(
        [np.cos(angle / 2.0), np.sin(angle / 2.0), 0.0, 0.0],
        dtype=np.float64,
    )


def test_threshold_constant_value() -> None:
    """The nlerp-fallback threshold is a single named constant."""
    assert SLERP_LERP_FALLBACK_THRESHOLD == 0.9995


def test_endpoints() -> None:
    q0 = _angle_quat(0.0)
    q1 = _angle_quat(np.pi / 2.0)
    assert np.allclose(slerp(q0, q1, 0.0), q0)
    assert np.allclose(slerp(q0, q1, 1.0), q1)


@pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_rotations_sibling_matches_canonical(t: float) -> None:
    q0 = _angle_quat(0.0)
    q1 = _angle_quat(np.pi / 3.0)
    assert np.allclose(rotations_slerp(q0, q1, t), slerp(q0, q1, t))


@pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_coordination_sibling_matches_canonical(t: float) -> None:
    cm = CooperativeManipulation(["r1"])
    q0 = _angle_quat(0.0)
    q1 = _angle_quat(np.pi / 3.0)
    assert np.allclose(cm._slerp(q0, q1, t), slerp(q0, q1, t))


@pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_skeleton_mapper_sibling_matches_canonical(t: float) -> None:
    # Inputs are already unit, so the mapper's pre-normalization is a no-op
    # and results must equal the canonical implementation exactly.
    q0 = _angle_quat(0.0)
    q1 = _angle_quat(np.pi / 3.0)
    assert np.allclose(SkeletonMapper.slerp(q0, q1, t), slerp(q0, q1, t))


def test_threshold_boundary_nlerp_fallback() -> None:
    """Just above the threshold the result is the normalized linear blend."""
    # Pick an angle whose cos(angle/2 * 2) = dot is just above 0.9995.
    small_angle = 2.0 * np.arccos(np.sqrt((SLERP_LERP_FALLBACK_THRESHOLD + 1.0) / 2.0))
    q0 = _angle_quat(0.0)
    q1 = _angle_quat(small_angle)
    dot = float(np.dot(q0, q1))
    assert dot > SLERP_LERP_FALLBACK_THRESHOLD
    expected = q0 + 0.5 * (q1 - q0)
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(slerp(q0, q1, 0.5), expected)


def test_negative_dot_takes_short_path() -> None:
    q0 = _angle_quat(0.0)
    q1 = -_angle_quat(np.pi / 2.0)
    result = slerp(q0, q1, 0.5)
    assert np.isclose(np.linalg.norm(result), 1.0)
    # Equivalent to interpolating against +q1 (double-cover shortest arc).
    assert np.allclose(result, slerp(q0, -q1, 0.5))
