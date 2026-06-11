"""Unit tests for the private quaternion helpers."""

from __future__ import annotations

import numpy as np
import pytest
from src.engines.physics_engines.opensim.python.opensim_golf.fk import (
    _rotmat_to_quat as opensim_rotmat_to_quat,
)
from src.engines.physics_engines.pinocchio.python.motion_matching.fit_swing import (
    rotmat_to_quat_wxyz as pinocchio_rotmat_to_quat,
)
from src.shared.python.motion_matching.loaders._quaternion import (
    _canonicalize_sign,
    quat_inverse_distance,
    rotmat_to_quat,
    slerp,
)
from src.shared.python.pose_interchange.se3 import matrix_to_quat as se3_matrix_to_quat
from src.shared.python.spatial_algebra.pose6dof.rotations import (
    rotation_matrix_to_quaternion as spatial_rotation_matrix_to_quaternion,
)


def test_quaternion_q0_nonnegative_after_canonicalize() -> None:
    q = np.array([-0.5, 0.5, 0.5, -0.5])
    q = q / np.linalg.norm(q)
    out = _canonicalize_sign(q)
    assert out[0] >= 0.0


def test_quaternion_inverse_distance_zero_for_q_and_neg_q() -> None:
    q = np.array([0.6, 0.0, 0.8, 0.0])
    q = q / np.linalg.norm(q)
    assert quat_inverse_distance(q, q) == pytest.approx(0.0, abs=1e-9)
    assert quat_inverse_distance(q, -q) == pytest.approx(0.0, abs=1e-9)


def test_rotmat_identity_to_quat() -> None:
    q = rotmat_to_quat(np.eye(3))
    assert q == pytest.approx(np.array([1.0, 0.0, 0.0, 0.0]), abs=1e-12)


def test_rotmat_z_180_to_quat() -> None:
    rz = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    q = rotmat_to_quat(rz)
    assert q[0] >= 0.0
    assert abs(q[3]) == pytest.approx(1.0, abs=1e-9)


def test_rotmat_stack_returns_canonical_signs() -> None:
    stack = np.stack([np.eye(3) for _ in range(4)])
    q = rotmat_to_quat(stack)
    assert q.shape == (4, 4)
    assert np.all(q[:, 0] >= 0.0)


def test_rotmat_invalid_shape_raises() -> None:
    with pytest.raises(ValueError):
        rotmat_to_quat(np.zeros((2, 3)))


def test_slerp_endpoints() -> None:
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = np.array([0.0, 1.0, 0.0, 0.0])
    assert slerp(q0, q1, 0.0) == pytest.approx(q0)
    assert slerp(q0, q1, 1.0) == pytest.approx(q1)


def test_rotation_matrix_quaternion_converters_share_canonical_contract() -> None:
    """All public Shepperd wrappers agree with the canonical helper."""
    rot = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    expected = rotmat_to_quat(rot)

    converters = (
        se3_matrix_to_quat,
        spatial_rotation_matrix_to_quaternion,
        opensim_rotmat_to_quat,
        pinocchio_rotmat_to_quat,
    )
    for converter in converters:
        observed = converter(rot)
        assert observed[0] >= 0.0
        np.testing.assert_allclose(observed, expected, atol=1e-12)
