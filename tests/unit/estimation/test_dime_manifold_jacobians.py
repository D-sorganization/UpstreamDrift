"""Behavioural tests for the DIME manifold chart and retraction Jacobians (#11550).

The chart under test is the right (body-frame) retraction used by
``dime_manifold``: ``retract(q, v) = q (x) Exp(v)`` on unit quaternions and the
product chart ``R^3 x SO(3)`` (translation first, then rotation) for
``SE3Manifold``. Every analytic Jacobian is compared against a central finite
difference at seeded random poses, at rotations near 0 and near pi, and at
nonzero tangent velocities.

Tolerance: the central difference with step ``H = 1e-6`` has truncation error
``O(H^2) ~ 1e-12`` and round-off error ``O(eps / H) ~ 1e-10`` for the O(1)
derivatives involved, so ``FD_ATOL = 1e-7`` leaves three orders of magnitude of
margin while still rejecting any Jacobian that is wrong in its O(|v|) or
O(|phi|) terms (the previous implementations were off by >1e-2 at these points).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
import pytest

from src.shared.python.contracts import PreconditionError
from src.shared.python.estimation.dime_manifold import (
    ManifoldContract,
    QuaternionManifold,
    SE3Manifold,
)

H = 1e-6
FD_ATOL = 1e-7
SEED = 11550


def _central_difference(
    func: Callable[[np.ndarray], np.ndarray], x0: np.ndarray
) -> np.ndarray:
    """Central finite-difference Jacobian of ``func`` at ``x0``."""
    f0 = np.asarray(func(x0))
    jac = np.zeros((f0.size, x0.size), dtype=np.float64)
    for k in range(x0.size):
        step = np.zeros_like(x0)
        step[k] = H
        jac[:, k] = (np.asarray(func(x0 + step)) - np.asarray(func(x0 - step))) / (
            2.0 * H
        )
    return jac


def _local_after_retract(
    manifold: ManifoldContract, q0: np.ndarray, q1: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Return ``d -> local_coordinates(q0, retract(q1, d))``."""

    def func(d: np.ndarray) -> np.ndarray:
        return manifold.local_coordinates(q0, manifold.retract(q1, d))

    return func


def _random_unit_quaternion(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    return q


def _rotvec_with_angle(rng: np.random.Generator, angle: float) -> np.ndarray:
    axis = rng.normal(size=3)
    return angle * axis / np.linalg.norm(axis)


# Rotation angles exercised: Taylor branch, near zero, generic, and near pi.
ANGLES = [1e-5, 1e-3, 0.4, 1.3, 2.5, np.pi - 1e-3]


@pytest.mark.unit
@pytest.mark.parametrize("angle", ANGLES)
def test_quaternion_retract_jacobian_matches_fd_at_nonzero_velocity(
    angle: float,
) -> None:
    rng = np.random.default_rng(SEED)
    manifold = QuaternionManifold()
    for _ in range(5):
        q = _random_unit_quaternion(rng)
        v = _rotvec_with_angle(rng, angle)
        expected = _central_difference(partial(manifold.retract, q), v)
        actual = manifold.retract_jacobian(q, v)
        assert actual.shape == (4, 3)
        np.testing.assert_allclose(actual, expected, atol=FD_ATOL, rtol=0.0)


@pytest.mark.unit
@pytest.mark.parametrize("angle", ANGLES)
def test_quaternion_local_coordinates_jacobian_matches_fd_away_from_identity(
    angle: float,
) -> None:
    rng = np.random.default_rng(SEED)
    manifold = QuaternionManifold()
    zero = np.zeros(3)
    for _ in range(5):
        q0 = _random_unit_quaternion(rng)
        q1 = np.asarray(manifold.retract(q0, _rotvec_with_angle(rng, angle)))
        expected = _central_difference(
            _local_after_retract(manifold, q0, q1),
            zero,
        )
        actual = manifold.local_coordinates_jacobian(q0, q1)
        assert actual.shape == (3, 3)
        np.testing.assert_allclose(actual, expected, atol=FD_ATOL, rtol=0.0)


@pytest.mark.unit
def test_quaternion_local_coordinates_jacobian_respects_sign_equivalence() -> None:
    rng = np.random.default_rng(SEED)
    manifold = QuaternionManifold()
    q0 = _random_unit_quaternion(rng)
    q1 = np.asarray(manifold.retract(q0, _rotvec_with_angle(rng, 1.1)))
    np.testing.assert_allclose(
        manifold.local_coordinates_jacobian(q0, -q1),
        manifold.local_coordinates_jacobian(q0, q1),
        atol=1e-12,
    )


@pytest.mark.unit
@pytest.mark.parametrize("angle", ANGLES)
def test_quaternion_round_trip_local_after_retract(angle: float) -> None:
    rng = np.random.default_rng(SEED)
    manifold = QuaternionManifold()
    for _ in range(5):
        q = _random_unit_quaternion(rng)
        v = _rotvec_with_angle(rng, angle)
        np.testing.assert_allclose(
            manifold.local_coordinates(q, manifold.retract(q, v)), v, atol=1e-9
        )


@pytest.mark.unit
def test_se3_jacobians_match_fd_at_random_pose_and_velocity() -> None:
    rng = np.random.default_rng(SEED)
    manifold = SE3Manifold()
    zero = np.zeros(6)
    for angle in ANGLES:
        q = np.concatenate([rng.normal(size=3), _random_unit_quaternion(rng)])
        v = np.concatenate([rng.normal(size=3), _rotvec_with_angle(rng, angle)])

        expected_r = _central_difference(partial(manifold.retract, q), v)
        np.testing.assert_allclose(
            manifold.retract_jacobian(q, v), expected_r, atol=FD_ATOL, rtol=0.0
        )

        q1 = np.asarray(manifold.retract(q, v))
        expected_l = _central_difference(
            _local_after_retract(manifold, q, q1),
            zero,
        )
        np.testing.assert_allclose(
            manifold.local_coordinates_jacobian(q, q1),
            expected_l,
            atol=FD_ATOL,
            rtol=0.0,
        )
        np.testing.assert_allclose(manifold.local_coordinates(q, q1), v, atol=1e-9)


@pytest.mark.unit
def test_jacobians_reject_malformed_inputs() -> None:
    quat = QuaternionManifold()
    se3 = SE3Manifold()
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(PreconditionError):
        quat.retract_jacobian(q_id, np.zeros(4))
    with pytest.raises(PreconditionError):
        quat.retract_jacobian(q_id, np.array([np.nan, 0.0, 0.0]))
    with pytest.raises(PreconditionError):
        quat.retract_jacobian(np.array([2.0, 0.0, 0.0, 0.0]), np.zeros(3))
    with pytest.raises(PreconditionError):
        quat.local_coordinates_jacobian(q_id, np.array([np.inf, 0.0, 0.0, 0.0]))
    with pytest.raises(PreconditionError):
        se3.retract_jacobian(np.zeros(6), np.zeros(6))
    with pytest.raises(PreconditionError):
        se3.local_coordinates_jacobian(np.zeros(7), np.zeros(7))
