"""Free-flyer base frames at the Pinocchio reference boundary (issue #12605).

Canonical-v2 base linear velocity is WORLD-frame while Pinocchio's free-flyer
``v[0:3]`` is LOCAL, so ``v_local = R^T v_world``. The reference adapter used to
pass the linear block through unchanged, which only agrees when the base is
unrotated. These tests use a rotated base so that bug is visible, with expected
values derived by hand (90 degrees about +z) or from the independent
``se3.quat_to_matrix`` rather than from the NSX-50 map the adapter delegates to.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from src.shared.python.pose_interchange import se3
from src.shared.python.pose_interchange.adapters.pinocchio_reference import (
    CanonicalV2State,
    PinocchioNativeState,
    PinocchioReferenceAdapter,
)

pytestmark = pytest.mark.unit

_HALF = np.sqrt(0.5)
# 90 degrees about world +z: body x -> world y, so R^T e_x = -e_y.
_QUAT_Z90_WXYZ = np.array([_HALF, 0.0, 0.0, _HALF])


class _RecordingBackend:
    """Linear stand-in backend that records the native state it receives."""

    def __init__(self) -> None:
        self.rnea_v: list[np.ndarray] = []
        self.rnea_a: list[np.ndarray] = []

    def fk(self, q: np.ndarray, frame_name: str) -> np.ndarray:  # pragma: no cover
        return np.eye(4)

    def jacobian(self, q: np.ndarray, frame_name: str) -> np.ndarray:
        n_v = q.shape[0] - 1
        return np.arange(6 * n_v, dtype=np.float64).reshape(6, n_v) / 10.0

    def rnea(self, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        self.rnea_v.append(v.copy())
        self.rnea_a.append(a.copy())
        return a + 0.1 * v

    def aba(self, q: np.ndarray, v: np.ndarray, tau: np.ndarray) -> np.ndarray:
        return tau - 0.1 * v


def _z90_state(v: list[float], a: list[float] | None = None) -> CanonicalV2State:
    q = np.concatenate([[0.1, 0.2, 0.3], _QUAT_Z90_WXYZ, [0.5]])
    accel = np.zeros(7) if a is None else np.asarray(a, dtype=np.float64)
    return CanonicalV2State(q=q, v=np.asarray(v, dtype=np.float64), a=accel)


def _random_state(seed: int) -> CanonicalV2State:
    rng = np.random.default_rng(seed)
    quat = se3.quat_exp(rng.normal(scale=0.8, size=3))
    q = np.concatenate([rng.normal(size=3), quat, rng.normal(size=2)])
    return CanonicalV2State(q=q, v=rng.normal(size=8), a=rng.normal(size=8))


def test_world_linear_velocity_is_rotated_into_local_frame() -> None:
    adapter = PinocchioReferenceAdapter(_RecordingBackend())
    native = adapter.from_canonical_v2(_z90_state([1.0, 0.0, 0.0, 0.4, 0.5, 0.6, 0.7]))

    np.testing.assert_allclose(native.v[0:3], [0.0, -1.0, 0.0], atol=1e-12)
    # Angular velocity is body-frame on both sides; joints pass through.
    np.testing.assert_allclose(native.v[3:7], [0.4, 0.5, 0.6, 0.7], atol=1e-12)


def test_base_linear_acceleration_carries_rotating_frame_drift() -> None:
    # Unit world-x velocity, unit body yaw rate, zero canonical acceleration:
    # d/dt(R^T v) = -w x (R^T v) = -(e_z x -e_y) = -e_x.
    adapter = PinocchioReferenceAdapter(_RecordingBackend())
    native = adapter.from_canonical_v2(_z90_state([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    np.testing.assert_allclose(native.a[0:3], [-1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(native.a[3:7], np.zeros(4), atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rotated_base_matches_independent_rotation_and_round_trips(
    seed: int,
) -> None:
    adapter = PinocchioReferenceAdapter(_RecordingBackend())
    canonical = _random_state(seed)
    rot = se3.quat_to_matrix(canonical.q[3:7])

    native = adapter.from_canonical_v2(canonical)

    np.testing.assert_allclose(rot @ native.v[0:3], canonical.v[0:3], atol=1e-12)
    np.testing.assert_allclose(native.v[3:], canonical.v[3:], atol=1e-12)
    omega = canonical.v[3:6]
    expected_lin_acc = rot.T @ canonical.a[0:3] - np.cross(
        omega, rot.T @ canonical.v[0:3]
    )
    np.testing.assert_allclose(native.a[0:3], expected_lin_acc, atol=1e-12)

    recovered = adapter.to_canonical_v2(native)
    np.testing.assert_allclose(recovered.q, canonical.q, atol=1e-12)
    np.testing.assert_allclose(recovered.v, canonical.v, atol=1e-12)
    np.testing.assert_allclose(recovered.a, canonical.a, atol=1e-12)


def test_jacobian_columns_act_on_canonical_velocity() -> None:
    backend = _RecordingBackend()
    adapter = PinocchioReferenceAdapter(backend)
    canonical = _random_state(3)
    native = adapter.from_canonical_v2(canonical)

    blocks = adapter.jacobian(canonical, "club_head")

    native_jacobian = backend.jacobian(native.q, "club_head")
    np.testing.assert_allclose(
        blocks["spatial"] @ canonical.v, native_jacobian @ native.v, atol=1e-12
    )


def test_inverse_and_forward_dynamics_see_local_base_motion() -> None:
    backend = _RecordingBackend()
    adapter = PinocchioReferenceAdapter(backend)
    canonical = _random_state(4)
    native = adapter.from_canonical_v2(canonical)

    tau = adapter.inverse_dynamics(canonical)

    np.testing.assert_allclose(backend.rnea_v[-1], native.v, atol=1e-12)
    np.testing.assert_allclose(backend.rnea_a[-1], native.a, atol=1e-12)
    # aba inverts this stand-in rnea, so forward dynamics recovers canonical a.
    np.testing.assert_allclose(
        adapter.forward_dynamics(canonical, tau), canonical.a, atol=1e-12
    )


def test_trajectory_overrides_are_mapped_per_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "upstream_pinocchio_id", raising=False)
    backend = _RecordingBackend()
    adapter = PinocchioReferenceAdapter(backend)
    states = [_random_state(seed) for seed in (5, 6)]
    q = np.vstack([s.q for s in states])
    qdot = np.vstack([s.v for s in states])
    qddot = np.vstack([s.a for s in states])

    result = adapter.inverse_dynamics_trajectory(
        q, np.array([0.0, 0.1]), qdot=qdot, qddot=qddot
    )

    for index, state in enumerate(states):
        native = adapter.from_canonical_v2(state)
        np.testing.assert_allclose(backend.rnea_v[index], native.v, atol=1e-12)
        np.testing.assert_allclose(backend.rnea_a[index], native.a, atol=1e-12)
    np.testing.assert_allclose(result.qdot, qdot)
    np.testing.assert_allclose(result.qddot, qddot)


def test_trajectory_qddot_override_requires_qdot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "upstream_pinocchio_id", raising=False)
    adapter = PinocchioReferenceAdapter(_RecordingBackend())
    q = np.vstack([_z90_state([0.0] * 7).q] * 3)

    with pytest.raises(ValueError, match="requires the matching canonical qdot"):
        adapter.inverse_dynamics_trajectory(
            q, np.array([0.0, 0.1, 0.2]), qddot=np.zeros((3, 7))
        )


def test_non_unit_base_quaternion_is_rejected() -> None:
    adapter = PinocchioReferenceAdapter(_RecordingBackend())
    native = PinocchioNativeState(
        q=np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0]),
        v=np.zeros(6),
        a=np.zeros(6),
    )

    with pytest.raises(ValueError, match="unit"):
        adapter.to_canonical_v2(native)
