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
import types

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


def test_non_unit_base_quaternion_is_rejected() -> None:
    adapter = PinocchioReferenceAdapter(_RecordingBackend())
    native = PinocchioNativeState(
        q=np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0]),
        v=np.zeros(6),
        a=np.zeros(6),
    )

    with pytest.raises(ValueError, match="unit"):
        adapter.to_canonical_v2(native)


# --- Finite-difference trajectory path (issue #12700) ----------------------
#
# Without overrides the adapter differentiates the sampled q trajectory. The
# result must be the canonical-v2 velocity/acceleration (WORLD linear, BODY
# angular, joint rates), not component rates of the raw Pinocchio q. The
# trajectory below has a rotated base translating with constant WORLD
# acceleration and spinning about a fixed body axis with constant angular
# acceleration, so every derivative is known in closed form.

_FD_P0 = np.array([0.1, -0.2, 0.3])
_FD_V0 = np.array([0.8, -0.5, 0.3])
_FD_ACC = np.array([0.4, 0.2, -0.6])
_FD_AXIS = np.array([1.0, 2.0, -0.5]) / np.linalg.norm([1.0, 2.0, -0.5])
_FD_OMEGA0 = 1.3
_FD_ALPHA = 0.7
_FD_QUAT0 = se3.quat_exp(np.array([0.3, -0.9, 1.2]))


def _fd_canonical(t: float) -> CanonicalV2State:
    """Analytic canonical-v2 state of the FD reference trajectory at ``t``."""

    theta = _FD_OMEGA0 * t + 0.5 * _FD_ALPHA * t * t
    quat = se3.quat_multiply(_FD_QUAT0, se3.quat_exp(theta * _FD_AXIS))
    q = np.concatenate(
        [_FD_P0 + _FD_V0 * t + 0.5 * _FD_ACC * t * t, quat, [np.sin(2.0 * t)]]
    )
    v = np.concatenate(
        [
            _FD_V0 + _FD_ACC * t,
            (_FD_OMEGA0 + _FD_ALPHA * t) * _FD_AXIS,
            [2.0 * np.cos(2.0 * t)],
        ]
    )
    a = np.concatenate([_FD_ACC, _FD_ALPHA * _FD_AXIS, [-4.0 * np.sin(2.0 * t)]])
    return CanonicalV2State(q=q, v=v, a=a)


def _fd_trajectory() -> tuple[np.ndarray, np.ndarray, list[CanonicalV2State]]:
    # Non-uniform spacing exercises the variable-step central formulas.
    times = np.cumsum(np.r_[0.0, np.tile([0.004, 0.006], 10)])
    states = [_fd_canonical(float(t)) for t in times]
    return np.vstack([s.q for s in states]), times, states


def _assert_fd_matches_analytic(
    result_qdot: np.ndarray,
    result_qddot: np.ndarray,
    backend: _RecordingBackend,
    adapter: PinocchioReferenceAdapter,
    states: list[CanonicalV2State],
) -> None:
    interior = slice(1, -1)
    expected_v = np.vstack([s.v for s in states])
    expected_a = np.vstack([s.a for s in states])
    # Base blocks are exact for this trajectory; the sin joint carries the
    # O(dt) truncation error of the non-uniform three-point stencil.
    np.testing.assert_allclose(
        result_qdot[interior, :6], expected_v[interior, :6], atol=1e-9
    )
    np.testing.assert_allclose(
        result_qdot[interior, 6], expected_v[interior, 6], atol=1e-4
    )
    np.testing.assert_allclose(
        result_qddot[interior, :6], expected_a[interior, :6], atol=1e-7
    )
    np.testing.assert_allclose(
        result_qddot[interior, 6], expected_a[interior, 6], atol=2e-2
    )
    # Pinocchio receives the LOCAL-frame native image of the canonical motion.
    for index in range(1, len(states) - 1):
        native = adapter.from_canonical_v2(
            CanonicalV2State(
                q=states[index].q, v=result_qdot[index], a=result_qddot[index]
            )
        )
        np.testing.assert_allclose(backend.rnea_v[index], native.v, atol=1e-12)
        np.testing.assert_allclose(backend.rnea_a[index], native.a, atol=1e-12)
        exact = adapter.from_canonical_v2(states[index])
        np.testing.assert_allclose(backend.rnea_v[index][:6], exact.v[:6], atol=1e-9)


def test_finite_difference_trajectory_yields_canonical_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "upstream_pinocchio_id", raising=False)
    backend = _RecordingBackend()
    adapter = PinocchioReferenceAdapter(backend)
    q, times, states = _fd_trajectory()

    result = adapter.inverse_dynamics_trajectory(q, times)

    assert result.backend == "numpy"
    _assert_fd_matches_analytic(result.qdot, result.qddot, backend, adapter, states)


def test_finite_difference_feeds_rust_driver_local_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def _inverse_dynamics(q, times, callback, qdot_override, qddot_override):
        # The adapter differentiates on the Lie group itself; the driver only
        # loops RNEA over the LOCAL-frame derivatives it is handed.
        calls["overrides"] = (qdot_override is not None, qddot_override is not None)
        tau = np.vstack(
            [
                callback(q[i], qdot_override[i], qddot_override[i])
                for i in range(q.shape[0])
            ]
        )
        return qdot_override, qddot_override, tau

    monkeypatch.setitem(
        sys.modules,
        "upstream_pinocchio_id",
        types.SimpleNamespace(inverse_dynamics=_inverse_dynamics),
    )
    backend = _RecordingBackend()
    adapter = PinocchioReferenceAdapter(backend)
    q, times, states = _fd_trajectory()

    result = adapter.inverse_dynamics_trajectory(q, times)

    assert result.backend == "rust"
    assert calls["overrides"] == (True, True)
    _assert_fd_matches_analytic(result.qdot, result.qddot, backend, adapter, states)


def test_finite_difference_qdot_with_qddot_override_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A qddot override no longer needs a qdot override: the FD velocity is
    # canonical, so it can drive the base acceleration map.
    monkeypatch.delitem(sys.modules, "upstream_pinocchio_id", raising=False)
    backend = _RecordingBackend()
    adapter = PinocchioReferenceAdapter(backend)
    q, times, states = _fd_trajectory()
    qddot = np.vstack([s.a for s in states])

    result = adapter.inverse_dynamics_trajectory(q, times, qddot=qddot)

    np.testing.assert_allclose(result.qddot, qddot)
    _assert_fd_matches_analytic(result.qdot, qddot, backend, adapter, states)


def test_pinocchio_difference_agrees_with_canonical_finite_difference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pin = pytest.importorskip("pinocchio")
    if type(pin).__module__ == "unittest.mock":
        pytest.skip("tests/unit/conftest.py stubs pinocchio with a MagicMock")
    monkeypatch.delitem(sys.modules, "upstream_pinocchio_id", raising=False)
    model = pin.Model()
    joint = model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "base")
    model.addJoint(joint, pin.JointModelRevoluteZ(), pin.SE3.Identity(), "j1")
    backend = _RecordingBackend()
    adapter = PinocchioReferenceAdapter(backend)
    q, times, _ = _fd_trajectory()

    adapter.inverse_dynamics_trajectory(q, times)

    q_native = [adapter.from_canonical_v2(_fd_canonical(float(t))).q for t in times]
    for k in range(1, len(times) - 1):
        forward = pin.difference(model, q_native[k], q_native[k + 1])
        backward = pin.difference(model, q_native[k], q_native[k - 1])
        dt_b, dt_f = times[k] - times[k - 1], times[k + 1] - times[k]
        # Same second-order variable-step stencil, on Pinocchio's own chart.
        v_pin = (forward * dt_b**2 - backward * dt_f**2) / (dt_b * dt_f * (dt_b + dt_f))
        np.testing.assert_allclose(backend.rnea_v[k][:6], v_pin[:6], atol=1e-4)
