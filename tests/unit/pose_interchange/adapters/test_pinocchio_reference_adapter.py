from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from src.shared.python.pose_interchange.adapters.pinocchio_reference import (
    CanonicalV2State,
    PinocchioReferenceAdapter,
    PinocchioReferenceCapability,
)

pytestmark = pytest.mark.unit


class _ReferenceBackend:
    def __init__(self) -> None:
        self.inertial = np.array([2.0, -3.0], dtype=np.float64)
        self.last_rnea_q: np.ndarray | None = None

    def fk(self, q: np.ndarray, frame_name: str) -> np.ndarray:
        assert frame_name == "club_head"
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = q[:3] + np.array([0.001, 0.002, 0.003])
        return transform

    def jacobian(self, q: np.ndarray, frame_name: str) -> np.ndarray:
        assert frame_name == "club_head"
        jacobian = np.arange(36, dtype=np.float64).reshape(6, 6)
        return jacobian[:, : q.shape[0] - 1]

    def rnea(self, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        self.last_rnea_q = q.copy()
        return a + 0.1 * v + 0.01 * q[: v.shape[0]] + self.inertial[0]

    def aba(self, q: np.ndarray, v: np.ndarray, tau: np.ndarray) -> np.ndarray:
        return tau - 0.2 * v - 0.01 * q[: v.shape[0]]

    def set_inertial(self, values: np.ndarray) -> None:
        self.inertial = values.copy()


def _state() -> CanonicalV2State:
    return CanonicalV2State(
        q=np.array(
            [1.0, 2.0, 3.0, 0.5, -0.5, 0.25, -0.25, 0.1],
            dtype=np.float64,
        ),
        v=np.array([10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 0.4], dtype=np.float64),
        a=np.array([0.1, 0.2, 0.3, 4.0, 5.0, 6.0, 0.7], dtype=np.float64),
    )


def test_canonical_v2_roundtrip_remaps_pinocchio_w_last_quaternion() -> None:
    adapter = PinocchioReferenceAdapter(_ReferenceBackend())
    canonical = _state()

    native = adapter.from_canonical_v2(canonical)

    np.testing.assert_allclose(native.q[:7], [1.0, 2.0, 3.0, -0.5, 0.25, -0.25, 0.5])
    np.testing.assert_allclose(native.v[:6], [1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    np.testing.assert_allclose(native.a[:6], [4.0, 5.0, 6.0, 0.1, 0.2, 0.3])

    recovered = adapter.to_canonical_v2(native)
    np.testing.assert_allclose(recovered.q, canonical.q, atol=1.0e-12)
    np.testing.assert_allclose(recovered.v, canonical.v, atol=1.0e-12)
    np.testing.assert_allclose(recovered.a, canonical.a, atol=1.0e-12)


def test_capabilities_declare_reference_dynamics_and_gradients() -> None:
    adapter = PinocchioReferenceAdapter(_ReferenceBackend())

    assert adapter.capabilities() == {
        PinocchioReferenceCapability.INVERSE_DYN,
        PinocchioReferenceCapability.FORWARD_DYN,
        PinocchioReferenceCapability.GRADIENTS,
    }
    caps = adapter.engine_capabilities()
    assert caps.has_parameter_gradients
    assert caps.has_state_control_gradients
    assert caps.has_forward_sim
    assert caps.to_dict()["inverse_dynamics"] == "full"


def test_fk_jacobian_rnea_and_aba_use_canonical_boundary() -> None:
    backend = _ReferenceBackend()
    adapter = PinocchioReferenceAdapter(backend)
    state = _state()

    fk = adapter.fk(state, "club_head")
    np.testing.assert_allclose(fk[:3, 3], [1.001, 2.002, 3.003])

    jacobian = adapter.jacobian(state, "club_head")
    native_jacobian = np.arange(36, dtype=np.float64).reshape(6, 6)
    np.testing.assert_allclose(jacobian["linear"], native_jacobian[:3])
    np.testing.assert_allclose(jacobian["angular"], native_jacobian[3:])
    np.testing.assert_allclose(
        jacobian["spatial"], np.vstack([native_jacobian[3:], native_jacobian[:3]])
    )

    tau = adapter.inverse_dynamics(state)
    native = adapter.from_canonical_v2(state)
    np.testing.assert_allclose(
        tau, native.a + 0.1 * native.v + 0.01 * native.q[:7] + 2.0
    )
    assert backend.last_rnea_q is not None
    np.testing.assert_allclose(backend.last_rnea_q[:7], native.q[:7])

    canonical_accel = adapter.forward_dynamics(state, tau)
    expected_native = tau - 0.2 * native.v - 0.01 * native.q[:7]
    np.testing.assert_allclose(canonical_accel[:6], expected_native[[3, 4, 5, 0, 1, 2]])


def test_inverse_dynamics_trajectory_uses_numpy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "upstream_pinocchio_id", raising=False)
    adapter = PinocchioReferenceAdapter(_ReferenceBackend())
    q = np.vstack([_state().q, _state().q + 0.01, _state().q + 0.03])
    times = np.array([0.0, 0.1, 0.2], dtype=np.float64)

    result = adapter.inverse_dynamics_trajectory(q, times)

    assert result.backend == "numpy"
    assert result.qdot.shape == (3, 7)
    assert result.qddot.shape == (3, 7)
    assert result.tau.shape == (3, 7)
    assert np.all(np.isfinite(result.tau))


def test_inverse_dynamics_trajectory_routes_through_rust_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust = types.SimpleNamespace()
    calls: dict[str, np.ndarray] = {}

    def _inverse_dynamics(q, times, callback, qdot_override, qddot_override):
        calls["q"] = q.copy()
        calls["times"] = times.copy()
        assert qdot_override is None
        assert qddot_override is None
        qdot = np.ones((q.shape[0], q.shape[1] - 1), dtype=np.float64)
        qddot = np.full_like(qdot, 2.0)
        tau = np.vstack(
            [
                callback(q[index], qdot[index], qddot[index])
                for index in range(q.shape[0])
            ]
        )
        return qdot, qddot, tau

    rust.inverse_dynamics = _inverse_dynamics
    monkeypatch.setitem(sys.modules, "upstream_pinocchio_id", rust)

    adapter = PinocchioReferenceAdapter(_ReferenceBackend())
    q = np.vstack([_state().q, _state().q + 0.01])
    result = adapter.inverse_dynamics_trajectory(q, np.array([0.0, 0.1]))

    assert result.backend == "rust"
    np.testing.assert_allclose(calls["q"][0, 3:7], [-0.5, 0.25, -0.25, 0.5])
    np.testing.assert_allclose(result.qdot[0, :6], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert result.tau.shape == (2, 7)


def test_gradient_fallback_covers_state_and_inertial_parameters() -> None:
    backend = _ReferenceBackend()
    adapter = PinocchioReferenceAdapter(backend)

    gradients = adapter.inverse_dynamics_gradients(
        _state(),
        inertial_parameters=backend.inertial,
        set_inertial_parameters=backend.set_inertial,
    )

    assert gradients.backend == "numpy"
    assert gradients.dtau_dq.shape == (7, 8)
    assert gradients.dtau_dv.shape == (7, 7)
    assert gradients.dtau_da.shape == (7, 7)
    assert gradients.dtau_dinertial.shape == (7, 2)
    np.testing.assert_allclose(np.diag(gradients.dtau_da), np.ones(7), atol=1.0e-9)
    np.testing.assert_allclose(np.diag(gradients.dtau_dv), np.full(7, 0.1), atol=1.0e-9)
    np.testing.assert_allclose(gradients.dtau_dinertial[:, 0], np.ones(7), atol=1.0e-9)
    np.testing.assert_allclose(gradients.dtau_dinertial[:, 1], np.zeros(7), atol=1.0e-9)
