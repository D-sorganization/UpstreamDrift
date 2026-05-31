from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.pose_interchange.adapters.mujoco import (
    CanonicalV2State,
    MujocoAdapter,
    MujocoCanonicalV2Capability,
    MujocoNativeState,
)

pytestmark = pytest.mark.unit


class _MujocoBackend:
    def __init__(self) -> None:
        self.forward_q: np.ndarray | None = None
        self.inverse_q: np.ndarray | None = None

    def forward_dynamics(
        self,
        q: np.ndarray,
        v: np.ndarray,
        u: np.ndarray | None = None,
    ) -> np.ndarray:
        self.forward_q = q.copy()
        applied = np.zeros_like(v) if u is None else u
        return applied - 0.2 * v + 0.01 * q[: v.shape[0]]

    def inverse_dynamics(
        self, q: np.ndarray, v: np.ndarray, a: np.ndarray
    ) -> np.ndarray:
        self.inverse_q = q.copy()
        return a + 0.1 * v + 0.01 * q[: v.shape[0]]


def _state() -> CanonicalV2State:
    return CanonicalV2State(
        q=np.array(
            [1.0, 2.0, 3.0, 0.5, -0.5, 0.5, -0.5, 0.1],
            dtype=np.float64,
        ),
        v=np.array([4.0, 5.0, 6.0, 0.7, 0.8, 0.9, 0.2], dtype=np.float64),
        a=np.array([0.4, 0.5, 0.6, 1.7, 1.8, 1.9, 1.2], dtype=np.float64),
        t=0.125,
    )


def test_canonical_v2_roundtrip_keeps_mujoco_w_first_quaternion() -> None:
    adapter = MujocoAdapter()
    canonical = _state()

    native = adapter.from_canonical_v2(canonical)

    np.testing.assert_allclose(native.qpos, canonical.q)
    np.testing.assert_allclose(native.qvel, canonical.v)
    np.testing.assert_allclose(native.qacc, canonical.a)

    recovered = adapter.to_canonical_v2(native)
    np.testing.assert_allclose(recovered.q, canonical.q, atol=1.0e-12)
    np.testing.assert_allclose(recovered.v, canonical.v, atol=1.0e-12)
    np.testing.assert_allclose(recovered.a, canonical.a, atol=1.0e-12)
    assert recovered.t == pytest.approx(canonical.t)


def test_native_state_documents_mujoco_qvel_linear_then_body_angular() -> None:
    adapter = MujocoAdapter()
    native = adapter.from_canonical_v2(_state())

    np.testing.assert_allclose(native.qvel[:3], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(native.qvel[3:6], [0.7, 0.8, 0.9])


def test_capabilities_declare_supported_cc10_operations() -> None:
    adapter = MujocoAdapter()

    assert adapter.capabilities() == {
        MujocoCanonicalV2Capability.FORWARD_DYN,
        MujocoCanonicalV2Capability.INVERSE_DYN,
        MujocoCanonicalV2Capability.CONTACT,
    }
    caps = adapter.engine_capabilities()
    assert caps.has_forward_sim
    assert caps.has_contact_forces
    assert caps.has_contact_step
    assert caps.to_dict()["inverse_dynamics"] == "full"
    assert caps.extra["native_quat_order"] == "wxyz"


def test_supported_dynamics_capabilities_route_through_backend() -> None:
    backend = _MujocoBackend()
    adapter = MujocoAdapter(backend)
    state = _state()
    tau: np.ndarray = np.arange(7, dtype=np.float64)

    qacc = adapter.forward_dynamics(state, tau)
    expected_qacc = tau - 0.2 * state.v + 0.01 * state.q[:7]
    np.testing.assert_allclose(qacc, expected_qacc)
    assert backend.forward_q is not None
    np.testing.assert_allclose(backend.forward_q[:7], state.q[:7])

    inverse_tau = adapter.inverse_dynamics(state)
    expected_tau = state.a + 0.1 * state.v + 0.01 * state.q[:7]
    np.testing.assert_allclose(inverse_tau, expected_tau)
    assert backend.inverse_q is not None
    np.testing.assert_allclose(backend.inverse_q[:7], state.q[:7])


def test_canonical_v2_state_rejects_invalid_free_joint_shapes() -> None:
    with pytest.raises(ValueError, match="one more entry than v"):
        CanonicalV2State(
            q=np.zeros(7, dtype=np.float64),
            v=np.zeros(7, dtype=np.float64),
            a=np.zeros(7, dtype=np.float64),
        )


def test_to_canonical_v2_rejects_zero_quaternion() -> None:
    adapter = MujocoAdapter()
    with pytest.raises(ValueError, match="quaternion must be non-zero"):
        adapter.to_canonical_v2(
            MujocoNativeState(
                qpos=np.zeros(8, dtype=np.float64),
                qvel=np.zeros(7, dtype=np.float64),
                qacc=np.zeros(7, dtype=np.float64),
            )
        )
