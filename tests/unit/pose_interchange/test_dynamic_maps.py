"""NSX-50: dual velocity / acceleration / generalized-force maps (#12578).

The canonical-v2 tangent vector ``(v_c, a_c, tau_c)`` is mapped to each engine's
native tangent coordinates by ``T(q)``.  Acceptance: power ``tau . v`` is
invariant, ``Tdot`` is the true time derivative of ``T`` along the motion,
and the maps round-trip.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.pose_interchange import dynamic_maps as dm
from src.shared.python.pose_interchange import se3
from src.shared.python.pose_interchange.adapters import (
    DrakeAdapter,
    MujocoAdapter,
    OpenSimAdapter,
    PinocchioAdapter,
)
from src.shared.python.pose_interchange.canonical_state import CanonicalState

pytestmark = pytest.mark.unit

ADAPTERS = {
    "pinocchio": PinocchioAdapter,
    "mujoco": MujocoAdapter,
    "drake": DrakeAdapter,
    "opensim": OpenSimAdapter,
}


def _canonical_state(seed: int) -> CanonicalState:
    rng = np.random.default_rng(seed)
    n_joints = len(dm.canonical_joint_names(PinocchioAdapter().joint_layout()))
    quat = se3.quat_exp(rng.normal(scale=0.4, size=3))
    q = np.concatenate([rng.normal(size=3), quat, rng.normal(size=n_joints)])
    nv = 6 + n_joints
    return CanonicalState(q=q, v=rng.normal(size=nv), a=rng.normal(size=nv))


def native_q(engine: str, state: CanonicalState) -> np.ndarray:
    """Native configuration (dynamic layout) of a canonical state."""
    pos, quat, joints = state.base_position, state.base_quat_wxyz, state.joint_q
    if engine == "pinocchio":
        return np.concatenate([pos, quat[[1, 2, 3, 0]], joints])
    if engine == "mujoco":
        return np.concatenate([pos, quat, joints])
    if engine == "drake":
        return np.concatenate([quat, pos, joints])
    euler = se3.matrix_to_euler_xyz_deg(se3.quat_to_matrix(quat))
    return np.concatenate([pos, euler, np.degrees(joints)])


@pytest.fixture(params=sorted(ADAPTERS))
def engine(request) -> str:
    return request.param


def test_adapters_implement_dynamic_map_protocol(engine: str) -> None:
    assert isinstance(ADAPTERS[engine](), dm.DynamicMapAdapter)


def test_map_shapes_and_invertibility(engine: str) -> None:
    state = _canonical_state(1)
    adapter = ADAPTERS[engine]()
    t_mat = adapter.velocity_map(native_q(engine, state))
    assert t_mat.shape == (state.nv, state.nv)
    assert np.linalg.matrix_rank(t_mat) == state.nv


def test_known_base_blocks() -> None:
    state = _canonical_state(2)
    rot = se3.quat_to_matrix(state.base_quat_wxyz)
    t_pin = PinocchioAdapter().velocity_map(native_q("pinocchio", state))
    np.testing.assert_allclose(t_pin[0:3, 0:3], rot.T, atol=1e-12)
    np.testing.assert_allclose(t_pin[3:6, 3:6], np.eye(3), atol=1e-12)
    t_mj = MujocoAdapter().velocity_map(native_q("mujoco", state))
    np.testing.assert_allclose(t_mj, np.eye(state.nv), atol=1e-12)
    t_dr = DrakeAdapter().velocity_map(native_q("drake", state))
    np.testing.assert_allclose(t_dr[0:3, 3:6], rot, atol=1e-12)
    np.testing.assert_allclose(t_dr[3:6, 0:3], np.eye(3), atol=1e-12)


@pytest.mark.parametrize("seed", [3, 4, 5])
def test_power_invariance(engine: str, seed: int) -> None:
    state = _canonical_state(seed)
    adapter = ADAPTERS[engine]()
    q_n = native_q(engine, state)
    t_mat = adapter.velocity_map(q_n)
    tau_n = np.random.default_rng(seed + 100).normal(size=state.nv)
    tau_c = adapter.generalized_force_map(q_n) @ tau_n
    v_n = dm.native_velocity(t_mat, state.v)
    assert tau_n @ v_n == pytest.approx(tau_c @ state.v, rel=1e-12, abs=1e-12)
    assert dm.check_power_invariance(t_mat, t_mat.T, state.v, tau_n) == pytest.approx(
        tau_n @ v_n
    )
    np.testing.assert_allclose(
        dm.canonical_force(t_mat, tau_n, v_canonical=state.v), tau_c
    )


def test_power_check_rejects_a_non_dual_force(engine: str) -> None:
    state = _canonical_state(6)
    adapter = ADAPTERS[engine]()
    q_n = native_q(engine, state)
    t_mat = adapter.velocity_map(q_n)
    tau_n = np.random.default_rng(6).normal(size=state.nv)
    wrong = adapter.generalized_force_map(q_n) + 1e-3
    with pytest.raises(ValueError, match="power not preserved"):
        dm.check_power_invariance(t_mat, wrong, state.v, tau_n)


def test_round_trips(engine: str) -> None:
    state = _canonical_state(7)
    adapter = ADAPTERS[engine]()
    q_n = native_q(engine, state)
    t_mat, t_dot = adapter.acceleration_map(q_n, state.v)
    v_n = dm.native_velocity(t_mat, state.v)
    np.testing.assert_allclose(dm.canonical_velocity(t_mat, v_n), state.v, atol=1e-10)
    a_n = dm.native_acceleration(t_mat, t_dot, state.a, state.v)
    np.testing.assert_allclose(
        dm.canonical_acceleration(t_mat, t_dot, a_n, state.v), state.a, atol=1e-10
    )
    tau_n = np.random.default_rng(8).normal(size=state.nv)
    tau_c = dm.canonical_force(t_mat, tau_n)
    np.testing.assert_allclose(dm.native_force(t_mat, tau_c), tau_n, atol=1e-10)


def test_tdot_is_time_derivative_of_t(engine: str) -> None:
    """Central difference of T along the constant canonical velocity."""
    state = _canonical_state(9)
    adapter = ADAPTERS[engine]()
    h = 1e-6

    def t_at(time: float) -> np.ndarray:
        moved = state.integrate(state.v * time)
        return adapter.velocity_map(native_q(engine, moved))

    _, t_dot = adapter.acceleration_map(native_q(engine, state), state.v)
    fd = (t_at(h) - t_at(-h)) / (2 * h)
    np.testing.assert_allclose(t_dot @ state.v, fd @ state.v, atol=1e-6)


def test_native_acceleration_matches_finite_difference(engine: str) -> None:
    """a_n = d/dt (T(q(t)) v_c(t)) with v_c(t) = v + a t (to first order)."""
    state = _canonical_state(10)
    adapter = ADAPTERS[engine]()
    h = 1e-6

    def v_native_at(time: float) -> np.ndarray:
        moved = state.integrate(state.v * time + 0.5 * state.a * time**2)
        return adapter.velocity_map(native_q(engine, moved)) @ (
            state.v + state.a * time
        )

    t_mat, t_dot = adapter.acceleration_map(native_q(engine, state), state.v)
    a_n = dm.native_acceleration(t_mat, t_dot, state.a, state.v)
    fd = (v_native_at(h) - v_native_at(-h)) / (2 * h)
    np.testing.assert_allclose(a_n, fd, atol=1e-6)


def test_opensim_euler_rate_matrix_matches_rotation_kinematics() -> None:
    """omega_body = S euler_rate, checked against R^T dR/dt."""
    euler = np.array([12.0, -25.0, 40.0])
    rate = np.array([0.3, -0.2, 0.5])  # deg/s
    h = 1e-6
    rot = lambda e: se3.euler_xyz_deg_to_matrix(e)  # noqa: E731
    dr = (rot(euler + h * rate) - rot(euler - h * rate)) / (2 * h)
    spin = rot(euler).T @ dr
    omega = np.array([spin[2, 1], spin[0, 2], spin[1, 0]])
    s_mat, _, _ = dm._euler_rate_matrix(np.radians(euler))
    np.testing.assert_allclose(s_mat @ np.radians(rate), omega, atol=1e-9)


def test_joint_units_and_sign_follow_the_slot() -> None:
    from dataclasses import replace

    adapter = OpenSimAdapter()
    layout = dict(adapter.joint_layout())
    first = dm.canonical_joint_names(layout)[0]
    layout[first] = replace(layout[first], sign=-1)
    state = _canonical_state(11)
    t_mat, _ = dm.velocity_map_matrices(
        adapter.dynamic_layout, native_q("opensim", state), layout
    )
    row = layout[first].start_index - 6 + 6
    assert t_mat[row, 6] == pytest.approx(-180.0 / np.pi)


def test_gimbal_lock_is_rejected() -> None:
    state = _canonical_state(12)
    q_n = native_q("opensim", state)
    q_n[4] = 90.0
    with pytest.raises(ValueError, match="singular"):
        OpenSimAdapter().velocity_map(q_n)


def test_preconditions() -> None:
    adapter = PinocchioAdapter()
    state = _canonical_state(13)
    q_n = native_q("pinocchio", state)
    with pytest.raises(ValueError):
        adapter.velocity_map(q_n[:5])
    with pytest.raises(ValueError):
        adapter.velocity_map(np.where(np.arange(q_n.size) == 0, np.nan, q_n))
    with pytest.raises(ValueError, match="length"):
        adapter.acceleration_map(q_n, np.zeros(3))
    bad = q_n.copy()
    bad[3:7] *= 2.0
    with pytest.raises(ValueError, match="unit"):
        adapter.velocity_map(bad)
    with pytest.raises(ValueError):
        dm.NativeDynamicLayout("x", "euler_xyz_deg", 3, 6, False, "world", "body")
