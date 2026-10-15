"""Unit tests for NSX-50 pose_interchange dynamic state and dual transforms (#12578)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.pose_interchange.adapters.drake import DrakeAdapter
from src.shared.python.pose_interchange.adapters.mujoco import MujocoAdapter
from src.shared.python.pose_interchange.adapters.opensim import OpenSimAdapter
from src.shared.python.pose_interchange.adapters.pinocchio import PinocchioAdapter
from src.shared.python.pose_interchange.dynamic_transforms import (
    euler_xyz_rate_matrix,
    euler_xyz_rate_matrix_dot,
)
from src.shared.python.pose_interchange.protocol import JointSlot
from src.shared.python.pose_interchange.se3 import (
    euler_xyz_deg_to_matrix,
    euler_xyz_deg_to_quat_wxyz,
    matrix_to_euler_xyz_deg,
)

pytestmark = pytest.mark.unit


def _sample_canonical_state(
    n_joints: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12578)
    pos = np.array([0.15, -0.25, 0.92], dtype=np.float64)
    rpy_deg = np.array([18.0, -24.0, 35.0], dtype=np.float64)
    quat_wxyz = euler_xyz_deg_to_quat_wxyz(rpy_deg)
    joints_q = rng.normal(scale=0.3, size=n_joints)
    q_c = np.concatenate([pos, quat_wxyz, joints_q])
    v_c = rng.normal(scale=1.2, size=6 + n_joints)
    a_c = rng.normal(scale=3.5, size=6 + n_joints)
    tau_c = rng.normal(scale=15.0, size=6 + n_joints)
    return q_c, v_c, a_c, tau_c


def _custom_model(
    base_offset: int, units: str = "rad"
) -> dict[str, dict[str, JointSlot]]:
    signs = [1, -1, 1, -1]
    layout = {
        f"j{i}": JointSlot(
            canonical_name=f"j{i}",
            engine_name=f"eng_j{i}",
            start_index=base_offset + i,
            length=1,
            units=units,
            sign=signs[i],
        )
        for i in range(4)
    }
    return {"joint_layout": layout}


def test_pinocchio_velocity_acceleration_and_dual_force_invariance() -> None:
    """Pinocchio adapter preserves virtual power tau . v and round-trips v, a, tau."""
    adapter = PinocchioAdapter()
    q_c, v_c, a_c, tau_c = _sample_canonical_state(4)
    model = _custom_model(base_offset=7, units="rad")
    q_pin = q_c.copy()
    q_pin[3:7] = np.array([q_c[4], q_c[5], q_c[6], q_c[3]])

    v_pin = adapter.velocity_map(q_c, v_c, model=model)
    a_pin = adapter.acceleration_map(q_c, v_c, a_c, model=model)
    tau_pin = adapter.generalized_force_map(q_c, tau_c, model=model)

    # Virtual power invariance: tau_pin . v_pin == tau_c . v_c
    np.testing.assert_allclose(
        float(tau_pin @ v_pin), float(tau_c @ v_c), atol=1e-12, rtol=1e-12
    )

    # Round-trip recovery from Pinocchio native q_pin
    v_rec = adapter.velocity_to_canonical(q_pin, v_pin, model=model)
    a_rec = adapter.acceleration_to_canonical(q_pin, v_pin, a_pin, model=model)
    tau_rec = adapter.generalized_force_to_canonical(q_pin, tau_pin, model=model)

    np.testing.assert_allclose(v_rec, v_c, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(a_rec, a_c, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(tau_rec, tau_c, atol=1e-12, rtol=1e-12)


def test_mujoco_velocity_acceleration_and_dual_force_invariance() -> None:
    """MuJoCo adapter preserves virtual power tau . v and round-trips v, a, tau."""
    adapter = MujocoAdapter()
    q_c, v_c, a_c, tau_c = _sample_canonical_state(4)
    model = _custom_model(base_offset=7, units="rad")

    v_mj = adapter.velocity_map(q_c, v_c, model=model)
    a_mj = adapter.acceleration_map(q_c, v_c, a_c, model=model)
    tau_mj = adapter.generalized_force_map(q_c, tau_c, model=model)

    np.testing.assert_allclose(
        float(tau_mj @ v_mj), float(tau_c @ v_c), atol=1e-12, rtol=1e-12
    )

    v_rec = adapter.velocity_to_canonical(q_c, v_mj, model=model)
    a_rec = adapter.acceleration_to_canonical(q_c, v_mj, a_mj, model=model)
    tau_rec = adapter.generalized_force_to_canonical(q_c, tau_mj, model=model)

    np.testing.assert_allclose(v_rec, v_c, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(a_rec, a_c, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(tau_rec, tau_c, atol=1e-12, rtol=1e-12)


def test_drake_quat_floating_and_rpy_power_invariance_and_round_trip() -> None:
    """Drake adapter preserves virtual power and round-trips in both quat_floating and rpy modes."""
    adapter = DrakeAdapter()
    q_c, v_c, a_c, tau_c = _sample_canonical_state(4)
    model = _custom_model(base_offset=6, units="rad")
    q_drake_quat = np.concatenate([q_c[3:7], q_c[0:3], q_c[7:]])

    v_dr = adapter.velocity_map(q_c, v_c, base_mode="quat_floating", model=model)
    a_dr = adapter.acceleration_map(
        q_c, v_c, a_c, base_mode="quat_floating", model=model
    )
    tau_dr = adapter.generalized_force_map(
        q_c, tau_c, base_mode="quat_floating", model=model
    )

    # Power invariance across [lin_w, ang_b] <-> [ang_w, lin_w]
    np.testing.assert_allclose(
        float(tau_dr @ v_dr), float(tau_c @ v_c), atol=1e-12, rtol=1e-12
    )

    v_rec = adapter.velocity_to_canonical(
        q_drake_quat, v_dr, base_mode="quat_floating", model=model
    )
    a_rec = adapter.acceleration_to_canonical(
        q_drake_quat, v_dr, a_dr, base_mode="quat_floating", model=model
    )
    tau_rec = adapter.generalized_force_to_canonical(
        q_drake_quat, tau_dr, base_mode="quat_floating", model=model
    )

    np.testing.assert_allclose(v_rec, v_c, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(a_rec, a_c, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(tau_rec, tau_c, atol=1e-12, rtol=1e-12)

    # Also test RPY mode (including T_dot v term)
    v_rpy = adapter.velocity_map(q_c, v_c, base_mode="rpy", model=model)
    a_rpy = adapter.acceleration_map(q_c, v_c, a_c, base_mode="rpy", model=model)
    tau_rpy = adapter.generalized_force_map(q_c, tau_c, base_mode="rpy", model=model)

    np.testing.assert_allclose(
        float(tau_rpy @ v_rpy), float(tau_c @ v_c), atol=1e-12, rtol=1e-12
    )
    v_rpy_rec = adapter.velocity_to_canonical(q_c, v_rpy, base_mode="rpy", model=model)
    a_rpy_rec = adapter.acceleration_to_canonical(
        q_c, v_rpy, a_rpy, base_mode="rpy", model=model
    )
    tau_rpy_rec = adapter.generalized_force_to_canonical(
        q_c, tau_rpy, base_mode="rpy", model=model
    )
    np.testing.assert_allclose(v_rpy_rec, v_c, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(a_rpy_rec, a_c, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(tau_rpy_rec, tau_c, atol=1e-12, rtol=1e-12)


def test_opensim_euler_deg_power_invariance_round_trip_and_tdot_v_term() -> None:
    """OpenSim adapter preserves virtual power tau . v, round-trips v/a/tau, and
    matches finite-differenced Euler acceleration (verifying the T_dot v term)."""
    adapter = OpenSimAdapter()
    q_c, v_c, a_c, tau_c = _sample_canonical_state(4)
    model = _custom_model(base_offset=6, units="deg")

    v_osim = adapter.velocity_map(q_c, v_c, model=model)
    a_osim = adapter.acceleration_map(q_c, v_c, a_c, model=model)
    tau_osim = adapter.generalized_force_map(q_c, tau_c, model=model)

    # Virtual power invariance across rad <-> deg and body-omega <-> Euler-deg rates
    np.testing.assert_allclose(
        float(tau_osim @ v_osim), float(tau_c @ v_c), atol=1e-11, rtol=1e-11
    )

    # Verify that the T_dot v term in acceleration_map is non-zero and matches
    # time differentiation of rpy_dot(t) = E(rpy(t))^{-1} omega_body(t)
    rpy_deg_0 = np.array([18.0, -24.0, 35.0], dtype=np.float64)
    rpy_rad_0 = np.radians(rpy_deg_0)
    omega_b = v_c[3:6]
    alpha_b = a_c[3:6]
    rpy_dot_rad = np.linalg.solve(euler_xyz_rate_matrix(rpy_rad_0), omega_b)

    dt = 1e-7
    rpy_plus = rpy_rad_0 + dt * rpy_dot_rad
    rpy_minus = rpy_rad_0 - dt * rpy_dot_rad
    omega_plus = omega_b + dt * alpha_b
    omega_minus = omega_b - dt * alpha_b
    rpy_dot_plus = np.linalg.solve(euler_xyz_rate_matrix(rpy_plus), omega_plus)
    rpy_dot_minus = np.linalg.solve(euler_xyz_rate_matrix(rpy_minus), omega_minus)
    rpy_ddot_fd_deg = np.degrees((rpy_dot_plus - rpy_dot_minus) / (2.0 * dt))

    np.testing.assert_allclose(a_osim[3:6], rpy_ddot_fd_deg, atol=1e-6, rtol=1e-6)

    # Also verify euler_xyz_rate_matrix_dot matches finite difference of R^T R_dot
    rot_plus = euler_xyz_deg_to_matrix(np.degrees(rpy_plus))
    rot_minus = euler_xyz_deg_to_matrix(np.degrees(rpy_minus))
    rot_0 = euler_xyz_deg_to_matrix(rpy_deg_0)
    skew_omega = rot_0.T @ ((rot_plus - rot_minus) / (2.0 * dt))
    omega_from_rot = np.array([skew_omega[2, 1], skew_omega[0, 2], skew_omega[1, 0]])
    np.testing.assert_allclose(omega_from_rot, omega_b, atol=1e-7, rtol=1e-7)

    e_dot_fd = (euler_xyz_rate_matrix(rpy_plus) - euler_xyz_rate_matrix(rpy_minus)) / (
        2.0 * dt
    )
    np.testing.assert_allclose(
        euler_xyz_rate_matrix_dot(rpy_rad_0, rpy_dot_rad),
        e_dot_fd,
        atol=1e-7,
        rtol=1e-7,
    )

    # Round-trip recovery from OpenSim 6+n_j q vector
    rot_0_deg = matrix_to_euler_xyz_deg(rot_0)
    q_osim = np.concatenate(
        [q_c[0:3], rot_0_deg, np.degrees(q_c[7:]) * np.array([1, -1, 1, -1])]
    )
    v_rec = adapter.velocity_to_canonical(q_osim, v_osim, model=model)
    a_rec = adapter.acceleration_to_canonical(q_osim, v_osim, a_osim, model=model)
    tau_rec = adapter.generalized_force_to_canonical(q_osim, tau_osim, model=model)

    np.testing.assert_allclose(v_rec, v_c, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(a_rec, a_c, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(tau_rec, tau_c, atol=1e-11, rtol=1e-11)


def test_dynamic_transforms_dbc_rejects_mismatched_dimensions_and_nan() -> None:
    """DbC preconditions reject mismatched vector dimensions, non-finite inputs, and gimbal lock."""
    q_c, v_c, a_c, _ = _sample_canonical_state(4)
    pin = PinocchioAdapter()
    osim = OpenSimAdapter()

    with pytest.raises(ValueError, match="length"):
        pin.velocity_map(q_c, v_c[:-1])

    with pytest.raises(ValueError, match="length"):
        pin.acceleration_map(q_c, v_c, a_c[:-1])

    with pytest.raises(ValueError, match="length"):
        pin.generalized_force_map(q_c, v_c[:-2])

    v_nan = v_c.copy()
    v_nan[2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        pin.velocity_map(q_c, v_nan)

    # Gimbal lock at pitch = +90 deg
    q_gimbal = np.array([0.0, 0.0, 0.0, 0.0, 90.0, 0.0, 0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="singular"):
        osim.velocity_to_canonical(q_gimbal, v_c)
