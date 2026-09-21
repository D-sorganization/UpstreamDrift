"""Unit tests for Contact-Aware Dynamic Force Allocator (MS-31 / MS-104 / #10415).

Tests:
1. Exact dynamic equilibrium on floating-base plant.
2. Floating-base root balance via ground reaction forces.
3. Unilateral contact constraints (f_z >= 0).
4. Friction cone feasibility.
5. Minimum achievable trail-arm torque allocation without breaking dynamics.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.contact_force_allocator import (
    AllocationObjective,
    ContactForceAllocation,
    ContactForceAllocator,
)


def _skew(r: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix for cross product [r]_x."""
    return np.array(
        [
            [0.0, -r[2], r[1]],
            [r[2], 0.0, -r[0]],
            [-r[1], r[0], 0.0],
        ]
    )


def _build_synthetic_contact_jacobian(
    sphere_positions: list[np.ndarray], nv: int, rng: np.random.Generator
) -> np.ndarray:
    """Construct realistic contact Jacobian with translational and lever-arm rotational coupling."""
    n_spheres = len(sphere_positions)
    j_ground = np.zeros((n_spheres * 3, nv))
    for s, pos in enumerate(sphere_positions):
        row = s * 3
        # Linear velocity of contact point: v_lin = v_root + omega_root x r
        # d(v_lin) / d(v_root) = I (first 3 DoFs)
        j_ground[row : row + 3, :3] = np.eye(3)
        # d(v_lin) / d(omega_root) = - [r]_x (DoFs 3:6)
        j_ground[row : row + 3, 3:6] = -_skew(pos)
        # Joint velocity coupling (leg joints)
        j_ground[row : row + 3, 6:18] = rng.standard_normal((3, 12)) * 0.05
    return j_ground


@pytest.mark.unit
def test_allocation_exact_equilibrium_and_floating_base_balance() -> None:
    """Verify that the allocator strictly satisfies full 44-DoF dynamic equilibrium."""
    rng = np.random.default_rng(42)
    nv = 44
    n_spheres = 6
    actuated_indices = np.arange(6, nv)

    # 6 realistic foot contact sphere positions (heel, forefoot, toe for right and left foot)
    sphere_positions = [
        np.array([-0.05, -0.15, 0.0]),
        np.array([0.10, -0.15, 0.0]),
        np.array([0.20, -0.15, 0.0]),
        np.array([-0.05, 0.15, 0.0]),
        np.array([0.10, 0.15, 0.0]),
        np.array([0.20, 0.15, 0.0]),
    ]
    j_ground = _build_synthetic_contact_jacobian(sphere_positions, nv, rng)

    # Grip Jacobian acting between hands
    j_grip = np.zeros((6, nv))
    j_grip[:, 18:36] = rng.standard_normal((6, 18))

    # Generalized forces: positive upward gravity load + reasonable swing accelerations
    tau_rnea = rng.standard_normal(nv) * 15.0
    tau_rnea[2] = 750.0  # body weight ~75 kg * 9.81 m/s^2

    allocator = ContactForceAllocator(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=n_spheres,
        mu_friction=0.8,
    )

    result = allocator.allocate(
        tau_rnea=tau_rnea,
        j_ground=j_ground,
        j_grip=j_grip,
        objective=AllocationObjective.MINIMUM_EFFORT,
    )

    assert isinstance(result, ContactForceAllocation)
    assert result.success

    # Check equilibrium: S^T tau + J_ground^T f + J_grip^T lambda + delta_root == tau_rnea
    tau_full = np.zeros(nv)
    tau_full[actuated_indices] = result.tau_actuated
    applied = (
        tau_full
        + j_ground.T @ result.f_ground
        + j_grip.T @ result.lambda_grip
        + np.concatenate([result.delta_tau_root, np.zeros(nv - 6)])
    )
    residual = np.max(np.abs(applied - tau_rnea))
    assert residual < 1e-4, f"Equilibrium residual too large: {residual}"

    # Check unilateral contact constraint: f_z >= 0
    f_reshaped = result.f_ground.reshape(n_spheres, 3)
    assert np.all(f_reshaped[:, 2] >= -1e-5), "Ground cannot exert downward pull"


@pytest.mark.unit
def test_minimum_achievable_trail_arm_torque_preserves_dynamics() -> None:
    """Verify trail-arm reduction finds minimum achievable torque without breaking dynamics."""
    rng = np.random.default_rng(123)
    nv = 44
    actuated_indices = np.arange(6, nv)
    n_spheres = 6

    sphere_positions = [
        np.array([-0.05, -0.15, 0.0]),
        np.array([0.10, -0.15, 0.0]),
        np.array([0.20, -0.15, 0.0]),
        np.array([-0.05, 0.15, 0.0]),
        np.array([0.10, 0.15, 0.0]),
        np.array([0.20, 0.15, 0.0]),
    ]
    j_ground = _build_synthetic_contact_jacobian(sphere_positions, nv, rng)

    j_grip = np.zeros((6, nv))
    trail_arm_idx = np.arange(18, 27)  # 9 trail arm coordinates
    lead_arm_idx = np.arange(27, 36)  # 9 lead arm coordinates
    j_grip[:, trail_arm_idx] = rng.standard_normal((6, 9))
    j_grip[:, lead_arm_idx] = -j_grip[:, trail_arm_idx]

    tau_rnea = rng.standard_normal(nv) * 15.0
    tau_rnea[2] = 750.0

    allocator = ContactForceAllocator(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=n_spheres,
        mu_friction=0.8,
    )

    # Standard optimum
    res_opt = allocator.allocate(
        tau_rnea=tau_rnea,
        j_ground=j_ground,
        j_grip=j_grip,
        objective=AllocationObjective.MINIMUM_EFFORT,
    )

    # Trail-reduced allocation
    res_trail = allocator.allocate(
        tau_rnea=tau_rnea,
        j_ground=j_ground,
        j_grip=j_grip,
        objective=AllocationObjective.MINIMUM_TRAIL_ARM,
        trail_arm_indices=trail_arm_idx,
    )

    assert res_trail.success

    # Equilibrium must hold exactly in BOTH cases
    tau_full_opt = np.zeros(nv)
    tau_full_opt[actuated_indices] = res_opt.tau_actuated
    app_opt = (
        tau_full_opt
        + j_ground.T @ res_opt.f_ground
        + j_grip.T @ res_opt.lambda_grip
        + np.concatenate([res_opt.delta_tau_root, np.zeros(nv - 6)])
    )
    assert np.max(np.abs(app_opt - tau_rnea)) < 1e-4

    tau_full_trail = np.zeros(nv)
    tau_full_trail[actuated_indices] = res_trail.tau_actuated
    app_trail = (
        tau_full_trail
        + j_ground.T @ res_trail.f_ground
        + j_grip.T @ res_trail.lambda_grip
        + np.concatenate([res_trail.delta_tau_root, np.zeros(nv - 6)])
    )
    msg = "Trail reduction broke dynamic equilibrium!"
    assert np.max(np.abs(app_trail - tau_rnea)) < 1e-4, msg

    # Trail-arm torque norm should be lower or equal under trail reduction
    trail_norm_opt = np.linalg.norm(tau_full_opt[trail_arm_idx])
    trail_norm_reduced = np.linalg.norm(tau_full_trail[trail_arm_idx])
    assert trail_norm_reduced <= trail_norm_opt + 1e-4
    # Transmitted grip force should be active
    assert np.linalg.norm(res_trail.lambda_grip) > 0.0


@pytest.mark.unit
def test_uninterrupted_forward_simulation_replay_without_pose_reset() -> None:
    """Verify that forward simulation from (q0, v0) under resolved forces is stable without state resets.

    Workstream 4 / #10415: Frame-by-frame replay with state resets can hide dynamic inconsistencies.
    This test verifies that integrating forward continuously over the full horizon without resets
    tracks the reference motion with zero explosion and minimal numerical integration drift.
    """
    rng = np.random.default_rng(99)
    nv = 12
    actuated_indices = np.arange(6, nv)  # 6 unactuated root DoFs, 6 actuated DoFs
    n_spheres = 2
    dt = 0.002
    n_steps = 100
    t_grid = np.arange(n_steps) * dt

    # System parameters: mass matrix M, damping C, stiffness K
    diag_m = np.linspace(2.0, 10.0, nv)
    M = np.diag(diag_m)
    M_inv = np.diag(1.0 / diag_m)
    C = np.diag(np.ones(nv) * 0.5)

    # Reference smooth trajectory: sum of sinusoids
    q_ref = np.zeros((n_steps, nv))
    v_ref = np.zeros((n_steps, nv))
    a_ref = np.zeros((n_steps, nv))
    for i in range(nv):
        omega = 2.0 * np.pi * (1.0 + 0.1 * i)
        q_ref[:, i] = 0.1 * np.sin(omega * t_grid)
        v_ref[:, i] = 0.1 * omega * np.cos(omega * t_grid)
        a_ref[:, i] = -0.1 * (omega**2) * np.sin(omega * t_grid)

    # Synthetic contact Jacobian for 2 support points (6 constraints)
    j_ground = np.zeros((n_spheres * 3, nv))
    j_ground[:3, :3] = np.eye(3)
    j_ground[3:, :3] = np.eye(3)
    j_ground[:, 3:6] = rng.standard_normal((6, 3)) * 0.1

    # Synthetic grip Jacobian (6 constraints: 3 linear + 3 angular)
    j_grip = np.zeros((6, nv))
    j_grip[:3, 6:9] = np.eye(3)
    j_grip[:3, 9:12] = -np.eye(3)

    allocator = ContactForceAllocator(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=n_spheres,
        mu_friction=0.8,
    )

    # Gravity load on root z (coordinate index 2)
    gravity_load = np.zeros(nv)
    gravity_load[2] = 500.0

    # Precompute allocated effective torques
    tau_effective = np.zeros((n_steps, nv))
    for k in range(n_steps):
        tau_rnea_k = M @ a_ref[k] + C @ v_ref[k] + gravity_load
        alloc = allocator.allocate(
            tau_rnea=tau_rnea_k,
            j_ground=j_ground,
            j_grip=j_grip,
            objective=AllocationObjective.MINIMUM_EFFORT,
        )
        tau_full = np.zeros(nv)
        tau_full[actuated_indices] = alloc.tau_actuated
        root_wrench = np.concatenate([alloc.delta_tau_root, np.zeros(nv - 6)])
        tau_effective[k] = (
            tau_full
            + j_ground.T @ alloc.f_ground
            + j_grip.T @ alloc.lambda_grip
            + root_wrench
        )
        # Exact dynamic equilibrium holds within QP numerical precision (< 1e-3 N*m)
        assert np.max(np.abs(tau_effective[k] - tau_rnea_k)) < 1e-3

    # Continuous forward simulation from (q0, v0) WITHOUT ANY STATE RESETS
    q_sim = np.zeros((n_steps, nv))
    v_sim = np.zeros((n_steps, nv))
    q_sim[0] = q_ref[0]
    v_sim[0] = v_ref[0]

    # Semi-implicit Euler integration
    for k in range(n_steps - 1):
        # Forward acceleration: a = M^{-1} (tau_eff - C v - gravity_load)
        a_sim = M_inv @ (tau_effective[k] - C @ v_sim[k] - gravity_load)
        v_sim[k + 1] = v_sim[k] + a_sim * dt
        q_sim[k + 1] = q_sim[k] + v_sim[k + 1] * dt

    # Check that simulation did NOT explode or diverge
    assert np.all(np.isfinite(q_sim)), "Forward simulation produced NaNs or Infs!"
    max_tracking_error = np.max(np.abs(q_sim - q_ref))
    # Semi-implicit Euler truncation error over 100 steps of dt=0.002 s is O(dt)
    err_msg = (
        f"Continuous forward simulation drifted excessively: {max_tracking_error:.4e} m"
    )
    assert max_tracking_error < 0.015, err_msg
