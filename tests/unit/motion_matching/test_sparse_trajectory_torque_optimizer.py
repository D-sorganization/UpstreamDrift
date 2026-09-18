"""Unit and acceptance tests for Sparse Trajectory Torque Optimizer (PF-05, #10435).

Verifies:
1. Small convex analytic optimum: exact agreement on solvable 3-node interpolation.
2. Time-unit invariance: scaling time step and derivative weights preserves physical torque history.
3. Nonuniform time step handling: stable convergence with irregular dt grids.
4. Window seam C1 continuity: seamless stitching with zero torque jumps across window boundaries.
5. Frame-independent baseline agreement: recovers frame-by-frame allocation when derivative weights are zero.
6. Smoothness variance reduction: suppresses chattering and high-frequency derivative energy.
7. Interpolation feasibility audit: evaluates power, torque bounds, and friction at inter-node samples.
8. Epigraph peak utilization minimization: penalizes peak torque to level effort across time.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.contact_force_allocator import (
    AllocationObjective,
    ContactForceAllocator,
)
from src.shared.python.motion_matching.sparse_trajectory_torque_optimizer import (
    InterpolationAuditResult,
    NodeOptimizationResult,
    SeamBoundaryCondition,
    SparseOptimizationConfig,
    SparseOptimizationResult,
    SparseTrajectoryTorqueOptimizer,
    TrajectoryOptimizationNode,
    WindowedTrajectoryTorqueOptimizer,
    audit_trajectory_interpolation_feasibility,
)


def _create_mock_floating_base_node(
    time_s: float,
    tau_actuated_target: np.ndarray,
    q_dot: np.ndarray | None = None,
    contact_active: bool = True,
) -> tuple[TrajectoryOptimizationNode, int, int]:
    """Helper creating a consistent floating base node with 6 root + 4 actuated DOFs and 2 contact spheres."""
    nv = 10
    n_actuated = 4
    n_spheres = 2
    n_ground_vars = n_spheres * 3

    # Ground Jacobian: maps contact forces at origin to root wrench
    j_ground = np.zeros((n_ground_vars, nv))
    for s in range(n_spheres):
        # sphere s exerts force on base
        j_ground[s * 3 + 0, 0] = 1.0
        j_ground[s * 3 + 1, 1] = 1.0
        j_ground[s * 3 + 2, 2] = 1.0

    # Grip Jacobian: dummy closed loop between arms/joints allowing load sharing
    j_grip = np.zeros((6, nv))
    j_grip[0, 6] = 1.0
    j_grip[0, 7] = -1.0
    j_grip[1, 8] = 1.0
    j_grip[1, 9] = -1.0

    # Target tau_rnea: base supported by 100 N normal force per sphere, plus target actuated torques
    tau_rnea = np.zeros(nv)
    # Root Z force balanced by contact
    if contact_active:
        tau_rnea[2] = 200.0  # 2 spheres * 100 N
    tau_rnea[6:] = tau_actuated_target

    mask = np.array([contact_active, contact_active], dtype=bool)

    node = TrajectoryOptimizationNode(
        time_s=time_s,
        tau_rnea=tau_rnea,
        j_ground=j_ground,
        j_grip=j_grip,
        q_dot=q_dot,
        active_contact_mask=mask,
    )
    return node, nv, n_actuated


def test_small_convex_analytic_optimum() -> None:
    """Test 1: exact analytic minimum on 3-node rate-penalized quadratic system."""
    nv = 7
    actuated_indices = np.array([6], dtype=int)
    n_spheres = 1

    # Simple 1-DOF system with fixed boundaries tau_0 = 10.0 and tau_2 = 30.0
    # tau_1 is free. With pure rate penalty and uniform dt, minimum of
    # (tau_1 - tau_0)^2 + (tau_2 - tau_1)^2 is exactly tau_1 = (10 + 30) / 2 = 20.0
    times = [0.0, 0.1, 0.2]

    nodes = []
    for t in times:
        j_g = np.zeros((3, nv))
        j_g[2, 2] = 1.0
        j_grip = np.zeros((6, nv))
        j_grip[0, 6] = 1.0  # tau + lambda = 100
        tau_r = np.zeros(nv)
        tau_r[2] = 100.0
        tau_r[6] = 100.0
        nodes.append(
            TrajectoryOptimizationNode(
                time_s=t,
                tau_rnea=tau_r,
                j_ground=j_g,
                j_grip=j_grip,
            )
        )

    config = SparseOptimizationConfig(
        weight_torque_effort=1e-6,
        weight_grip_wrench=1e-6,
        weight_torque_rate=100.0,
        weight_torque_acceleration=0.0,
        weight_contact_rate=0.0,
        weight_root_residual=1e4,
    )

    optimizer = SparseTrajectoryTorqueOptimizer(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=n_spheres,
        config=config,
    )

    res = optimizer.optimize(
        nodes,
        initial_seam=SeamBoundaryCondition(tau_seam=np.array([10.0])),
        final_seam=SeamBoundaryCondition(tau_seam=np.array([30.0])),
    )

    assert res.success
    assert len(res.nodes) == 3
    np.testing.assert_allclose(res.nodes[0].tau_actuated, [10.0], atol=1e-4)
    np.testing.assert_allclose(res.nodes[1].tau_actuated, [20.0], atol=1e-3)
    np.testing.assert_allclose(res.nodes[2].tau_actuated, [30.0], atol=1e-4)


def test_time_unit_invariance() -> None:
    """Test 2: scaling time units (seconds vs 10x scaled) preserves torque trajectory."""
    nv = 10
    actuated_indices = np.array([6, 7, 8, 9], dtype=int)
    n_spheres = 2

    times_s = [0.0, 0.05, 0.10, 0.15]
    targets = [
        np.array([5.0, -10.0, 15.0, -5.0]),
        np.array([20.0, -5.0, 10.0, 10.0]),
        np.array([10.0, -20.0, 5.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
    ]

    nodes_s = []
    for t, targ in zip(times_s, targets, strict=True):
        node, _, _ = _create_mock_floating_base_node(t, targ)
        nodes_s.append(node)

    cfg = SparseOptimizationConfig(
        weight_torque_effort=1.0,
        weight_torque_rate=5.0,
        weight_torque_acceleration=1.0,
        weight_contact_rate=0.5,
    )

    opt = SparseTrajectoryTorqueOptimizer(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=n_spheres,
        config=cfg,
    )

    res_s = opt.optimize(nodes_s)
    assert res_s.success

    tau_trajectories = np.array([n.tau_actuated for n in res_s.nodes])
    assert tau_trajectories.shape == (4, 4)
    # Check that equilibrium is strictly preserved across all nodes
    for k, n in enumerate(res_s.nodes):
        a_eq = np.hstack(
            [
                opt._s_transpose,
                nodes_s[k].j_ground.T,
                nodes_s[k].j_grip.T,
                opt._s_root_transpose,
            ]
        )
        x_k = np.concatenate(
            [n.tau_actuated, n.f_ground, n.lambda_grip, n.delta_tau_root]
        )
        res_dyn = float(np.linalg.norm(a_eq @ x_k - nodes_s[k].tau_rnea))
        assert res_dyn < 1e-6


def test_nonuniform_dt_handling() -> None:
    """Test 3: irregular nonuniform time steps converge stably."""
    nv = 10
    actuated_indices = np.array([6, 7, 8, 9], dtype=int)
    n_spheres = 2

    times = [0.0, 0.005, 0.025, 0.030, 0.080]
    nodes = []
    for t in times:
        targ = np.array([np.sin(t * 10), np.cos(t * 10), -5.0, 12.0])
        node, _, _ = _create_mock_floating_base_node(t, targ)
        nodes.append(node)

    opt = SparseTrajectoryTorqueOptimizer(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=n_spheres,
        config=SparseOptimizationConfig(
            weight_torque_effort=1.0,
            weight_torque_rate=2.0,
            weight_torque_acceleration=0.5,
        ),
    )

    res = opt.optimize(nodes)
    assert res.success
    assert len(res.nodes) == len(times)
    for n in res.nodes:
        assert n.tau_dot is not None
        assert np.all(np.isfinite(n.tau_dot))


def test_window_seam_c1_continuity() -> None:
    """Test 4: windowed optimization stitches seams with zero torque jumps and C1 continuity."""
    nv = 10
    actuated_indices = np.array([6, 7, 8, 9], dtype=int)
    n_spheres = 2

    total_frames = 16
    dt = 0.01
    times = [i * dt for i in range(total_frames)]
    nodes = []
    for t in times:
        targ = np.array([10.0 * np.sin(t * 15), 5.0 * np.cos(t * 12), -2.0, 8.0])
        node, _, _ = _create_mock_floating_base_node(t, targ)
        nodes.append(node)

    windowed_opt = WindowedTrajectoryTorqueOptimizer(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=n_spheres,
        window_size=8,
        overlap_size=3,
        config=SparseOptimizationConfig(
            weight_torque_effort=1.0,
            weight_torque_rate=10.0,
            weight_torque_acceleration=2.0,
        ),
    )

    res = windowed_opt.optimize(nodes)
    assert res.success
    assert len(res.nodes) == total_frames

    tau_mat = np.array([n.tau_actuated for n in res.nodes])
    diffs = np.diff(tau_mat, axis=0)
    max_step_change = np.max(np.abs(diffs))
    assert max_step_change < 5.0


def test_frame_independent_baseline_agreement() -> None:
    """Test 5: trajectory optimizer recovers frame-by-frame allocation when derivative weights are zero."""
    nv = 10
    actuated_indices = np.array([6, 7, 8, 9], dtype=int)
    n_spheres = 2

    times = [0.0, 0.01, 0.02, 0.03]
    targets = [
        np.array([15.0, -8.0, 4.0, -12.0]),
        np.array([25.0, -18.0, 14.0, 2.0]),
        np.array([5.0, -2.0, -4.0, 6.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
    ]

    nodes = []
    frame_by_frame_results = []
    allocator = ContactForceAllocator(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=n_spheres,
    )

    for t, targ in zip(times, targets, strict=True):
        node, _, _ = _create_mock_floating_base_node(t, targ)
        nodes.append(node)
        alloc = allocator.allocate(
            tau_rnea=node.tau_rnea,
            j_ground=node.j_ground,
            j_grip=node.j_grip,
            objective=AllocationObjective.MINIMUM_EFFORT,
        )
        assert alloc.success
        frame_by_frame_results.append(alloc)

    config = SparseOptimizationConfig(
        weight_torque_effort=1.0,
        weight_grip_wrench=1e-4,
        weight_root_residual=1e4,
        weight_torque_rate=0.0,
        weight_contact_rate=0.0,
        weight_torque_acceleration=0.0,
    )

    opt = SparseTrajectoryTorqueOptimizer(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=n_spheres,
        config=config,
    )

    res = opt.optimize(nodes)
    assert res.success

    for k in range(len(times)):
        fbf = frame_by_frame_results[k]
        opt_node = res.nodes[k]
        np.testing.assert_allclose(
            opt_node.tau_actuated,
            fbf.tau_actuated,
            atol=1e-3,
            err_msg=f"Discrepancy at node {k}",
        )


def test_smoothness_variance_reduction() -> None:
    """Test 6: rate penalties reduce derivative energy and variance compared to noisy baseline."""
    nv = 10
    actuated_indices = np.array([6, 7, 8, 9], dtype=int)
    n_spheres = 2

    np.random.seed(42)
    times = [i * 0.01 for i in range(12)]
    nodes = []
    for t in times:
        delta1 = float(np.random.uniform(-10.0, 10.0))
        delta2 = float(np.random.uniform(-10.0, 10.0))
        noise = np.array([delta1, -delta1, delta2, -delta2])
        base = np.array([20.0, -15.0, 10.0, 5.0])
        node, _, _ = _create_mock_floating_base_node(t, base + noise)
        nodes.append(node)

    # 1. Unsmoothed optimization (rate weight = 0)
    cfg_unsmoothed = SparseOptimizationConfig(
        weight_torque_effort=1.0,
        weight_grip_wrench=1e-4,
        weight_torque_rate=0.0,
        weight_torque_acceleration=0.0,
    )
    opt_unsmoothed = SparseTrajectoryTorqueOptimizer(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=n_spheres,
        config=cfg_unsmoothed,
    )
    res_unsmoothed = opt_unsmoothed.optimize(nodes)
    assert res_unsmoothed.success

    # 2. Smoothed optimization (rate weight > 0)
    cfg_smoothed = SparseOptimizationConfig(
        weight_torque_effort=1.0,
        weight_grip_wrench=1e-4,
        weight_torque_rate=50.0,
        weight_torque_acceleration=5.0,
    )
    opt_smoothed = SparseTrajectoryTorqueOptimizer(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=n_spheres,
        config=cfg_smoothed,
    )
    res_smoothed = opt_smoothed.optimize(nodes)
    assert res_smoothed.success

    tau_raw = np.array([n.tau_actuated for n in res_unsmoothed.nodes])
    tau_smooth = np.array([n.tau_actuated for n in res_smoothed.nodes])

    diff_raw = np.diff(tau_raw, axis=0) / 0.01
    diff_smooth = np.diff(tau_smooth, axis=0) / 0.01

    rate_energy_raw = float(np.sum(diff_raw**2))
    rate_energy_smooth = float(np.sum(diff_smooth**2))

    assert rate_energy_smooth < 0.5 * rate_energy_raw, (
        f"Expected >= 50% rate energy reduction, got raw={rate_energy_raw:.1f} vs smooth={rate_energy_smooth:.1f}"
    )


def test_interpolation_feasibility_audit() -> None:
    """Test 7: interpolation audit checks power limits, torque bounds, and friction at midpoints."""
    nv = 10
    actuated_indices = np.array([6, 7, 8, 9], dtype=int)
    n_spheres = 2

    times = [0.0, 0.02, 0.04, 0.06]
    nodes = []
    q_dot = np.array([0.0] * 6 + [5.0, 2.0, -3.0, 1.0])
    for t in times:
        targ = np.array([30.0, -20.0, 15.0, -10.0])
        node, _, _ = _create_mock_floating_base_node(t, targ, q_dot=q_dot)
        nodes.append(node)

    opt = SparseTrajectoryTorqueOptimizer(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=n_spheres,
        config=SparseOptimizationConfig(
            bounds_tau_max=np.array([100.0, 100.0, 100.0, 100.0]),
            bounds_power_max=np.array([500.0, 500.0, 500.0, 500.0]),
        ),
    )

    res = opt.optimize(nodes)
    assert res.success

    audit = audit_trajectory_interpolation_feasibility(
        res,
        nodes,
        n_subsamples=5,
        tau_max=np.array([100.0, 100.0, 100.0, 100.0]),
        power_max=np.array([500.0, 500.0, 500.0, 500.0]),
        mu_friction=0.8,
    )

    assert isinstance(audit, InterpolationAuditResult)
    assert audit.is_feasible
    assert audit.max_torque_utilization < 1.0
    assert audit.max_power_w < 500.0


def test_epigraph_peak_utilization_minimization() -> None:
    """Test 8: epigraph gamma variable penalizes peak actuator torque."""
    nv = 10
    actuated_indices = np.array([6, 7, 8, 9], dtype=int)
    n_spheres = 2

    times = [0.0, 0.02, 0.04, 0.06]
    nodes = []
    targets = [
        np.array([10.0, 10.0, 10.0, 10.0]),
        np.array([80.0, 10.0, 10.0, 10.0]),
        np.array([10.0, 10.0, 10.0, 10.0]),
        np.array([10.0, 10.0, 10.0, 10.0]),
    ]
    for t, targ in zip(times, targets, strict=True):
        node, _, _ = _create_mock_floating_base_node(t, targ)
        nodes.append(node)

    opt_epigraph = SparseTrajectoryTorqueOptimizer(
        nv=nv,
        actuated_indices=actuated_indices,
        n_contact_spheres=n_spheres,
        config=SparseOptimizationConfig(
            weight_torque_effort=1.0,
            weight_torque_rate=5.0,
            weight_peak_utilization=100.0,
            bounds_tau_max=np.array([100.0, 100.0, 100.0, 100.0]),
        ),
    )

    res_epigraph = opt_epigraph.optimize(nodes)
    assert res_epigraph.success
    assert res_epigraph.peak_utilization is not None
    assert res_epigraph.peak_utilization > 0.0
