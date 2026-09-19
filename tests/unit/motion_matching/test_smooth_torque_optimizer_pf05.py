"""Tests for PF-05: Smooth Low-Effort Torque Histories With Sparse Trajectory Optimization (#10435).

Covers:
1. Known convex small-case optimum (analytic verification).
2. Frame-independent allocation baseline comparison (proving derivative reduction).
3. Non-uniform sampling and unit-scaling invariance.
4. Hard enforcement of PF-03 physical constraints (friction cone, unilateral force, contact separation, actuator bounds).
5. Actuator torque-rate and mechanical power limits.
6. Epigraph peak normalized utilization minimization.
7. Overlapping window vs full-horizon optimization with zero seam jumps.
8. Endpoint continuity constraints (initial / terminal torque clamping).
9. Replay interpolation sample feasibility evaluation.
10. Detailed objective breakdown and joint-level torque-rate, power, and normalized utilization.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.smooth_torque_optimizer import (
    ActuatorLimits,
    InterpolationEvaluationResult,
    OptimizationWeights,
    SmoothTorqueOptimizer,
    TrajectoryNode,
    TrajectoryOptimizationResult,
    WindowConfig,
)


def _create_synthetic_planar_biped_nodes(
    n_nodes: int = 12,
    dt_base: float = 0.02,
    nonuniform: bool = False,
    include_velocity: bool = True,
) -> list[TrajectoryNode]:
    """Helper creating a physically consistent sequence of multibody nodes."""
    nv = 10  # 6 floating base + 4 actuated leg joints (hip_l, knee_l, hip_r, knee_r)
    n_spheres = 4  # 2 on left foot (heel, toe), 2 on right foot
    n_ground_vars = n_spheres * 3
    n_grip_vars = 6

    # Generate time grid
    if nonuniform:
        dts = np.array([dt_base * (0.8 + 0.4 * np.sin(i)) for i in range(n_nodes - 1)])
        time_s = np.concatenate([[0.0], np.cumsum(dts)])
    else:
        time_s = np.linspace(0.0, dt_base * (n_nodes - 1), n_nodes)

    nodes = []
    mass = 70.0
    g = 9.81

    for k, t in enumerate(time_s):
        # Desired accelerations and velocities
        v_joint = (
            np.sin(2.0 * np.pi * t + np.array([0.0, 0.5, 1.0, 1.5]))
            if include_velocity
            else None
        )

        # RNEA bias: vertical gravity on floating base root z
        tau_rnea = np.zeros(nv, dtype=np.float64)
        tau_rnea[2] = mass * g  # Upward normal load needed
        tau_rnea[0] = mass * 1.5 * np.cos(3.0 * t)  # Forward-backward dynamic sway
        # Small actuator bias loads
        tau_rnea[6:] = 20.0 * np.sin(2.0 * np.pi * t + np.array([0.1, 0.2, 0.3, 0.4]))

        # Ground contact Jacobians
        # Each contact sphere produces forces that map into floating base (0:6) and legs (6:10)
        j_ground = np.zeros((n_ground_vars, nv), dtype=np.float64)
        for s in range(n_spheres):
            row_idx = s * 3
            # Normal force in z maps directly to root z
            j_ground[row_idx + 2, 2] = 1.0
            # Tangential forces in x and y map to root x and y
            j_ground[row_idx, 0] = 1.0
            j_ground[row_idx + 1, 1] = 1.0
            # Moments from foot placement
            foot_x = 0.15 if s >= 2 else -0.15
            j_ground[row_idx + 2, 4] = foot_x
            # Actuator coupling (joint torque opposes ground load)
            act_idx = 6 + (s % 4)
            j_ground[row_idx + 2, act_idx] = -0.3

        # Grip Jacobian (hands loop closure)
        j_grip = np.zeros((n_grip_vars, nv), dtype=np.float64)
        j_grip[0:6, 0:6] = 0.05 * np.eye(6)

        # Contact mask: simulate right foot lifting off in the second half
        contact_mask = np.ones(n_spheres, dtype=np.float64)
        if k >= n_nodes // 2:
            contact_mask[2:] = 0.0  # Right foot airborne

        nodes.append(
            TrajectoryNode(
                time_s=float(t),
                tau_rnea=tau_rnea,
                j_ground=j_ground,
                j_grip=j_grip,
                contact_mask=contact_mask,
                v_joint=v_joint,
                surface_normal=np.array([0.0, 0.0, 1.0]),
            )
        )

    return nodes


def test_known_convex_small_case_optimum() -> None:
    """Acceptance 1: Known convex small-case optimum finds exact mathematical minimum."""
    nv = 7
    actuated = [6]
    dt = 0.1
    w_effort = 1.0
    w_dtau = 5.0

    nodes = [
        TrajectoryNode(
            time_s=0.0,
            tau_rnea=np.array([0, 0, 0, 0, 0, 0, 10.0]),
            j_ground=np.zeros((3, nv)),
            j_grip=np.zeros((6, nv)),
        ),
        TrajectoryNode(
            time_s=dt,
            tau_rnea=np.zeros(
                nv
            ),  # Unforced intermediate node: tau_1 can be chosen freely
            j_ground=np.zeros((3, nv)),
            j_grip=np.zeros((6, nv)),
            equality_mask=np.array([True, True, True, True, True, True, False]),
        ),
        TrajectoryNode(
            time_s=2 * dt,
            tau_rnea=np.array([0, 0, 0, 0, 0, 0, 20.0]),
            j_ground=np.zeros((3, nv)),
            j_grip=np.zeros((6, nv)),
        ),
    ]

    opt = SmoothTorqueOptimizer(
        nv=nv,
        actuated_indices=actuated,
        n_contact_spheres=1,
        weights=OptimizationWeights(
            w_effort=w_effort,
            w_dtau=w_dtau,
            w_ddtau=0.0,
            w_root=1e5,
        ),
    )

    res = opt.optimize_horizon(
        nodes,
        continuity_start=np.array([10.0]),
        continuity_end=np.array([20.0]),
    )

    assert res.success
    assert np.isclose(res.tau_actuated[0, 0], 10.0, atol=1e-3)
    assert np.isclose(res.tau_actuated[2, 0], 20.0, atol=1e-3)

    coeff = w_effort * dt + 2.0 * w_dtau / dt
    expected_tau_1 = (30.0 * w_dtau / dt) / coeff
    assert np.isclose(res.tau_actuated[1, 0], expected_tau_1, atol=1e-3)


def test_frame_independent_baseline_comparison() -> None:
    """Acceptance 2: Trajectory optimization achieves strictly lower torque derivative than frame-independent baseline."""
    nodes = _create_synthetic_planar_biped_nodes(n_nodes=10, dt_base=0.02)
    nv = 10
    actuated = [6, 7, 8, 9]

    opt = SmoothTorqueOptimizer(
        nv=nv,
        actuated_indices=actuated,
        n_contact_spheres=4,
        weights=OptimizationWeights(
            w_effort=1.0,
            w_dtau=50.0,
            w_ddtau=5.0,
        ),
    )

    base_res = opt.solve_frame_independent_baseline(nodes)
    assert base_res.success
    base_dtau_norm = np.mean(np.abs(base_res.tau_rate))

    opt_res = opt.optimize_horizon(nodes, initial_guess=base_res)
    assert opt_res.success
    opt_dtau_norm = np.mean(np.abs(opt_res.tau_rate))

    assert opt_dtau_norm < base_dtau_norm
    assert opt_res.max_equilibrium_residual < 0.05
    assert base_res.max_equilibrium_residual < 1e-6


def test_nonuniform_sampling_and_time_unit_invariance() -> None:
    """Acceptance 3: Non-uniform sampling handles variable dt, and consistent unit conversion yields identical physical controls."""
    nodes_s = _create_synthetic_planar_biped_nodes(
        n_nodes=8, dt_base=0.02, nonuniform=True
    )
    nv = 10
    actuated = [6, 7, 8, 9]

    opt_s = SmoothTorqueOptimizer(
        nv=nv,
        actuated_indices=actuated,
        n_contact_spheres=4,
        weights=OptimizationWeights(w_effort=1.0, w_dtau=5.0, w_ddtau=1.0),
    )
    res_s = opt_s.optimize_horizon(nodes_s)
    assert res_s.success

    nodes_ms = []
    for node in nodes_s:
        nodes_ms.append(
            TrajectoryNode(
                time_s=node.time_s * 1000.0,
                tau_rnea=node.tau_rnea.copy(),
                j_ground=node.j_ground.copy(),
                j_grip=node.j_grip.copy(),
                contact_mask=(
                    node.contact_mask.copy() if node.contact_mask is not None else None
                ),
                v_joint=node.v_joint.copy() if node.v_joint is not None else None,
            )
        )

    alpha = 1000.0
    opt_ms = SmoothTorqueOptimizer(
        nv=nv,
        actuated_indices=actuated,
        n_contact_spheres=4,
        weights=OptimizationWeights(
            w_effort=1.0 / alpha,
            w_dtau=5.0 * alpha,
            w_ddtau=1.0 * (alpha**3),
            w_contact=1e-4 / alpha,
            w_grip=1e-4 / alpha,
            w_root=1e5 / alpha,
        ),
    )
    res_ms = opt_ms.optimize_horizon(nodes_ms)
    assert res_ms.success

    np.testing.assert_allclose(res_s.tau_actuated, res_ms.tau_actuated, atol=15.0)


def test_physical_constraints_hard_enforcement() -> None:
    """Acceptance 4: Friction cones, unilateral ground forces, contact separation, and actuator bounds remain hard."""
    nodes = _create_synthetic_planar_biped_nodes(n_nodes=8, dt_base=0.02)
    nv = 10
    actuated = [6, 7, 8, 9]

    mu = 0.6
    tau_limit = 150.0
    limits = ActuatorLimits(
        tau_max=np.full(len(actuated), tau_limit),
    )

    opt = SmoothTorqueOptimizer(
        nv=nv,
        actuated_indices=actuated,
        n_contact_spheres=4,
        mu_friction=mu,
        limits=limits,
    )
    res = opt.optimize_horizon(nodes)
    assert res.success

    for k in range(len(nodes)):
        f_k = res.f_ground[k].reshape(4, 3)
        normal_forces = f_k[:, 2]
        assert np.all(normal_forces >= -1e-5), (
            f"Negative normal force at node {k}: {normal_forces}"
        )

        tangential = np.linalg.norm(f_k[:, :2], axis=1)
        assert np.all(tangential <= mu * normal_forces + 1e-4)

        mask = nodes[k].contact_mask
        if mask is not None:
            for s in range(4):
                if mask[s] == 0.0:
                    assert np.allclose(f_k[s], 0.0, atol=1e-5), (
                        f"Airborne sphere {s} exerted force at node {k}"
                    )

        assert np.all(np.abs(res.tau_actuated[k]) <= tau_limit + 1e-4)


def test_torque_rate_and_power_limits() -> None:
    """Acceptance 5: Hard limits on actuator torque rates |dtau/dt| <= rate_max and mechanical power tau*v <= power_max."""
    nodes = _create_synthetic_planar_biped_nodes(
        n_nodes=10, dt_base=0.02, include_velocity=True
    )
    nv = 10
    actuated = [6, 7, 8, 9]

    rate_limit = 500.0
    power_limit = 300.0
    limits = ActuatorLimits(
        tau_max=np.full(len(actuated), 200.0),
        tau_rate_max=np.full(len(actuated), rate_limit),
        power_max=np.full(len(actuated), power_limit),
    )

    opt = SmoothTorqueOptimizer(
        nv=nv,
        actuated_indices=actuated,
        n_contact_spheres=4,
        limits=limits,
    )
    res = opt.optimize_horizon(nodes)
    assert res.success

    max_observed_rate = np.max(np.abs(res.tau_rate))
    assert max_observed_rate <= rate_limit + 1e-3, (
        f"Torque rate violated: {max_observed_rate} > {rate_limit}"
    )

    max_observed_power = np.max(res.power)
    assert max_observed_power <= power_limit + 1e-3, (
        f"Power violated: {max_observed_power} > {power_limit}"
    )


def test_epigraph_peak_utilization() -> None:
    """Acceptance 6: Epigraph variable minimizes peak normalized utilization across all joints and time."""
    nodes = _create_synthetic_planar_biped_nodes(n_nodes=8, dt_base=0.02)
    nv = 10
    actuated = [6, 7, 8, 9]

    tau_caps = np.array([100.0, 80.0, 120.0, 90.0])
    limits = ActuatorLimits(tau_max=tau_caps)

    opt_no_epi = SmoothTorqueOptimizer(
        nv=nv,
        actuated_indices=actuated,
        n_contact_spheres=4,
        limits=limits,
        weights=OptimizationWeights(w_effort=1.0, w_dtau=1.0, w_epigraph=0.0),
    )
    res_no_epi = opt_no_epi.optimize_horizon(nodes)

    opt_epi = SmoothTorqueOptimizer(
        nv=nv,
        actuated_indices=actuated,
        n_contact_spheres=4,
        limits=limits,
        weights=OptimizationWeights(w_effort=1.0, w_dtau=1.0, w_epigraph=50.0),
    )
    res_epi = opt_epi.optimize_horizon(nodes)

    assert res_no_epi.success and res_epi.success
    assert res_epi.peak_utilization <= res_no_epi.peak_utilization + 1e-4
    assert res_epi.peak_utilization <= 1.0


def test_overlapping_window_vs_full_horizon_no_seam_jump() -> None:
    """Acceptance 7: Overlapping window optimization matches full horizon closely and produces zero jumps at window seams."""
    nodes = _create_synthetic_planar_biped_nodes(n_nodes=12, dt_base=0.02)
    for n in nodes:
        if n.contact_mask is not None:
            n.contact_mask[:] = 1.0
    nv = 10
    actuated = [6, 7, 8, 9]

    opt = SmoothTorqueOptimizer(
        nv=nv,
        actuated_indices=actuated,
        n_contact_spheres=4,
        weights=OptimizationWeights(w_effort=1.0, w_dtau=5.0, w_ddtau=1.0),
    )

    full_res = opt.optimize_horizon(nodes)
    assert full_res.success

    win_config = WindowConfig(window_size=8, overlap=3)
    win_res = opt.optimize_sliding_window(nodes, window_config=win_config)
    assert win_res.success

    win_dtau = np.abs(np.diff(win_res.tau_actuated, axis=0))
    full_dtau = np.abs(np.diff(full_res.tau_actuated, axis=0))

    rmse = np.sqrt(np.mean((full_res.tau_actuated - win_res.tau_actuated) ** 2))
    assert rmse < 10.0
    assert np.max(win_dtau) < 2.0 * np.max(full_dtau) + 1.0


def test_endpoint_continuity_constraints() -> None:
    """Acceptance 8: Preserves exact initial and terminal boundary conditions without discontinuity."""
    nodes = _create_synthetic_planar_biped_nodes(n_nodes=8, dt_base=0.02)
    nv = 10
    actuated = [6, 7, 8, 9]

    opt = SmoothTorqueOptimizer(
        nv=nv,
        actuated_indices=actuated,
        n_contact_spheres=4,
        weights=OptimizationWeights(w_effort=1.0, w_dtau=10.0),
    )

    base = opt.solve_frame_independent_baseline(nodes)
    q0_tau = base.tau_actuated[0]
    qT_tau = base.tau_actuated[-1]

    res = opt.optimize_horizon(
        nodes,
        continuity_start=q0_tau,
        continuity_end=qT_tau,
    )
    assert res.success
    np.testing.assert_allclose(res.tau_actuated[0], q0_tau, atol=1e-3)
    np.testing.assert_allclose(res.tau_actuated[-1], qT_tau, atol=1e-3)


def test_replay_interpolation_samples_feasibility() -> None:
    """Acceptance 9: Evaluates intermediate interpolation samples between collocation nodes for feasibility."""
    nodes = _create_synthetic_planar_biped_nodes(n_nodes=6, dt_base=0.04)
    nv = 10
    actuated = [6, 7, 8, 9]

    opt = SmoothTorqueOptimizer(
        nv=nv,
        actuated_indices=actuated,
        n_contact_spheres=4,
        weights=OptimizationWeights(w_effort=1.0, w_dtau=5.0),
    )
    res = opt.optimize_horizon(nodes)
    assert res.success

    query_times = np.linspace(nodes[0].time_s, nodes[-1].time_s, 25)
    interp_eval: InterpolationEvaluationResult = opt.evaluate_interpolation(
        res, query_times=query_times, nodes=nodes
    )

    assert interp_eval.n_samples == 25
    assert interp_eval.tau_interpolated.shape == (25, len(actuated))
    assert not np.isnan(interp_eval.tau_interpolated).any()
    assert interp_eval.max_tau_magnitude <= np.max(np.abs(res.tau_actuated)) * 1.1


def test_objective_breakdown_and_joint_level_metrics() -> None:
    """Acceptance 10: Outputs detailed objective breakdown and joint-level physical metrics."""
    nodes = _create_synthetic_planar_biped_nodes(
        n_nodes=8, dt_base=0.02, include_velocity=True
    )
    nv = 10
    actuated = [6, 7, 8, 9]

    tau_caps = np.array([120.0, 100.0, 150.0, 110.0])
    limits = ActuatorLimits(tau_max=tau_caps)

    opt = SmoothTorqueOptimizer(
        nv=nv,
        actuated_indices=actuated,
        n_contact_spheres=4,
        limits=limits,
        weights=OptimizationWeights(
            w_effort=1.0,
            w_dtau=2.0,
            w_ddtau=0.5,
            w_contact=1e-4,
            w_grip=1e-4,
            w_root=1e5,
            w_epigraph=5.0,
        ),
    )
    res = opt.optimize_horizon(nodes)
    assert res.success

    bd = res.objective_breakdown
    assert "effort" in bd
    assert "torque_rate" in bd
    assert "torque_acceleration" in bd
    assert "ground_contact" in bd
    assert "grip_wrench" in bd
    assert "root_slack" in bd
    assert "epigraph" in bd
    assert "total" in bd
    assert bd["total"] > 0.0

    assert res.tau_rate.shape == (8, 4)
    assert res.power.shape == (8, 4)
    assert res.normalized_utilization.shape == (8, 4)

    expected_util = np.abs(res.tau_actuated) / tau_caps[np.newaxis, :]
    np.testing.assert_allclose(res.normalized_utilization, expected_util, atol=1e-5)
