"""Unit tests for native counterfactual qualification (epic #10286 / CF-1, CF-2, CF-3).

Tests versioned spatial wrench contracts, moment transport, pointwise ZTCF/ZVCF
evaluations under constraints, conservation, and 3D power/work accounting.
"""

from __future__ import annotations

from collections.abc import Mapping
import numpy as np
import pytest

from src.shared.python.motion_matching.counterfactual import (
    CounterfactualTrajectory,
    NativeConstrainedCounterfactualProvider,
    PointwiseCounterfactualSample,
    SpatialWrench,
    solve_constrained_dynamics,
)

pytestmark = pytest.mark.unit


# ============================================================================
# CF-1: Spatial Wrench Contracts & Moment Transport
# ============================================================================


def test_spatial_wrench_creation_and_properties() -> None:
    force = np.array([10.0, -5.0, 20.0])
    torque = np.array([1.0, 2.0, -3.0])
    point = np.array([0.1, 0.2, 0.3])

    wrench = SpatialWrench(
        force_N=force,
        torque_Nm=torque,
        point_of_application_m=point,
        frame="world",
        action_direction="proximal_on_distal",
    )

    assert np.allclose(wrench.force_N, force)
    assert np.allclose(wrench.torque_Nm, torque)
    assert np.allclose(wrench.point_of_application_m, point)
    assert wrench.frame == "world"
    assert wrench.action_direction == "proximal_on_distal"
    assert wrench.units == "SI"

    # 6D vector layout: [Fx, Fy, Fz, Mx, My, Mz]
    vec = wrench.vector
    assert vec.shape == (6,)
    assert np.allclose(vec[:3], force)
    assert np.allclose(vec[3:], torque)


def test_spatial_wrench_moment_transport() -> None:
    """Moment transport rule: M_P = M_O + r_{P/O} x F, where r_{P/O} = O - P."""
    # Consider force at origin O = [0, 0, 0] with F = [0, 10, 0] and M_O = [0, 0, 0]
    # At P = [1, 0, 0], lever from P to O is r = O - P = [-1, 0, 0]
    # Torque about P is r x F = [-1, 0, 0] x [0, 10, 0] = [0, 0, -10]
    wrench = SpatialWrench(
        force_N=np.array([0.0, 10.0, 0.0]),
        torque_Nm=np.array([0.0, 0.0, 0.0]),
        point_of_application_m=np.array([0.0, 0.0, 0.0]),
        frame="world",
    )

    new_point = np.array([1.0, 0.0, 0.0])
    transported = wrench.change_point_of_application(new_point)

    assert np.allclose(transported.force_N, wrench.force_N)
    assert np.allclose(transported.point_of_application_m, new_point)
    expected_torque = np.array([0.0, 0.0, -10.0])
    assert np.allclose(transported.torque_Nm, expected_torque)

    # Invariance check: transporting back to origin recovers original torque
    reverted = transported.change_point_of_application(np.array([0.0, 0.0, 0.0]))
    assert np.allclose(reverted.torque_Nm, wrench.torque_Nm)


def test_spatial_wrench_action_reaction_negation() -> None:
    wrench = SpatialWrench(
        force_N=np.array([12.0, -4.0, 8.0]),
        torque_Nm=np.array([1.5, -2.5, 3.5]),
        point_of_application_m=np.array([0.2, 0.4, 0.6]),
        frame="local_weld",
        action_direction="proximal_on_distal",
    )

    reaction = wrench.negate()
    assert np.allclose(reaction.force_N, -wrench.force_N)
    assert np.allclose(reaction.torque_Nm, -wrench.torque_Nm)
    assert np.allclose(reaction.point_of_application_m, wrench.point_of_application_m)
    assert reaction.action_direction == "distal_on_proximal"


def test_spatial_wrench_rejects_invalid_inputs() -> None:
    valid = np.zeros(3)
    with pytest.raises(ValueError, match="finite"):
        SpatialWrench(
            force_N=np.array([np.nan, 0, 0]),
            torque_Nm=valid,
            point_of_application_m=valid,
        )

    with pytest.raises(ValueError, match="shape"):
        SpatialWrench(
            force_N=np.zeros(2), torque_Nm=valid, point_of_application_m=valid
        )

    with pytest.raises(ValueError, match="frame"):
        SpatialWrench(
            force_N=valid, torque_Nm=valid, point_of_application_m=valid, frame=""
        )


# ============================================================================
# CF-2: Pointwise Constrained Counterfactual Provider
# ============================================================================


class _AnalyticalConstrainedLoopModel:
    """Mock constrained dynamics model satisfying M a + h = B u + J^T lambda, J a + Jdot v = 0."""

    def __init__(self) -> None:
        self.coordinates = ["theta1", "theta2"]
        self.mass = 2.0
        self.length = 1.0
        self.g = 9.81

    def evaluate_dynamics(
        self,
        q: Mapping[str, float],
        v: Mapping[str, float],
        u: Mapping[str, float],
    ) -> tuple[dict[str, float], SpatialWrench]:
        """Solves constrained KKT system analytically for a 2-DOF closed kinematic loop."""
        q1, q2 = q["theta1"], q["theta2"]
        v1, v2 = v["theta1"], v["theta2"]
        u1, u2 = u.get("theta1", 0.0), u.get("theta2", 0.0)

        # Invertible 2x2 mass matrix M = diag(2.0, 1.0)
        M = np.diag([2.0, 1.0])
        # h(q, v) = Coriolis + gravity
        # Coriolis: C = diag(0.5 * v1, 0.2 * v2)
        # Gravity: g = [mass * g * sin(q1), mass * g * sin(q2)]
        h = np.array(
            [
                0.5 * v1**2 + self.mass * self.g * np.sin(q1),
                0.2 * v2**2 + self.g * np.sin(q2),
            ]
        )
        tau = np.array([u1, u2])

        # 1D holonomic constraint: q1 - q2 = 0 -> J = [1.0, -1.0], Jdot v = 0
        J = np.array([[1.0, -1.0]])
        # KKT system:
        # [ M   -J^T ] [ a      ] = [ tau - h ]
        # [ J     0  ] [ lambda ]   [ 0       ]
        KKT = np.array(
            [
                [M[0, 0], M[0, 1], -J[0, 0]],
                [M[1, 0], M[1, 1], -J[0, 1]],
                [J[0, 0], J[0, 1], 0.0],
            ]
        )
        rhs = np.array([tau[0] - h[0], tau[1] - h[1], 0.0])
        sol = np.linalg.solve(KKT, rhs)
        a = sol[:2]
        lam = sol[2]

        accel_dict = {"theta1": float(a[0]), "theta2": float(a[1])}
        # Reaction wrench at weld point [0, length, 0]
        wrench = SpatialWrench(
            force_N=np.array([float(lam), 0.0, 0.0]),
            torque_Nm=np.array([0.0, 0.0, float(lam * self.length)]),
            point_of_application_m=np.array([0.0, self.length, 0.0]),
            frame="world",
            action_direction="proximal_on_distal",
        )
        return accel_dict, wrench


def test_pointwise_counterfactual_sample_closure() -> None:
    model = _AnalyticalConstrainedLoopModel()
    provider = NativeConstrainedCounterfactualProvider(model)

    q = {"theta1": 0.3, "theta2": 0.3}
    v = {"theta1": 1.2, "theta2": 1.2}
    u = {"theta1": 15.0, "theta2": -5.0}

    sample: PointwiseCounterfactualSample = provider.evaluate_pointwise(
        q, v, u, time_s=0.25
    )

    # 1. State retention: inputs must not be mutated
    assert sample.time_s == 0.25
    assert sample.coordinates == q
    assert sample.rates == v
    assert sample.applied_efforts == u

    # 2. Control split closure: actual = ztcf + control_increment
    for name in ("theta1", "theta2"):
        act = sample.actual_acceleration[name]
        ztcf = sample.ztcf_acceleration[name]
        ctrl = sample.control_increment_acceleration[name]
        assert np.isclose(act, ztcf + ctrl, rtol=1e-9, atol=1e-10)

    # 3. Reaction wrench control split closure: lambda_act = lambda_ztcf + lambda_ctrl
    act_w = sample.actual_reaction_wrench.vector
    ztcf_w = sample.ztcf_reaction_wrench.vector
    ctrl_w = sample.control_increment_reaction_wrench.vector
    assert np.allclose(act_w, ztcf_w + ctrl_w, rtol=1e-9, atol=1e-10)

    # 4. ZVCF evaluation: v = 0 and u = 0
    # Under v = 0, Coriolis terms vanish, so a_zvcf != a_ztcf when v != 0
    zvcf_a1 = sample.zvcf_acceleration["theta1"]
    ztcf_a1 = sample.ztcf_acceleration["theta1"]
    assert not np.isclose(zvcf_a1, ztcf_a1)

    # 5. Immutability & non-mutation
    with pytest.raises(TypeError):
        sample.coordinates["theta1"] = 99.0  # type: ignore[index]


# ============================================================================
# CF-3: 3D Spatial Wrench Power, Work, Impulse Accounting
# ============================================================================


def test_counterfactual_trajectory_power_work_and_impulse() -> None:
    times = np.linspace(0.0, 1.0, 101)
    n_samples = len(times)
    names = ("joint1", "joint2")

    # Construct synthetic smooth trajectory
    # F = [10 * cos(t), 0, 5], v = [2 * sin(t), 0, 0]
    # M = [0, 2, 0], omega = [0, 1, 0]
    F_total = np.zeros((n_samples, 3))
    F_total[:, 0] = 10.0 * np.cos(times)
    F_total[:, 2] = 5.0
    M_total = np.zeros((n_samples, 3))
    M_total[:, 1] = 2.0

    F_drift = np.zeros((n_samples, 3))
    F_drift[:, 2] = 5.0
    M_drift = np.zeros((n_samples, 3))

    F_ctrl = F_total - F_drift
    M_ctrl = M_total - M_drift

    w_total = np.hstack([F_total, M_total])
    w_drift = np.hstack([F_drift, M_drift])
    w_ctrl = np.hstack([F_ctrl, M_ctrl])
    w_zvcf = np.zeros((n_samples, 6))

    twist = np.zeros((n_samples, 6))
    twist[:, 0] = 2.0 * np.sin(times)  # linear vx
    twist[:, 4] = 1.0  # angular wy

    a_total = np.zeros((n_samples, 2))
    a_drift = np.zeros((n_samples, 2))
    a_ctrl = np.zeros((n_samples, 2))
    a_zvcf = np.zeros((n_samples, 2))

    traj = CounterfactualTrajectory(
        time_s=times,
        coordinate_names=names,
        actual_accelerations=a_total,
        ztcf_accelerations=a_drift,
        control_accelerations=a_ctrl,
        zvcf_accelerations=a_zvcf,
        actual_wrenches=w_total,
        ztcf_wrenches=w_drift,
        control_wrenches=w_ctrl,
        zvcf_wrenches=w_zvcf,
        twist=twist,
        parent_run_id="simscape-returned102",
    )

    # 1. Power P = F . v + M . omega
    # For actual: F . v = (10 cos(t)) * (2 sin(t)) = 20 sin(t)cos(t) = 10 sin(2t)
    #             M . omega = 2.0 * 1.0 = 2.0
    #             P = 10 sin(2t) + 2.0
    power_act = traj.power("actual")
    expected_power = 10.0 * np.sin(2.0 * times) + 2.0
    assert np.allclose(power_act, expected_power, atol=1e-12)

    # For drift: F . v = 0 (orthogonal), M . omega = 0 -> P_drift = 0
    power_drift = traj.power("ztcf")
    assert np.allclose(power_drift, 0.0, atol=1e-12)

    # Control power: P_ctrl = P_act - P_drift
    power_ctrl = traj.power("control")
    assert np.allclose(power_ctrl, power_act, atol=1e-12)

    # 2. Work W(t) = integral P(tau) dtau
    work_act = traj.work("actual")
    assert work_act.shape == (n_samples,)
    assert work_act[0] == 0.0
    # Integral of 10 sin(2t) + 2 is -5 cos(2t) + 2t + 5
    expected_work = -5.0 * np.cos(2.0 * times) + 2.0 * times + 5.0
    # Numerical trapezoidal quadrature matches within 1e-4 on 101 points
    assert np.allclose(work_act, expected_work, atol=1e-3)

    # 3. Linear impulse: integral F dt
    impulse_act = traj.linear_impulse("actual")
    assert impulse_act.shape == (n_samples, 3)
    assert np.allclose(impulse_act[0], 0.0)

    # 4. Conversion to InteractionEvidenceTrajectory
    evidence = traj.to_interaction_evidence(interface_name="lead_hand_weld")
    assert evidence.sample_count == n_samples
    assert evidence.interface_count == 1
    assert np.allclose(evidence.wrench_total[:, 0, :], w_total)
    assert np.allclose(evidence.wrench_drift[:, 0, :], w_drift)
    assert np.allclose(evidence.wrench_control[:, 0, :], w_ctrl)


def test_provider_with_native_model_interface() -> None:
    """Verify that NativeConstrainedCounterfactualProvider works with models providing

    accelerations(q, v, u) and closure_reaction_wrench('world').
    """

    class MockNativePinocchio:
        def __init__(self) -> None:
            self._q = {"c0": 0.1}
            self._v = {"c0": 0.2}
            self._u = {"c0": 5.0}

        def accelerations(
            self, q: Mapping[str, float], v: Mapping[str, float], u: Mapping[str, float]
        ) -> dict[str, float]:
            val = u["c0"] - 9.81 * np.sin(q["c0"]) - 0.1 * v["c0"] ** 2
            return {"c0": val}

        def closure_reaction_wrench(
            self, frame: str = "world"
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            f = np.array([10.0, 0.0, 0.0])
            tau = np.array([0.0, 5.0, 0.0])
            point = np.array([0.0, 0.5, 0.0])
            return f, tau, point

    mock_model = MockNativePinocchio()
    provider = NativeConstrainedCounterfactualProvider(mock_model)
    sample = provider.evaluate_pointwise({"c0": 0.1}, {"c0": 0.2}, {"c0": 5.0})

    assert "c0" in sample.actual_acceleration
    assert "c0" in sample.ztcf_acceleration
    assert "c0" in sample.zvcf_acceleration
    assert np.allclose(sample.actual_reaction_wrench.force_N, [10.0, 0.0, 0.0])
    assert np.allclose(sample.actual_reaction_wrench.torque_Nm, [0.0, 5.0, 0.0])


def test_wrench_moment_origin_power_invariance() -> None:
    """Verify that power P = F . v_P + M_P . omega is strictly invariant under change of moment origin."""
    point_A = np.array([0.5, -0.2, 0.8])
    force = np.array([25.0, -15.0, 10.0])
    torque_A = np.array([2.0, -4.0, 6.0])

    wrench_A = SpatialWrench(
        force_N=force,
        torque_Nm=torque_A,
        point_of_application_m=point_A,
    )

    # Rigid body twist: angular velocity omega and linear velocity v_A at point A
    omega = np.array([1.5, -2.0, 0.8])
    v_A = np.array([3.0, 1.2, -0.5])

    # Power evaluated at point A
    power_A = wrench_A.instantaneous_power(v_A, omega)

    # Transport moment to point B
    point_B = np.array([1.2, 0.4, -0.1])
    wrench_B = wrench_A.change_point_of_application(point_B)

    # Rigid kinematic transport of linear velocity from A to B: v_B = v_A + omega x (B - A)
    v_B = v_A + np.cross(omega, point_B - point_A)

    # Power evaluated at point B
    power_B = wrench_B.instantaneous_power(v_B, omega)

    assert power_A == pytest.approx(power_B, abs=1e-12)


def test_constrained_dynamics_kkt_solver() -> None:
    """Validate constrained forward dynamics solver (Ma + h = Bu + J^T lambda).

    Validates:
    - Force balance: Ma + h - Bu - J^T lambda = 0
    - Constraint acceleration: Ja + J_dot v = 0
    - Action/reaction: equal and opposite constraint reactions
    - Power invariance of constraint forces: lambda^T (J v) = 0
    """
    rng = np.random.default_rng(12345)
    n = 6  # generalized coordinates
    m = 2  # holonomic constraints

    # Symmetric positive-definite mass matrix
    A_mat = rng.normal(0, 1, size=(n, n))
    M = A_mat.T @ A_mat + 2.0 * np.eye(n)

    h = rng.normal(0, 10, size=n)
    B = np.eye(n)
    u = rng.normal(0, 5, size=n)

    # Constraint Jacobian of full row rank
    J = rng.normal(0, 1, size=(m, n))
    gamma = rng.normal(0, 2, size=m)  # J_dot * v

    # Solve constrained dynamics
    accel, lambdas = solve_constrained_dynamics(
        mass_matrix=M,
        coriolis_gravity=h,
        actuation_matrix=B,
        applied_torques=u,
        constraint_jacobian=J,
        constraint_drift=gamma,
    )

    assert accel.shape == (n,)
    assert lambdas.shape == (m,)

    # 1. Force balance: M a + h - B u - J^T lambda = 0
    force_residual = M @ accel + h - B @ u - J.T @ lambdas
    np.testing.assert_allclose(force_residual, 0.0, atol=1e-12)

    # 2. Constraint acceleration: J a + gamma = 0
    constraint_acc_residual = J @ accel + gamma
    np.testing.assert_allclose(constraint_acc_residual, 0.0, atol=1e-12)

    # 3. Action / Reaction: Generalized constraint reaction is J^T lambda
    # For opposing bodies in weld, J_1 = -J_2 => F_1 = J_1^T lambda = - (J_2^T lambda) = -F_2
    J_opposing = -J
    f_action = J.T @ lambdas
    f_reaction = J_opposing.T @ lambdas
    np.testing.assert_allclose(f_action + f_reaction, 0.0, atol=1e-12)

    # 4. Power of constraint forces under valid velocities J v = 0 is zero
    # Construct a velocity v in nullspace of J: J v = 0
    u_null, s_null, vh_null = np.linalg.svd(J)
    null_basis = vh_null[m:].T  # n x (n-m)
    v_valid = null_basis @ rng.normal(0, 1, size=n - m)
    np.testing.assert_allclose(J @ v_valid, 0.0, atol=1e-12)

    # Constraint power: (J^T lambda)^T v = lambda^T (J v) = 0
    p_constraint = float(f_action @ v_valid)
    assert p_constraint == pytest.approx(0.0, abs=1e-12)


def test_saved_bundle_rejects_baseline_acceleration_relabeling() -> None:
    """Verify that from_saved_simscape_bundle explicitly rejects baseline-acceleration relabeling when torques are unavailable."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    manifest = (
        repo_root
        / "docs/development/simscape_tour_matching/native_evidence/simscape_returned102.replay.json"
    )
    if not manifest.exists():
        pytest.skip("run102 manifest not found")

    # When torques are unavailable in bundle, must raise ValueError
    with pytest.raises(
        ValueError, match="applied actuator torques 'tau' are non-finite or unavailable"
    ):
        CounterfactualTrajectory.from_saved_simscape_bundle(manifest)

    # With explicit allow_unverified_relabeling flag, it allows inspection
    traj = CounterfactualTrajectory.from_saved_simscape_bundle(
        manifest, allow_unverified_relabeling=True
    )
    assert traj.parent_run_id == "simscape-returned102"
    assert traj.model_tier == "saved_simscape_bundle"
    assert traj.time_s.size == 307
