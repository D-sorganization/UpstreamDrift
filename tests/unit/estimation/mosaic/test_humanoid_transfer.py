"""Tests for MOSAIC humanoid transfer, payload identification, and WBC export (MOSAIC-12)."""

from __future__ import annotations

import json
import numpy as np
import pytest

from src.shared.python.core.contracts import ContractViolationError
from src.shared.python.estimation.mosaic.humanoid_transfer import (
    NON_CLAIMS,
    PayloadIdentificationResult,
    PayloadObservabilityReport,
    WBCSchedule,
    export_wbc_schedule,
    floating_base_actuation_mask,
    identify_payload_inertia,
    spatial_payload_regressor,
    validate_wbc_schedule,
)
from src.shared.python.estimation.mosaic.inertial import (
    SPATIAL_PARAMETERS_PER_BODY,
    is_spatial_consistent,
)
from src.shared.python.estimation.mosaic.local_policy import (
    linearize_along_trajectory,
    tvlqr_gains,
)
from src.shared.python.estimation.mosaic.planar_chain import PlanarChain

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _synthetic_spatial_trajectory(
    n_samples: int = 50, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate rich 3-D rigid-body spatial trajectory (lin_acc, ang_vel, ang_acc)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0, n_samples)
    # Linear acceleration (N, 3)
    lin_acc = np.stack(
        [
            1.5 * np.sin(2.0 * np.pi * 0.8 * t),
            0.8 * np.cos(2.0 * np.pi * 1.2 * t),
            2.0 * np.sin(2.0 * np.pi * 0.5 * t) + 1.0,
        ],
        axis=-1,
    )
    # Angular velocity (N, 3)
    ang_vel = np.stack(
        [
            1.2 * np.cos(2.0 * np.pi * 0.6 * t),
            1.0 * np.sin(2.0 * np.pi * 0.9 * t),
            0.7 * np.cos(2.0 * np.pi * 1.4 * t),
        ],
        axis=-1,
    )
    # Angular acceleration (N, 3)
    ang_acc = np.stack(
        [
            -1.2 * 2.0 * np.pi * 0.6 * np.sin(2.0 * np.pi * 0.6 * t),
            1.0 * 2.0 * np.pi * 0.9 * np.cos(2.0 * np.pi * 0.9 * t),
            -0.7 * 2.0 * np.pi * 1.4 * np.sin(2.0 * np.pi * 1.4 * t),
        ],
        axis=-1,
    )
    return lin_acc, ang_vel, ang_acc


def _synthetic_payload_inertia() -> np.ndarray:
    """Physically consistent 10-parameter spatial inertia: [m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]."""
    mass = 2.5
    com = np.array([0.05, -0.02, 0.08])
    h = mass * com
    # Positive-definite central inertia
    i_bar = np.array([0.04, 0.002, -0.001, 0.035, 0.001, 0.05])
    # Parallel axis theorem: I_origin = I_bar - m [com]_x^2
    com_skew = np.array(
        [
            [0.0, -com[2], com[1]],
            [com[2], 0.0, -com[0]],
            [-com[1], com[0], 0.0],
        ]
    )
    pat = -mass * (com_skew @ com_skew)
    i_orig = np.array(
        [
            i_bar[0] + pat[0, 0],
            i_bar[1] + pat[0, 1],
            i_bar[2] + pat[0, 2],
            i_bar[3] + pat[1, 1],
            i_bar[4] + pat[1, 2],
            i_bar[5] + pat[2, 2],
        ]
    )
    pi = np.array([mass, *h, *i_orig], dtype=np.float64)
    assert is_spatial_consistent(pi[None])[0]
    return pi


# ---------------------------------------------------------------------------
# Payload Identification Tests
# ---------------------------------------------------------------------------


def test_identify_payload_inertia_recovers_synthetic_object() -> None:
    """Recover 10 spatial inertial parameters of unknown payload to < 10^{-6} error."""
    lin_acc, ang_vel, ang_acc = _synthetic_spatial_trajectory(n_samples=60)
    pi_true = _synthetic_payload_inertia()

    # Generate synthetic wrench: (N, 6)
    regressor = spatial_payload_regressor(lin_acc, ang_vel, ang_acc)
    assert regressor.shape == (60, 6, 10)
    wrench = np.einsum("tij,j->ti", regressor, pi_true)

    result: PayloadIdentificationResult = identify_payload_inertia(
        wrench=wrench,
        linear_acceleration=lin_acc,
        angular_velocity=ang_vel,
        angular_acceleration=ang_acc,
    )

    # Condition report
    report: PayloadObservabilityReport = result.report
    assert report.is_fully_observable
    assert report.observable_dimension == 10
    assert report.observable_fraction == 1.0
    assert report.condition_number > 1.0
    assert report.condition_number < 1e5

    # Measured recovery error < 10^{-6}
    error = np.linalg.norm(result.parameters - pi_true)
    assert error < 1e-6
    assert result.residual_rms < 1e-6


def test_identify_payload_inertia_with_known_tool_anchor() -> None:
    """Identify payload when transmitted wrench passes through a known tool anchor."""
    lin_acc, ang_vel, ang_acc = _synthetic_spatial_trajectory(n_samples=80)
    pi_payload = _synthetic_payload_inertia()
    # Known gripper tool inertia
    pi_tool = np.array(
        [1.2, 0.0, 0.0, 0.05, 0.02, 0.0, 0.0, 0.02, 0.0, 0.015],
        dtype=np.float64,
    )

    reg = spatial_payload_regressor(lin_acc, ang_vel, ang_acc)
    wrench_payload = np.einsum("tij,j->ti", reg, pi_payload)
    wrench_tool = np.einsum("tij,j->ti", reg, pi_tool)
    total_wrench = wrench_payload + wrench_tool

    result = identify_payload_inertia(
        wrench=total_wrench,
        linear_acceleration=lin_acc,
        angular_velocity=ang_vel,
        angular_acceleration=ang_acc,
        known_tool_inertia=pi_tool,
    )

    np.testing.assert_allclose(result.parameters, pi_payload, atol=1e-6)
    assert result.report.is_fully_observable


# ---------------------------------------------------------------------------
# Whole-Body Controller (WBC) Schedule Export & Validation
# ---------------------------------------------------------------------------


def test_wbc_schedule_export_round_trip() -> None:
    """Export and round-trip serialize structured WBC schedule to JSON artifact."""
    t_nodes = 20
    n_q = 3
    n_u = 2
    times = np.linspace(0.0, 1.0, t_nodes)
    q_ref = np.zeros((t_nodes, n_q))
    v_ref = np.zeros((t_nodes, n_q))
    a_ref = np.zeros((t_nodes, n_q))
    u_ff = np.zeros((t_nodes, n_u))
    gains_k = np.zeros((t_nodes, n_u, 2 * n_q))
    pi = np.ones(12)
    geometry = np.array([0.9, 0.6, 0.4])

    schedule = export_wbc_schedule(
        times=times,
        q_ref=q_ref,
        v_ref=v_ref,
        a_ref=a_ref,
        u_feedforward=u_ff,
        gains_k=gains_k,
        inertial_parameters=pi,
        geometry_parameters=geometry,
        model_name="planar_humanoid_arm",
    )

    json_str = schedule.to_json()
    assert isinstance(json_str, str)
    data = json.loads(json_str)
    assert data["model_name"] == "planar_humanoid_arm"
    assert "non_claims" in data["metadata"]

    loaded = WBCSchedule.from_json(json_str)
    np.testing.assert_allclose(loaded.times, times)
    np.testing.assert_allclose(loaded.q_ref, q_ref)
    np.testing.assert_allclose(loaded.gains_k, gains_k)
    np.testing.assert_allclose(loaded.inertial_parameters, pi)


def test_wbc_schedule_closed_loop_replay_bounded_effort() -> None:
    """Simulate closed-loop tracking on WBC schedule verifying bounded feedback effort."""
    chain = PlanarChain(
        link_lengths=np.array([0.8, 0.5]),
        actuated=np.array([True, True]),
    )
    pi = np.array([1.2, 0.4, 0.0, 0.15, 0.8, 0.2, 0.0, 0.08], dtype=np.float64)
    dt = 0.01
    n_steps = 30
    times = np.arange(n_steps, dtype=np.float64) * dt

    # Reference trajectory from rollout
    x0 = np.array([[0.2, -0.3, 0.0, 0.0]])
    u_nom = np.zeros((1, n_steps, 2))
    u_nom[0, :, 0] = 0.3 * np.sin(2 * np.pi * times)
    q_traj, v_traj = chain.rollout(x0, u_nom[:, :-1], pi, dt)
    states_ref = np.concatenate([q_traj[0], v_traj[0]], axis=-1)  # (n_steps, 4)

    # Compute TVLQR gains
    def step_fn(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return chain.step(x, u, pi, dt)

    lin = linearize_along_trajectory(step_fn, states_ref[:-1], u_nom[0, :-1], eps=1e-6)
    q_w = 10.0 * np.eye(4)
    r_w = 1.0 * np.eye(2)
    k_gains = tvlqr_gains(lin, q_w, r_w)

    schedule = export_wbc_schedule(
        times=times[:-1],
        q_ref=states_ref[:-1, :2],
        v_ref=states_ref[:-1, 2:],
        a_ref=np.zeros_like(states_ref[:-1, :2]),
        u_feedforward=u_nom[0, :-1],
        gains_k=k_gains,
        inertial_parameters=pi,
        geometry_parameters=chain.link_lengths,
        model_name="two_link_arm",
    )

    # Validate tracking from a perturbed initial condition
    perturbation = np.array([0.05, -0.04, 0.0, 0.0])
    replay_report = validate_wbc_schedule(
        schedule=schedule,
        step_fn=step_fn,
        initial_perturbation=perturbation,
        position_tolerance=0.1,
        dt=dt,
    )

    assert replay_report.closed_loop_accepted
    assert replay_report.closed_loop_rms < replay_report.open_loop_rms
    assert replay_report.closed_loop_rms < 0.1
    # Feedback effort is bounded and non-zero
    assert 0.0 < replay_report.feedback_effort_rms < 2.0


# ---------------------------------------------------------------------------
# Floating-Base Unactuated Rows Tests
# ---------------------------------------------------------------------------


def test_floating_base_unactuated_rows_validation() -> None:
    """Validate 6-DoF unactuated floating base mask for humanoid models."""
    n_dof = 18  # 6 floating base + 12 actuated joints
    mask = floating_base_actuation_mask(n_dof=n_dof, n_floating_dof=6)
    assert mask.shape == (n_dof,)
    assert not np.any(mask[:6])  # First 6 are unactuated
    assert np.all(mask[6:])  # Remaining 12 are actuated


def test_non_claims_documentation() -> None:
    """Verify non-goals and non-claims are documented without real-time/global optimality claims."""
    assert len(NON_CLAIMS) >= 2
    assert any("real-time" in claim.lower() for claim in NON_CLAIMS)
    assert any("global" in claim.lower() for claim in NON_CLAIMS)


def test_preconditions() -> None:
    """Enforce contract preconditions."""
    lin_acc, ang_vel, ang_acc = _synthetic_spatial_trajectory(n_samples=10)
    # Mismatched wrench shape
    with pytest.raises(ContractViolationError):
        identify_payload_inertia(
            wrench=np.zeros((8, 6)),
            linear_acceleration=lin_acc,
            angular_velocity=ang_vel,
            angular_acceleration=ang_acc,
        )

    # Inconsistent tool inertia shape
    with pytest.raises(ContractViolationError):
        identify_payload_inertia(
            wrench=np.zeros((10, 6)),
            linear_acceleration=lin_acc,
            angular_velocity=ang_vel,
            angular_acceleration=ang_acc,
            known_tool_inertia=np.zeros(5),
        )

    # Invalid floating base count
    with pytest.raises(ContractViolationError):
        floating_base_actuation_mask(n_dof=5, n_floating_dof=6)
