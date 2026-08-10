"""Mechanics checks for the double-pendulum interaction-force study."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.counterfactual_ensemble import (
    controls_at_times,
    evaluate_killswitch_case,
    evaluate_killswitch_ensemble,
)
from scripts.research.proximal_distal_energy.extract_wscg_charts import (
    extract_series,
    verify_sources,
)
from scripts.research.proximal_distal_energy.interaction_forces import (
    force_power_decomposition,
    matched_state_killswitch,
    reaction_force_decomposition,
)
from scripts.research.proximal_distal_energy.run_experiments import rollout_program
from scripts.research.proximal_distal_energy.run_interaction_force_study import (
    build_evidence,
)
from scripts.research.proximal_distal_energy.swing_model import PlanarInertials
from scripts.research.proximal_distal_energy.torque_programs import (
    restrain_then_drive_program,
)
from src.shared.python.simulation_backends import GolfModelParams, make_backend


@pytest.fixture
def params() -> GolfModelParams:
    return GolfModelParams.default()


@pytest.fixture
def inertials(params: GolfModelParams) -> PlanarInertials:
    return PlanarInertials.from_params(params)


@pytest.mark.unit
def test_force_components_reconstruct_newton_balance(
    params: GolfModelParams, inertials: PlanarInertials
) -> None:
    q = np.array([[-1.6, -0.8], [-0.9, -0.35], [-0.3, 0.15]])
    v = np.array([[2.0, 1.0], [5.0, 8.0], [9.0, 14.0]])
    u = np.array([[60.0, -8.0], [60.0, 0.0], [60.0, 15.0]])
    backend = make_backend("ode", params)
    qdd = np.vstack(
        [
            backend.forward_dynamics(qk, vk, uk)
            for qk, vk, uk in zip(q, v, u, strict=True)
        ]
    )

    result = reaction_force_decomposition(inertials, q, v, qdd)
    component_sum = sum(result.components.values())
    gravity = np.array([0.0, -inertials.g_proj])

    np.testing.assert_allclose(component_sum, result.total, atol=1e-12)
    np.testing.assert_allclose(
        result.total + inertials.m2 * gravity,
        inertials.m2 * result.club_com_acceleration,
        atol=1e-12,
    )


@pytest.mark.unit
def test_drift_and_control_forces_reconstruct_total(
    params: GolfModelParams, inertials: PlanarInertials
) -> None:
    q = np.array([[-1.1, -0.6]])
    v = np.array([[7.0, 11.0]])
    u = np.array([[60.0, 15.0]])
    backend = make_backend("ode", params)
    drift_qdd = backend.forward_dynamics(q[0], v[0], np.zeros(2))[None, :]
    total_qdd = backend.forward_dynamics(q[0], v[0], u[0])[None, :]

    total = reaction_force_decomposition(inertials, q, v, total_qdd)
    drift = reaction_force_decomposition(inertials, q, v, drift_qdd)
    control = total.total - drift.total

    np.testing.assert_allclose(drift.total + control, total.total, atol=1e-12)


@pytest.mark.unit
def test_force_power_components_reconstruct_transfer_power(
    params: GolfModelParams, inertials: PlanarInertials
) -> None:
    program = restrain_then_drive_program(60.0, 15.0, 10.0, 0.10)
    _, q, v, u = rollout_program(params, program)
    backend = make_backend("ode", params)
    qdd = np.vstack(
        [
            backend.forward_dynamics(qk, vk, uk)
            for qk, vk, uk in zip(q, v, u, strict=True)
        ]
    )
    forces = reaction_force_decomposition(inertials, q, v, qdd)

    powers = force_power_decomposition(inertials, q, v, forces)

    np.testing.assert_allclose(
        sum(powers.components.values()), powers.total, atol=1e-10
    )


@pytest.mark.unit
def test_club_moment_balance(
    params: GolfModelParams, inertials: PlanarInertials
) -> None:
    q = np.array([[-1.2, -0.5], [-0.6, -0.2]])
    v = np.array([[4.0, 3.0], [8.0, 9.0]])
    u = np.array([[60.0, -5.0], [60.0, 15.0]])
    backend = make_backend("ode", params)
    qdd = np.vstack(
        [
            backend.forward_dynamics(qk, vk, uk)
            for qk, vk, uk in zip(q, v, u, strict=True)
        ]
    )
    forces = reaction_force_decomposition(inertials, q, v, qdd)
    phi = q.sum(axis=1)
    wrist_to_com = inertials.lc2 * np.column_stack((np.sin(phi), -np.cos(phi)))
    com_to_wrist = -wrist_to_com
    force_moment = (
        com_to_wrist[:, 0] * forces.total[:, 1]
        - com_to_wrist[:, 1] * forces.total[:, 0]
    )
    wrist_moment = u[:, 1] - inertials.damping_wrist * v[:, 1]

    np.testing.assert_allclose(
        force_moment + wrist_moment,
        inertials.i2_com * qdd.sum(axis=1),
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.unit
def test_killswitch_starts_at_matched_state_and_matches_pointwise_drift(
    params: GolfModelParams,
) -> None:
    program = restrain_then_drive_program(60.0, 15.0, 10.0, 0.10)
    t, q, v, u = rollout_program(params, program)
    cut = 210
    result = matched_state_killswitch(
        params, t, q, v, u, cut_index=cut, horizon=80, dt=1.0e-3
    )
    backend = make_backend("ode", params)

    np.testing.assert_array_equal(result.commanded.q[0], result.zero_torque.q[0])
    np.testing.assert_array_equal(result.commanded.v[0], result.zero_torque.v[0])
    np.testing.assert_allclose(
        result.zero_torque_initial_qdd,
        backend.forward_dynamics(q[cut], v[cut], np.zeros(2)),
        atol=1e-12,
    )
    assert np.linalg.norm(result.commanded.q[-1] - result.zero_torque.q[-1]) > 1e-4


@pytest.mark.unit
def test_force_contract_rejects_nonfinite_input(
    inertials: PlanarInertials,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        reaction_force_decomposition(
            inertials,
            np.array([[np.nan, 0.0]]),
            np.zeros((1, 2)),
            np.zeros((1, 2)),
        )


@pytest.mark.unit
def test_wscg_source_package_is_hash_verified_and_complete() -> None:
    hashes = verify_sources()
    rows = extract_series()
    names = {str(row["series"]) for row in rows}

    assert len(hashes) == 2
    assert len(rows) == 1625
    assert len(names) == 13
    assert "Wrist Torque" in names
    assert "LeadHandCFAxial" in names


@pytest.mark.unit
def test_interaction_force_evidence_pins_mechanism_metrics() -> None:
    arrays, summary = build_evidence()

    assert arrays["force_total"].shape == arrays["q"].shape
    assert summary["force"]["peak_total_n_to_impact"] == pytest.approx(
        315.3915, abs=0.01
    )
    assert summary["transfer"]["net_work_late_half_j"] == pytest.approx(
        131.7204, abs=0.01
    )
    assert summary["killswitch"]["terminal_q_separation_rad"] > 0.5


@pytest.mark.unit
def test_controls_are_sampled_in_absolute_source_time() -> None:
    program = restrain_then_drive_program(60.0, 15.0, 10.0, 0.10)
    controls = controls_at_times(program, np.array([0.099, 0.100, 0.101]))

    np.testing.assert_array_equal(controls[:, 0], np.full(3, 60.0))
    np.testing.assert_array_equal(controls[:, 1], np.array([-10.0, 15.0, 15.0]))


@pytest.mark.unit
def test_killswitch_case_reports_all_divergence_channels(
    params: GolfModelParams, inertials: PlanarInertials
) -> None:
    program = restrain_then_drive_program(60.0, 15.0, 10.0, 0.10)
    t, q, v, _ = rollout_program(params, program)
    cut = 220
    case = evaluate_killswitch_case(
        params,
        inertials,
        program,
        source_time_s=float(t[cut]),
        source_q=q[cut],
        source_v=v[cut],
        horizon_s=0.08,
        dt_s=0.001,
    )

    np.testing.assert_array_equal(case.commanded.q[0], case.zero_torque.q[0])
    np.testing.assert_array_equal(case.commanded.v[0], case.zero_torque.v[0])
    assert set(case.metrics) >= {
        "terminal_state_distance",
        "terminal_force_distance_n",
        "terminal_power_difference_w",
        "force_work_difference_j",
        "terminal_clubhead_speed_difference_m_s",
    }
    assert case.metrics["terminal_state_distance"] > 0.0


@pytest.mark.unit
def test_killswitch_timestep_sensitivity_converges(
    params: GolfModelParams, inertials: PlanarInertials
) -> None:
    program = restrain_then_drive_program(60.0, 15.0, 10.0, 0.10)
    t, q, v, _ = rollout_program(params, program)
    cut = 250
    fine = evaluate_killswitch_case(
        params,
        inertials,
        program,
        source_time_s=float(t[cut]),
        source_q=q[cut],
        source_v=v[cut],
        horizon_s=0.08,
        dt_s=0.0005,
    )
    baseline = evaluate_killswitch_case(
        params,
        inertials,
        program,
        source_time_s=float(t[cut]),
        source_q=q[cut],
        source_v=v[cut],
        horizon_s=0.08,
        dt_s=0.001,
    )

    assert (
        abs(
            fine.metrics["terminal_state_distance"]
            - baseline.metrics["terminal_state_distance"]
        )
        < 1e-5
    )
    assert (
        abs(
            fine.metrics["force_work_difference_j"]
            - baseline.metrics["force_work_difference_j"]
        )
        < 0.05
    )


@pytest.mark.unit
def test_killswitch_ensemble_covers_cut_time_and_timestep_grid(
    params: GolfModelParams, inertials: PlanarInertials
) -> None:
    program = restrain_then_drive_program(60.0, 15.0, 10.0, 0.10)
    t, q, v, _ = rollout_program(params, program)
    rows = evaluate_killswitch_ensemble(
        params,
        inertials,
        program,
        t,
        q,
        v,
        cut_times_s=(0.12, 0.20, 0.28),
        horizons_s=(0.04, 0.08),
        timesteps_s=(0.0005, 0.001, 0.002),
    )

    assert len(rows) == 18
    assert {row["cut_time_s"] for row in rows} == {0.12, 0.20, 0.28}
    assert {row["dt_s"] for row in rows} == {0.0005, 0.001, 0.002}
    assert all(row["matched_initial_state_error"] == 0.0 for row in rows)
