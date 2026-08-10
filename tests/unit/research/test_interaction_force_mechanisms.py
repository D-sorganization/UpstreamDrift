"""Mechanics checks for the double-pendulum interaction-force study."""

from __future__ import annotations

import numpy as np
import pytest

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
