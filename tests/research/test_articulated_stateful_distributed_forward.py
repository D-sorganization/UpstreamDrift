"""Timestamp and ledger tests for stateful distributed forward integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_distributed_grip import (
    DistributedGripConfig,
)
from scripts.research.proximal_distal_energy.articulated_stateful_distributed_forward import (
    STATEFUL_OPERATOR_SPLIT,
    StatefulDistributedForwardConfig,
    StatefulDistributedIntegrationCase,
    integrate_stateful_distributed_grip,
)
from scripts.research.proximal_distal_energy.articulated_stateful_friction import (
    StatefulFrictionConfig,
    TangentialRegime,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _case_inputs() -> tuple[object, dict[str, object], np.ndarray, float]:
    model, metadata = build_subject_scaled_model(default_synthetic_profiles()[0])
    with np.load(DATA / "subject_scaled_closed_contact.npz") as source:
        q = np.asarray(source["solution_q"][0, 6], dtype=float)
        grip_span_m = float(source["case_grip_span_m"][0])
    return model, metadata, q, grip_span_m


def _run(
    *,
    slack_distance_m: float = 0.0,
    initial_state: float = 0.0,
    time_step_s: float = 0.0005,
    duration_s: float = 0.002,
):
    model, metadata, q, grip_span_m = _case_inputs()
    grip = DistributedGripConfig(
        station_count_per_hand=3,
        station_width_m=0.03,
        friction_coefficient=0.4,
        slack_distance_m=slack_distance_m,
    )
    case = StatefulDistributedIntegrationCase(
        q=q,
        qd=np.zeros(model.nq),
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
        time_step_s=time_step_s,
        initial_club_displacement_m=0.001 if slack_distance_m == 0.0 else 0.0,
        initial_club_velocity_m_s=0.05,
        engine="mujoco",
        grip=grip,
        friction=StatefulFrictionConfig(
            tangential_stiffness_n_m=600.0,
            friction_coefficient=0.4,
        ),
        initial_elastic_displacement_m=np.full((2, 3, 3), initial_state),
    )
    return integrate_stateful_distributed_grip(
        model,
        case,
        StatefulDistributedForwardConfig(
            duration_s=duration_s,
            time_steps_s=(0.001, 0.0005, 0.00025),
        ),
    )


def test_stateful_forward_configuration_fails_closed() -> None:
    with pytest.raises(ValueError, match="duration_s"):
        StatefulDistributedForwardConfig(duration_s=0.0)
    with pytest.raises(ValueError, match="time_steps_s"):
        StatefulDistributedForwardConfig(time_steps_s=(0.0005, 0.001))


def test_forward_trace_separates_node_and_interval_timestamps() -> None:
    trace = _run()

    assert trace["operator_split"] == STATEFUL_OPERATOR_SPLIT
    assert trace["force_timestamp"] == "interval_end_state_at_left_node_kinematics"
    assert trace["mechanical_step"] == "semi_implicit_euler"
    assert trace["node_time_s"].shape == (5,)
    assert trace["interval_time_start_s"].shape == (4,)
    assert trace["node_q"].shape[0] == 5
    assert trace["interval_generalized_contact_force"].shape[0] == 4
    assert trace["interval_force_on_club_n"].shape == (4, 2, 3, 3)
    assert trace["interval_friction_limit_n"].shape == (4, 2, 3)
    assert trace["node_elastic_displacement_m"].shape == (5, 2, 3, 3)
    assert trace["interval_regime"].shape == (4, 2, 3)
    np.testing.assert_allclose(
        trace["interval_elastic_displacement_before_m"],
        trace["node_elastic_displacement_m"][:-1],
    )
    np.testing.assert_allclose(
        trace["interval_elastic_displacement_after_m"],
        trace["node_elastic_displacement_m"][1:],
    )
    assert np.all(np.isfinite(trace["node_q"]))
    assert np.max(np.abs(trace["interval_virtual_power_residual_w"])) <= 1.0e-10
    assert trace["static_stick_modeled"] is True
    assert trace["human_or_anatomical_inference"] is False


def test_interval_constitutive_ledgers_close_exactly() -> None:
    trace = _run()

    np.testing.assert_allclose(
        trace["interval_constitutive_work_j"],
        trace["interval_tangential_elastic_energy_change_j"]
        + trace["interval_frictional_dissipation_j"]
        + trace["interval_release_dissipation_j"],
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        trace["node_tangential_strain_energy_j"][1:]
        - trace["node_tangential_strain_energy_j"][:-1],
        np.sum(trace["interval_tangential_elastic_energy_change_j"], axis=(1, 2)),
        atol=1.0e-14,
    )
    assert np.all(trace["interval_frictional_dissipation_j"] >= 0.0)
    assert np.all(trace["interval_release_dissipation_j"] >= 0.0)


def test_opening_releases_initial_tangential_state_without_carryover() -> None:
    trace = _run(slack_distance_m=0.01, initial_state=0.001)

    assert np.all(trace["interval_regime"][0] == TangentialRegime.OPEN.value)
    assert np.sum(trace["interval_release_dissipation_j"][0]) > 0.0
    np.testing.assert_allclose(trace["node_elastic_displacement_m"][1], 0.0)
    np.testing.assert_allclose(trace["node_tangential_strain_energy_j"][1], 0.0)


def test_passive_energy_defect_decreases_under_time_step_refinement() -> None:
    traces = [
        _run(time_step_s=time_step_s, duration_s=0.005)
        for time_step_s in (0.001, 0.0005, 0.00025)
    ]
    final_defects = np.asarray(
        [abs(trace["passive_energy_balance_residual_j"][-1]) for trace in traces]
    )
    coupling_defects = np.asarray(
        [
            np.max(np.abs(trace["interval_tangential_coupling_work_residual_j"]))
            for trace in traces
        ]
    )

    assert np.all(np.diff(final_defects) < 0.0), final_defects
    assert np.all(np.diff(coupling_defects) < 0.0), coupling_defects
