"""Contracts for active-set event location on a declared discrete path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_contact_events import (
    ContactEventKind,
    ContactEventLocationConfig,
    align_state_trace_to_events,
    locate_contact_events,
)
from scripts.research.proximal_distal_energy.articulated_distributed_event_attribution import (
    attribute_distributed_contact_trajectory,
    locate_distributed_trace_events,
)
from scripts.research.proximal_distal_energy.articulated_distributed_forward import (
    DistributedForwardConfig,
    DistributedIntegrationCase,
    integrate_distributed_grip,
)
from scripts.research.proximal_distal_energy.articulated_distributed_grip import (
    DistributedGripConfig,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _linear_gap(position: np.ndarray) -> np.ndarray:
    return np.array([[position[0] - 0.5]])


@pytest.mark.parametrize(
    ("positions", "active", "expected_kind"),
    [
        (np.array([[0.0], [1.0]]), [False, True], ContactEventKind.REATTACHMENT),
        (np.array([[1.0], [0.0]]), [True, False], ContactEventKind.OPENING),
    ],
)
def test_locates_opening_and_reattachment_on_interpolated_path(
    positions: np.ndarray,
    active: list[bool],
    expected_kind: ContactEventKind,
) -> None:
    gaps = np.array([[[_linear_gap(row)[0, 0]]] for row in positions])
    events = locate_contact_events(
        time_s=np.array([0.0, 1.0]),
        positions=positions,
        velocities=np.array([[2.0], [4.0]]),
        station_signed_gap_m=gaps,
        station_active=np.asarray(active, dtype=bool)[:, None, None],
        gap_evaluator=_linear_gap,
    )

    assert len(events) == 1
    event = events[0]
    assert event.kind is expected_kind
    assert event.time_s == pytest.approx(0.5, abs=1.0e-10)
    assert event.position[0] == pytest.approx(0.5, abs=1.0e-10)
    assert event.velocity[0] == pytest.approx(3.0, abs=1.0e-10)
    assert abs(event.gap_residual_m) <= 1.0e-10
    assert event.final_bracket_width_s <= 1.0e-12
    assert event.path_model == "linear_state_interpolant"


def test_no_active_set_change_emits_no_event() -> None:
    events = locate_contact_events(
        time_s=np.array([0.0, 1.0]),
        positions=np.array([[0.0], [0.25]]),
        velocities=np.zeros((2, 1)),
        station_signed_gap_m=np.array([[[-0.5]], [[-0.25]]]),
        station_active=np.zeros((2, 1, 1), dtype=bool),
        gap_evaluator=_linear_gap,
    )

    assert events == ()


def test_event_alignment_duplicates_pre_post_state_without_crossing_event() -> None:
    positions = np.array([[0.0], [1.0]])
    velocities = np.array([[2.0], [4.0]])
    events = locate_contact_events(
        time_s=np.array([0.0, 1.0]),
        positions=positions,
        velocities=velocities,
        station_signed_gap_m=np.array([[[-0.5]], [[0.5]]]),
        station_active=np.array([[[False]], [[True]]]),
        gap_evaluator=_linear_gap,
    )

    aligned = align_state_trace_to_events(
        time_s=np.array([0.0, 1.0]),
        positions=positions,
        velocities=velocities,
        events=events,
    )

    np.testing.assert_allclose(aligned.time_s, [0.0, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(aligned.positions[:, 0], [0.0, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(aligned.velocities[:, 0], [2.0, 3.0, 3.0, 4.0])
    np.testing.assert_array_equal(aligned.segment_ids, [0, 0, 1, 1])
    np.testing.assert_array_equal(aligned.event_record_offsets, [0])


def test_inconsistent_active_state_and_gap_fail_closed() -> None:
    with pytest.raises(ValueError, match="active state must equal"):
        locate_contact_events(
            time_s=np.array([0.0, 1.0]),
            positions=np.array([[0.0], [1.0]]),
            velocities=np.zeros((2, 1)),
            station_signed_gap_m=np.array([[[-0.5]], [[0.5]]]),
            station_active=np.zeros((2, 1, 1), dtype=bool),
            gap_evaluator=_linear_gap,
        )


def test_transition_without_a_bracketed_root_fails_closed() -> None:
    with pytest.raises(ValueError, match="does not bracket"):
        locate_contact_events(
            time_s=np.array([0.0, 1.0]),
            positions=np.array([[0.0], [1.0]]),
            velocities=np.zeros((2, 1)),
            station_signed_gap_m=np.array([[[-0.5]], [[-0.25]]]),
            station_active=np.array([[[False]], [[True]]]),
            gap_evaluator=lambda position: np.array([[-0.5 + 0.25 * position[0]]]),
            config=ContactEventLocationConfig(validate_active_gap_consistency=False),
        )


def test_invalid_gap_evaluator_shape_fails_closed() -> None:
    with pytest.raises(ValueError, match="invalid station-gap array"):
        locate_contact_events(
            time_s=np.array([0.0, 1.0]),
            positions=np.array([[0.0], [1.0]]),
            velocities=np.zeros((2, 1)),
            station_signed_gap_m=np.array([[[-0.5]], [[0.5]]]),
            station_active=np.array([[[False]], [[True]]]),
            gap_evaluator=lambda _position: np.array([0.0, 1.0]),
        )


def test_registered_distributed_probe_locates_opening_and_reattachment() -> None:
    model, metadata = build_subject_scaled_model(default_synthetic_profiles()[0])
    with np.load(DATA / "subject_scaled_closed_contact.npz") as source:
        q = np.asarray(source["solution_q"][0, 6], dtype=float)
        grip_span_m = float(source["case_grip_span_m"][0])
    grip = DistributedGripConfig(
        station_count_per_hand=1,
        station_width_m=0.0,
        slack_distance_m=0.0015,
    )
    case = DistributedIntegrationCase(
        q=q,
        qd=np.zeros(model.nq),
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
        time_step_s=0.001,
        initial_club_displacement_m=0.001,
        initial_club_velocity_m_s=-0.8,
        engine="mujoco",
        grip=grip,
    )
    config = DistributedForwardConfig(
        duration_s=0.05,
        time_steps_s=(0.001, 0.0005),
    )
    trace = integrate_distributed_grip(model, case, config)

    events = locate_distributed_trace_events(model=model, case=case, trace=trace)

    kinds = {event.kind for event in events}
    assert ContactEventKind.OPENING in kinds
    assert ContactEventKind.REATTACHMENT in kinds
    assert max(abs(event.gap_residual_m) for event in events) <= 1.0e-10
    assert all(event.final_bracket_width_s <= 1.0e-12 for event in events)

    evidence = attribute_distributed_contact_trajectory(
        model=model,
        case=case,
        config=config,
    )
    assert len(evidence.events) == len(events)
    assert np.count_nonzero(np.diff(evidence.aligned.time_s) == 0.0) >= 2
    assert np.allclose(evidence.attribution.total_event_impulse, 0.0)
    assert evidence.attribution.total_event_work_j == pytest.approx(0.0)
    assert np.max(evidence.pointwise_force_closure_residual) <= 1.0e-12
