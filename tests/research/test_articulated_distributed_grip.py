from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_distributed_grip import (
    DistributedGripConfig,
    distributed_reference_lengths,
    evaluate_distributed_grip,
)
from scripts.research.proximal_distal_energy.articulated_slack_contact import (
    AttachmentLawConfig,
    AttachmentLawKind,
    evaluate_attachment_law,
    evaluate_slack_projection,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _closed_state() -> tuple[object, dict[str, object], np.ndarray, float]:
    model, metadata = build_subject_scaled_model(default_synthetic_profiles()[0])
    with np.load(DATA / "subject_scaled_closed_contact.npz") as source:
        q = np.asarray(source["solution_q"][0, 6], dtype=float)
        grip_span_m = float(source["case_grip_span_m"][0])
    return model, metadata, q, grip_span_m


def test_distributed_grip_configuration_fails_closed() -> None:
    with pytest.raises(ValueError, match="positive odd"):
        DistributedGripConfig(station_count_per_hand=2)
    with pytest.raises(ValueError, match="station_width_m"):
        DistributedGripConfig(station_count_per_hand=3, station_width_m=0.0)
    with pytest.raises(ValueError, match="total_stiffness_n_m"):
        DistributedGripConfig(total_stiffness_n_m=0.0)


def test_tension_reference_length_separates_preload_from_extension() -> None:
    law = AttachmentLawConfig(
        kind=AttachmentLawKind.TENSION_ONLY,
        stiffness=2000.0,
        damping=20.0,
    )
    open_snapshot = evaluate_attachment_law(
        displacement_m=np.array([0.01, 0.0, 0.0]),
        relative_velocity_m_s=np.zeros(3),
        config=law,
        reference_length_m=0.01,
    )
    loaded_snapshot = evaluate_attachment_law(
        displacement_m=np.array([0.011, 0.0, 0.0]),
        relative_velocity_m_s=np.zeros(3),
        config=law,
        reference_length_m=0.01,
    )

    assert open_snapshot.active is False
    assert loaded_snapshot.extension_m == pytest.approx(0.001)
    assert loaded_snapshot.force_on_club_n[0] == pytest.approx(2.0)


def test_one_fiber_reduces_to_point_tension_law() -> None:
    model, metadata, q, grip_span_m = _closed_state()
    hand_x = float(metadata["hand_contact_local_x_m"])
    config = DistributedGripConfig(station_count_per_hand=1, station_width_m=0.0)
    references = distributed_reference_lengths(
        model,
        q,
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=hand_x,
        config=config,
    )
    perturbed = q.copy()
    perturbed[14] += 1.0e-3
    qd = np.zeros(model.nq)
    distributed = evaluate_distributed_grip(
        model,
        perturbed,
        qd,
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=hand_x,
        reference_lengths_m=references,
        config=config,
    )
    point = evaluate_slack_projection(
        model,
        perturbed,
        qd,
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=hand_x,
        law=AttachmentLawConfig(
            kind=AttachmentLawKind.TENSION_ONLY,
            stiffness=1800.0,
            damping=18.0,
        ),
    )

    assert np.allclose(
        distributed.generalized_contact_force,
        point.generalized_contact_force,
        atol=1.0e-12,
    )
    assert distributed.maximum_station_force_n == pytest.approx(
        point.maximum_contact_force_n
    )
    assert distributed.strain_energy_j == pytest.approx(point.strain_energy_j)


def test_multi_fiber_projection_closes_power_and_passivity() -> None:
    model, metadata, q, grip_span_m = _closed_state()
    hand_x = float(metadata["hand_contact_local_x_m"])
    config = DistributedGripConfig(station_count_per_hand=5)
    references = distributed_reference_lengths(
        model,
        q,
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=hand_x,
        config=config,
    )
    perturbed = q.copy()
    perturbed[14] += 1.0e-3
    qd = np.zeros(model.nq)
    qd[14] = 0.05
    snapshot = evaluate_distributed_grip(
        model,
        perturbed,
        qd,
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=hand_x,
        reference_lengths_m=references,
        config=config,
    )

    assert snapshot.force_on_club_n.shape == (2, 5, 3)
    assert snapshot.active_station_count > 0
    assert snapshot.action_reaction_residual_n <= 1.0e-12
    assert snapshot.coincident_couple_residual_nm <= 1.0e-12
    assert snapshot.reversed_couple_sign_residual_nm <= 1.0e-12
    assert snapshot.virtual_power_residual_w <= 1.0e-12
    assert snapshot.dissipation_power_w <= 0.0
    assert 0.0 < snapshot.load_concentration <= 1.0
