from __future__ import annotations

import numpy as np
import pytest

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

pytestmark = pytest.mark.scientific


def test_attachment_law_configuration_fails_closed() -> None:
    with pytest.raises(ValueError, match="stiffness"):
        AttachmentLawConfig(kind=AttachmentLawKind.BILATERAL, stiffness=0.0)
    with pytest.raises(ValueError, match="slack_distance_m"):
        AttachmentLawConfig(
            kind=AttachmentLawKind.DEAD_ZONE_TENSION,
            slack_distance_m=0.0,
        )
    with pytest.raises(ValueError, match="zero slack"):
        AttachmentLawConfig(
            kind=AttachmentLawKind.TENSION_ONLY,
            slack_distance_m=1.0e-3,
        )


def test_dead_zone_transmits_no_force_while_open() -> None:
    snapshot = evaluate_attachment_law(
        displacement_m=np.array([5.0e-4, 0.0, 0.0]),
        relative_velocity_m_s=np.array([0.2, 0.0, 0.0]),
        config=AttachmentLawConfig(
            kind=AttachmentLawKind.DEAD_ZONE_TENSION,
            stiffness=2000.0,
            damping=20.0,
            slack_distance_m=1.0e-3,
        ),
    )

    assert snapshot.active is False
    assert snapshot.extension_m == 0.0
    assert np.array_equal(snapshot.force_on_club_n, np.zeros(3))
    assert snapshot.strain_energy_j == 0.0
    assert snapshot.interface_power_w == 0.0


def test_tension_law_is_equal_opposite_and_passive() -> None:
    displacement = np.array([2.0e-3, 0.0, 0.0])
    relative_velocity = np.array([0.1, 0.0, 0.0])
    snapshot = evaluate_attachment_law(
        displacement_m=displacement,
        relative_velocity_m_s=relative_velocity,
        config=AttachmentLawConfig(
            kind=AttachmentLawKind.DEAD_ZONE_TENSION,
            stiffness=2000.0,
            damping=20.0,
            slack_distance_m=1.0e-3,
        ),
    )

    assert snapshot.active is True
    assert snapshot.extension_m == pytest.approx(1.0e-3)
    assert snapshot.force_on_club_n[0] == pytest.approx(4.0)
    assert np.allclose(snapshot.force_on_hand_n, -snapshot.force_on_club_n)
    assert snapshot.dissipation_power_w <= 0.0
    assert snapshot.interface_power_w == pytest.approx(
        snapshot.storage_power_w + snapshot.dissipation_power_w
    )


def test_tension_law_does_not_add_compressive_damping_during_unloading() -> None:
    snapshot = evaluate_attachment_law(
        displacement_m=np.array([2.0e-3, 0.0, 0.0]),
        relative_velocity_m_s=np.array([-1.0, 0.0, 0.0]),
        config=AttachmentLawConfig(
            kind=AttachmentLawKind.TENSION_ONLY,
            stiffness=2000.0,
            damping=20.0,
        ),
    )

    assert snapshot.active is True
    assert snapshot.force_on_club_n[0] == pytest.approx(4.0)
    assert snapshot.dissipation_power_w == 0.0
    assert snapshot.interface_power_w == pytest.approx(snapshot.storage_power_w)


def test_bilateral_law_reproduces_vector_kelvin_voigt_contract() -> None:
    displacement = np.array([1.0e-3, -2.0e-3, 3.0e-3])
    relative_velocity = np.array([0.2, -0.1, 0.05])
    config = AttachmentLawConfig(
        kind=AttachmentLawKind.BILATERAL,
        stiffness=1800.0,
        damping=18.0,
    )
    snapshot = evaluate_attachment_law(
        displacement_m=displacement,
        relative_velocity_m_s=relative_velocity,
        config=config,
    )

    expected_force = (
        config.stiffness * displacement + config.damping * relative_velocity
    )
    assert np.allclose(snapshot.force_on_club_n, expected_force)
    assert snapshot.interface_power_w == pytest.approx(
        -float(expected_force @ relative_velocity)
    )


def test_open_two_hand_projection_has_zero_generalized_force() -> None:
    model, metadata = build_subject_scaled_model(default_synthetic_profiles()[0])
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    authority = (
        root
        / "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.npz"
    )
    with np.load(authority) as source:
        q = np.asarray(source["solution_q"][0, 6], dtype=float)
        grip_span_m = float(source["case_grip_span_m"][0])
    q[14] += 5.0e-4
    snapshot = evaluate_slack_projection(
        model,
        q,
        np.zeros(model.nq),
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
        law=AttachmentLawConfig(
            kind=AttachmentLawKind.DEAD_ZONE_TENSION,
            slack_distance_m=1.0e-3,
        ),
    )

    assert snapshot.active_interface_count == 0
    assert np.linalg.norm(snapshot.generalized_contact_force) == pytest.approx(0.0)
    assert snapshot.maximum_contact_force_n == 0.0
