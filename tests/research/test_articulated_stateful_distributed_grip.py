"""Stateful distributed-grip adapter tests for #9153."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.research.proximal_distal_energy.articulated_distributed_grip import (
    DistributedGripConfig,
    distributed_reference_lengths,
)
from scripts.research.proximal_distal_energy.articulated_stateful_distributed_grip import (
    StatefulDistributedGripInput,
    evaluate_stateful_distributed_grip,
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


def _case() -> tuple[object, dict[str, object], np.ndarray, float]:
    model, metadata = build_subject_scaled_model(default_synthetic_profiles()[0])
    with np.load(DATA / "subject_scaled_closed_contact.npz") as source:
        q = np.asarray(source["solution_q"][0, 6], dtype=float)
        grip_span = float(source["case_grip_span_m"][0])
    return model, metadata, q, grip_span


def test_stateful_adapter_returns_combined_load_and_exact_energy_ledgers() -> None:
    model, metadata, q, grip_span = _case()
    grip = DistributedGripConfig(
        station_count_per_hand=3,
        station_width_m=0.03,
        friction_coefficient=0.4,
    )
    references = distributed_reference_lengths(
        model,
        q,
        grip_span_m=grip_span,
        hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
        config=grip,
    )
    perturbed = q.copy()
    perturbed[14] += 0.002
    velocity = np.zeros(model.nq)
    velocity[15] = 0.5
    result = evaluate_stateful_distributed_grip(
        model,
        perturbed,
        velocity,
        np.zeros((2, 3, 3)),
        inputs=StatefulDistributedGripInput(
            grip_span_m=grip_span,
            hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
            reference_lengths_m=references,
            grip_config=grip,
            friction_config=StatefulFrictionConfig(
                tangential_stiffness_n_m=600.0,
                friction_coefficient=0.4,
            ),
            time_step_s=0.001,
        ),
    )

    assert result.elastic_displacement_m.shape == (2, 3, 3)
    assert result.regimes.shape == (2, 3)
    assert result.force_on_club_n.shape == (2, 3, 3)
    assert result.active_station.shape == (2, 3)
    assert result.station_signed_gap_m.shape == (2, 3)
    assert np.array_equal(result.station_signed_gap_m > 0.0, result.active_station)
    assert result.generalized_contact_force.shape == (model.nq,)
    np.testing.assert_allclose(
        result.generalized_contact_force,
        result.normal_generalized_contact_force
        + result.tangential_generalized_contact_force,
        atol=1.0e-14,
    )
    assert np.all(np.isfinite(result.generalized_contact_force))
    assert np.all(result.frictional_dissipation_j >= 0.0)
    assert np.all(result.release_dissipation_j >= 0.0)
    np.testing.assert_allclose(
        result.constitutive_work_j,
        result.elastic_energy_change_j
        + result.frictional_dissipation_j
        + result.release_dissipation_j,
        atol=1.0e-14,
    )
    assert result.static_stick_modeled is True
    assert result.human_or_anatomical_inference is False
    assert result.normal_strain_energy_j >= 0.0
    assert result.normal_dissipation_power_w <= 0.0


def test_open_station_resets_state_and_retains_release_dissipation() -> None:
    model, metadata, q, grip_span = _case()
    grip = DistributedGripConfig(
        station_count_per_hand=1,
        station_width_m=0.0,
        friction_coefficient=0.4,
        slack_distance_m=0.01,
    )
    references = distributed_reference_lengths(
        model,
        q,
        grip_span_m=grip_span,
        hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
        config=grip,
    )
    result = evaluate_stateful_distributed_grip(
        model,
        q,
        np.zeros(model.nq),
        np.full((2, 1, 3), 0.001),
        inputs=StatefulDistributedGripInput(
            grip_span_m=grip_span,
            hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
            reference_lengths_m=references,
            grip_config=grip,
            friction_config=StatefulFrictionConfig(
                tangential_stiffness_n_m=1000.0,
                friction_coefficient=0.4,
            ),
            time_step_s=0.001,
        ),
    )

    assert np.all(result.regimes == TangentialRegime.OPEN.value)
    np.testing.assert_allclose(result.elastic_displacement_m, 0.0)
    assert np.all(result.release_dissipation_j > 0.0)
