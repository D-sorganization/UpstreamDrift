"""Manufactured tests for event-aligned articulated forward attribution."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_contact_projection import (
    ArticulatedContactProjectionConfig,
)
from scripts.research.proximal_distal_energy.articulated_forward_attribution import (
    differentiate_mass_along_velocity,
    differentiate_mass_matrices,
    integrate_forward_attribution,
    require_forward_attribution_closure,
    scale_forward_attribution_inputs,
)
from scripts.research.proximal_distal_energy.articulated_forward_contract import (
    ArticulatedForwardContactConfig,
)
from scripts.research.proximal_distal_energy.articulated_forward_integration import (
    ForwardIntegrationCase,
)
from scripts.research.proximal_distal_energy.articulated_rigid_forward_attribution import (
    attribute_rigid_contact_trajectory,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

pytestmark = pytest.mark.scientific


def test_constant_force_closes_impulse_and_work() -> None:
    time_s = np.array([0.0, 0.5, 1.0])
    velocity = np.array([[0.0], [1.0], [2.0]])
    mass = np.ones((3, 1, 1))
    force = np.full((3, 1, 1), 2.0)

    result = integrate_forward_attribution(
        time_s=time_s,
        mass_matrices=mass,
        mass_matrix_rates=np.zeros_like(mass),
        velocities=velocity,
        generalized_forces=force,
        contribution_names=("contact",),
        segment_ids=np.zeros(3, dtype=np.int64),
        event_impulses=np.empty((0, 1)),
        event_work_j=np.empty(0),
    )

    np.testing.assert_allclose(result.continuous_impulses[0], [2.0])
    np.testing.assert_allclose(result.generalized_work_j, [2.0])
    np.testing.assert_allclose(result.momentum_change, [2.0])
    np.testing.assert_allclose(result.transport_impulse, [0.0])
    assert result.momentum_closure_residual == pytest.approx(0.0)
    assert result.work_closure_residual_j == pytest.approx(0.0)
    assert result.momentum_reference_norm == pytest.approx(2.0)
    assert result.momentum_closure_relative_residual == pytest.approx(0.0)
    assert result.work_reference_j == pytest.approx(2.0)
    assert result.work_closure_relative_residual == pytest.approx(0.0)


def test_variable_mass_retains_euler_lagrange_transport_term() -> None:
    time_s = np.array([0.0, 0.5, 1.0])
    mass_scalar = 1.0 + time_s
    velocity_scalar = 2.0 + time_s
    mass = mass_scalar[:, None, None]
    velocity = velocity_scalar[:, None]
    force = mass_scalar[:, None, None]

    result = integrate_forward_attribution(
        time_s=time_s,
        mass_matrices=mass,
        mass_matrix_rates=np.ones_like(mass),
        velocities=velocity,
        generalized_forces=force,
        contribution_names=("active",),
        segment_ids=np.zeros(3, dtype=np.int64),
        event_impulses=np.empty((0, 1)),
        event_work_j=np.empty(0),
    )

    np.testing.assert_allclose(result.continuous_impulses[0], [1.5])
    np.testing.assert_allclose(result.transport_impulse, [2.5])
    np.testing.assert_allclose(result.momentum_change, [4.0])
    assert result.momentum_closure_residual == pytest.approx(0.0)


def test_variable_mass_constant_velocity_requires_kinetic_transport_work() -> None:
    time_s = np.array([0.0, 0.5, 1.0])
    mass = (1.0 + time_s)[:, None, None]
    velocity = np.full((3, 1), 2.0)

    result = integrate_forward_attribution(
        time_s=time_s,
        mass_matrices=mass,
        mass_matrix_rates=np.ones_like(mass),
        velocities=velocity,
        generalized_forces=np.zeros((3, 1, 1)),
        contribution_names=("active",),
        segment_ids=np.zeros(3, dtype=np.int64),
        event_impulses=np.empty((0, 1)),
        event_work_j=np.empty(0),
    )

    assert result.kinetic_transport_work_j == pytest.approx(2.0)
    assert result.kinetic_energy_change_j == pytest.approx(2.0)
    assert result.work_closure_residual_j == pytest.approx(0.0)
    assert result.work_component_names == ("active", "kinetic_transport", "event")


def test_duplicate_event_time_separates_continuous_and_impulsive_terms() -> None:
    result = integrate_forward_attribution(
        time_s=np.array([0.0, 1.0, 1.0, 2.0]),
        mass_matrices=np.ones((4, 1, 1)),
        mass_matrix_rates=np.zeros((4, 1, 1)),
        velocities=np.array([[0.0], [1.0], [3.0], [4.0]]),
        generalized_forces=np.ones((4, 1, 1)),
        contribution_names=("contact",),
        segment_ids=np.array([0, 0, 1, 1]),
        event_impulses=np.array([[2.0]]),
        event_work_j=np.array([4.0]),
    )

    np.testing.assert_allclose(result.continuous_impulses[0], [2.0])
    np.testing.assert_allclose(result.total_event_impulse, [2.0])
    np.testing.assert_allclose(result.momentum_change, [4.0])
    assert result.continuous_work_j == pytest.approx(4.0)
    assert result.total_event_work_j == pytest.approx(4.0)
    assert result.kinetic_energy_change_j == pytest.approx(8.0)
    assert result.work_closure_residual_j == pytest.approx(0.0)


def test_coordinate_scaling_preserves_work_and_transforms_impulse() -> None:
    arguments = {
        "time_s": np.array([0.0, 0.5, 1.0]),
        "mass_matrices": np.repeat(np.eye(2)[None, :, :], 3, axis=0),
        "mass_matrix_rates": np.zeros((3, 2, 2)),
        "velocities": np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]),
        "generalized_forces": np.repeat(np.array([[[2.0, 3.0]]]), 3, axis=0),
        "contribution_names": ("contact",),
        "segment_ids": np.zeros(3, dtype=np.int64),
        "event_impulses": np.empty((0, 2)),
        "event_work_j": np.empty(0),
    }
    reference = integrate_forward_attribution(**arguments)
    scaled_arguments = scale_forward_attribution_inputs(
        **arguments,
        coordinate_scale=np.array([2.0, 0.5]),
    )
    scaled = integrate_forward_attribution(**scaled_arguments)

    np.testing.assert_allclose(scaled.generalized_work_j, reference.generalized_work_j)
    np.testing.assert_allclose(scaled.continuous_work_j, reference.continuous_work_j)
    np.testing.assert_allclose(
        scaled.continuous_impulses[0],
        reference.continuous_impulses[0] / np.array([2.0, 0.5]),
    )


def test_duplicate_time_without_event_boundary_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate times require"):
        integrate_forward_attribution(
            time_s=np.array([0.0, 1.0, 1.0]),
            mass_matrices=np.ones((3, 1, 1)),
            mass_matrix_rates=np.zeros((3, 1, 1)),
            velocities=np.ones((3, 1)),
            generalized_forces=np.ones((3, 1, 1)),
            contribution_names=("contact",),
            segment_ids=np.zeros(3, dtype=np.int64),
            event_impulses=np.empty((0, 1)),
            event_work_j=np.empty(0),
        )


def test_event_count_must_match_segment_transitions() -> None:
    with pytest.raises(ValueError, match="one row per segment transition"):
        integrate_forward_attribution(
            time_s=np.array([0.0, 1.0, 1.0]),
            mass_matrices=np.ones((3, 1, 1)),
            mass_matrix_rates=np.zeros((3, 1, 1)),
            velocities=np.ones((3, 1)),
            generalized_forces=np.ones((3, 1, 1)),
            contribution_names=("contact",),
            segment_ids=np.array([0, 0, 1]),
            event_impulses=np.empty((0, 1)),
            event_work_j=np.empty(0),
        )


def test_mass_matrix_rate_is_differentiated_within_each_segment() -> None:
    time_s = np.array([0.0, 0.5, 1.0, 1.0, 1.5, 2.0])
    scalar_mass = 1.0 + 2.0 * time_s
    rates = differentiate_mass_matrices(
        time_s=time_s,
        mass_matrices=scalar_mass[:, None, None],
        segment_ids=np.array([0, 0, 0, 1, 1, 1]),
    )

    np.testing.assert_allclose(rates[:, 0, 0], 2.0)


def test_mass_rate_directional_derivative_is_independent_of_event_sampling() -> None:
    positions = np.array([[0.5], [1.0]])
    velocities = np.array([[3.0], [-2.0]])

    rates = differentiate_mass_along_velocity(
        positions=positions,
        velocities=velocities,
        mass_evaluator=lambda q: np.array([[1.0 + 2.0 * q[0]]]),
        directional_step_s=1.0e-6,
    )

    np.testing.assert_allclose(rates[:, 0, 0], [6.0, -4.0], atol=1.0e-9)


def test_planted_force_corruption_fails_closed() -> None:
    result = integrate_forward_attribution(
        time_s=np.array([0.0, 0.5, 1.0]),
        mass_matrices=np.ones((3, 1, 1)),
        mass_matrix_rates=np.zeros((3, 1, 1)),
        velocities=np.array([[0.0], [1.0], [2.0]]),
        generalized_forces=np.array([[[2.0]], [[2.2]], [[2.0]]]),
        contribution_names=("contact",),
        segment_ids=np.zeros(3, dtype=np.int64),
        event_impulses=np.empty((0, 1)),
        event_work_j=np.empty(0),
    )

    assert result.momentum_closure_residual > 0.0
    assert result.momentum_closure_relative_residual > 0.0
    with pytest.raises(ValueError, match="momentum closure"):
        require_forward_attribution_closure(
            result,
            momentum_tolerance=1.0e-12,
            work_tolerance_j=1.0,
        )


def test_rigid_contact_trace_replays_registered_force_contributions() -> None:
    model, metadata = build_subject_scaled_model(default_synthetic_profiles()[0])
    q = np.zeros(model.nq)
    case = ForwardIntegrationCase(
        q=q,
        qd=np.zeros(model.nq),
        grip_span_m=0.18,
        hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
        time_step_s=0.0005,
        contact_stiffness=1800.0,
        contact_damping=18.0,
        initial_club_displacement_m=0.001,
        initial_club_velocity_m_s=0.05,
        engine="mujoco",
    )

    evidence = attribute_rigid_contact_trajectory(
        model,
        case,
        ArticulatedForwardContactConfig(
            duration_s=0.002,
            time_steps_s=(0.001, 0.0005),
        ),
        ArticulatedContactProjectionConfig(
            contact_stiffness=case.contact_stiffness,
            contact_damping=case.contact_damping,
        ),
    )

    assert evidence.attribution.contribution_names == (
        "configuration",
        "velocity",
        "contact",
        "active",
    )
    assert evidence.generalized_forces.shape == (5, 4, model.nq)
    assert np.allclose(evidence.generalized_forces[:, 3], 0.0)
    assert np.max(np.abs(evidence.pointwise_force_closure_residual)) <= 1.0e-12
    assert np.all(np.isfinite(evidence.attribution.generalized_work_j))


def test_signed_shares_expose_cancellation_without_clipping() -> None:
    time_s = np.array([0.0, 0.5, 1.0])
    velocity = time_s[:, None]
    forces = np.repeat(np.array([[[2.0], [-1.0]]]), 3, axis=0)

    result = integrate_forward_attribution(
        time_s=time_s,
        mass_matrices=np.ones((3, 1, 1)),
        mass_matrix_rates=np.zeros((3, 1, 1)),
        velocities=velocity,
        generalized_forces=forces,
        contribution_names=("positive", "negative"),
        segment_ids=np.zeros(3, dtype=np.int64),
        event_impulses=np.empty((0, 1)),
        event_work_j=np.empty(0),
    )

    np.testing.assert_allclose(result.impulse_shares[:, 0], [2.0, -1.0, 0.0, 0.0])
    np.testing.assert_allclose(result.work_shares, [2.0, -1.0, 0.0, 0.0])
    assert result.impulse_cancellation_indices[0] == pytest.approx(3.0)
    assert result.work_cancellation_index == pytest.approx(3.0)
    assert result.impulse_share_adequacy[0]
    assert result.work_share_adequate


def test_near_zero_share_denominators_are_suppressed() -> None:
    result = integrate_forward_attribution(
        time_s=np.array([0.0, 0.5, 1.0]),
        mass_matrices=np.ones((3, 1, 1)),
        mass_matrix_rates=np.zeros((3, 1, 1)),
        velocities=np.zeros((3, 1)),
        generalized_forces=np.zeros((3, 1, 1)),
        contribution_names=("contact",),
        segment_ids=np.zeros(3, dtype=np.int64),
        event_impulses=np.empty((0, 1)),
        event_work_j=np.empty(0),
        share_denominator_floor=1.0e-10,
    )

    assert not result.impulse_share_adequacy[0]
    assert not result.work_share_adequate
    assert np.all(np.isnan(result.impulse_shares))
    assert np.all(np.isnan(result.work_shares))
    assert np.isnan(result.impulse_cancellation_indices[0])
    assert np.isnan(result.work_cancellation_index)
