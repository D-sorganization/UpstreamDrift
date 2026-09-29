"""Contracts for the spatial full-body common-state experiment."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.spatial_full_body import (
    SpatialExperimentConfig,
    _populate_mujoco_full_mass_matrix,
    build_spatial_model,
    evaluate_hand_wrenches,
    run_cross_formulation_experiment,
)

pytestmark = pytest.mark.scientific


def test_mujoco_mass_matrix_adapter_supports_legacy_and_csr_apis() -> None:
    class LegacyData:
        qM = object()

    class CsrData:
        pass

    class FakeMujoco:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def mj_fullM(self, *args: object) -> None:
            self.calls.append(args)

    module = FakeMujoco()
    model = object()
    matrix = np.empty((2, 2))
    legacy = LegacyData()
    modern = CsrData()

    _populate_mujoco_full_mass_matrix(module, model, legacy, matrix)
    _populate_mujoco_full_mass_matrix(module, model, modern, matrix)

    assert module.calls == [
        (model, matrix, legacy.qM),
        (model, modern, matrix),
    ]


def test_spatial_model_is_nonplanar_and_contains_body_and_free_club() -> None:
    model = build_spatial_model()

    assert model.nq == 20
    assert {joint.kind for joint in model.joints} == {"revolute", "prismatic"}
    assert {body.region for body in model.bodies} >= {
        "pelvis",
        "torso",
        "lead_arm",
        "trail_arm",
        "lower_body",
        "club",
    }
    axes = np.stack([joint.axis for joint in model.joints])
    assert np.linalg.matrix_rank(axes) == 3
    assert model.club_dof_indices.shape == (6,)


def test_hand_wrench_action_reaction_and_power_close() -> None:
    model = build_spatial_model()
    sample = evaluate_hand_wrenches(model, time_s=0.215, coincident_hands=False)

    np.testing.assert_allclose(
        sample.club_wrench + sample.body_wrench,
        np.zeros(6),
        atol=1e-12,
    )
    assert abs(sample.action_reaction_power_residual_w) < 1e-10
    assert np.linalg.norm(sample.club_wrench[:3]) > 0.0
    assert abs(sample.club_wrench[5]) > 0.0
    assert abs(sample.club_wrench[3]) + abs(sample.club_wrench[4]) > 0.0


def test_coincident_hand_negative_control_removes_force_couple() -> None:
    model = build_spatial_model()
    baseline = evaluate_hand_wrenches(model, time_s=0.215, coincident_hands=False)
    control = evaluate_hand_wrenches(model, time_s=0.215, coincident_hands=True)

    assert baseline.force_generated_couple_nm < 0.0
    assert control.force_generated_couple_nm == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(
        baseline.lead_force_n + baseline.trail_force_n,
        control.lead_force_n + control.trail_force_n,
        atol=1e-12,
    )


def test_reversing_contact_geometry_reverses_couple_without_force_change() -> None:
    model = build_spatial_model()
    baseline = evaluate_hand_wrenches(model, time_s=0.215, coincident_hands=False)
    reversed_geometry = evaluate_hand_wrenches(
        model,
        time_s=0.215,
        coincident_hands=False,
        reverse_geometry=True,
    )

    np.testing.assert_allclose(
        baseline.lead_force_n,
        reversed_geometry.lead_force_n,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        baseline.trail_force_n,
        reversed_geometry.trail_force_n,
        atol=1e-12,
    )
    assert reversed_geometry.force_generated_couple_nm == pytest.approx(
        -baseline.force_generated_couple_nm,
        abs=1e-12,
    )


@pytest.mark.requires_mujoco
def test_mujoco_and_lagrange_inverse_dynamics_agree_within_predeclared_bound() -> None:
    result = run_cross_formulation_experiment(
        SpatialExperimentConfig(duration_s=0.24, sample_dt_s=0.004)
    )

    assert result.formulation_names == (
        "lagrange_christoffel",
        "mujoco_native_inverse_dynamics",
    )
    assert result.model_hashes[0] == result.model_hashes[1]
    assert result.out_of_plane_motion_m > 1e-3
    assert result.max_relative_inverse_dynamics_error <= result.tolerance.relative
    assert result.max_absolute_generalized_force_error <= result.tolerance.absolute
    assert result.intervention_event_grid_error_s <= result.tolerance.event_time_s
    assert result.classification in {"equivalent", "structural_discrepancy"}


def test_configuration_rejects_post_outcome_tolerance_changes() -> None:
    with pytest.raises(ValueError, match="predeclared"):
        SpatialExperimentConfig(
            duration_s=0.24,
            sample_dt_s=0.004,
            tolerance_source="chosen_after_outcome",
        )
