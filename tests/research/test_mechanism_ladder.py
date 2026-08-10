"""Contracts for common interaction observables across model-fidelity tiers."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.mechanism_ladder import (
    InteractionSample,
    closed_loop_grip_jacobian,
    embed_planar_sample,
    mobile_hub_force_shift,
    rotation_matrix,
)
from src.engines.common.jacobian_diagnostics import compute_constraint_diagnostics

pytestmark = pytest.mark.unit


def _sample() -> InteractionSample:
    return InteractionSample(
        model_tier="planar-double-pendulum",
        time_s=0.25,
        frame="world",
        reference_point_m=np.array([0.2, -0.1, 0.3]),
        force_n=np.array([12.0, -7.0, 3.0]),
        couple_nm=np.array([1.0, 2.0, -4.0]),
        linear_velocity_m_s=np.array([0.5, 1.2, -0.4]),
        angular_velocity_rad_s=np.array([2.0, -1.0, 5.0]),
    )


def test_wrench_transport_preserves_power_with_transport_velocity() -> None:
    sample = _sample()
    new_point = np.array([-0.4, 0.3, 0.8])

    moved = sample.transport(new_point)

    assert moved.total_power_w == pytest.approx(sample.total_power_w, abs=1e-12)
    expected_moment = sample.couple_nm - np.cross(
        new_point - sample.reference_point_m, sample.force_n
    )
    assert moved.couple_nm == pytest.approx(expected_moment)


def test_rigid_frame_rotation_preserves_force_moment_and_total_power() -> None:
    sample = _sample()
    transform = rotation_matrix(np.array([1.0, -2.0, 0.5]), 1.17)

    rotated = sample.rotate(transform, frame="rotated")

    assert np.linalg.norm(rotated.force_n) == pytest.approx(
        np.linalg.norm(sample.force_n), abs=1e-12
    )
    assert np.linalg.norm(rotated.couple_nm) == pytest.approx(
        np.linalg.norm(sample.couple_nm), abs=1e-12
    )
    assert rotated.total_power_w == pytest.approx(sample.total_power_w, abs=1e-12)


def test_planar_embedding_matches_scalar_force_and_couple_power() -> None:
    sample = embed_planar_sample(
        model_tier="planar",
        time_s=0.1,
        reference_point_xy_m=np.array([0.3, -0.2]),
        force_xy_n=np.array([20.0, 5.0]),
        couple_z_nm=-3.0,
        linear_velocity_xy_m_s=np.array([1.0, -2.0]),
        angular_velocity_z_rad_s=4.0,
    )

    assert sample.force_power_w == pytest.approx(10.0)
    assert sample.couple_power_w == pytest.approx(-12.0)
    assert sample.total_power_w == pytest.approx(-2.0)


def test_mobile_hub_force_shift_obeys_newtons_second_law() -> None:
    acceleration = np.array([3.0, -1.5, 0.25])

    shift = mobile_hub_force_shift(0.35, acceleration)

    assert shift == pytest.approx(np.array([1.05, -0.525, 0.0875]))


def test_closed_loop_grip_jacobian_has_expected_rank_and_nullspace() -> None:
    jacobian = closed_loop_grip_jacobian(
        lead_angle_rad=-0.8,
        trail_angle_rad=-0.5,
        grip_angle_rad=0.25,
        lead_arm_length_m=0.75,
        trail_arm_length_m=0.78,
        grip_separation_m=0.25,
    )

    diagnostics = compute_constraint_diagnostics(jacobian, expected_dof=1)

    assert jacobian.shape == (4, 5)
    assert diagnostics.constraint_rank == 4
    assert diagnostics.nullspace_dim == 1
    assert not diagnostics.is_overconstrained


@pytest.mark.parametrize(
    ("axis", "angle"),
    [([1.0, 0.0, 0.0], 0.4), ([0.0, 1.0, 1.0], -0.7), ([1.0, 2.0, 3.0], 2.1)],
)
def test_rotation_matrix_is_proper_orthogonal(axis: list[float], angle: float) -> None:
    transform = rotation_matrix(np.array(axis), angle)

    assert transform.T @ transform == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(transform) == pytest.approx(1.0, abs=1e-12)


def test_interaction_sample_rejects_nonfinite_vectors() -> None:
    with pytest.raises(ValueError, match="finite"):
        InteractionSample(
            model_tier="invalid",
            time_s=0.0,
            frame="world",
            reference_point_m=np.zeros(3),
            force_n=np.array([np.nan, 0.0, 0.0]),
            couple_nm=np.zeros(3),
            linear_velocity_m_s=np.zeros(3),
            angular_velocity_rad_s=np.zeros(3),
        )
