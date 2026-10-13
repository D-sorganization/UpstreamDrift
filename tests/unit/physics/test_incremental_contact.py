"""Energetic restitution and incremental friction contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.shared.python.core.physics_constants import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_RADIUS_M,
)
from src.shared.python.physics.incremental_contact import (
    contact_backend_info,
    solve_incremental_contact,
)
from src.shared.python.physics.rigid_contact import RigidContactInput

pytestmark = pytest.mark.unit


def _state(**changes) -> RigidContactInput:
    base = RigidContactInput(
        club_velocity_mps=np.array([45.0, 0.0, 0.0]),
        club_angular_velocity_rad_s=np.zeros(3),
        ball_velocity_mps=np.zeros(3),
        ball_angular_velocity_rad_s=np.zeros(3),
        normal=np.array([1.0, 0.0, 0.0]),
        club_lever_m=np.array([0.04, 0.0, 0.0]),
        club_inertia_world_kg_m2=np.diag([0.00012, 0.00018, 0.00020]),
        club_mass_kg=0.2,
        restitution=0.8,
        friction_coefficient=0.4,
    )
    return replace(base, **changes)


@pytest.mark.parametrize("restitution", [0.0, 0.4, 0.8, 1.0])
def test_central_frictionless_recovers_reduced_mass_and_velocity_cor(
    restitution,
) -> None:
    state = _state(restitution=restitution, friction_coefficient=0.0)
    result = solve_incremental_contact(
        state, energetic_restitution=restitution, steps_per_compression=32
    )
    expected_j = (1 + restitution) * 45.0 / (1 / 0.2 + 1 / GOLF_BALL_MASS_KG)
    assert result.ball_impulse_n_s[0] == pytest.approx(expected_j, abs=1e-8)
    assert result.post_relative_normal_mps == pytest.approx(-restitution * 45, abs=1e-8)
    assert result.post_kinetic_energy_j <= result.pre_kinetic_energy_j + 1e-8
    assert result.contact_law == "incremental_coulomb_energetic"


def test_offcenter_contact_conserves_momentum_and_remains_passive() -> None:
    state = _state(
        club_velocity_mps=np.array([45.0, -3.0, -1.0]),
        club_angular_velocity_rad_s=np.array([1.0, -2.0, 3.0]),
        club_lever_m=np.array([0.04, 0.02, 0.01]),
        restitution=1.0,
    )
    result = solve_incremental_contact(
        state, energetic_restitution=1.0, steps_per_compression=128
    )
    before_linear = (
        state.club_mass_kg * state.club_velocity_mps
        + GOLF_BALL_MASS_KG * state.ball_velocity_mps
    )
    after_linear = (
        state.club_mass_kg * result.club_velocity_mps
        + GOLF_BALL_MASS_KG * result.ball_velocity_mps
    )
    np.testing.assert_allclose(after_linear, before_linear, atol=1e-10)
    center = state.club_lever_m + GOLF_BALL_RADIUS_M * state.normal
    ball_i = 0.4 * GOLF_BALL_MASS_KG * GOLF_BALL_RADIUS_M**2
    before_angular = (
        state.club_inertia_world_kg_m2 @ state.club_angular_velocity_rad_s
        + ball_i * state.ball_angular_velocity_rad_s
        + np.cross(center, GOLF_BALL_MASS_KG * state.ball_velocity_mps)
    )
    after_angular = (
        state.club_inertia_world_kg_m2 @ result.club_angular_velocity_rad_s
        + ball_i * result.ball_angular_velocity_rad_s
        + np.cross(center, GOLF_BALL_MASS_KG * result.ball_velocity_mps)
    )
    np.testing.assert_allclose(after_angular, before_angular, atol=1e-10)
    assert result.post_kinetic_energy_j <= result.pre_kinetic_energy_j + 1e-7
    assert result.compression_work_j > 0
    assert result.rebound_work_j == pytest.approx(result.compression_work_j, rel=1e-7)


def test_step_refinement_converges_for_oblique_contact() -> None:
    state = _state(
        club_velocity_mps=np.array([45.0, 3.0, 0.0]),
        club_lever_m=np.array([0.04, 0.02, 0.01]),
        friction_coefficient=0.2,
    )
    coarse = solve_incremental_contact(
        state, energetic_restitution=0.8, steps_per_compression=64
    )
    fine = solve_incremental_contact(
        state, energetic_restitution=0.8, steps_per_compression=128
    )
    finer = solve_incremental_contact(
        state, energetic_restitution=0.8, steps_per_compression=256
    )
    first_error = np.linalg.norm(coarse.ball_velocity_mps - fine.ball_velocity_mps)
    second_error = np.linalg.norm(fine.ball_velocity_mps - finer.ball_velocity_mps)
    assert second_error < first_error
    assert second_error < 0.05


def test_incompatible_restitution_argument_rejected() -> None:
    with pytest.raises(ValueError, match="energetic restitution"):
        solve_incremental_contact(_state(), energetic_restitution=0.5)


def test_rotation_covariance_with_offcenter_spin() -> None:
    state = _state(
        club_velocity_mps=np.array([45.0, 2.0, -1.0]),
        club_angular_velocity_rad_s=np.array([1.0, -2.0, 3.0]),
        ball_angular_velocity_rad_s=np.array([5.0, 6.0, -4.0]),
        club_lever_m=np.array([0.04, 0.02, 0.01]),
        friction_coefficient=0.2,
    )
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed = replace(
        state,
        club_velocity_mps=rotation @ state.club_velocity_mps,
        club_angular_velocity_rad_s=rotation @ state.club_angular_velocity_rad_s,
        ball_velocity_mps=rotation @ state.ball_velocity_mps,
        ball_angular_velocity_rad_s=rotation @ state.ball_angular_velocity_rad_s,
        normal=rotation @ state.normal,
        club_lever_m=rotation @ state.club_lever_m,
        club_inertia_world_kg_m2=rotation @ state.club_inertia_world_kg_m2 @ rotation.T,
    )
    original = solve_incremental_contact(
        state, energetic_restitution=0.8, steps_per_compression=128
    )
    rotated = solve_incremental_contact(
        transformed, energetic_restitution=0.8, steps_per_compression=128
    )
    for field in (
        "club_velocity_mps",
        "club_angular_velocity_rad_s",
        "ball_velocity_mps",
        "ball_angular_velocity_rad_s",
        "ball_impulse_n_s",
    ):
        np.testing.assert_allclose(
            getattr(rotated, field), rotation @ getattr(original, field), atol=1e-8
        )


def test_proven_center_fastpath_matches_refined_incremental_result() -> None:
    state = _state(
        club_velocity_mps=np.array([45.0, 5.0, -3.0]),
        club_lever_m=np.array([0.04, 0.0, 0.0]),
        friction_coefficient=0.4,
    )
    exact = solve_incremental_contact(
        state, energetic_restitution=0.8, steps_per_compression=128
    )
    integrated = solve_incremental_contact(
        state,
        energetic_restitution=0.8,
        steps_per_compression=256,
        use_analytic_fastpath=False,
    )
    assert exact.fastpath_used
    assert not integrated.fastpath_used
    np.testing.assert_allclose(
        exact.ball_velocity_mps, integrated.ball_velocity_mps, atol=1e-6
    )
    np.testing.assert_allclose(
        exact.ball_angular_velocity_rad_s,
        integrated.ball_angular_velocity_rad_s,
        atol=1e-6,
    )


def test_frictionless_offcenter_fastpath_preserves_energetic_work() -> None:
    state = _state(club_lever_m=np.array([0.04, 0.02, 0.01]), friction_coefficient=0.0)
    exact = solve_incremental_contact(
        state, energetic_restitution=0.8, steps_per_compression=64
    )
    integrated = solve_incremental_contact(
        state,
        energetic_restitution=0.8,
        steps_per_compression=256,
        use_analytic_fastpath=False,
    )
    assert exact.fastpath_used
    np.testing.assert_allclose(
        exact.ball_impulse_n_s, integrated.ball_impulse_n_s, atol=1e-6
    )
    assert exact.rebound_work_j == pytest.approx(
        0.64 * exact.compression_work_j, rel=1e-10
    )


def test_compiled_kernel_matches_python_reference_for_offcenter_cases() -> None:
    pytest.importorskip("numba")
    rng = np.random.default_rng(90210)
    for _ in range(40):
        state = _state(
            club_velocity_mps=np.array([45.0, *rng.uniform(-8, 8, 2)]),
            club_angular_velocity_rad_s=rng.uniform(-4, 4, 3),
            ball_angular_velocity_rad_s=rng.uniform(-100, 100, 3),
            club_lever_m=np.array([0.04, *rng.uniform(-0.02, 0.02, 2)]),
            friction_coefficient=float(rng.choice([0.05, 0.2, 0.4, 1.0])),
        )
        reference = solve_incremental_contact(
            state,
            energetic_restitution=0.8,
            steps_per_compression=128,
            use_analytic_fastpath=False,
            backend="python_reference",
        )
        compiled = solve_incremental_contact(
            state,
            energetic_restitution=0.8,
            steps_per_compression=128,
            use_analytic_fastpath=False,
            backend="numba",
        )
        np.testing.assert_allclose(
            compiled.ball_impulse_n_s, reference.ball_impulse_n_s, atol=1e-9
        )
        np.testing.assert_allclose(
            compiled.ball_velocity_mps, reference.ball_velocity_mps, atol=1e-8
        )
        assert compiled.compression_work_j == pytest.approx(
            reference.compression_work_j, rel=1e-10
        )
        assert compiled.rebound_work_j == pytest.approx(
            reference.rebound_work_j, rel=1e-10
        )
    info = contact_backend_info()
    assert info["active_backend"] == "numba"
    assert info["fastmath"] is False
