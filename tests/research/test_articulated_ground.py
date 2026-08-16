from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_ground import (
    ArticulatedGroundConfig,
    augmented_ground_mass_matrix,
    build_articulated_ground,
    evaluate_ground_wrench,
    ground_extra_potential_gradient,
    ground_mass_increment_coriolis,
    ground_state_energy,
)
from scripts.research.proximal_distal_energy.articulated_shaft import (
    ArticulatedShaftConfig,
    augmented_mass_matrix,
    build_articulated_shaft,
)
from scripts.research.proximal_distal_energy.spatial_full_body import mass_matrix
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)


def _authority():
    model = build_subject_scaled_model(default_synthetic_profiles()[0])[0]
    q = np.zeros(model.nq)
    rigid = mass_matrix(model, q)
    shaft = build_articulated_shaft(model, ArticulatedShaftConfig())
    shaft_mass = augmented_mass_matrix(model, q, rigid, shaft)
    return model, q, shaft, shaft_mass


def test_ground_contract_fails_closed() -> None:
    with pytest.raises(ValueError, match="activation"):
        ArticulatedGroundConfig(activation="rolling")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive pairs"):
        ArticulatedGroundConfig(translation_stiffness_n_m=(0.0, 1.0))
    with pytest.raises(ValueError, match="finite pair"):
        ArticulatedGroundConfig(center_of_pressure_xz_m=(np.nan, 0.0))


@pytest.mark.parametrize(
    ("activation", "labels"),
    [
        ("fixed", ()),
        ("translation", ("base_x", "base_z")),
        ("free_moment", ("base_pitch_y",)),
        ("coupled", ("base_x", "base_z", "base_pitch_y")),
    ],
)
def test_ground_activation_selects_only_declared_pathways(
    activation: str, labels: tuple[str, ...]
) -> None:
    properties = build_articulated_ground(
        ArticulatedGroundConfig(activation=activation)  # type: ignore[arg-type]
    )
    assert properties.active_labels == labels
    assert properties.coordinate_count == len(labels)
    assert properties.stiffness.shape == (len(labels), len(labels))


def test_fixed_base_reduces_exactly_and_finite_base_mass_is_positive() -> None:
    model, q, shaft, shaft_mass = _authority()
    fixed = build_articulated_ground(ArticulatedGroundConfig(activation="fixed"))
    np.testing.assert_array_equal(
        augmented_ground_mass_matrix(model, q, shaft_mass, shaft, fixed),
        shaft_mass,
    )

    coupled = build_articulated_ground()
    matrix = augmented_ground_mass_matrix(model, q, shaft_mass, shaft, coupled)
    np.testing.assert_allclose(matrix, matrix.T, atol=1.0e-13)
    assert matrix.shape == (model.nq + shaft.coordinate_count + 3,) * 2
    assert np.min(np.linalg.eigvalsh(matrix)) > 0.0


def test_ground_wrench_closes_power_and_separates_reference_transport() -> None:
    config = ArticulatedGroundConfig()
    properties = build_articulated_ground(config)
    base = np.array([0.002, -0.001, 0.003])
    velocity = np.array([0.04, -0.02, 0.05])
    wrench = evaluate_ground_wrench(base, velocity, properties)

    assert wrench.intrinsic_free_moment_nm == pytest.approx(
        -config.free_moment_stiffness_nm_rad * base[2]
        - config.free_moment_damping_nm_s_rad * velocity[2]
    )
    assert wrench.damping_power_w <= 0.0
    assert wrench.power_residual_w == pytest.approx(0.0, abs=1.0e-12)

    reversed_cop = build_articulated_ground(
        replace(
            config,
            center_of_pressure_xz_m=(
                -config.center_of_pressure_xz_m[0],
                -config.center_of_pressure_xz_m[1],
            ),
        )
    )
    reversed_wrench = evaluate_ground_wrench(base, velocity, reversed_cop)
    np.testing.assert_array_equal(reversed_wrench.force_n, wrench.force_n)
    assert reversed_wrench.intrinsic_free_moment_nm == wrench.intrinsic_free_moment_nm
    assert reversed_wrench.transported_moment_nm != wrench.transported_moment_nm


def test_zero_base_state_reduces_energy_to_shaft_authority() -> None:
    model, q, shaft, _ = _authority()
    ground = build_articulated_ground()
    qd = np.linspace(-0.05, 0.05, model.nq)
    eta = np.zeros(shaft.coordinate_count)
    energy = ground_state_energy(
        model,
        q,
        qd,
        eta,
        eta,
        np.zeros(ground.coordinate_count),
        np.zeros(ground.coordinate_count),
        shaft,
        ground,
    )
    assert energy.base_kinetic_j == pytest.approx(0.0, abs=1.0e-15)
    assert energy.ground_strain_j == pytest.approx(0.0, abs=1.0e-15)
    assert energy.extra_gravitational_j == pytest.approx(0.0, abs=1.0e-15)
    assert energy.total_mechanical_j == pytest.approx(
        energy.shaft_energy.total_mechanical_j, abs=1.0e-15
    )


def test_ground_bias_and_gravity_gradient_are_finite_and_pathway_separated() -> None:
    model, q, shaft, _ = _authority()
    ground = build_articulated_ground()
    qd = np.linspace(-0.03, 0.03, model.nq)
    eta_dot = np.linspace(-0.01, 0.01, shaft.coordinate_count)
    base_velocity = np.array([0.01, -0.02, 0.03])
    base = np.array([0.001, -0.002, 0.003])

    coriolis = ground_mass_increment_coriolis(
        model, q, qd, eta_dot, base_velocity, shaft, ground
    )
    gradient = ground_extra_potential_gradient(model, q, base, shaft, ground)
    assert coriolis.shape == gradient.shape == (model.nq + shaft.coordinate_count + 3,)
    assert np.all(np.isfinite(coriolis))
    assert np.all(np.isfinite(gradient))
    assert gradient[model.nq + shaft.coordinate_count + 1] > 0.0

    fixed = build_articulated_ground(ArticulatedGroundConfig(activation="fixed"))
    np.testing.assert_array_equal(
        ground_mass_increment_coriolis(
            model, q, qd, eta_dot, np.zeros(0), shaft, fixed
        ),
        np.zeros(model.nq + shaft.coordinate_count),
    )
    np.testing.assert_array_equal(
        ground_extra_potential_gradient(model, q, np.zeros(0), shaft, fixed),
        np.zeros(model.nq + shaft.coordinate_count),
    )
