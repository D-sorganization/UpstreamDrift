from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_shaft import (
    ArticulatedShaftConfig,
    augmented_mass_matrix,
    build_articulated_shaft,
    shaft_state_energy,
)
from scripts.research.proximal_distal_energy.spatial_full_body import mass_matrix
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)


def _model():
    return build_subject_scaled_model(default_synthetic_profiles()[0])[0]


def test_articulated_shaft_contract_fails_closed() -> None:
    with pytest.raises(ValueError, match="activation"):
        ArticulatedShaftConfig(activation="all")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="poisson_ratio"):
        ArticulatedShaftConfig(poisson_ratio=0.5)
    with pytest.raises(ValueError, match="shaft_length_m"):
        ArticulatedShaftConfig(shaft_length_m=0.0)
    with pytest.raises(ValueError, match="bending_frequency_scale"):
        ArticulatedShaftConfig(bending_frequency_scale=0.0)
    with pytest.raises(ValueError, match="torsional_stiffness_scale"):
        ArticulatedShaftConfig(torsional_stiffness_scale=-1.0)


def test_modal_and_torsional_properties_are_positive_and_traceable() -> None:
    properties = build_articulated_shaft(_model(), ArticulatedShaftConfig())

    assert properties.active_labels == ("bend_x", "bend_y", "torsion")
    assert properties.bending_frequency_hz > 0.0
    assert properties.torsion_frequency_hz > properties.bending_frequency_hz
    assert properties.fe_bending_frequency_relative_error < 1.0e-12
    assert (
        properties.calibration_status == "synthetic_reference_not_equipment_calibrated"
    )
    np.testing.assert_allclose(properties.elastic_mass, properties.elastic_mass.T)
    np.testing.assert_allclose(
        properties.elastic_stiffness, properties.elastic_stiffness.T
    )
    assert np.min(np.linalg.eigvalsh(properties.elastic_mass)) > 0.0
    assert np.min(np.linalg.eigvalsh(properties.elastic_stiffness)) > 0.0


def test_registered_modal_scales_change_only_the_declared_frequencies() -> None:
    model = _model()
    nominal = build_articulated_shaft(model)
    scaled = build_articulated_shaft(
        model,
        ArticulatedShaftConfig(
            bending_frequency_scale=0.8,
            torsional_stiffness_scale=1.44,
        ),
    )

    assert scaled.bending_frequency_hz == pytest.approx(
        0.8 * nominal.bending_frequency_hz
    )
    assert scaled.torsion_frequency_hz == pytest.approx(
        1.2 * nominal.torsion_frequency_hz
    )
    np.testing.assert_allclose(scaled.elastic_mass, nominal.elastic_mass)


@pytest.mark.parametrize(
    ("activation", "labels"),
    [
        ("rigid", ()),
        ("bending", ("bend_x", "bend_y")),
        ("torsion", ("torsion",)),
        ("coupled", ("bend_x", "bend_y", "torsion")),
    ],
)
def test_activation_selects_only_declared_elastic_coordinates(
    activation: str, labels: tuple[str, ...]
) -> None:
    properties = build_articulated_shaft(
        _model(),
        ArticulatedShaftConfig(activation=activation),  # type: ignore[arg-type]
    )
    assert properties.active_labels == labels
    assert properties.coordinate_count == len(labels)


def test_augmented_mass_is_symmetric_positive_and_rigid_reduces_exactly() -> None:
    model = _model()
    q = np.zeros(model.nq)
    rigid = mass_matrix(model, q)
    rigid_properties = build_articulated_shaft(
        model, ArticulatedShaftConfig(activation="rigid")
    )
    coupled = build_articulated_shaft(model, ArticulatedShaftConfig())

    np.testing.assert_array_equal(
        augmented_mass_matrix(model, q, rigid, rigid_properties), rigid
    )
    matrix = augmented_mass_matrix(model, q, rigid, coupled)
    np.testing.assert_allclose(matrix, matrix.T, atol=1.0e-13)
    assert np.min(np.linalg.eigvalsh(matrix)) > 0.0


def test_shaft_energy_closes_to_rigid_value_at_zero_elastic_state() -> None:
    model = _model()
    config = ArticulatedShaftConfig()
    properties = build_articulated_shaft(model, config)
    q = np.zeros(model.nq)
    qd = np.linspace(-0.1, 0.1, model.nq)
    energy = shaft_state_energy(
        model,
        q,
        qd,
        np.zeros(properties.coordinate_count),
        np.zeros(properties.coordinate_count),
        properties,
    )
    assert energy.elastic_kinetic_j == pytest.approx(0.0, abs=1.0e-15)
    assert energy.elastic_strain_j == pytest.approx(0.0, abs=1.0e-15)
    assert energy.extra_gravitational_j == pytest.approx(0.0, abs=1.0e-15)

    undamped = build_articulated_shaft(model, replace(config, damping_ratio=0.0))
    assert np.all(undamped.elastic_damping == 0.0)
