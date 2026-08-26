"""Analytical contracts for the double-pendulum identifiability boundary."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.double_pendulum_identifiability import (
    BASE_COEFFICIENT_NAMES,
    PHYSICAL_PARAMETER_NAMES,
    CoefficientScaleContract,
    DoublePendulumPhysicalParameters,
    coefficient_uncertainty_lower_bound,
    exact_invariance_counterexamples,
    inverse_dynamics_regressor,
    nondimensional_regressor,
    parameter_map_jacobian,
    physical_parameter_rank_witness,
    stacked_inverse_dynamics_regressor,
)
from src.shared.python.simulation_backends import GolfModelParams, make_backend

pytestmark = pytest.mark.scientific


def _unit_scales() -> CoefficientScaleContract:
    return CoefficientScaleContract(
        coefficient_scales=(1.0,) * len(BASE_COEFFICIENT_NAMES),
        torque_scale_nm=1.0,
    )


def test_base_regressor_reconstructs_canonical_inverse_dynamics() -> None:
    model = GolfModelParams.default()
    physical = DoublePendulumPhysicalParameters.from_model(model)
    backend = make_backend("ode", model)
    cases = (
        (np.array([-0.8, 1.1]), np.array([2.5, -1.2]), np.array([4.0, -7.0])),
        (np.array([-1.7, -0.4]), np.array([7.0, 3.2]), np.array([-12.0, 9.0])),
        (np.array([0.3, 0.9]), np.array([-4.0, 5.0]), np.array([3.0, 2.0])),
    )

    for q, v, acceleration in cases:
        expected = backend.mass_matrix(q) @ acceleration + backend.bias_forces(q, v)
        reconstructed = (
            inverse_dynamics_regressor(q, v, acceleration)
            @ physical.base_coefficients()
        )
        np.testing.assert_allclose(reconstructed, expected, rtol=1e-12, atol=1e-12)


def test_physical_parameter_map_has_exact_rank_deficit() -> None:
    physical = DoublePendulumPhysicalParameters.from_model(GolfModelParams.default())
    jacobian = parameter_map_jacobian(physical)

    assert jacobian.shape == (
        len(BASE_COEFFICIENT_NAMES),
        len(PHYSICAL_PARAMETER_NAMES),
    )
    assert np.linalg.matrix_rank(jacobian, tol=1e-10) == 7
    assert jacobian.shape[1] - np.linalg.matrix_rank(jacobian, tol=1e-10) == 4


def test_physical_parameter_rank_has_nonzero_analytical_minor() -> None:
    physical = DoublePendulumPhysicalParameters.from_model(GolfModelParams.default())
    jacobian = parameter_map_jacobian(physical)
    witness = physical_parameter_rank_witness(physical)
    columns = [
        PHYSICAL_PARAMETER_NAMES.index(name) for name in witness.parameter_columns
    ]

    assert witness.closed_form == "m2^2 * r1 * r2 * (g*cos(phi))^2"
    assert witness.determinant > 0.0
    assert abs(np.linalg.det(jacobian[:, columns])) == pytest.approx(
        witness.determinant, rel=1e-12
    )


def test_physical_parameter_map_jacobian_matches_finite_differences() -> None:
    physical = DoublePendulumPhysicalParameters.from_model(GolfModelParams.default())
    analytical = parameter_map_jacobian(physical)

    numerical = np.zeros_like(analytical)
    for column, name in enumerate(PHYSICAL_PARAMETER_NAMES):
        value = getattr(physical, name)
        step = 1e-6 * max(1.0, abs(value))
        lower = replace(physical, **{name: value - step})
        upper = replace(physical, **{name: value + step})
        numerical[:, column] = (
            upper.base_coefficients() - lower.base_coefficients()
        ) / (2.0 * step)

    np.testing.assert_allclose(analytical, numerical, rtol=1e-7, atol=1e-8)


def test_exact_counterexamples_preserve_all_base_coefficients() -> None:
    baseline = DoublePendulumPhysicalParameters.from_model(GolfModelParams.default())
    alternatives = exact_invariance_counterexamples(baseline)

    assert set(alternatives) == {
        "gravity_plane_tradeoff",
        "lower_mass_com_coupling",
        "upper_mass_com_tradeoff",
    }
    for alternative in alternatives.values():
        assert alternative != baseline
        np.testing.assert_allclose(
            alternative.base_coefficients(),
            baseline.base_coefficients(),
            rtol=1e-12,
            atol=1e-12,
        )


def test_stacked_regressor_rejects_inconsistent_sample_shapes() -> None:
    with pytest.raises(ValueError, match=r"matching \(n, 2\) shapes"):
        stacked_inverse_dynamics_regressor(
            np.zeros((2, 2)), np.zeros((3, 2)), np.zeros((2, 2))
        )


def test_coefficient_uncertainty_bound_has_declared_closed_form_scaling() -> None:
    coefficient_count = len(BASE_COEFFICIENT_NAMES)
    regressor = np.tile(np.eye(coefficient_count), (4, 1))
    coefficients = np.arange(1.0, coefficient_count + 1.0)

    low_noise = coefficient_uncertainty_lower_bound(
        regressor, coefficients, scales=_unit_scales(), torque_noise_sd_nm=0.5
    )
    high_noise = coefficient_uncertainty_lower_bound(
        regressor, coefficients, scales=_unit_scales(), torque_noise_sd_nm=1.0
    )

    assert low_noise.full_rank is True
    np.testing.assert_allclose(low_noise.standard_errors, 0.25)
    np.testing.assert_allclose(
        high_noise.ci95_relative_half_widths,
        2.0 * np.asarray(low_noise.ci95_relative_half_widths),
    )
    assert low_noise.max_abs_parameter_correlation == pytest.approx(0.0)


def test_coefficient_uncertainty_bound_fails_closed_without_rank() -> None:
    diagnostic = coefficient_uncertainty_lower_bound(
        np.zeros((20, len(BASE_COEFFICIENT_NAMES))),
        np.ones(len(BASE_COEFFICIENT_NAMES)),
        scales=_unit_scales(),
        torque_noise_sd_nm=1.0,
    )

    assert diagnostic.full_rank is False
    assert diagnostic.standard_errors is None
    assert diagnostic.ci95_relative_half_widths is None
    assert diagnostic.max_abs_parameter_correlation is None


@pytest.mark.parametrize("noise", [-1.0, 0.0, float("nan")])
def test_coefficient_uncertainty_bound_rejects_invalid_noise(noise: float) -> None:
    with pytest.raises(ValueError, match="torque_noise_sd_nm must be positive"):
        coefficient_uncertainty_lower_bound(
            np.eye(len(BASE_COEFFICIENT_NAMES)),
            np.ones(len(BASE_COEFFICIENT_NAMES)),
            scales=_unit_scales(),
            torque_noise_sd_nm=noise,
        )


def test_nondimensional_regressor_is_invariant_to_coefficient_units() -> None:
    regressor = np.arange(1.0, 22.0).reshape(3, len(BASE_COEFFICIENT_NAMES))
    reference_scales = np.arange(1.0, 8.0)
    conversion = np.array([1e3, 1e3, 1e3, 1e2, 1e2, 60.0, 60.0])
    reference = CoefficientScaleContract(
        coefficient_scales=tuple(reference_scales), torque_scale_nm=50.0
    )
    converted = CoefficientScaleContract(
        coefficient_scales=tuple(reference_scales * conversion),
        torque_scale_nm=50.0,
    )

    np.testing.assert_allclose(
        nondimensional_regressor(regressor, reference),
        nondimensional_regressor(regressor / conversion, converted),
        rtol=0.0,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    ("coefficient_scales", "torque_scale"),
    [((1.0,) * 6, 1.0), ((1.0,) * 6 + (0.0,), 1.0), ((1.0,) * 7, 0.0)],
)
def test_coefficient_scale_contract_fails_closed(
    coefficient_scales: tuple[float, ...], torque_scale: float
) -> None:
    with pytest.raises(ValueError, match="positive"):
        CoefficientScaleContract(
            coefficient_scales=coefficient_scales, torque_scale_nm=torque_scale
        )
