"""Machine-readable payload builders for identifiability evidence."""

from __future__ import annotations

import numpy as np

from scripts.research.proximal_distal_energy.double_pendulum_identifiability import (
    BASE_COEFFICIENT_NAMES,
    BASE_COEFFICIENT_UNITS,
    PHYSICAL_PARAMETER_NAMES,
    CoefficientScaleContract,
    CoefficientUncertaintyLowerBound,
    DoublePendulumPhysicalParameters,
    coefficient_uncertainty_lower_bound,
    exact_invariance_counterexamples,
    nondimensional_regressor,
)
from scripts.research.proximal_distal_energy.double_pendulum_identifiability_contract import (
    COEFFICIENT_UNIT_CONVERSION_FIXTURE,
    COUNTEREXAMPLE_NAMES,
    RANK_TOLERANCE,
    REFERENCE_NOISE_SD_NM,
    TORQUE_NOISE_LEVELS_NM,
)
from scripts.research.proximal_distal_energy.local_linear_diagnostics import (
    RankDiagnostic,
    rank_diagnostic,
)


def rank_payload(diagnostic: RankDiagnostic) -> dict[str, object]:
    """Serialize a rank decision without recomputing it after rounding."""
    return {
        "full_rank": diagnostic.full_rank,
        "matrix_shape": list(diagnostic.matrix_shape),
        "rank": diagnostic.rank,
        "retained_condition_number": diagnostic.retained_condition_number,
        "singular_values": list(diagnostic.singular_values),
        "smallest_retained": diagnostic.smallest_retained,
        "threshold": diagnostic.threshold,
    }


def parameter_record(
    parameters: DoublePendulumPhysicalParameters,
) -> dict[str, float]:
    """Serialize physical parameters in their registered order."""
    return {
        name: float(value)
        for name, value in zip(
            PHYSICAL_PARAMETER_NAMES, parameters.vector(), strict=True
        )
    }


def counterexample_payload(
    baseline: DoublePendulumPhysicalParameters,
) -> list[dict[str, object]]:
    """Serialize the three exact parameter non-uniqueness families."""
    reference = baseline.base_coefficients()
    alternatives = exact_invariance_counterexamples(baseline)
    records: list[dict[str, object]] = []
    for name in COUNTEREXAMPLE_NAMES:
        alternative = alternatives[name]
        delta = alternative.vector() - baseline.vector()
        records.append(
            {
                "alternative_parameters": parameter_record(alternative),
                "base_coefficient_max_abs_difference": float(
                    np.max(np.abs(alternative.base_coefficients() - reference))
                ),
                "changed_parameters": {
                    parameter_name: float(change)
                    for parameter_name, change in zip(
                        PHYSICAL_PARAMETER_NAMES, delta, strict=True
                    )
                    if change != 0.0
                },
                "name": name,
            }
        )
    return records


def cumulative_rank_payload(regressor: np.ndarray) -> list[dict[str, object]]:
    """Return registered finite-window rank diagnostics."""
    records: list[dict[str, object]] = []
    sample_count = regressor.shape[0] // 2
    for fraction in (0.10, 0.30, 0.70, 1.00):
        retained_samples = max(1, int(np.ceil(sample_count * fraction)))
        diagnostic = rank_diagnostic(regressor[: 2 * retained_samples], RANK_TOLERANCE)
        records.append(
            {
                "fraction": fraction,
                "rank": diagnostic.rank,
                "retained_condition_number": diagnostic.retained_condition_number,
                "sample_count": retained_samples,
            }
        )
    return records


def _uncertainty_payload(
    diagnostic: CoefficientUncertaintyLowerBound,
) -> dict[str, object]:
    standard_errors = diagnostic.standard_errors
    relative_bounds = diagnostic.ci95_relative_half_widths
    worst_index = (
        int(np.argmax(relative_bounds)) if relative_bounds is not None else None
    )
    worst_relative_bound = (
        float(relative_bounds[worst_index])
        if relative_bounds is not None and worst_index is not None
        else None
    )
    return {
        "ci95_relative_half_widths": (
            dict(zip(BASE_COEFFICIENT_NAMES, relative_bounds, strict=True))
            if relative_bounds is not None
            else None
        ),
        "coefficient_standard_errors": (
            dict(zip(BASE_COEFFICIENT_NAMES, standard_errors, strict=True))
            if standard_errors is not None
            else None
        ),
        "full_rank": diagnostic.full_rank,
        "dimensionless_retained_condition_number": (
            diagnostic.dimensionless_retained_condition_number
        ),
        "dimensionless_singular_values": list(diagnostic.dimensionless_singular_values),
        "matrix_shape": list(diagnostic.matrix_shape),
        "max_abs_parameter_correlation": diagnostic.max_abs_parameter_correlation,
        "rank": diagnostic.rank,
        "torque_noise_sd_nm": diagnostic.torque_noise_sd_nm,
        "worst_ci95_relative_half_width": worst_relative_bound,
        "worst_conditioned_coefficient": (
            BASE_COEFFICIENT_NAMES[worst_index] if worst_index is not None else None
        ),
    }


def noise_aware_lower_bound_payload(
    regressor: np.ndarray,
    coefficients: np.ndarray,
    scales: CoefficientScaleContract,
) -> dict[str, object]:
    """Serialize noise and finite-window adverse cases for the oracle bound."""
    cases = [
        _uncertainty_payload(
            coefficient_uncertainty_lower_bound(
                regressor,
                coefficients,
                scales=scales,
                torque_noise_sd_nm=noise_sd,
                absolute_tolerance=RANK_TOLERANCE.absolute,
                relative_tolerance=RANK_TOLERANCE.relative,
            )
        )
        for noise_sd in TORQUE_NOISE_LEVELS_NM
    ]
    sample_count = regressor.shape[0] // 2
    windows: list[dict[str, object]] = []
    for fraction in (0.10, 0.30, 0.70, 1.00):
        retained_samples = max(1, int(np.ceil(sample_count * fraction)))
        payload = _uncertainty_payload(
            coefficient_uncertainty_lower_bound(
                regressor[: 2 * retained_samples],
                coefficients,
                scales=scales,
                torque_noise_sd_nm=REFERENCE_NOISE_SD_NM,
                absolute_tolerance=RANK_TOLERANCE.absolute,
                relative_tolerance=RANK_TOLERANCE.relative,
            )
        )
        windows.append(
            {"fraction": fraction, "sample_count": retained_samples, **payload}
        )
    return {
        "assumptions": [
            "positions, velocities, and accelerations are exact",
            "the analytical model form and event alignment are exact",
            "generalized-torque noise is independent, Gaussian, homoscedastic, and known",
            "the bound applies only to the seven base coefficients",
        ],
        "classification": "oracle_kinematics_fisher_lower_bound",
        "confidence_level": 0.95,
        "full_record_cases": cases,
        "inference_boundary": (
            "These conditional Gaussian 95% half-widths are Cramer-Rao-style "
            "best-case lower bounds. They omit kinematic differentiation noise, "
            "correlated sensor errors, model discrepancy, event-time uncertainty, "
            "priors, repeated participants, and held-out data; therefore they "
            "cannot establish practical or participant identifiability."
        ),
        "reference_window_noise_sd_nm": REFERENCE_NOISE_SD_NM,
        "status": "conditional_lower_bound_only",
        "window_cases": windows,
    }


def registered_scales(
    coefficients: np.ndarray, controls: np.ndarray
) -> CoefficientScaleContract:
    """Derive positive coefficient and torque coordinates from the fixture."""
    values = np.abs(np.asarray(coefficients, dtype=float))
    return CoefficientScaleContract(
        coefficient_scales=tuple(float(value) for value in values),
        torque_scale_nm=max(float(np.max(np.abs(controls))), 1.0),
    )


def scale_contract_payload(scales: CoefficientScaleContract) -> dict[str, object]:
    """Serialize dimensional coordinates and their dimensionless transform."""
    return {
        "coefficient_coordinates": list(BASE_COEFFICIENT_NAMES),
        "coefficient_scale_derivation": "absolute registered base-coefficient values",
        "coefficient_scales": list(scales.coefficient_scales),
        "coefficient_units": list(BASE_COEFFICIENT_UNITS),
        "dimensionless_regressor_equation": "Y_bar = Y * diag(s_theta) / s_tau",
        "rank_basis": "nondimensional_base_coefficient_regressor",
        "torque_coordinate": "generalized_torque",
        "torque_scale_derivation": (
            "maximum absolute registered generalized torque with 1 N*m floor"
        ),
        "torque_scale_nm": scales.torque_scale_nm,
        "torque_unit": "N*m",
    }


def scale_sensitivity_payload(
    regressor: np.ndarray, scales: CoefficientScaleContract
) -> list[dict[str, object]]:
    """Retain rank decisions under three declared positive scale choices."""
    factors = {
        "registered": np.ones(len(BASE_COEFFICIENT_NAMES)),
        "alternating_half_double": np.array([0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 0.5]),
        "alternating_double_half": np.array([2.0, 0.5, 2.0, 0.5, 2.0, 0.5, 2.0]),
    }
    records: list[dict[str, object]] = []
    for name, factor in factors.items():
        trial = CoefficientScaleContract(
            coefficient_scales=tuple(
                float(value) for value in scales.coefficient_array() * factor
            ),
            torque_scale_nm=scales.torque_scale_nm,
        )
        diagnostic = rank_diagnostic(
            nondimensional_regressor(regressor, trial), RANK_TOLERANCE
        )
        records.append(
            {
                "coefficient_scale_factors": list(factor),
                "name": name,
                **rank_payload(diagnostic),
            }
        )
    return records


def unit_invariance_payload(
    regressor: np.ndarray, scales: CoefficientScaleContract
) -> dict[str, object]:
    """Verify equivalent coefficient units preserve the transformed matrix."""
    conversion = np.asarray(COEFFICIENT_UNIT_CONVERSION_FIXTURE, dtype=float)
    reference = nondimensional_regressor(regressor, scales)
    converted_scales = CoefficientScaleContract(
        coefficient_scales=tuple(
            float(value) for value in scales.coefficient_array() * conversion
        ),
        torque_scale_nm=scales.torque_scale_nm,
    )
    converted = nondimensional_regressor(regressor / conversion, converted_scales)
    return {
        "coefficient_coordinate_conversion_factors": list(conversion),
        "fixture": "equivalent_coefficient_units",
        "max_abs_dimensionless_regressor_difference": float(
            np.max(np.abs(reference - converted))
        ),
        "rank_after_conversion": rank_diagnostic(converted, RANK_TOLERANCE).rank,
        "rank_before_conversion": rank_diagnostic(reference, RANK_TOLERANCE).rank,
    }
