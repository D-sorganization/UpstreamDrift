"""Pure tests for the offline retained activation-law diagnostic."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.opensim.analyze_retained_activation_residual import (
    _hermite_simpson_defects,
    _full_cell_allocation,
    _integrate_channel,
    activation_derivative,
)


PARAMETERS = {
    "tau_activation": 0.01,
    "tau_deactivation": 0.04,
    "minimum_activation": 0.01,
}

pytestmark = pytest.mark.unit


def test_equal_excitation_uses_zero_rate_deactivation_branch() -> None:
    assert activation_derivative(0.2, 0.2, **PARAMETERS) == 0.0


def test_excitation_strictly_above_activation_uses_activation_time_constant() -> None:
    activation = 0.2
    excitation = np.nextafter(activation, 1.0)
    scale = 0.5 + 1.5 * activation
    expected = (excitation - activation) / (PARAMETERS["tau_activation"] * scale)
    assert (
        activation_derivative(activation, float(excitation), **PARAMETERS) == expected
    )


def test_excitation_below_activation_uses_deactivation_time_constant() -> None:
    activation = 0.2
    excitation = np.nextafter(activation, 0.0)
    scale = 0.5 + 1.5 * activation
    expected = (excitation - activation) / (PARAMETERS["tau_deactivation"] / scale)
    assert (
        activation_derivative(activation, float(excitation), **PARAMETERS) == expected
    )


def test_activation_is_clamped_before_derivative() -> None:
    assert activation_derivative(0.0, 0.01, **PARAMETERS) == 0.0
    assert activation_derivative(1.2, 1.0, **PARAMETERS) == 0.0


def test_constant_excitation_at_initial_activation_remains_constant() -> None:
    times = np.linspace(0.0, 0.01, 11)
    excitations = np.full_like(times, 0.05)
    actual = _integrate_channel(
        times, excitations, 0.05, PARAMETERS, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_array_equal(actual, np.full_like(times, 0.05))


def test_reconstructed_hermite_simpson_defect_is_zero_for_constant_state() -> None:
    times = np.array([0.0, 0.005, 0.01])
    activation = np.array([0.2, 0.2, 0.2])
    excitation = np.array([0.2, 0.2, 0.2])
    defects = _hermite_simpson_defects(times, activation, excitation, PARAMETERS)
    np.testing.assert_array_equal(defects["midpoint_state"], np.array([0.0]))
    np.testing.assert_array_equal(defects["simpson_dynamics"], np.array([0.0]))


def test_hermite_simpson_cell_uses_endpoint_midpoint_endpoint_equations() -> None:
    times = np.array([0.0, 0.0005, 0.001])
    activation = np.array([0.1, 0.1005, 0.101])
    scale = 0.5 + 1.5 * activation
    excitation = activation + PARAMETERS["tau_activation"] * scale
    defects = _hermite_simpson_defects(times, activation, excitation, PARAMETERS)
    assert abs(defects["midpoint_state"][0]) < 1e-16
    assert abs(defects["simpson_dynamics"][0]) < 1e-16


def test_hermite_simpson_defect_requires_actual_midpoint_rows() -> None:
    with pytest.raises(ValueError, match="endpoint/midpoint triples"):
        _hermite_simpson_defects(
            np.array([0.0, 0.01]),
            np.array([0.2, 0.2]),
            np.array([0.2, 0.2]),
            PARAMETERS,
        )


def test_full_cell_allocation_restarts_at_optimized_left_state_and_records_crossing() -> (
    None
):
    times = np.array([0.0, 0.0005, 0.001])
    optimized = np.array([0.2, 0.2001, 0.2002])
    excitation = np.array([0.1, 0.3, 0.1])
    result = _full_cell_allocation(times, optimized, excitation, PARAMETERS)
    assert result["cell_count"] == 1
    local_values = result["local_scalar_activation_endpoint_midpoint_endpoint"]
    assert local_values[0][0] == optimized[0]
    assert result["local_tolerance_sensitivity_midpoint_endpoint"][0][0] < 1e-12
    crossings = result["sampled_chord_branch_crossings"]
    assert len(crossings) == 2
    assert crossings[0]["classification"] == "linear sign change estimate"


def test_nonfinite_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        activation_derivative(float("nan"), 0.1, **PARAMETERS)


def test_nonfinite_activation_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="parameters must be finite"):
        activation_derivative(
            0.1,
            0.2,
            tau_activation=float("inf"),
            tau_deactivation=0.04,
            minimum_activation=0.01,
        )
