"""Pure tests for the offline retained activation-law diagnostic."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from scripts.opensim.analyze_retained_activation_residual import (
    _analysis_method,
    _hermite_simpson_defects,
    _full_cell_allocation,
    _integrate_channel,
    _validate_saved_time_grid,
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


def test_nonuniform_twenty_one_cell_grid_reports_actual_widths_and_midpoints() -> None:
    normalized_mesh = np.concatenate(([0.0, 0.025], np.arange(0.05, 1.0001, 0.05)))
    assert len(normalized_mesh) - 1 == 21
    times = np.empty(43, dtype=np.float64)
    times[::2] = normalized_mesh * 0.01
    times[1::2] = (times[:-2:2] + times[2::2]) / 2.0

    method = _analysis_method(
        {
            "mesh_intervals": 21,
            "transcription_scheme": "hermite-simpson",
            "replay_control_interpolation": "piecewise_linear",
        },
        times,
    )

    assert method["trajectory_row_count"] == 43
    assert method["collocation_cell_count"] == 21
    widths = method["saved_collocation_cell_widths_seconds"]
    assert widths[0] == pytest.approx(0.00025)
    assert widths[1] == pytest.approx(0.00025)
    assert widths[2] == pytest.approx(0.0005)
    assert method["mean_collocation_cell_duration_seconds"] == pytest.approx(0.01 / 21)
    np.testing.assert_array_equal(
        method["saved_collocation_midpoint_times_seconds"], times[1::2]
    )


def test_nonuniform_hermite_simpson_uses_each_exact_cell_width() -> None:
    times = np.array([0.0, 0.000125, 0.00025, 0.0005, 0.00075])
    activation = np.full(5, 0.2)
    excitation = np.full(5, 0.2)
    defects = _hermite_simpson_defects(times, activation, excitation, PARAMETERS)
    np.testing.assert_array_equal(defects["midpoint_state"], np.zeros(2))
    np.testing.assert_array_equal(defects["simpson_dynamics"], np.zeros(2))


@pytest.mark.parametrize(
    "times, mesh_intervals, message",
    [
        (np.array([0.0, np.nan, 1.0]), 1, "finite"),
        (np.array([0.0, 0.5, 0.5]), 1, "strictly increasing"),
        (np.array([0.0, 0.5, 1.0, 1.5]), 1, "triples"),
        (np.array([0.0, 0.25, 0.5, 0.75, 1.0]), 1, "requested mesh"),
    ],
)
def test_saved_grid_rejects_invalid_time_or_mesh_consistency(
    times: np.ndarray, mesh_intervals: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_saved_time_grid(times, mesh_intervals)


def test_retained_refinement_analysis_binds_exact_saved_grid(tmp_path) -> None:
    from scripts.opensim.analyze_retained_activation_residual import analyze

    repository = Path(__file__).resolve().parents[2]
    snapshot = (
        repository
        / "docs/development/feedback_controls/experiments/native_moco_refinement_solve_cecfcd98_root_20261011T080500Z/solution_control_replay_inputs_final"
    )
    audit = (
        repository
        / "docs/development/feedback_controls/experiments/native_moco_refinement_solve_cecfcd98_root_20261011T080500Z/solution_control_replay_diagnostic_3bc787e36d_root/retained_replay/native_audit/fresh_replay_diagnostic.json"
    )
    report = analyze(snapshot, audit, tmp_path / "refinement-analysis.json")

    assert report["replay_status"] == "failed"
    assert report["sample_count"] == report["trajectory_row_count"] == 43
    assert report["interval_count"] == 42
    assert "adjacent saved-row spans" in report["interval_count_semantics"]
    assert report["collocation_cell_count"] == 21
    method = report["method"]
    assert method["collocation_cell_count"] == 21
    assert len(method["saved_collocation_cell_widths_seconds"]) == 21
    assert len(method["saved_collocation_midpoint_times_seconds"]) == 21
    assert "41 trajectory rows" not in method["collocation_row_pattern"]
    assert report["input_sha256"]["replay_audit"] == (
        "dfcb7498c0aecd03928d82cade03c7bf1d867be1c88c44c2c2dd469c917ca97d"
    )
    assert report["input_sha256"]["source_osim"] == (
        "6a098517bdba09b19b3eb99e4da7e795f27d10be8f42c0fca04fee20ce67e143"
    )
    assert (
        report["source_properties"]["flexor"]["effective_from_xml_or_4_6_defaults"][
            "activation_time_constant"
        ]
        == 0.01
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
