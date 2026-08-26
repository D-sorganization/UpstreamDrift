"""Finite-time and event-sensitivity contracts for issue #9116."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.phase_event_stability import (
    StateScales,
    direct_transition_control,
    event_time_sensitivity,
    finite_time_spectra,
    first_positive_crossing,
    normalized_transition,
    periodicity_gate,
    propagate_state_transition,
    saltation_matrix,
)
from src.shared.python.simulation_backends import GolfModelParams

pytestmark = pytest.mark.unit


def test_state_scales_reject_nonpositive_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="four finite positive"):
        StateScales((1.0, 1.0, 1.0, 0.0))
    with pytest.raises(ValueError, match="four finite positive"):
        StateScales((1.0, 1.0, 1.0, float("nan")))
    with pytest.raises(ValueError, match="four finite positive"):
        StateScales((1.0, 1.0, 1.0))


def test_normalized_transition_is_invariant_to_equivalent_units() -> None:
    physical = np.array(
        [
            [1.0, 0.2, 0.1, 0.0],
            [0.0, 0.8, 0.0, 0.3],
            [0.0, 0.0, 1.2, 0.1],
            [0.0, 0.0, 0.0, 0.9],
        ]
    )
    scales = StateScales((2.0, 3.0, 4.0, 5.0))
    unit_change = np.diag([100.0, 100.0, 0.01, 0.01])
    converted = unit_change @ physical @ np.linalg.inv(unit_change)
    converted_scales = StateScales(tuple(unit_change.diagonal() * scales.array))

    before = normalized_transition(physical, scales)
    after = normalized_transition(converted, converted_scales)

    np.testing.assert_allclose(after, before, rtol=1e-13, atol=1e-13)


def test_finite_time_spectra_retains_amplification_and_contraction() -> None:
    transitions = np.stack(
        [
            np.eye(2),
            np.diag([2.0, 0.5]),
            np.diag([4.0, 0.25]),
        ]
    )
    spectra = finite_time_spectra(transitions, np.array([0.0, 1.0, 2.0]))

    np.testing.assert_allclose(spectra.singular_values[1], [2.0, 0.5])
    np.testing.assert_allclose(spectra.exponents_per_s[1], [np.log(2.0), np.log(0.5)])
    np.testing.assert_allclose(spectra.exponents_per_s[2], [np.log(2.0), np.log(0.5)])
    assert np.all(np.isnan(spectra.exponents_per_s[0]))


def test_transverse_event_sensitivity_matches_implicit_formula() -> None:
    transition = np.array(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    scales = StateScales((0.5, 1.0, 2.0, 3.0))
    result = event_time_sensitivity(
        transition,
        event_flow=np.array([3.0, 1.0, 0.0, 0.0]),
        guard_gradient=np.array([1.0, 1.0, 0.0, 0.0]),
        state_scales=scales,
        transversality_threshold=1e-8,
    )

    assert result.status == "transverse"
    assert result.transversality_per_s == pytest.approx(4.0)
    assert result.derivative_s_per_scaled_state is not None
    np.testing.assert_allclose(
        result.derivative_s_per_scaled_state,
        [-0.25, -0.25, 0.0, 0.0],
    )


def test_near_grazing_event_suppresses_derivative() -> None:
    result = event_time_sensitivity(
        np.eye(4),
        event_flow=np.array([1.0, -1.0, 0.0, 0.0]),
        guard_gradient=np.array([1.0, 1.0, 0.0, 0.0]),
        state_scales=StateScales((1.0, 1.0, 1.0, 1.0)),
        transversality_threshold=1e-6,
    )

    assert result.status == "near_grazing"
    assert result.derivative_s_per_scaled_state is None


def test_time_guard_identity_reset_has_identity_saltation() -> None:
    flow = np.array([1.0, 2.0, 3.0, 4.0])
    matrix = saltation_matrix(
        reset_jacobian=np.eye(4),
        flow_before=flow,
        flow_after=flow,
        guard_gradient=np.zeros(4),
        guard_time_derivative=1.0,
        reset_time_derivative=np.zeros(4),
    )
    np.testing.assert_array_equal(matrix, np.eye(4))


def test_saltation_rejects_grazing_denominator() -> None:
    with pytest.raises(ValueError, match="transverse"):
        saltation_matrix(
            reset_jacobian=np.eye(2),
            flow_before=np.array([1.0, 0.0]),
            flow_after=np.array([1.0, 0.0]),
            guard_gradient=np.array([0.0, 1.0]),
            guard_time_derivative=0.0,
            reset_time_derivative=np.zeros(2),
        )


def test_periodicity_gate_suppresses_floquet_for_open_trajectory() -> None:
    result = periodicity_gate(
        initial_state=np.zeros(4),
        final_state=np.array([0.1, 0.0, 0.0, 0.0]),
        state_scales=StateScales((1.0, 1.0, 1.0, 1.0)),
        tolerance=1e-8,
    )

    assert result.periodic is False
    assert result.floquet_eligible is False
    assert result.normalized_residual == pytest.approx(0.1)


def test_first_positive_crossing_is_interpolated_and_typed() -> None:
    crossing = first_positive_crossing(
        np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
        np.array([-2.0, -1.0, 1.0, -1.0, 1.0]),
    )

    assert crossing.status == "multiple"
    assert crossing.crossing_count == 2
    assert crossing.sample_index == 1
    assert crossing.fraction == pytest.approx(0.5)
    assert crossing.time_s == pytest.approx(0.15)


def test_state_transition_matches_direct_double_pendulum_perturbations() -> None:
    params = GolfModelParams.default()
    controls = np.tile(np.array([60.0, -10.0]), (24, 1))
    initial_state = np.array([-2.2, -1.57, 0.0, 0.0])
    scales = StateScales((np.pi, np.pi, 10.0, 10.0))
    rollout = propagate_state_transition(
        params,
        initial_state=initial_state,
        controls=controls,
        dt_s=1e-3,
        state_steps=np.array([1e-6, 1e-6, 1e-5, 1e-5]),
    )
    direct = direct_transition_control(
        params,
        initial_state=initial_state,
        controls=controls,
        dt_s=1e-3,
        state_scales=scales,
        perturbation_scale=1e-6,
    )
    predicted = normalized_transition(rollout.transition[-1], scales)

    assert rollout.state.shape == (25, 4)
    assert rollout.transition.shape == (25, 4, 4)
    np.testing.assert_array_equal(rollout.transition[0], np.eye(4))
    np.testing.assert_allclose(direct, predicted, rtol=2e-5, atol=2e-7)
