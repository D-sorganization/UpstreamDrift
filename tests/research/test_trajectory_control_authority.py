"""Trajectory-varying control-authority contracts for issue #9123."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.phase_event_stability import StateScales
from scripts.research.proximal_distal_energy.torque_programs import (
    restrain_then_drive_program,
)
from scripts.research.proximal_distal_energy.trajectory_control_authority import (
    ControlScales,
    event_conditioned_gramian,
    frozen_local_gramian,
    linearize_trajectory,
    propagated_terminal_input_sensitivity,
    reachability_history,
    direct_terminal_pulse_sensitivity,
    direct_variable_terminal_pulse_sensitivity,
    refine_guard_crossing,
    scale_step_matrices,
    step_linearization,
)
from src.shared.python.simulation_backends import GolfModelParams

pytestmark = pytest.mark.unit


def test_control_scales_require_two_finite_positive_values() -> None:
    for values in ((1.0,), (1.0, 0.0), (1.0, float("nan"))):
        with pytest.raises(ValueError, match="two finite positive"):
            ControlScales(values)


def test_exact_rk4_step_linearization_is_deterministic_and_input_immutable() -> None:
    params = GolfModelParams.default()
    state = np.array([-2.0, -1.3, 2.0, 5.0])
    control = np.array([55.0, -8.0])
    state_before = state.copy()
    control_before = control.copy()
    kwargs = {
        "params": params,
        "state": state,
        "control": control,
        "time_s": 0.125,
        "dt_s": 1e-3,
        "state_steps": np.array([1e-6, 1e-6, 1e-5, 1e-5]),
        "control_steps": np.array([1e-4, 1e-4]),
        "state_scales": StateScales((np.pi, np.pi, 10.0, 10.0)),
        "control_scales": ControlScales((100.0, 100.0)),
    }

    first = step_linearization(**kwargs)
    second = step_linearization(**kwargs)

    assert first.state_matrix.shape == (4, 4)
    assert first.input_matrix.shape == (4, 2)
    assert first.scaled_state_matrix.shape == (4, 4)
    assert first.scaled_sample_input_matrix.shape == (4, 2)
    assert first.scaled_energy_input_matrix.shape == (4, 2)
    for name in first.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    np.testing.assert_array_equal(state, state_before)
    np.testing.assert_array_equal(control, control_before)


def test_continuous_energy_normalization_converges_for_scalar_integrator() -> None:
    results = []
    for dt_s in (0.1, 0.05, 0.025):
        count = round(1.0 / dt_s)
        state_matrices = np.ones((count, 1, 1))
        energy_inputs = np.full((count, 1, 1), np.sqrt(dt_s))
        results.append(reachability_history(state_matrices, energy_inputs)[-1, 0, 0])

    np.testing.assert_allclose(results, np.ones(3), rtol=0.0, atol=1e-14)


def test_channel_gramians_add_to_full_trajectory_varying_gramian() -> None:
    state_matrices = np.array(
        [
            [[1.0, 0.2], [0.0, 1.0]],
            [[0.9, -0.1], [0.3, 1.1]],
            [[1.2, 0.0], [-0.2, 0.8]],
        ]
    )
    input_matrices = np.array(
        [
            [[1.0, 0.2], [0.0, 0.5]],
            [[0.4, -0.3], [0.7, 0.1]],
            [[0.2, 0.6], [-0.5, 0.9]],
        ]
    )

    full = reachability_history(state_matrices, input_matrices)
    shoulder = reachability_history(
        state_matrices, input_matrices, channel_mask=np.array([1.0, 0.0])
    )
    wrist = reachability_history(
        state_matrices, input_matrices, channel_mask=np.array([0.0, 1.0])
    )
    zero = reachability_history(
        state_matrices, input_matrices, channel_mask=np.zeros(2)
    )

    np.testing.assert_allclose(full, shoulder + wrist, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(zero, np.zeros_like(zero))


def test_event_conditioning_uses_explicit_orthonormal_tangent_basis() -> None:
    gramian = np.diag([4.0, 3.0, 2.0, 1.0])
    result = event_conditioned_gramian(
        gramian,
        event_flow=np.array([2.0, 0.5, -0.2, 0.1]),
        guard_gradient=np.array([1.0, 0.0, 0.0, 0.0]),
        transversality_threshold=1e-10,
    )

    assert result.status == "transverse"
    assert result.tangent_basis is not None
    assert result.tangent_gramian is not None
    assert result.projection is not None
    assert result.tangent_basis.shape == (4, 3)
    assert result.tangent_gramian.shape == (3, 3)
    np.testing.assert_allclose(
        result.tangent_basis.T @ result.tangent_basis,
        np.eye(3),
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        np.array([1.0, 0.0, 0.0, 0.0]) @ result.tangent_basis,
        np.zeros(3),
        atol=1e-13,
    )
    np.testing.assert_allclose(result.projection @ result.projection, result.projection)


def test_event_conditioning_fails_closed_for_near_grazing_guard() -> None:
    result = event_conditioned_gramian(
        np.eye(4),
        event_flow=np.array([0.0, 1.0, 0.0, 0.0]),
        guard_gradient=np.array([1.0, 0.0, 0.0, 0.0]),
        transversality_threshold=1e-8,
    )

    assert result.status == "near_grazing"
    assert result.projection is None
    assert result.tangent_basis is None
    assert result.tangent_gramian is None


def test_frozen_local_countermodel_is_not_trajectory_varying_authority() -> None:
    state_matrices = np.array([[[1.0]], [[2.0]]])
    input_matrices = np.ones((2, 1, 1))

    varying = reachability_history(state_matrices, input_matrices)[-1]
    frozen = frozen_local_gramian(state_matrices[0], input_matrices[0], step_count=2)

    assert varying[0, 0] == pytest.approx(5.0)
    assert frozen[0, 0] == pytest.approx(2.0)


def test_equivalent_state_and_control_units_preserve_scaled_step_matrices() -> None:
    state_matrix = np.array(
        [
            [1.0, 0.1, 0.2, 0.0],
            [0.0, 1.0, 0.0, 0.3],
            [0.0, 0.0, 0.9, 0.2],
            [0.1, 0.0, -0.2, 1.1],
        ]
    )
    input_matrix = np.array([[0.1, 0.2], [0.0, 0.1], [0.5, -0.2], [0.3, 0.4]])
    state_units = np.diag([100.0, 100.0, 0.01, 0.01])
    control_units = np.diag([1000.0, 0.001])
    state_scales = StateScales((2.0, 3.0, 4.0, 5.0))
    control_scales = ControlScales((6.0, 7.0))

    baseline = scale_step_matrices(
        state_matrix,
        input_matrix,
        dt_s=0.02,
        state_scales=state_scales,
        control_scales=control_scales,
    )
    converted = scale_step_matrices(
        state_units @ state_matrix @ np.linalg.inv(state_units),
        state_units @ input_matrix @ np.linalg.inv(control_units),
        dt_s=0.02,
        state_scales=StateScales(tuple(state_units.diagonal() * state_scales.array)),
        control_scales=ControlScales(
            tuple(control_units.diagonal() * control_scales.array)
        ),
    )

    for before, after in zip(baseline, converted, strict=True):
        np.testing.assert_allclose(after, before, rtol=1e-13, atol=1e-13)


def test_trajectory_linearization_matches_direct_terminal_pulse() -> None:
    params = GolfModelParams.default()
    controls = np.tile(np.array([60.0, -10.0]), (12, 1))
    scales = StateScales((np.pi, np.pi, 10.0, 10.0))
    control_scales = ControlScales((100.0, 100.0))
    trajectory = linearize_trajectory(
        params=params,
        initial_state=np.array([-2.2, -1.57, 0.0, 0.0]),
        controls=controls,
        dt_s=1e-3,
        state_steps=np.array([1e-6, 1e-6, 1e-5, 1e-5]),
        control_steps=np.array([1e-4, 1e-4]),
        state_scales=scales,
        control_scales=control_scales,
    )
    predicted = propagated_terminal_input_sensitivity(
        trajectory.scaled_state_matrices,
        trajectory.scaled_energy_input_matrices,
        pulse_step=4,
        channel_index=1,
    )
    direct = direct_terminal_pulse_sensitivity(
        params=params,
        initial_state=trajectory.state[0],
        controls=controls,
        dt_s=1e-3,
        state_scales=scales,
        control_scales=control_scales,
        pulse_step=4,
        channel_index=1,
        perturbation_scale=1e-6,
    )
    variable_direct = direct_variable_terminal_pulse_sensitivity(
        params=params,
        initial_state=trajectory.state[0],
        controls=controls,
        step_durations_s=np.full(controls.shape[0], 1e-3),
        state_scales=scales,
        control_scales=control_scales,
        pulse_step=4,
        channel_index=1,
        perturbation_scale=1e-6,
    )

    assert trajectory.state.shape == (13, 4)
    assert trajectory.scaled_state_matrices.shape == (12, 4, 4)
    assert trajectory.scaled_energy_input_matrices.shape == (12, 4, 2)
    np.testing.assert_allclose(direct, predicted, rtol=3e-5, atol=3e-8)
    np.testing.assert_array_equal(variable_direct, direct)


def test_registered_guard_crossing_is_refined_inside_bracket() -> None:
    params = GolfModelParams.default()
    controls = restrain_then_drive_program(60.0, 15.0, 10.0, 0.10).controls(900, 1e-3)
    trajectory = linearize_trajectory(
        params=params,
        initial_state=np.array([-2.2, -1.57, 0.0, 0.0]),
        controls=controls,
        dt_s=1e-3,
        state_steps=np.array([1e-6, 1e-6, 1e-5, 1e-5]),
        control_steps=np.array([1e-4, 1e-4]),
        state_scales=StateScales((np.pi, np.pi, 10.0, 10.0)),
        control_scales=ControlScales((100.0, 100.0)),
    )
    guard = trajectory.state[:, 0] + trajectory.state[:, 1]
    indices = np.flatnonzero((guard[:-1] < 0.0) & (guard[1:] >= 0.0))
    assert indices.size == 1
    index = int(indices[0])

    crossing = refine_guard_crossing(
        params=params,
        state_before=trajectory.state[index],
        control=controls[index],
        time_before_s=trajectory.time_s[index],
        bracket_dt_s=1e-3,
        guard_gradient=np.array([1.0, 1.0, 0.0, 0.0]),
        guard_tolerance=1e-11,
        time_tolerance_s=1e-12,
    )

    assert crossing.status == "transverse_candidate"
    assert 0.0 < crossing.partial_dt_s <= 1e-3
    assert crossing.time_s == pytest.approx(
        trajectory.time_s[index] + crossing.partial_dt_s
    )
    assert abs(crossing.guard_residual) <= 1e-11


@pytest.mark.parametrize(
    ("state_matrices", "input_matrices", "message"),
    (
        (np.ones((2, 2)), np.ones((2, 2, 1)), "state_matrices"),
        (np.ones((2, 2, 2)), np.ones((3, 2, 1)), "same step count"),
        (np.ones((2, 2, 2)), np.full((2, 2, 1), np.nan), "finite"),
    ),
)
def test_reachability_history_rejects_invalid_arrays(
    state_matrices: np.ndarray,
    input_matrices: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        reachability_history(state_matrices, input_matrices)
