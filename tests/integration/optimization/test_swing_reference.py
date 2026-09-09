"""Independent endpoint and refusal contracts for the #9830 reference path."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.optimization import _swing_reference as reference

pytestmark = pytest.mark.unit


def _controls() -> reference.ReferenceControls:
    return reference.ReferenceControls(1e-9, 1e-11, 1e-9, 0.002)


def _oscillator(time: float, state: np.ndarray) -> np.ndarray:
    return np.array([state[1], -(80**2) * state[0]])


def test_oscillatory_endpoint_matches_independent_trigonometric_solution() -> None:
    result = reference.resolved_endpoint(_oscillator, [1.0, 0.0], 0.12, _controls())
    expected = [np.cos(9.6), -80 * np.sin(9.6)]
    np.testing.assert_allclose(result.state, expected, atol=2e-8, rtol=1e-10)
    assert max(result.normalized_refinement) <= 1
    assert all(count > 0 for count in result.rhs_evaluations)
    assert result.status == "refinement_supported"


def test_damped_fast_mode_matches_closed_form() -> None:
    frequency = np.sqrt(750000.0)
    time = 0.01

    def rhs(t: float, state: np.ndarray) -> np.ndarray:
        return np.array([state[1], -1e6 * state[0] - 1000 * state[1]])

    result = reference.resolved_endpoint(rhs, [1, 0], time, _controls())
    decay = np.exp(-500 * time)
    position = decay * (
        np.cos(frequency * time) + 500 / frequency * np.sin(frequency * time)
    )
    velocity = -decay * 1e6 / frequency * np.sin(frequency * time)
    np.testing.assert_allclose(result.state, [position, velocity], atol=1e-8, rtol=1e-9)


def test_endpoint_is_the_actual_interval_end_and_inputs_are_owned() -> None:
    initial = np.array([1.0, 2.0])

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        assert not state.flags.writeable
        return np.array([state[1], 2.0])

    result = reference.resolved_endpoint(rhs, initial, 0.37, _controls())
    np.testing.assert_allclose(result.state, [1 + 2 * 0.37 + 0.37**2, 2.74], atol=1e-12)
    np.testing.assert_array_equal(initial, [1, 2])
    with pytest.raises(FrozenInstanceError):
        result.status = "physical validation"
    with pytest.raises(TypeError):
        result.state[0] = 0


def test_refinement_controls_preserve_distinct_component_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def solve(fun, times, initial, **options):
        calls.append(options)
        return SimpleNamespace(
            success=True, t=np.array([times[1]]), y=np.array(initial)[:, None]
        )

    monkeypatch.setattr(reference, "solve_ivp", solve)
    reference.resolved_endpoint(_oscillator, [1, 0], 0.12, _controls())
    # Local truncation tolerances must leave margin for endpoint accumulation.
    np.testing.assert_allclose(calls[0]["atol"], [1e-12, 1e-10], rtol=1e-15)
    np.testing.assert_allclose(calls[1]["atol"], [1e-13, 1e-11], rtol=1e-15)
    assert calls[0]["rtol"] == pytest.approx(_controls().rtol / 10)
    assert calls[1]["rtol"] == pytest.approx(calls[0]["rtol"] / 10)
    assert calls[1]["max_step"] == calls[0]["max_step"] / 2
    assert calls[0]["t_eval"] == [0.12]


def test_endpoint_budget_is_distinct_from_local_solver_tolerances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter([1.0 + 5e-10, 1.0])

    def solve(fun, times, initial, **options):
        return SimpleNamespace(
            success=True, t=np.array([times[1]]), y=np.array([[next(outputs)], [0]])
        )

    monkeypatch.setattr(reference, "solve_ivp", solve)
    result = reference.resolved_endpoint(_oscillator, [1, 0], 0.12, _controls())
    assert result.controls == _controls()
    assert result.normalized_refinement[0] == pytest.approx(5e-10 / 1.01e-9)


def test_disagreeing_refinements_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = iter([1.001, 1.0])

    def solve(fun, times, initial, **options):
        return SimpleNamespace(
            success=True, t=np.array([times[1]]), y=np.array([[next(outputs)], [0]])
        )

    monkeypatch.setattr(reference, "solve_ivp", solve)
    with pytest.raises(reference.ReferenceIntegrationError, match="refinement"):
        reference.resolved_endpoint(_oscillator, [1, 0], 0.12, _controls())


def test_large_initial_state_does_not_mask_small_endpoint_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoints = iter([1e-6, 0.0])

    def solve(fun, times, initial, **options):
        return SimpleNamespace(
            success=True, t=np.array([times[1]]), y=np.array([[next(endpoints)], [0]])
        )

    monkeypatch.setattr(reference, "solve_ivp", solve)
    with pytest.raises(reference.ReferenceIntegrationError, match="refinement"):
        reference.resolved_endpoint(_oscillator, [1e6, 0], 0.12, _controls())


@pytest.mark.parametrize("fault", ["failure", "nan", "shape", "complex", "endpoint"])
def test_failed_or_malformed_solver_result_is_never_qualified(
    fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = SimpleNamespace(
        success=True, message="injected failure", t=np.array([0.12]), y=np.ones((2, 1))
    )
    if fault == "failure":
        result.success = False
    elif fault == "nan":
        result.y[0, 0] = np.nan
    elif fault == "shape":
        result.y = np.ones((1, 1))
    elif fault == "complex":
        result.y = result.y * 1j
    else:
        result.t = np.array([0.11])
    monkeypatch.setattr(reference, "solve_ivp", lambda *args, **kwargs: result)
    with pytest.raises(reference.ReferenceIntegrationError):
        reference.resolved_endpoint(_oscillator, [1, 0], 0.12, _controls())


@pytest.mark.parametrize("output", [[np.nan, 0], [1], [1j, 0], [True, 0]])
def test_invalid_rhs_is_refused(output: list) -> None:
    with pytest.raises(reference.ReferenceIntegrationError):
        reference.resolved_endpoint(lambda t, x: output, [1, 0], 0.12, _controls())


def test_rhs_work_budget_is_enforced() -> None:
    controls = replace(_controls(), max_rhs_evaluations=2)
    with pytest.raises(reference.ReferenceIntegrationError, match="budget"):
        reference.resolved_endpoint(_oscillator, [1, 0], 0.12, controls)


@pytest.mark.parametrize(
    "field", ["rtol", "position_atol", "velocity_atol", "max_step_s"]
)
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_controls_refuse_invalid_numeric_domains(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        replace(_controls(), **{field: value})


@pytest.mark.parametrize("value", [True, "0.1", 1j])
def test_control_types_are_not_coerced(value: object) -> None:
    with pytest.raises(TypeError):
        replace(_controls(), rtol=value)


@pytest.mark.parametrize("value", [1.0, 1e-16])
def test_relative_tolerance_does_not_trigger_silent_solver_clipping(
    value: float,
) -> None:
    with pytest.raises(ValueError):
        replace(_controls(), rtol=value)


@pytest.mark.parametrize(
    "value, exception",
    [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)],
)
def test_rhs_budget_is_a_positive_integer(value: object, exception: type) -> None:
    with pytest.raises(exception):
        replace(_controls(), max_rhs_evaluations=value)


@pytest.mark.parametrize(
    "state, exception",
    [
        ([], ValueError),
        ([1, 2, 3], ValueError),
        ([[1, 0]], ValueError),
        ([np.nan, 0], ValueError),
        ([True, 0], TypeError),
        (["1", "0"], TypeError),
    ],
)
def test_state_contract(state: object, exception: type) -> None:
    with pytest.raises(exception):
        reference.resolved_endpoint(_oscillator, state, 0.12, _controls())


@pytest.mark.parametrize(
    "duration, exception",
    [
        (0, ValueError),
        (-1, ValueError),
        (np.inf, ValueError),
        (True, TypeError),
        ("1", TypeError),
    ],
)
def test_duration_contract(duration: object, exception: type) -> None:
    with pytest.raises(exception):
        reference.resolved_endpoint(_oscillator, [1, 0], duration, _controls())


def test_callable_and_controls_are_checked() -> None:
    with pytest.raises(TypeError):
        reference.resolved_endpoint(None, [1, 0], 0.12, _controls())
    with pytest.raises(TypeError):
        reference.resolved_endpoint(_oscillator, [1, 0], 0.12, None)
