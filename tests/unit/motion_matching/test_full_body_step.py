"""Pure numerical tests for the differentiable full-body RK4 step."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.motion_matching.full_body_step import (
    FullBodyStepOptions,
    NativeFullBodyStep,
)
from src.shared.python.motion_matching.polynomial_actuation import (
    ROOT_COORDINATES,
    FullBodyPolynomialControl,
)

pytestmark = pytest.mark.unit


class LinearPlant:
    def __init__(self, names: tuple[str, ...], aq: np.ndarray, av: np.ndarray):
        self.coordinate_order = tuple(reversed(names))
        self.aq = aq
        self.av = av
        self.derivative_calls = 0

    def accelerations(self, coordinates, rates, primitive_efforts):
        names = tuple(coordinates)
        q = np.array([coordinates[name] for name in names])
        v = np.array([rates[name] for name in names])
        effort = np.array([primitive_efforts[name] for name in names])
        acceleration = self.aq @ q + self.av @ v + effort
        return {name: acceleration[index] for index, name in enumerate(names)}

    def acceleration_derivatives(self, coordinates, rates, primitive_efforts):
        del rates, primitive_efforts
        self.derivative_calls += 1
        names = tuple(coordinates)
        return SimpleNamespace(
            names=names,
            dq=self.aq.copy(),
            dv=self.av.copy(),
            deffort=np.eye(len(names)),
        )

    def contact_effort_derivatives(self, coordinates, rates):
        del rates
        names = tuple(coordinates)
        zeros = np.zeros((len(names), len(names)))
        return SimpleNamespace(names=names, dq=zeros, dv=zeros, differentiable=True)


def _system(substeps: int = 1):
    names = (*ROOT_COORDINATES, "joint")
    control = FullBodyPolynomialControl(names, ("joint",), 3.0)
    aq = np.diag(np.linspace(-0.2, 0.3, len(names)))
    av = np.diag(np.linspace(0.1, -0.15, len(names)))
    plant = LinearPlant(names, aq, av)
    stepper = NativeFullBodyStep(plant, control, FullBodyStepOptions(substeps))
    return stepper, plant, control


def test_constant_acceleration_solution_and_summed_coefficient_sensitivity() -> None:
    stepper, plant, control = _system(substeps=3)
    plant.aq[:] = 0.0
    plant.av[:] = 0.0
    n = control.n_coordinates
    state = np.r_[np.linspace(-0.3, 0.4, n), np.linspace(0.2, -0.1, n)]
    parameters = np.full(control.n_parameters, 1.75)
    time_s = 0.8
    dt_s = 0.3

    result = stepper.linearize(state, parameters, time_s=time_s, dt_s=dt_s)

    expected = state.copy()
    actuator = control.coordinate_names.index("joint")
    expected[:n] += state[n:] * dt_s
    expected[actuator] += 0.5 * 1.75 * dt_s**2
    expected[n + actuator] += 1.75 * dt_s
    np.testing.assert_allclose(result.next_state, expected, rtol=0.0, atol=2e-15)
    summed = result.dnext_dcoefficients[:, :7].sum(axis=1)
    expected_sensitivity = np.zeros(2 * n)
    expected_sensitivity[actuator] = 0.5 * dt_s**2
    expected_sensitivity[n + actuator] = dt_s
    np.testing.assert_allclose(summed, expected_sensitivity, rtol=0.0, atol=2e-15)


def test_linearization_matches_independent_directional_differences() -> None:
    stepper, _, control = _system(substeps=4)
    rng = np.random.default_rng(10265)
    state = rng.normal(scale=0.2, size=2 * control.n_coordinates)
    parameters = rng.normal(scale=0.5, size=control.n_parameters)
    result = stepper.linearize(state, parameters, time_s=0.41, dt_s=0.23)

    for argument, jacobian in (
        (state, result.dnext_dstate),
        (parameters, result.dnext_dcoefficients),
    ):
        direction = rng.normal(size=argument.size)
        direction /= np.linalg.norm(direction)
        epsilon = 2e-7
        if argument is state:
            plus = stepper.step(
                state + epsilon * direction, parameters, time_s=0.41, dt_s=0.23
            )
            minus = stepper.step(
                state - epsilon * direction, parameters, time_s=0.41, dt_s=0.23
            )
        else:
            plus = stepper.step(
                state, parameters + epsilon * direction, time_s=0.41, dt_s=0.23
            )
            minus = stepper.step(
                state, parameters - epsilon * direction, time_s=0.41, dt_s=0.23
            )
        measured = (plus - minus) / (2.0 * epsilon)
        np.testing.assert_allclose(jacobian @ direction, measured, rtol=3e-7, atol=3e-8)

    for array in result:
        if isinstance(array, np.ndarray):
            assert array.flags.owndata
            assert not array.flags.writeable


def test_linear_autonomous_state_uses_closed_form_rk4_polynomial() -> None:
    stepper, plant, control = _system(substeps=1)
    n = control.n_coordinates
    state = np.linspace(-0.4, 0.5, 2 * n)
    dt_s = 0.17
    system = np.block([[np.zeros((n, n)), np.eye(n)], [plant.aq, plant.av]])
    scaled = dt_s * system
    expected_map = (
        np.eye(2 * n)
        + scaled
        + scaled @ scaled / 2.0
        + scaled @ scaled @ scaled / 6.0
        + scaled @ scaled @ scaled @ scaled / 24.0
    )

    result = stepper.linearize(
        state,
        np.zeros(control.n_parameters),
        time_s=0.61,
        dt_s=dt_s,
    )

    np.testing.assert_allclose(result.next_state, expected_map @ state, atol=2e-15)
    np.testing.assert_allclose(result.dnext_dstate, expected_map, atol=2e-15)


def test_value_path_avoids_derivative_calls_and_inputs_are_not_mutated() -> None:
    stepper, plant, control = _system(substeps=2)
    state = np.linspace(-0.2, 0.2, 2 * control.n_coordinates)
    parameters = np.linspace(-1.0, 1.0, control.n_parameters)
    saved_state = state.copy()
    saved_parameters = parameters.copy()

    result = stepper.step(state, parameters, time_s=0.2, dt_s=0.1)

    assert plant.derivative_calls == 0
    np.testing.assert_array_equal(state, saved_state)
    np.testing.assert_array_equal(parameters, saved_parameters)
    assert result.flags.owndata
    assert not result.flags.writeable


def test_contact_branch_flag_is_aggregated_across_rk4_stages() -> None:
    stepper, plant, control = _system()
    original = plant.contact_effort_derivatives
    calls = 0

    def contact(coordinates, rates):
        nonlocal calls
        calls += 1
        result = original(coordinates, rates)
        result.differentiable = calls != 3
        return result

    plant.contact_effort_derivatives = contact
    result = stepper.linearize(
        np.zeros(2 * control.n_coordinates),
        np.zeros(control.n_parameters),
        time_s=0.0,
        dt_s=0.1,
    )

    assert not result.differentiable
    assert calls == 4


@pytest.mark.parametrize("substeps", [0, -1, 1.0, True])
def test_options_require_positive_exact_integer(substeps) -> None:
    with pytest.raises(ValueError, match="substeps"):
        FullBodyStepOptions(substeps)


@pytest.mark.parametrize(
    ("state", "time_s", "dt_s", "match"),
    [
        (np.zeros(13), 0.0, 0.1, "state"),
        (np.r_[np.zeros(13), np.nan], 0.0, 0.1, "finite"),
        (np.zeros(14), -0.1, 0.1, "time_s"),
        (np.zeros(14), 0.0, 0.0, "dt_s"),
        (np.zeros(14), 2.95, 0.1, "horizon"),
    ],
)
def test_step_input_validation(state, time_s, dt_s, match) -> None:
    stepper, _, control = _system()
    with pytest.raises(ValueError, match=match):
        stepper.step(
            state,
            np.zeros(control.n_parameters),
            time_s=time_s,
            dt_s=dt_s,
        )


def test_derivative_names_and_shapes_are_checked_before_composition() -> None:
    stepper, plant, control = _system()
    original = plant.acceleration_derivatives

    def reordered(*args):
        result = original(*args)
        result.names = tuple(reversed(result.names))
        return result

    plant.acceleration_derivatives = reordered
    with pytest.raises(ValueError, match="derivative names"):
        stepper.linearize(
            np.zeros(2 * control.n_coordinates),
            np.zeros(control.n_parameters),
            time_s=0.0,
            dt_s=0.1,
        )
