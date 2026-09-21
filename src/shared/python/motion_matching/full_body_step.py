"""Differentiable RK4 stepping for polynomially actuated full-body plants."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray

from .polynomial_actuation import FullBodyPolynomialControl

Array = NDArray[np.float64]


class FullBodyPlant(Protocol):
    """Narrow public plant surface required by the polynomial step."""

    @property
    def coordinate_order(self) -> tuple[str, ...]: ...

    def accelerations(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> Mapping[str, float]: ...

    def acceleration_derivatives(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> Any: ...

    def contact_effort_derivatives(
        self, coordinates: Mapping[str, float], rates: Mapping[str, float]
    ) -> Any: ...


@dataclass(frozen=True)
class FullBodyStepOptions:
    """Fixed numerical choices for one public RK4 step."""

    substeps: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.substeps, bool)
            or not isinstance(self.substeps, Integral)
            or self.substeps <= 0
        ):
            raise ValueError("substeps must be a positive integer")
        object.__setattr__(self, "substeps", int(self.substeps))


class StepLinearization(NamedTuple):
    """Detached RK4 result and exact discrete first derivatives."""

    next_state: Array
    dnext_dstate: Array
    dnext_dcoefficients: Array
    differentiable: bool


class _Dynamics(NamedTuple):
    value: Array
    dstate: Array | None
    dcoefficients: Array | None
    differentiable: bool


class _Substep(NamedTuple):
    state: Array
    dstate: Array | None
    dcoefficients: Array | None
    differentiable: bool


def _readonly(values: Any) -> Array:
    result = np.array(values, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("RK4 result must be finite")
    result.setflags(write=False)
    return result


def _matrix(value: Any, shape: tuple[int, int], label: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must have finite shape {shape}")
    return result


class NativeFullBodyStep:
    """Advance canonical ``[q, v]`` state under global polynomial efforts."""

    def __init__(
        self,
        plant: FullBodyPlant,
        control: FullBodyPolynomialControl,
        options: FullBodyStepOptions,
    ) -> None:
        plant_names = tuple(plant.coordinate_order)
        if len(set(plant_names)) != len(plant_names):
            raise ValueError("plant coordinate_order contains duplicate names")
        if set(plant_names) != set(control.coordinate_names):
            raise ValueError("plant and control coordinate inventories differ")
        self.plant = plant
        self.control = control
        self.options = options

    def _inputs(
        self, state: Any, parameters: Any, time_s: float, dt_s: float
    ) -> tuple[Array, Array, float, float]:
        n = self.control.n_coordinates
        state_array = np.asarray(state, dtype=float)
        if state_array.shape != (2 * n,):
            raise ValueError(f"state must have shape ({2 * n},)")
        if not np.isfinite(state_array).all():
            raise ValueError("state must be finite")
        if (
            isinstance(time_s, bool)
            or not isinstance(time_s, Real)
            or not np.isfinite(time_s)
            or time_s < 0.0
        ):
            raise ValueError("time_s must be finite and nonnegative")
        if (
            isinstance(dt_s, bool)
            or not isinstance(dt_s, Real)
            or not np.isfinite(dt_s)
            or dt_s <= 0.0
        ):
            raise ValueError("dt_s must be finite and positive")
        tolerance = 8.0 * np.finfo(float).eps * max(1.0, self.control.duration_s)
        if float(time_s) + float(dt_s) > self.control.duration_s + tolerance:
            raise ValueError("step end exceeds the polynomial control horizon")
        parameters_array = np.asarray(parameters, dtype=float)
        self.control.efforts(parameters_array, float(time_s))
        return state_array.copy(), parameters_array.copy(), float(time_s), float(dt_s)

    def _maps(self, state: Array) -> tuple[dict[str, float], dict[str, float]]:
        names = self.control.coordinate_names
        n = len(names)
        coordinates = {
            name: float(value) for name, value in zip(names, state[:n], strict=True)
        }
        rates = {
            name: float(value) for name, value in zip(names, state[n:], strict=True)
        }
        return coordinates, rates

    def _dynamics(
        self, state: Array, parameters: Array, time_s: float, linearize: bool
    ) -> _Dynamics:
        names = self.control.coordinate_names
        n = len(names)
        coordinates, rates = self._maps(state)
        efforts = self.control.efforts(parameters, time_s)
        raw = self.plant.accelerations(coordinates, rates, efforts)
        if set(raw) != set(names):
            raise ValueError(
                "plant acceleration names differ from requested coordinates"
            )
        acceleration = np.asarray([raw[name] for name in names], dtype=float)
        if acceleration.shape != (n,) or not np.isfinite(acceleration).all():
            raise ValueError("plant acceleration must be a finite coordinate vector")
        value = np.concatenate((state[n:], acceleration))
        if not linearize:
            return _Dynamics(value, None, None, True)
        derivatives = self.plant.acceleration_derivatives(coordinates, rates, efforts)
        if tuple(derivatives.names) != names:
            raise ValueError("plant derivative names differ from requested coordinates")
        dq = _matrix(derivatives.dq, (n, n), "acceleration dq")
        dv = _matrix(derivatives.dv, (n, n), "acceleration dv")
        deffort = _matrix(derivatives.deffort, (n, n), "acceleration deffort")
        contact = self.plant.contact_effort_derivatives(coordinates, rates)
        if tuple(contact.names) != names:
            raise ValueError(
                "contact derivative names differ from requested coordinates"
            )
        _matrix(contact.dq, (n, n), "contact dq")
        _matrix(contact.dv, (n, n), "contact dv")
        dstate = np.zeros((2 * n, 2 * n))
        dstate[:n, n:] = np.eye(n)
        dstate[n:, :n] = dq
        dstate[n:, n:] = dv
        dcoefficients = np.zeros((2 * n, self.control.n_parameters))
        dcoefficients[n:] = deffort @ self.control.effort_jacobian(time_s)
        return _Dynamics(value, dstate, dcoefficients, bool(contact.differentiable))

    def _rk4_substep(
        self,
        state: Array,
        parameters: Array,
        time_s: float,
        step_s: float,
        linearize: bool,
    ) -> _Substep:
        dimension = state.size
        identity = np.eye(dimension)
        k1 = self._dynamics(state, parameters, time_s, linearize)
        stage2 = state + 0.5 * step_s * k1.value
        k2 = self._dynamics(stage2, parameters, time_s + 0.5 * step_s, linearize)
        stage3 = state + 0.5 * step_s * k2.value
        k3 = self._dynamics(stage3, parameters, time_s + 0.5 * step_s, linearize)
        stage4 = state + step_s * k3.value
        k4 = self._dynamics(stage4, parameters, time_s + step_s, linearize)
        next_state = state + (step_s / 6.0) * (
            k1.value + 2.0 * k2.value + 2.0 * k3.value + k4.value
        )
        differentiable = all(stage.differentiable for stage in (k1, k2, k3, k4))
        if not linearize:
            return _Substep(next_state, None, None, differentiable)
        assert k1.dstate is not None and k1.dcoefficients is not None
        assert k2.dstate is not None and k2.dcoefficients is not None
        assert k3.dstate is not None and k3.dcoefficients is not None
        assert k4.dstate is not None and k4.dcoefficients is not None
        k1_x, k1_p = k1.dstate, k1.dcoefficients
        k2_x = k2.dstate @ (identity + 0.5 * step_s * k1_x)
        k2_p = k2.dstate @ (0.5 * step_s * k1_p) + k2.dcoefficients
        k3_x = k3.dstate @ (identity + 0.5 * step_s * k2_x)
        k3_p = k3.dstate @ (0.5 * step_s * k2_p) + k3.dcoefficients
        k4_x = k4.dstate @ (identity + step_s * k3_x)
        k4_p = k4.dstate @ (step_s * k3_p) + k4.dcoefficients
        dstate = identity + (step_s / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        dcoefficients = (step_s / 6.0) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)
        return _Substep(next_state, dstate, dcoefficients, differentiable)

    def step(self, state: Any, parameters: Any, *, time_s: float, dt_s: float) -> Array:
        """Advance one interval without evaluating derivative-only plant APIs."""
        current, coefficients, start, duration = self._inputs(
            state, parameters, time_s, dt_s
        )
        step_s = duration / self.options.substeps
        for index in range(self.options.substeps):
            result = self._rk4_substep(
                current, coefficients, start + index * step_s, step_s, False
            )
            current = result.state
        return _readonly(current)

    def linearize(
        self, state: Any, parameters: Any, *, time_s: float, dt_s: float
    ) -> StepLinearization:
        """Advance one interval and differentiate every RK4 stage and substep."""
        current, coefficients, start, duration = self._inputs(
            state, parameters, time_s, dt_s
        )
        dimension = current.size
        total_x: Array = np.eye(dimension)
        total_p: Array = np.zeros((dimension, self.control.n_parameters))
        differentiable = True
        step_s = duration / self.options.substeps
        for index in range(self.options.substeps):
            result = self._rk4_substep(
                current, coefficients, start + index * step_s, step_s, True
            )
            assert result.dstate is not None and result.dcoefficients is not None
            total_p = result.dstate @ total_p + result.dcoefficients
            total_x = result.dstate @ total_x
            current = result.state
            differentiable = differentiable and result.differentiable
        return StepLinearization(
            _readonly(current),
            _readonly(total_x),
            _readonly(total_p),
            differentiable,
        )
