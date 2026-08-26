"""Local first-order observability and controllability diagnostics.

The routines deliberately require caller-supplied perturbation steps and rank
tolerances. Numerical rank of one local linearization is not structural
identifiability, global nonlinear observability, or global controllability.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.simulation_backends import GolfModelParams, make_backend


FloatArray: TypeAlias = npt.NDArray[np.float64]
DynamicsFunction = Callable[[FloatArray, FloatArray], npt.ArrayLike]
OutputFunction = Callable[[FloatArray], npt.ArrayLike]

INFERENCE_BOUNDARY = (
    "This is a local first-order numerical rank diagnostic at one declared "
    "state, control, nondimensional scaling, finite-difference step, and "
    "tolerance. Raw dimensional matrices are retained for traceability but "
    "are not a basis for conditioning comparisons. It does "
    "not establish structural identifiability, practical identifiability, "
    "global nonlinear observability, or global controllability."
)


@dataclass(frozen=True, slots=True)
class NondimensionalScales:
    """Characteristic coordinate scales for a dimensionless linearization."""

    state: tuple[float, ...]
    control: tuple[float, ...]
    output: tuple[float, ...]
    characteristic_time_s: float

    def __post_init__(self) -> None:
        for name in ("state", "control", "output"):
            values = tuple(float(value) for value in getattr(self, name))
            if not values or any(
                not math.isfinite(value) or value <= 0.0 for value in values
            ):
                raise ValueError(f"{name} scales must be finite and positive")
            object.__setattr__(self, name, values)
        if (
            not math.isfinite(self.characteristic_time_s)
            or self.characteristic_time_s <= 0.0
        ):
            raise ValueError("characteristic time must be finite and positive")

    def arrays(
        self, *, state_dimension: int, control_dimension: int, output_dimension: int
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return validated scale vectors for declared matrix dimensions."""
        return (
            _positive_steps("state scales", self.state, state_dimension),
            _positive_steps("control scales", self.control, control_dimension),
            _positive_steps("output scales", self.output, output_dimension),
        )


@dataclass(frozen=True, slots=True)
class RankTolerance:
    """Absolute and relative thresholds for an SVD rank decision."""

    absolute: float
    relative: float

    def __post_init__(self) -> None:
        for name, value in (("absolute", self.absolute), ("relative", self.relative)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} rank tolerance must be finite and positive")


@dataclass(frozen=True, slots=True)
class RankDiagnostic:
    """Inspectable numerical-rank result for one scaled matrix."""

    rank: int
    full_rank: bool
    matrix_shape: tuple[int, int]
    threshold: float
    singular_values: tuple[float, ...]
    smallest_retained: float | None
    retained_condition_number: float | None


@dataclass(frozen=True, slots=True)
class LocalLinearAudit:
    """Local linearization and its state-observability/control rank decisions."""

    state_dimension: int
    control_dimension: int
    output_dimension: int
    state: tuple[float, ...]
    control: tuple[float, ...]
    state_steps: tuple[float, ...]
    control_steps: tuple[float, ...]
    state_matrix: FloatArray
    input_matrix: FloatArray
    output_matrix: FloatArray
    dimensionless_state_matrix: FloatArray
    dimensionless_input_matrix: FloatArray
    dimensionless_output_matrix: FloatArray
    observability_matrix: FloatArray
    controllability_matrix: FloatArray
    observability: RankDiagnostic
    controllability: RankDiagnostic
    scales: NondimensionalScales
    inference_boundary: str = INFERENCE_BOUNDARY

    def __post_init__(self) -> None:
        expected = {
            "state_matrix": (self.state_dimension, self.state_dimension),
            "input_matrix": (self.state_dimension, self.control_dimension),
            "output_matrix": (self.output_dimension, self.state_dimension),
            "dimensionless_state_matrix": (
                self.state_dimension,
                self.state_dimension,
            ),
            "dimensionless_input_matrix": (
                self.state_dimension,
                self.control_dimension,
            ),
            "dimensionless_output_matrix": (
                self.output_dimension,
                self.state_dimension,
            ),
        }
        for name, shape in expected.items():
            array = np.asarray(getattr(self, name), dtype=float)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite with shape {shape}")
            array = array.copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        for name in ("observability_matrix", "controllability_matrix"):
            array = np.asarray(getattr(self, name), dtype=float)
            if array.ndim != 2 or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be a finite matrix")
            array = array.copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)


def _finite_vector(name: str, value: npt.ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def _positive_steps(name: str, value: npt.ArrayLike, size: int) -> FloatArray:
    steps = _finite_vector(name, value)
    if steps.shape != (size,) or np.any(steps <= 0.0):
        raise ValueError(f"{name} must contain {size} finite positive values")
    return steps


def central_jacobian(
    function: Callable[[FloatArray], npt.ArrayLike],
    point: npt.ArrayLike,
    steps: npt.ArrayLike,
) -> FloatArray:
    """Return a central-difference Jacobian using explicit coordinate steps."""
    x = _finite_vector("point", point)
    h = _positive_steps("steps", steps, x.size)
    baseline = _finite_vector("function output", function(x.copy()))
    jacobian = np.empty((baseline.size, x.size), dtype=float)
    for index, step in enumerate(h):
        positive = x.copy()
        negative = x.copy()
        positive[index] += step
        negative[index] -= step
        upper = _finite_vector("function output", function(positive))
        lower = _finite_vector("function output", function(negative))
        if upper.shape != baseline.shape or lower.shape != baseline.shape:
            raise ValueError("function output dimension changed during linearization")
        jacobian[:, index] = (upper - lower) / (2.0 * step)
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("finite-difference Jacobian contains non-finite values")
    return jacobian


def observability_matrix(
    state_matrix: npt.ArrayLike, output_matrix: npt.ArrayLike
) -> FloatArray:
    """Construct ``[C; C A; ...; C A^(n-1)]`` for an n-state system."""
    state = _square_matrix("state_matrix", state_matrix)
    output = _matrix("output_matrix", output_matrix)
    if output.shape[1] != state.shape[0]:
        raise ValueError("output_matrix column count must equal state dimension")
    power = np.eye(state.shape[0])
    blocks: list[FloatArray] = []
    for _ in range(state.shape[0]):
        blocks.append(output @ power)
        power = power @ state
    return np.vstack(blocks)


def controllability_matrix(
    state_matrix: npt.ArrayLike, input_matrix: npt.ArrayLike
) -> FloatArray:
    """Construct ``[B, A B, ..., A^(n-1) B]`` for an n-state system."""
    state = _square_matrix("state_matrix", state_matrix)
    control = _matrix("input_matrix", input_matrix)
    if control.shape[0] != state.shape[0]:
        raise ValueError("input_matrix row count must equal state dimension")
    blocks: list[FloatArray] = []
    propagated = control.copy()
    for _ in range(state.shape[0]):
        blocks.append(propagated)
        propagated = state @ propagated
    return np.hstack(blocks)


def rank_diagnostic(matrix: npt.ArrayLike, tolerance: RankTolerance) -> RankDiagnostic:
    """Classify matrix rank using an inspectable mixed tolerance."""
    array = _matrix("matrix", matrix)
    singular_values = np.linalg.svd(array, compute_uv=False)
    leading = float(singular_values[0]) if singular_values.size else 0.0
    threshold = max(tolerance.absolute, tolerance.relative * leading)
    rank = int(np.count_nonzero(singular_values > threshold))
    retained = float(singular_values[rank - 1]) if rank else None
    condition = leading / retained if retained is not None else None
    return RankDiagnostic(
        rank=rank,
        full_rank=rank == min(array.shape),
        matrix_shape=(int(array.shape[0]), int(array.shape[1])),
        threshold=threshold,
        singular_values=tuple(float(value) for value in singular_values),
        smallest_retained=retained,
        retained_condition_number=condition,
    )


def audit_local_linearization(
    *,
    dynamics: DynamicsFunction,
    output: OutputFunction,
    state: npt.ArrayLike,
    control: npt.ArrayLike,
    state_steps: npt.ArrayLike,
    control_steps: npt.ArrayLike,
    scales: NondimensionalScales,
    tolerance: RankTolerance,
) -> LocalLinearAudit:
    """Linearize a continuous system locally and audit first-order rank."""
    x = _finite_vector("state", state)
    u = _finite_vector("control", control)
    x_steps = _positive_steps("state_steps", state_steps, x.size)
    u_steps = _positive_steps("control_steps", control_steps, u.size)

    def state_dynamics(candidate: FloatArray) -> npt.ArrayLike:
        return dynamics(candidate, u.copy())

    def control_dynamics(candidate: FloatArray) -> npt.ArrayLike:
        return dynamics(x.copy(), candidate)

    state_matrix = central_jacobian(state_dynamics, x, x_steps)
    input_matrix = central_jacobian(control_dynamics, u, u_steps)
    output_matrix = central_jacobian(output, x, x_steps)
    if state_matrix.shape[0] != x.size:
        raise ValueError("dynamics output dimension must equal state dimension")
    state_scales, control_scales, output_scales = scales.arrays(
        state_dimension=x.size,
        control_dimension=u.size,
        output_dimension=output_matrix.shape[0],
    )
    time_scale = scales.characteristic_time_s
    dimensionless_state = (
        time_scale
        * state_matrix
        * state_scales[np.newaxis, :]
        / state_scales[:, np.newaxis]
    )
    dimensionless_input = (
        time_scale
        * input_matrix
        * control_scales[np.newaxis, :]
        / state_scales[:, np.newaxis]
    )
    dimensionless_output = (
        output_matrix * state_scales[np.newaxis, :] / output_scales[:, np.newaxis]
    )
    observable = observability_matrix(dimensionless_state, dimensionless_output)
    controllable = controllability_matrix(dimensionless_state, dimensionless_input)
    return LocalLinearAudit(
        state_dimension=x.size,
        control_dimension=u.size,
        output_dimension=output_matrix.shape[0],
        state=tuple(float(value) for value in x),
        control=tuple(float(value) for value in u),
        state_steps=tuple(float(value) for value in x_steps),
        control_steps=tuple(float(value) for value in u_steps),
        state_matrix=state_matrix,
        input_matrix=input_matrix,
        output_matrix=output_matrix,
        dimensionless_state_matrix=dimensionless_state,
        dimensionless_input_matrix=dimensionless_input,
        dimensionless_output_matrix=dimensionless_output,
        observability_matrix=observable,
        controllability_matrix=controllable,
        observability=rank_diagnostic(observable, tolerance),
        controllability=rank_diagnostic(controllable, tolerance),
        scales=scales,
    )


def audit_double_pendulum_configuration_state(
    params: GolfModelParams,
    *,
    state: npt.ArrayLike,
    control: npt.ArrayLike,
    state_steps: npt.ArrayLike,
    control_steps: npt.ArrayLike,
    scales: NondimensionalScales,
    tolerance: RankTolerance,
    generalized_control_map: npt.ArrayLike | None = None,
    output_map: npt.ArrayLike | None = None,
) -> LocalLinearAudit:
    """Audit the ODE double pendulum under declared sensing and actuation maps."""
    backend = make_backend("ode", params)
    command = _finite_vector("control", control)
    control_mapping = (
        np.eye(2, dtype=float)
        if generalized_control_map is None
        else _matrix("generalized_control_map", generalized_control_map)
    )
    if control_mapping.shape != (2, command.size):
        raise ValueError(f"generalized_control_map must have shape {(2, command.size)}")
    measurement_mapping = (
        np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
        if output_map is None
        else _matrix("output_map", output_map)
    )
    if measurement_mapping.shape[1] != 4:
        raise ValueError("output_map must have four state columns")

    def dynamics(
        candidate_state: FloatArray, candidate_control: FloatArray
    ) -> FloatArray:
        derivative = np.empty(4, dtype=float)
        derivative[:2] = candidate_state[2:]
        derivative[2:] = backend.forward_dynamics(
            candidate_state[:2],
            candidate_state[2:],
            control_mapping @ candidate_control,
        )
        return derivative

    return audit_local_linearization(
        dynamics=dynamics,
        output=lambda candidate: measurement_mapping @ candidate,
        state=state,
        control=command,
        state_steps=state_steps,
        control_steps=control_steps,
        scales=scales,
        tolerance=tolerance,
    )


def _matrix(name: str, value: npt.ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or 0 in array.shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite matrix")
    return array


def _square_matrix(name: str, value: npt.ArrayLike) -> FloatArray:
    array = _matrix(name, value)
    if array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be square")
    return array
