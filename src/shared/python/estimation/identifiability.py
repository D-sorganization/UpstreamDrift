"""Identifiability probes for synthetic estimation targets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import check_finite, require

FloatArray: TypeAlias = npt.NDArray[np.float64]
ObservationModel = Callable[[FloatArray], FloatArray]


@dataclass(frozen=True)
class ParameterSpec:
    """Named parameter vector layout for probe reports."""

    names: tuple[str, ...]

    def __post_init__(self) -> None:
        require(len(self.names) > 0, "at least one parameter is required")
        require(len(set(self.names)) == len(self.names), "parameter names unique")
        require(all(name.strip() for name in self.names), "parameter names non-empty")


@dataclass(frozen=True)
class IdentifiabilityReport:
    """SVD summary of a stacked observation Jacobian."""

    parameter_names: tuple[str, ...]
    jacobian: FloatArray
    singular_values: FloatArray
    right_singular_vectors: FloatArray
    rank: int
    tolerance: float

    @property
    def condition_number(self) -> float:
        """Return ``sigma_max / sigma_min`` or infinity when rank deficient."""
        if self.singular_values.size == 0 or self.singular_values[-1] <= 0.0:
            return float("inf")
        return float(self.singular_values[0] / self.singular_values[-1])

    @property
    def nullspace_directions(self) -> dict[str, FloatArray]:
        """Return right-singular directions whose singular value is tiny."""
        directions: dict[str, FloatArray] = {}
        for index, sigma in enumerate(self.singular_values):
            if sigma <= self.tolerance:
                directions[f"sv_{index}"] = self.right_singular_vectors[:, index]
        return directions

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly report summary."""
        return {
            "parameter_names": list(self.parameter_names),
            "singular_values": self.singular_values.tolist(),
            "rank": self.rank,
            "tolerance": self.tolerance,
            "condition_number": self.condition_number,
            "nullspace_directions": {
                key: value.tolist() for key, value in self.nullspace_directions.items()
            },
        }


def finite_difference_jacobian(
    model: ObservationModel,
    parameters: npt.ArrayLike,
    *,
    step: float = 1.0e-6,
) -> FloatArray:
    """Return central finite-difference Jacobian of flattened observations."""
    params = _finite_vector(parameters, "parameters")
    require(step > 0.0 and np.isfinite(step), "step must be positive and finite")
    baseline = _finite_vector(model(params.copy()), "model(parameters)")
    jacobian = np.empty((baseline.size, params.size), dtype=np.float64)
    for col in range(params.size):
        delta = np.zeros_like(params)
        local_step = step * max(1.0, abs(float(params[col])))
        delta[col] = local_step
        plus = _finite_vector(model(params + delta), "model(parameters + delta)")
        minus = _finite_vector(model(params - delta), "model(parameters - delta)")
        require(plus.shape == baseline.shape, "model output shape changed")
        require(minus.shape == baseline.shape, "model output shape changed")
        jacobian[:, col] = (plus - minus) / (2.0 * local_step)
    return jacobian


def probe_identifiability(
    model: ObservationModel,
    parameters: npt.ArrayLike,
    spec: ParameterSpec,
    *,
    step: float = 1.0e-6,
    tolerance: float | None = None,
) -> IdentifiabilityReport:
    """Compute SVD of the stacked finite-difference observation Jacobian."""
    params = _finite_vector(parameters, "parameters")
    require(len(spec.names) == params.size, "spec names must match parameters")
    jacobian = finite_difference_jacobian(model, params, step=step)
    _, singular_values, vt = np.linalg.svd(jacobian, full_matrices=False)
    tol = _rank_tolerance(jacobian, singular_values, tolerance)
    rank = int(np.sum(singular_values > tol))
    return IdentifiabilityReport(
        parameter_names=spec.names,
        jacobian=jacobian,
        singular_values=singular_values,
        right_singular_vectors=vt.T,
        rank=rank,
        tolerance=tol,
    )


def sweep_parameter(
    model: ObservationModel,
    parameters: npt.ArrayLike,
    parameter_index: int,
    values: npt.ArrayLike,
) -> FloatArray:
    """Evaluate flattened observations while sweeping one parameter."""
    params = _finite_vector(parameters, "parameters")
    sweep_values = _finite_vector(values, "values")
    require(0 <= parameter_index < params.size, "parameter_index out of range")
    outputs: list[FloatArray] = []
    for value in sweep_values:
        candidate = params.copy()
        candidate[parameter_index] = value
        outputs.append(_finite_vector(model(candidate), "model(candidate)"))
    first_shape = outputs[0].shape
    require(all(out.shape == first_shape for out in outputs), "model shape changed")
    return np.vstack(outputs)


def plot_singular_values(report: IdentifiabilityReport):
    """Return a matplotlib figure for the singular-value spectrum."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.semilogy(np.arange(report.singular_values.size), report.singular_values, "o-")
    ax.axhline(report.tolerance, color="tab:red", linestyle="--", linewidth=1.0)
    ax.set_xlabel("singular vector")
    ax.set_ylabel("singular value")
    ax.set_title("Observation Jacobian singular spectrum")
    return fig


def _finite_vector(value: npt.ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    require(arr.size > 0, f"{name} must be non-empty")
    require(check_finite(arr), f"{name} must be finite")
    return arr


def _rank_tolerance(
    jacobian: FloatArray, singular_values: FloatArray, tolerance: float | None
) -> float:
    if tolerance is not None:
        require(tolerance >= 0.0 and np.isfinite(tolerance), "tolerance invalid")
        return tolerance
    if singular_values.size == 0:
        return 0.0
    scale = max(jacobian.shape) * np.finfo(np.float64).eps
    return float(scale * singular_values[0])
