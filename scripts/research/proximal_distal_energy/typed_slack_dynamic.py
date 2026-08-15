"""Dynamic excitation and identifiability audits for typed slack classes.

The audit operates on synthetic scalar channels. Mechanical passivity applies
only to the four mechanical constitutive classes; a control deadband is audited
as a delayed signal-transmission map and is never relabelled as stored energy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import numpy.typing as npt

from .typed_slack import SlackParameters, energy_residual, evaluate_slack

FloatArray = npt.NDArray[np.float64]
ExcitationKind = Literal["slow_sine", "multisine_reversal"]


@dataclass(frozen=True, slots=True)
class DynamicSlackParameters:
    """One constitutive class plus a control-channel time constant."""

    constitutive: SlackParameters
    time_constant_s: float = 0.02

    def __post_init__(self) -> None:
        if not np.isfinite(self.time_constant_s) or self.time_constant_s <= 0.0:
            raise ValueError("time_constant_s must be finite and positive")


@dataclass(frozen=True, slots=True)
class DynamicSlackResult:
    """Trace and ledger for one excitation of one declared slack class."""

    transmitted: FloatArray
    stored_energy_j: FloatArray
    engaged: npt.NDArray[np.bool_]
    input_work_j: float
    dissipative_work_j: float
    energy_residual_j: float
    loop_area_j: float
    passivity_applicable: bool
    activation_delay_s: float


@dataclass(frozen=True, slots=True)
class SensitivityAudit:
    """Dimensionless local output-sensitivity singular-value audit."""

    parameter_names: tuple[str, ...]
    scaled_singular_values: tuple[float, ...]
    rank: int
    condition_number: float | None
    minimum_scaled_singular_value: float


def excitation(time_s: npt.ArrayLike, kind: ExcitationKind) -> FloatArray:
    """Return a closed-cycle slow or persistently exciting displacement."""

    time = np.asarray(time_s, dtype=np.float64)
    if time.ndim != 1 or time.size < 3 or np.any(np.diff(time) <= 0.0):
        raise ValueError("time_s must be a strictly increasing one-dimensional array")
    phase = (time - time[0]) / (time[-1] - time[0])
    if kind == "slow_sine":
        return 0.026 * np.sin(2.0 * np.pi * phase)
    if kind == "multisine_reversal":
        return (
            0.018 * np.sin(2.0 * np.pi * phase)
            + 0.008 * np.sin(6.0 * np.pi * phase)
            + 0.004 * np.sin(10.0 * np.pi * phase)
        )
    raise ValueError(f"unsupported excitation kind: {kind}")


def _control_response(
    time: FloatArray,
    command: FloatArray,
    parameters: DynamicSlackParameters,
) -> DynamicSlackResult:
    constitutive = parameters.constitutive
    target = constitutive.stiffness * (
        np.sign(command) * np.maximum(np.abs(command) - constitutive.threshold, 0.0)
    )
    transmitted = np.zeros_like(target)
    for index in range(1, time.size):
        step = time[index] - time[index - 1]
        alpha = 1.0 - np.exp(-step / parameters.time_constant_s)
        transmitted[index] = transmitted[index - 1] + alpha * (
            target[index] - transmitted[index - 1]
        )
    return DynamicSlackResult(
        transmitted=transmitted,
        stored_energy_j=np.zeros_like(target),
        engaged=np.abs(command) > constitutive.threshold,
        input_work_j=0.0,
        dissipative_work_j=0.0,
        energy_residual_j=0.0,
        loop_area_j=0.0,
        passivity_applicable=False,
        activation_delay_s=parameters.time_constant_s,
    )


def simulate_dynamic_slack(
    time_s: npt.ArrayLike,
    signal: npt.ArrayLike,
    parameters: DynamicSlackParameters,
) -> DynamicSlackResult:
    """Evaluate one class under a shared excitation without class mixing."""

    time = np.asarray(time_s, dtype=np.float64)
    displacement = np.asarray(signal, dtype=np.float64)
    if time.ndim != 1 or time.size < 3 or displacement.shape != time.shape:
        raise ValueError("time_s and signal must be matching one-dimensional arrays")
    if np.any(np.diff(time) <= 0.0) or not np.all(np.isfinite(displacement)):
        raise ValueError("time_s must increase and signal must be finite")
    if parameters.constitutive.kind == "control_deadband":
        return _control_response(time, displacement, parameters)

    rate = np.gradient(displacement, time, edge_order=2)
    trace = evaluate_slack(displacement, rate, parameters.constitutive)
    input_work = float(np.trapezoid(trace.transmitted * rate, x=time))
    dissipative_work = float(np.trapezoid(trace.dissipative * rate, x=time))
    return DynamicSlackResult(
        transmitted=trace.transmitted,
        stored_energy_j=trace.stored_energy,
        engaged=trace.engaged,
        input_work_j=input_work,
        dissipative_work_j=dissipative_work,
        energy_residual_j=float(energy_residual(time, rate, trace)),
        loop_area_j=input_work,
        passivity_applicable=True,
        activation_delay_s=0.0,
    )


def _parameter_names(parameters: DynamicSlackParameters) -> tuple[str, ...]:
    kind = parameters.constitutive.kind
    if kind == "structural_preload":
        return ("stiffness", "damping", "preload")
    if kind == "control_deadband":
        return ("threshold", "stiffness", "time_constant_s")
    return ("threshold", "stiffness", "damping")


def _replace_parameter(
    parameters: DynamicSlackParameters,
    name: str,
    value: float,
) -> DynamicSlackParameters:
    if name == "time_constant_s":
        return replace(parameters, time_constant_s=value)
    return replace(
        parameters, constitutive=replace(parameters.constitutive, **{name: value})
    )


def scaled_sensitivity_audit(
    time_s: npt.ArrayLike,
    signal: npt.ArrayLike,
    parameters: DynamicSlackParameters,
) -> SensitivityAudit:
    """Audit local practical identifiability with dimensionless sensitivities."""

    names = _parameter_names(parameters)
    columns = []
    for name in names:
        value = (
            parameters.time_constant_s
            if name == "time_constant_s"
            else float(getattr(parameters.constitutive, name))
        )
        scale = max(abs(value), 1e-6)
        step = max(scale * 1e-4, 1e-8)
        lower = max(value - step, 1e-10) if name != "preload" else value - step
        upper = value + step
        low = simulate_dynamic_slack(
            time_s,
            signal,
            _replace_parameter(parameters, name, lower),
        ).transmitted
        high = simulate_dynamic_slack(
            time_s,
            signal,
            _replace_parameter(parameters, name, upper),
        ).transmitted
        columns.append(scale * (high - low) / (upper - lower))
    jacobian = np.column_stack(columns)
    output_scale = max(float(np.linalg.norm(jacobian, ord="fro")), 1e-12)
    singular_values = np.linalg.svd(jacobian / output_scale, compute_uv=False)
    tolerance = max(float(singular_values[0]) * 1e-7, 1e-10)
    rank = int(np.sum(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if rank == len(names) and singular_values[-1] > 0.0
        else None
    )
    return SensitivityAudit(
        parameter_names=names,
        scaled_singular_values=tuple(float(value) for value in singular_values),
        rank=rank,
        condition_number=condition,
        minimum_scaled_singular_value=float(singular_values[-1]),
    )
