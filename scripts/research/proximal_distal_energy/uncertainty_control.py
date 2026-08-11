"""Coupled uncertainty, identifiability, and bounded-control primitives.

The module deliberately separates numerical strategy comparison from claims
about human physiology.  ``ActuatorLimits`` defines a transparent engineering
surrogate: pure delay, first-order activation, torque-rate, asymmetric
torque--velocity capacity, and joint impedance.  Its effort metric is therefore
a declared model proxy, not metabolic energy or muscle activation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from scripts.research.proximal_distal_energy.two_arm_closed_loop import TwoArmControl

FloatArray: TypeAlias = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ControlProgram:
    """One predeclared open-loop timing and impedance strategy."""

    name: str
    wrist_onset_s: float
    early_wrist_nm: float
    late_wrist_nm: float
    shoulder_scale: float
    elbow_scale: float
    impedance_nms_rad: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.wrist_onset_s,
                self.early_wrist_nm,
                self.late_wrist_nm,
                self.shoulder_scale,
                self.elbow_scale,
                self.impedance_nms_rad,
            ],
            dtype=np.float64,
        )
        if not self.name.strip():
            raise ValueError("name must be nonempty")
        if not np.all(np.isfinite(values)):
            raise ValueError("control-program fields must be finite")
        if not (0.0 <= self.wrist_onset_s <= 0.4):
            raise ValueError("wrist_onset_s must be in [0, 0.4]")
        if self.shoulder_scale <= 0.0 or self.elbow_scale <= 0.0:
            raise ValueError("control scales must be positive")
        if self.impedance_nms_rad < 0.0:
            raise ValueError("impedance_nms_rad must be nonnegative")


@dataclass(frozen=True, slots=True)
class ActuatorLimits:
    """Predeclared delayed actuator-surrogate bounds."""

    delay_s: float = 0.025
    time_constant_s: float = 0.030
    maximum_torque_rate_nm_s: float = 120.0
    concentric_velocity_rad_s: float = 20.0
    eccentric_torque_ratio: float = 1.30

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.delay_s,
                self.time_constant_s,
                self.maximum_torque_rate_nm_s,
                self.concentric_velocity_rad_s,
                self.eccentric_torque_ratio,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("actuator limits must be finite")
        if self.delay_s < 0.0:
            raise ValueError("delay_s must be nonnegative")
        if self.time_constant_s <= 0.0:
            raise ValueError("time_constant_s must be positive")
        if self.maximum_torque_rate_nm_s <= 0.0:
            raise ValueError("maximum_torque_rate_nm_s must be positive")
        if self.concentric_velocity_rad_s <= 0.0:
            raise ValueError("concentric_velocity_rad_s must be positive")
        if self.eccentric_torque_ratio < 1.0:
            raise ValueError("eccentric_torque_ratio must be at least one")


def latin_hypercube(samples: int, dimensions: int, *, seed: int) -> FloatArray:
    """Return a deterministic centered-jitter Latin hypercube in ``(0, 1)``."""

    if samples < 2 or dimensions < 1:
        raise ValueError("samples must be >= 2 and dimensions must be >= 1")
    rng = np.random.default_rng(seed)
    design = np.empty((samples, dimensions), dtype=np.float64)
    for column in range(dimensions):
        permutation = rng.permutation(samples)
        jitter = rng.uniform(0.15, 0.85, size=samples)
        design[:, column] = (permutation + jitter) / samples
    return design


def _rank(values: npt.ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("rank input must be one-dimensional and finite")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def partial_rank_correlations(
    design: npt.ArrayLike, output: npt.ArrayLike
) -> FloatArray:
    """Compute partial rank correlations by residualizing other inputs."""

    matrix = np.asarray(design, dtype=np.float64)
    response = np.asarray(output, dtype=np.float64)
    if matrix.ndim != 2 or response.shape != (matrix.shape[0],):
        raise ValueError("design/output shapes are incompatible")
    if matrix.shape[0] <= matrix.shape[1] + 2:
        raise ValueError("PRCC requires more rows than dimensions plus two")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(response)):
        raise ValueError("PRCC inputs must be finite")
    ranked_x = np.column_stack(
        [_rank(matrix[:, index]) for index in range(matrix.shape[1])]
    )
    ranked_y = _rank(response)
    result = np.empty(matrix.shape[1], dtype=np.float64)
    for index in range(matrix.shape[1]):
        others = np.delete(ranked_x, index, axis=1)
        regressors = np.column_stack([np.ones(matrix.shape[0]), others])
        x_residual = (
            ranked_x[:, index]
            - regressors
            @ np.linalg.lstsq(regressors, ranked_x[:, index], rcond=None)[0]
        )
        y_residual = (
            ranked_y - regressors @ np.linalg.lstsq(regressors, ranked_y, rcond=None)[0]
        )
        denominator = float(np.linalg.norm(x_residual) * np.linalg.norm(y_residual))
        result[index] = (
            0.0
            if denominator <= np.finfo(float).eps
            else x_residual @ y_residual / denominator
        )
    return result


def planar_two_hand_wrench_map(
    right_offset_m: float, left_offset_m: float
) -> FloatArray:
    """Map four individual-hand force components to one planar net wrench."""

    offsets = np.asarray([right_offset_m, left_offset_m], dtype=np.float64)
    if not np.all(np.isfinite(offsets)) or np.isclose(offsets[0], offsets[1]):
        raise ValueError("finite distinct grip offsets are required")
    return np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, offsets[0], 0.0, offsets[1]],
        ],
        dtype=np.float64,
    )


def nondominated_indices(objectives: npt.ArrayLike) -> tuple[int, ...]:
    """Return indices not Pareto-dominated when every column is minimized."""

    values = np.asarray(objectives, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("objectives must be a nonempty finite matrix")
    keep: list[int] = []
    for index, row in enumerate(values):
        dominated = any(
            other != index
            and np.all(values[other] <= row)
            and np.any(values[other] < row)
            for other in range(values.shape[0])
        )
        if not dominated:
            keep.append(index)
    return tuple(keep)


def _command_target(
    program: ControlProgram, time_s: float, duration_s: float
) -> FloatArray:
    if time_s < 0.0:
        return np.zeros(6, dtype=np.float64)
    phase = float(np.clip(time_s / duration_s, 0.0, 1.0))
    transition = float(np.clip((time_s - program.wrist_onset_s) / 0.035, 0.0, 1.0))
    transition = transition * transition * (3.0 - 2.0 * transition)
    wrist = (
        1.0 - transition
    ) * program.early_wrist_nm + transition * program.late_wrist_nm
    return np.array(
        [
            program.shoulder_scale * (18.0 + 4.0 * phase),
            program.elbow_scale * (7.0 - 1.5 * phase),
            wrist,
            program.shoulder_scale * (16.0 + 3.0 * phase),
            program.elbow_scale * (6.0 - phase),
            -0.65 * wrist,
        ],
        dtype=np.float64,
    )


def _actuator_velocities(qdot: FloatArray) -> FloatArray:
    if qdot.shape != (10,) or not np.all(np.isfinite(qdot)):
        raise ValueError("qdot must contain ten finite values")
    return np.array(
        [
            qdot[0],
            qdot[1],
            qdot[8] - qdot[0] - qdot[1],
            qdot[2],
            qdot[3],
            qdot[8] - qdot[2] - qdot[3],
        ],
        dtype=np.float64,
    )


def delayed_control_law(
    program: ControlProgram,
    limits: ActuatorLimits,
    *,
    duration_s: float,
    step_s: float,
) -> Callable[[float, FloatArray, FloatArray], TwoArmControl]:
    """Build a deterministic delayed and bounded state-feedback control law."""

    if duration_s <= 0.0 or step_s <= 0.0 or duration_s / step_s < 2.0:
        raise ValueError("duration and step must define at least two intervals")
    time = np.arange(0.0, duration_s + 0.5 * step_s, step_s, dtype=np.float64)
    activation = np.zeros((time.size, 6), dtype=np.float64)
    maximum_change = limits.maximum_torque_rate_nm_s * step_s
    for index in range(1, time.size):
        delayed_time = float(time[index] - limits.delay_s)
        target = _command_target(program, delayed_time, duration_s)
        unconstrained_change = (
            step_s / limits.time_constant_s * (target - activation[index - 1])
        )
        activation[index] = activation[index - 1] + np.clip(
            unconstrained_change, -maximum_change, maximum_change
        )

    isometric_capacity = np.array([36.0, 20.0, 12.0, 34.0, 18.0, 12.0])

    def law(time_s: float, q: FloatArray, qdot: FloatArray) -> TwoArmControl:
        state = np.asarray(q, dtype=np.float64)
        velocity = np.asarray(qdot, dtype=np.float64)
        if state.shape != (10,) or not np.all(np.isfinite(state)):
            raise ValueError("q must contain ten finite values")
        raw = np.array(
            [np.interp(time_s, time, activation[:, index]) for index in range(6)]
        )
        actuator_velocity = _actuator_velocities(velocity)
        raw -= program.impedance_nms_rad * actuator_velocity
        concentric = raw * actuator_velocity > 0.0
        capacity = np.where(
            concentric,
            isometric_capacity
            * np.clip(
                1.0 - np.abs(actuator_velocity) / limits.concentric_velocity_rad_s,
                0.0,
                1.0,
            ),
            isometric_capacity * limits.eccentric_torque_ratio,
        )
        delivered = np.clip(raw, -capacity, capacity)
        return TwoArmControl(
            right_shoulder_nm=float(delivered[0]),
            right_elbow_nm=float(delivered[1]),
            right_wrist_nm=float(delivered[2]),
            left_shoulder_nm=float(delivered[3]),
            left_elbow_nm=float(delivered[4]),
            left_wrist_nm=float(delivered[5]),
        )

    return law


__all__ = [
    "ActuatorLimits",
    "ControlProgram",
    "delayed_control_law",
    "latin_hypercube",
    "nondominated_indices",
    "partial_rank_correlations",
    "planar_two_hand_wrench_map",
]
