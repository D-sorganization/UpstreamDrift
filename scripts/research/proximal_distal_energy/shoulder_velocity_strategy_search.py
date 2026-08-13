"""Trajectory search for proximal-speed and wrist-release coordination.

The search varies when proximal drive is reduced, the residual proximal torque,
and wrist-release time. It reports speed together with grip braking, peak load,
and exact pointwise drift/control work closure. The proximal coordinate remains
the first link of a fixed-hub planar model, not anatomical torso motion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from scripts.research.proximal_distal_energy.double_pendulum_attribution import (
    double_pendulum_joint_transfer_trajectory,
)
from scripts.research.proximal_distal_energy.run_experiments import (
    DT,
    HORIZON,
    rollout_controls,
)
from scripts.research.proximal_distal_energy.shoulder_velocity_transfer import (
    nondominated_indices,
)
from scripts.research.proximal_distal_energy.swing_model import (
    PlanarInertials,
    clubhead_speed,
    find_impact,
)
from src.shared.python.biomechanics.drift_control_transfer import compute_power_and_work
from src.shared.python.simulation_backends import GolfModelParams

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ShoulderVelocityProgram:
    """Piecewise-constant proximal drive and distal release program."""

    shoulder_cut_s: float
    shoulder_torque_before_nm: float
    shoulder_torque_after_nm: float
    wrist_release_s: float
    wrist_restrain_nm: float
    wrist_drive_nm: float

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.shoulder_cut_s,
                self.shoulder_torque_before_nm,
                self.shoulder_torque_after_nm,
                self.wrist_release_s,
                self.wrist_restrain_nm,
                self.wrist_drive_nm,
            ),
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("program values must be finite")
        if self.shoulder_cut_s < 0.0 or self.wrist_release_s < 0.0:
            raise ValueError("event times must be finite and non-negative")
        if self.wrist_restrain_nm < 0.0 or self.wrist_drive_nm < 0.0:
            raise ValueError("wrist torque magnitudes must be non-negative")


@dataclass(frozen=True, slots=True)
class ShoulderVelocityOutcome:
    """Finite trajectory outcomes for one declared program."""

    program: ShoulderVelocityProgram
    valid_impact: bool
    impact_time_s: float
    impact_speed_m_s: float
    proximal_velocity_at_release_rad_s: float
    proximal_velocity_at_impact_rad_s: float
    total_grip_work_j: float
    drift_grip_work_j: float
    control_grip_work_j: float
    transfer_work_closure_residual_j: float
    braking_grip_work_j: float
    peak_grip_force_n: float
    model_tier: str = "exact_planar_double_pendulum_fixed_hub"


def build_controls(
    program: ShoulderVelocityProgram, *, horizon: int = HORIZON, dt_s: float = DT
) -> FloatArray:
    """Build a bounded control history from one validated program."""
    if not isinstance(program, ShoulderVelocityProgram):
        raise TypeError("program must be a ShoulderVelocityProgram")
    if not isinstance(horizon, int) or horizon < 2:
        raise ValueError("horizon must be an integer of at least two")
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be positive and finite")
    time = np.arange(horizon, dtype=float) * dt_s
    controls = np.empty((horizon, 2), dtype=float)
    controls[:, 0] = np.where(
        time < program.shoulder_cut_s,
        program.shoulder_torque_before_nm,
        program.shoulder_torque_after_nm,
    )
    controls[:, 1] = np.where(
        time < program.wrist_release_s,
        -program.wrist_restrain_nm,
        program.wrist_drive_nm,
    )
    return controls


def _window_work(power: FloatArray, time: FloatArray, start: int, end: int) -> float:
    return float(np.trapezoid(power[start : end + 1], time[start : end + 1]))


def _evaluate_program(
    program: ShoulderVelocityProgram, params: GolfModelParams
) -> ShoulderVelocityOutcome:
    controls = build_controls(program)
    time, q, velocity, applied = rollout_controls(params, controls)
    inertials = PlanarInertials.from_params(params)
    impact = find_impact(time, q, velocity, inertials)
    valid = impact is not None
    impact_time = float(impact[0]) if impact is not None else float(time[-1])
    impact_index = int(np.searchsorted(time, impact_time, side="right") - 1)
    impact_index = max(1, min(impact_index, len(time) - 1))
    release_index = int(np.searchsorted(time, program.wrist_release_s, side="left"))
    release_index = max(0, min(release_index, impact_index - 1))

    trajectory = double_pendulum_joint_transfer_trajectory(
        time, q, velocity, applied, params
    )
    power = compute_power_and_work(trajectory)
    total = _window_work(
        power.force_power_total[:, 1], time, release_index, impact_index
    )
    drift = _window_work(
        power.force_power_drift[:, 1], time, release_index, impact_index
    )
    control = _window_work(
        power.force_power_control[:, 1], time, release_index, impact_index
    )
    total_power = power.force_power_total[release_index : impact_index + 1, 1]
    braking = -float(
        np.trapezoid(
            np.minimum(total_power, 0.0), time[release_index : impact_index + 1]
        )
    )
    grip_force = trajectory.force_total[release_index : impact_index + 1, 1]
    speed = clubhead_speed(inertials, q, velocity)
    return ShoulderVelocityOutcome(
        program=program,
        valid_impact=valid,
        impact_time_s=impact_time,
        impact_speed_m_s=float(impact[1]) if impact is not None else float(speed[-1]),
        proximal_velocity_at_release_rad_s=float(velocity[release_index, 0]),
        proximal_velocity_at_impact_rad_s=float(velocity[impact_index, 0]),
        total_grip_work_j=total,
        drift_grip_work_j=drift,
        control_grip_work_j=control,
        transfer_work_closure_residual_j=total - drift - control,
        braking_grip_work_j=braking,
        peak_grip_force_n=float(np.max(np.linalg.norm(grip_force, axis=1))),
    )


def evaluate_programs(
    programs: tuple[ShoulderVelocityProgram, ...], params: GolfModelParams
) -> tuple[ShoulderVelocityOutcome, ...]:
    """Evaluate programs without discarding invalid impact attempts."""
    if not programs or any(
        not isinstance(item, ShoulderVelocityProgram) for item in programs
    ):
        raise ValueError("programs must contain ShoulderVelocityProgram values")
    if not isinstance(params, GolfModelParams):
        raise TypeError("params must be GolfModelParams")
    return tuple(_evaluate_program(program, params) for program in programs)


def pareto_program_indices(values: object) -> npt.NDArray[np.int64]:
    """Return programs nondominated on speed, braking work, and peak force."""
    return nondominated_indices(values, maximize=(True, False, False))


__all__ = [
    "ShoulderVelocityOutcome",
    "ShoulderVelocityProgram",
    "build_controls",
    "evaluate_programs",
    "pareto_program_indices",
]
