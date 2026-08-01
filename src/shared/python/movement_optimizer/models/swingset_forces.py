# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Force/torque estimates for the swingset rider model.

Derived from the public swingset rollout outputs. Joint torques reuse
:func:`estimate_swingset_joint_torques`; gravity and chain tension are recovered
from the center-of-mass trajectory.

Coordinate convention (inherited from :mod:`swingset`): the swing hangs from the
pivot toward **+y**, so gravity acts in **+y**. Force vectors are returned in
this model frame; the canvas projector handles the screen flip. ``com_height_m``
is reported as ``-y`` so that "up" is positive for plotting.

Design Principles:
    DBC -- every public function validates its preconditions (ValueError).
    LoD -- consumes only swingset public outputs; no GUI imports.
    DRY -- reuses estimate_swingset_joint_torques; shared finite-difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .swingset import (
    CONTROL_DIMENSION,
    SWING_POLICY_JOINT_NAMES,
    SwingRollout,
    SwingSetConfig,
    estimate_swingset_joint_torques,
)

FloatArray: TypeAlias = NDArray[np.float64]

# Maps each policy joint to the body point where its torque indicator is drawn.
_JOINT_POINT_KEYS: dict[str, str] = {
    "torso": "waist",
    "hip": "hip",
    "knee": "knee",
    "shoulder": "shoulder",
    "elbow": "elbow",
}


def _require_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class SwingForceField:
    """Force/marker vectors for a single swing frame (world frame)."""

    com_m: FloatArray
    gravity_n: FloatArray
    chain_tension_n: FloatArray
    joint_torque_nm: FloatArray
    joint_points_m: dict[str, FloatArray]


@dataclass(frozen=True)
class SwingForceHistory:
    """Time history of swing torques, power, and COM motion for plotting."""

    time_s: FloatArray
    joint_torque_nm: FloatArray
    joint_power_w: FloatArray
    swing_angle_rad: FloatArray
    com_height_m: FloatArray
    com_path_m: FloatArray
    energy_j: FloatArray


def _total_mass(config: SwingSetConfig) -> float:
    return config.seat_mass_kg + config.rider_mass_kg


def _com_positions(rollout: SwingRollout) -> FloatArray:
    """Return ``(F, 2)`` center-of-mass positions across all snapshots."""
    return np.asarray(
        [snapshot.center_of_mass_m for snapshot in rollout.snapshots],
        dtype=np.float64,
    )


def _com_accelerations(rollout: SwingRollout, dt_s: float) -> FloatArray:
    positions = _com_positions(rollout)
    return _com_accelerations_from_positions(positions, dt_s)


def _com_accelerations_from_positions(positions: FloatArray, dt_s: float) -> FloatArray:
    if positions.shape[0] < 2:  # pragma: no cover - rollouts always have >= 2 snapshots
        return np.zeros_like(positions)
    return np.gradient(positions, dt_s, axis=0)


def swing_force_fields(
    config: SwingSetConfig,
    rollout: SwingRollout,
    dt_s: float,
) -> tuple[SwingForceField, ...]:
    """Return force vectors for every frame in ``rollout``.

    Preconditions:
        ``dt_s`` is positive and the rollout contains at least one snapshot.
    """
    _require_positive("dt_s", dt_s)
    if not rollout.snapshots:
        raise ValueError("rollout must contain at least one snapshot")

    mass = _total_mass(config)
    gravity_vec = np.asarray([0.0, mass * config.gravity_m_s2], dtype=np.float64)
    com_positions = _com_positions(rollout)
    accelerations = _com_accelerations_from_positions(com_positions, dt_s)
    torques = estimate_swingset_joint_torques(config, rollout, dt_s)

    fields: list[SwingForceField] = []
    for frame_index, snapshot in enumerate(rollout.snapshots):
        torque_index = min(frame_index, torques.shape[0] - 1)
        chain_tension = mass * accelerations[frame_index] - gravity_vec
        joint_points = {
            joint: np.asarray(snapshot.points[_JOINT_POINT_KEYS[joint]], dtype=np.float64)
            for joint in SWING_POLICY_JOINT_NAMES
        }
        fields.append(
            SwingForceField(
                com_m=np.asarray(snapshot.center_of_mass_m, dtype=np.float64),
                gravity_n=gravity_vec,
                chain_tension_n=chain_tension,
                joint_torque_nm=torques[torque_index],
                joint_points_m=joint_points,
            )
        )
    return tuple(fields)


def swing_force_field(
    config: SwingSetConfig,
    rollout: SwingRollout,
    dt_s: float,
    frame_index: int,
) -> SwingForceField:
    """Return rider force vectors at ``frame_index``.

    Preconditions:
        ``dt_s`` is positive and ``0 <= frame_index < len(rollout.snapshots)``.
    """
    frame_count = len(rollout.snapshots)
    if not 0 <= frame_index < frame_count:
        raise ValueError(f"frame_index must be in [0, {frame_count})")
    return swing_force_fields(config, rollout, dt_s)[frame_index]


def swing_force_history(
    config: SwingSetConfig,
    rollout: SwingRollout,
    dt_s: float,
) -> SwingForceHistory:
    """Return time-series torques, power, and COM motion for the analysis panel.

    Preconditions:
        ``dt_s`` is positive and ``rollout.controls`` has shape ``(N, 5)``.
    """
    _require_positive("dt_s", dt_s)
    controls = rollout.controls
    if controls.ndim != 2 or controls.shape[1] != CONTROL_DIMENSION:
        raise ValueError("rollout.controls must have shape (N, 5)")

    torques = estimate_swingset_joint_torques(config, rollout, dt_s)
    steps = torques.shape[0]
    power = torques * controls[:steps]

    com_positions = _com_positions(rollout)[:steps]
    swing_angles = np.asarray(rollout.swing_angles_rad[:steps], dtype=np.float64)
    inertia = _total_mass(config) * config.chain_length_m**2
    velocities = np.asarray(
        [state.swing_angular_velocity_rad_s for state in rollout.states[:steps]],
        dtype=np.float64,
    )
    energy = 0.5 * inertia * velocities**2

    return SwingForceHistory(
        time_s=np.arange(steps, dtype=np.float64) * dt_s,
        joint_torque_nm=torques,
        joint_power_w=power,
        swing_angle_rad=swing_angles,
        com_height_m=-com_positions[:, 1],
        com_path_m=com_positions,
        energy_j=energy,
    )
