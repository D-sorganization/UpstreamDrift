# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Planar segmented-chain dynamics for whip-motion analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TypeAlias

import numpy as np
from numpy.typing import NDArray

GRAVITY_M_S2: Final[float] = 9.80665
DEFAULT_SEGMENT_COUNT: Final[int] = 12
DEFAULT_SEGMENT_LENGTH_M: Final[float] = 0.18
DEFAULT_LINK_MASS_KG: Final[float] = 0.12
DEFAULT_DAMPING: Final[float] = 0.0015
DEFAULT_COUPLING: Final[float] = 0.025
DEFAULT_BEND_DAMPING: Final[float] = 0.001
MIN_SEGMENTS_FOR_CATENARY: Final[int] = 2
INTEGRATION_SUBSTEPS: Final[int] = 32

FloatArray: TypeAlias = NDArray[np.float64]


def _require_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def _as_float_array(name: str, values: FloatArray, expected_size: int) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if array.size != expected_size:
        raise ValueError(f"{name} must contain {expected_size} values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class ChainConfig:
    """Physical configuration for a side-view segmented chain.

    Preconditions:
        ``segment_count`` is at least 1. Length, mass, and gravity are positive.
        Damping and coupling are non-negative.
    """

    segment_count: int = DEFAULT_SEGMENT_COUNT
    segment_length_m: float = DEFAULT_SEGMENT_LENGTH_M
    link_mass_kg: float = DEFAULT_LINK_MASS_KG
    gravity_m_s2: float = GRAVITY_M_S2
    damping: float = DEFAULT_DAMPING
    coupling: float = DEFAULT_COUPLING
    bend_damping: float = DEFAULT_BEND_DAMPING

    def __post_init__(self) -> None:
        if self.segment_count < 1:
            raise ValueError("segment_count must be at least 1")
        _require_positive("segment_length_m", self.segment_length_m)
        _require_positive("link_mass_kg", self.link_mass_kg)
        _require_positive("gravity_m_s2", self.gravity_m_s2)
        if self.damping < 0.0:
            raise ValueError("damping must be non-negative")
        if self.coupling < 0.0:
            raise ValueError("coupling must be non-negative")
        if self.bend_damping < 0.0:
            raise ValueError("bend_damping must be non-negative")

    @property
    def total_length_m(self) -> float:
        return self.segment_count * self.segment_length_m


@dataclass(frozen=True)
class ChainStateMetrics:
    """Derived chain metrics useful for whip-motion analysis."""

    tip_speed_m_s: float
    center_of_mass_m: FloatArray
    max_curvature_rad: float


@dataclass(frozen=True)
class ChainState:
    """Angles and angular velocities for all chain links.

    Angles are measured from vertical-down in radians.
    """

    angles_rad: FloatArray
    angular_velocities_rad_s: FloatArray

    @classmethod
    def stationary(cls, config: ChainConfig, angle_rad: float = 0.0) -> ChainState:
        angles = np.full(config.segment_count, angle_rad, dtype=np.float64)
        velocities = np.zeros(config.segment_count, dtype=np.float64)
        return cls(angles_rad=angles, angular_velocities_rad_s=velocities)

    def validated(self, config: ChainConfig) -> ChainState:
        angles = _as_float_array("angles_rad", self.angles_rad, config.segment_count)
        velocities = _as_float_array(
            "angular_velocities_rad_s",
            self.angular_velocities_rad_s,
            config.segment_count,
        )
        return ChainState(angles_rad=angles, angular_velocities_rad_s=velocities)

    def node_positions(
        self,
        config: ChainConfig,
        anchor_xy_m: tuple[float, float] = (0.0, 0.0),
    ) -> FloatArray:
        checked = self.validated(config)
        offsets = segment_vectors(config, checked.angles_rad)
        endpoints = np.vstack(
            (
                np.zeros((1, 2), dtype=np.float64),
                np.cumsum(offsets, axis=0),
            )
        )
        anchor = np.asarray(anchor_xy_m, dtype=np.float64)
        if anchor.shape != (2,) or not np.all(np.isfinite(anchor)):
            raise ValueError("anchor_xy_m must contain two finite coordinates")
        return endpoints + anchor

    def metrics(self, config: ChainConfig) -> ChainStateMetrics:
        checked = self.validated(config)
        positions = checked.node_positions(config)
        velocities = node_velocities(config, checked)
        midpoints = link_midpoints(positions)
        curvature = np.diff(checked.angles_rad)
        max_curvature = float(np.max(np.abs(curvature))) if curvature.size else 0.0
        return ChainStateMetrics(
            tip_speed_m_s=float(np.linalg.norm(velocities[-1])),
            center_of_mass_m=np.mean(midpoints, axis=0),
            max_curvature_rad=max_curvature,
        )


@dataclass(frozen=True)
class ChainRollout:
    """Time history returned by :func:`simulate_chain`."""

    states: tuple[ChainState, ...]
    positions: FloatArray
    energy_j: FloatArray
    tip_speed_m_s: FloatArray


def segment_vectors(config: ChainConfig, angles_rad: FloatArray) -> FloatArray:
    angles = _as_float_array("angles_rad", angles_rad, config.segment_count)
    return config.segment_length_m * np.column_stack((np.sin(angles), np.cos(angles)))


def link_midpoints(node_positions: FloatArray) -> FloatArray:
    nodes = np.asarray(node_positions, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1] != 2 or nodes.shape[0] < 2:
        raise ValueError("node_positions must have shape (n >= 2, 2)")
    return 0.5 * (nodes[:-1] + nodes[1:])


def node_velocities(config: ChainConfig, state: ChainState) -> FloatArray:
    checked = state.validated(config)
    derivatives = config.segment_length_m * np.column_stack(
        (
            np.cos(checked.angles_rad) * checked.angular_velocities_rad_s,
            -np.sin(checked.angles_rad) * checked.angular_velocities_rad_s,
        )
    )
    return np.vstack(
        (
            np.zeros((1, 2), dtype=np.float64),
            np.cumsum(derivatives, axis=0),
        )
    )


def total_energy(config: ChainConfig, state: ChainState) -> float:
    checked = state.validated(config)
    positions = checked.node_positions(config)
    velocities = node_velocities(config, checked)
    kinetic = 0.5 * config.link_mass_kg * np.sum(velocities[1:] ** 2)
    potential_height = config.total_length_m - positions[1:, 1]
    potential = config.link_mass_kg * config.gravity_m_s2 * np.sum(potential_height)
    return float(kinetic + potential)


def initial_catenary_angles(segment_count: int, sag_rad: float) -> FloatArray:
    if segment_count < MIN_SEGMENTS_FOR_CATENARY:
        raise ValueError("segment_count must be at least 2")
    if sag_rad < 0.0:
        raise ValueError("sag_rad must be non-negative")
    return np.linspace(-sag_rad, sag_rad, segment_count, dtype=np.float64)


def initial_tip_kick_velocities(segment_count: int, amplitude_rad_s: float) -> FloatArray:
    """Return a smooth initial angular-velocity profile concentrated at the tip.

    Preconditions:
        ``segment_count`` is at least 1 and ``amplitude_rad_s`` is non-negative.
    """

    if segment_count < 1:
        raise ValueError("segment_count must be at least 1")
    if amplitude_rad_s < 0.0:
        raise ValueError("amplitude_rad_s must be non-negative")
    if segment_count == 1:
        return np.asarray([amplitude_rad_s], dtype=np.float64)
    normalized_link_position = np.linspace(0.0, 1.0, segment_count, dtype=np.float64)
    return amplitude_rad_s * normalized_link_position**2


def random_wadded_chain_state(
    config: ChainConfig,
    *,
    angle_span_rad: float = np.pi,
    velocity_span_rad_s: float = 0.0,
    seed: int | None = None,
) -> ChainState:
    """Return a deterministic random curled-chain start for analysis.

    Preconditions:
        ``angle_span_rad`` and ``velocity_span_rad_s`` are non-negative.
    """

    if angle_span_rad < 0.0:
        raise ValueError("angle_span_rad must be non-negative")
    if velocity_span_rad_s < 0.0:
        raise ValueError("velocity_span_rad_s must be non-negative")
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-angle_span_rad, angle_span_rad, config.segment_count)
    velocities = rng.uniform(-velocity_span_rad_s, velocity_span_rad_s, config.segment_count)
    return ChainState(angles.astype(np.float64), velocities.astype(np.float64)).validated(config)


def _angular_acceleration(
    config: ChainConfig,
    state: ChainState,
    torques_nm: FloatArray,
) -> FloatArray:
    checked = state.validated(config)
    torques = _as_float_array("torques_nm", torques_nm, config.segment_count)
    inertia = config.link_mass_kg * config.segment_length_m**2 / 3.0
    gravity_torque = (
        -config.link_mass_kg
        * config.gravity_m_s2
        * (config.segment_length_m / 2.0)
        * np.sin(checked.angles_rad)
    )
    damping_torque = -config.damping * checked.angular_velocities_rad_s
    neighbor_sum = np.zeros(config.segment_count, dtype=np.float64)
    neighbor_sum[:-1] += checked.angles_rad[1:] - checked.angles_rad[:-1]
    neighbor_sum[1:] += checked.angles_rad[:-1] - checked.angles_rad[1:]
    coupling_torque = config.coupling * neighbor_sum
    neighbor_velocity_sum = np.zeros(config.segment_count, dtype=np.float64)
    neighbor_velocity_sum[:-1] += (
        checked.angular_velocities_rad_s[1:] - checked.angular_velocities_rad_s[:-1]
    )
    neighbor_velocity_sum[1:] += (
        checked.angular_velocities_rad_s[:-1] - checked.angular_velocities_rad_s[1:]
    )
    bend_damping_torque = config.bend_damping * neighbor_velocity_sum
    return (
        gravity_torque + damping_torque + coupling_torque + bend_damping_torque + torques
    ) / inertia


def step_chain(
    config: ChainConfig,
    state: ChainState,
    dt_s: float,
    torques_nm: FloatArray | None = None,
) -> ChainState:
    """Advance the chain one semi-implicit Euler step.

    Preconditions:
        ``dt_s`` is positive and optional torques match the segment count.
    """

    _require_positive("dt_s", dt_s)
    current = state.validated(config)
    torques = (
        np.zeros(config.segment_count, dtype=np.float64)
        if torques_nm is None
        else np.asarray(torques_nm, dtype=np.float64)
    )
    sub_dt = dt_s / INTEGRATION_SUBSTEPS
    for _substep in range(INTEGRATION_SUBSTEPS):
        acceleration = _angular_acceleration(config, current, torques)
        next_velocities = current.angular_velocities_rad_s + acceleration * sub_dt
        next_angles = current.angles_rad + next_velocities * sub_dt
        current = ChainState(next_angles, next_velocities)
    return current


def simulate_chain(
    config: ChainConfig,
    initial_state: ChainState,
    steps: int,
    dt_s: float,
    torque_history_nm: FloatArray | None = None,
) -> ChainRollout:
    """Simulate a chain for a fixed number of time steps.

    Preconditions:
        ``steps`` and ``dt_s`` are positive. ``torque_history_nm`` has shape
        ``(steps, segment_count)`` when supplied.
    """

    if steps < 1:
        raise ValueError("steps must be at least 1")
    _require_positive("dt_s", dt_s)
    torques: FloatArray
    if torque_history_nm is None:
        torques = np.zeros((steps, config.segment_count), dtype=np.float64)
    else:
        torques = np.asarray(torque_history_nm, dtype=np.float64)
        if torques.shape != (steps, config.segment_count):
            raise ValueError("torque_history_nm has incompatible shape")
    states = [initial_state.validated(config)]
    for step_index in range(steps):
        states.append(step_chain(config, states[-1], dt_s, torques[step_index]))
    positions = np.stack([state.node_positions(config) for state in states])
    energy = np.asarray([total_energy(config, state) for state in states])
    tip_speed = np.asarray([state.metrics(config).tip_speed_m_s for state in states])
    return ChainRollout(
        states=tuple(states),
        positions=positions,
        energy_j=energy,
        tip_speed_m_s=tip_speed,
    )


def steps_for_duration(duration_s: float, dt_s: float) -> int:
    """Return the number of integration steps needed for a requested duration.

    Preconditions:
        ``duration_s`` and ``dt_s`` are positive.
    """

    _require_positive("duration_s", duration_s)
    _require_positive("dt_s", dt_s)
    return max(1, int(np.ceil(duration_s / dt_s)))


def simulate_chain_for_duration(
    config: ChainConfig,
    initial_state: ChainState,
    duration_s: float,
    dt_s: float,
    torque_history_nm: FloatArray | None = None,
) -> ChainRollout:
    """Simulate a chain for a user-requested physical duration.

    Preconditions:
        ``duration_s`` and ``dt_s`` are positive. ``torque_history_nm`` has shape
        ``(steps_for_duration(duration_s, dt_s), segment_count)`` when supplied.
    """

    steps = steps_for_duration(duration_s, dt_s)
    return simulate_chain(config, initial_state, steps, dt_s, torque_history_nm)
