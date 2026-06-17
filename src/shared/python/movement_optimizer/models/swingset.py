# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Side-view swingset kinematics and trainable policy rollouts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Final, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, differential_evolution, minimize

from .chain_dynamics import (
    GRAVITY_M_S2,
    ChainConfig,
    ChainState,
    initial_catenary_angles,
)

FloatArray: TypeAlias = NDArray[np.float64]
Policy: TypeAlias = Callable[["SwingSetState", float], "SwingControlAction"]
ProgressCallback: TypeAlias = Callable[[int, int, float, "CyclicPolicyParameters"], None]

DEFAULT_CHAIN_SEGMENTS: Final[int] = 14
DEFAULT_CHAIN_LENGTH_M: Final[float] = 2.4
DEFAULT_SEAT_MASS_KG: Final[float] = 4.5
DEFAULT_LINK_MASS_KG: Final[float] = 0.16
DEFAULT_SEAT_PLACEMENT_THIGH_FRACTION: Final[float] = 0.35
DEFAULT_DAMPING: Final[float] = 0.04
DEFAULT_PUMP_GAIN: Final[float] = 0.65
MAX_BODY_RATE_RAD_S: Final[float] = 2.4
MAX_EXTERNAL_TORQUE_NM: Final[float] = 70.0
CONTROL_DIMENSION: Final[int] = 5
DEFAULT_POLICY_STEPS: Final[int] = 300
DEFAULT_POLICY_DT_S: Final[float] = 0.02
JOINT_ENDSTOP_MARGIN_FRACTION: Final[float] = 0.20

# Iterative optimizer (differential evolution + local refine) budget controls.
DEFAULT_OPTIMIZER_BUDGET: Final[int] = 600
MAX_OPTIMIZER_BUDGET: Final[int] = 2000
DEFAULT_OPTIMIZER_SEED: Final[int] = 0
_OPTIMIZER_POPSIZE: Final[int] = 15
SWING_TORSO_LEAN_LIMITS_RAD: Final[tuple[float, float]] = (-0.35, 0.20)
SWING_HIP_LIMITS_RAD: Final[tuple[float, float]] = (-0.15, 1.25)
SWING_KNEE_LIMITS_RAD: Final[tuple[float, float]] = (-1.35, 0.10)
SWING_SHOULDER_LIMITS_RAD: Final[tuple[float, float]] = (-0.45, 0.25)
SWING_ELBOW_LIMITS_RAD: Final[tuple[float, float]] = (0.0, 0.35)
# Minimum perpendicular elbow-offset factor at full elbow flexion. A neutral
# elbow uses the full geometric offset (1.0); a fully flexed elbow tucks toward
# the shoulder->hand line down to this factor. See ``_elbow_offset_bias`` (#489).
ELBOW_OFFSET_BIAS_MIN: Final[float] = 0.35
SWING_POLICY_JOINT_NAMES: Final[tuple[str, ...]] = (
    "torso",
    "hip",
    "knee",
    "shoulder",
    "elbow",
)


def _require_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def _require_range(name: str, lower: float, upper: float) -> None:
    _require_positive(f"{name}_min", lower)
    _require_positive(f"{name}_max", upper)
    if upper < lower:
        raise ValueError(f"{name}_max must be greater than or equal to {name}_min")


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _segment_vector(length_m: float, angle_rad: float) -> FloatArray:
    return np.asarray(
        [length_m * np.sin(angle_rad), length_m * np.cos(angle_rad)],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class HumanSegmentSpec:
    """Length and mass for one body segment.

    Preconditions:
        ``length_m`` and ``mass_kg`` are positive.
    """

    length_m: float
    mass_kg: float

    def __post_init__(self) -> None:
        _require_positive("length_m", self.length_m)
        _require_positive("mass_kg", self.mass_kg)


@dataclass(frozen=True)
class SwingSetConfig:
    """Physical configuration for a trainable swingset model."""

    chain_segments: int = DEFAULT_CHAIN_SEGMENTS
    chain_length_m: float = DEFAULT_CHAIN_LENGTH_M
    chain_link_mass_kg: float = DEFAULT_LINK_MASS_KG
    seat_mass_kg: float = DEFAULT_SEAT_MASS_KG
    seat_placement_thigh_fraction: float = DEFAULT_SEAT_PLACEMENT_THIGH_FRACTION
    torso: HumanSegmentSpec = HumanSegmentSpec(0.62, 28.0)
    thigh: HumanSegmentSpec = HumanSegmentSpec(0.46, 8.0)
    shank: HumanSegmentSpec = HumanSegmentSpec(0.45, 5.5)
    upper_arm: HumanSegmentSpec = HumanSegmentSpec(0.30, 2.0)
    forearm: HumanSegmentSpec = HumanSegmentSpec(0.28, 1.6)
    gravity_m_s2: float = GRAVITY_M_S2
    damping: float = DEFAULT_DAMPING
    pump_gain: float = DEFAULT_PUMP_GAIN

    def __post_init__(self) -> None:
        if self.chain_segments < 1:
            raise ValueError("chain_segments must be at least 1")
        _require_positive("chain_length_m", self.chain_length_m)
        _require_positive("chain_link_mass_kg", self.chain_link_mass_kg)
        _require_positive("seat_mass_kg", self.seat_mass_kg)
        if not 0.0 < self.seat_placement_thigh_fraction <= 1.0:
            raise ValueError("seat_placement_thigh_fraction must be in (0, 1]")
        _require_positive("gravity_m_s2", self.gravity_m_s2)
        if self.damping < 0.0:
            raise ValueError("damping must be non-negative")
        if self.pump_gain < 0.0:
            raise ValueError("pump_gain must be non-negative")

    def chain_config(self) -> ChainConfig:
        return ChainConfig(
            segment_count=self.chain_segments,
            segment_length_m=self.chain_length_m / self.chain_segments,
            link_mass_kg=self.chain_link_mass_kg,
            gravity_m_s2=self.gravity_m_s2,
            damping=self.damping,
        )

    @property
    def rider_mass_kg(self) -> float:
        paired_arms = 2.0 * (self.upper_arm.mass_kg + self.forearm.mass_kg)
        paired_legs = 2.0 * (self.thigh.mass_kg + self.shank.mass_kg)
        return self.torso.mass_kg + paired_arms + paired_legs


@dataclass(frozen=True)
class SwingPose:
    """Generalized coordinates for the side-view human and swing."""

    swing_angle_rad: float = 0.0
    torso_lean_rad: float = 0.0
    hip_angle_rad: float = 0.0
    knee_angle_rad: float = 0.0
    shoulder_angle_rad: float = 0.0
    elbow_angle_rad: float = 0.0


@dataclass(frozen=True)
class SwingControlAction:
    """Control rates and external torque applied during one rollout step."""

    torso_lean_rate_rad_s: float = 0.0
    hip_rate_rad_s: float = 0.0
    knee_rate_rad_s: float = 0.0
    shoulder_rate_rad_s: float = 0.0
    elbow_rate_rad_s: float = 0.0
    external_torque_nm: float = 0.0

    def clipped(self) -> SwingControlAction:
        rate = MAX_BODY_RATE_RAD_S
        torque = MAX_EXTERNAL_TORQUE_NM
        return SwingControlAction(
            torso_lean_rate_rad_s=_clamp(self.torso_lean_rate_rad_s, -rate, rate),
            hip_rate_rad_s=_clamp(self.hip_rate_rad_s, -rate, rate),
            knee_rate_rad_s=_clamp(self.knee_rate_rad_s, -rate, rate),
            shoulder_rate_rad_s=_clamp(self.shoulder_rate_rad_s, -rate, rate),
            elbow_rate_rad_s=_clamp(self.elbow_rate_rad_s, -rate, rate),
            external_torque_nm=_clamp(self.external_torque_nm, -torque, torque),
        )

    def vector(self) -> FloatArray:
        return np.asarray(
            [
                self.torso_lean_rate_rad_s,
                self.hip_rate_rad_s,
                self.knee_rate_rad_s,
                self.shoulder_rate_rad_s,
                self.elbow_rate_rad_s,
            ],
            dtype=np.float64,
        )


def constrain_swing_pose(pose: SwingPose) -> SwingPose:
    """Clamp rider pose to realistic side-view swingset range of motion.

    Preconditions:
        All pose fields are finite.
    """

    values = np.asarray(
        [
            pose.swing_angle_rad,
            pose.torso_lean_rad,
            pose.hip_angle_rad,
            pose.knee_angle_rad,
            pose.shoulder_angle_rad,
            pose.elbow_angle_rad,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("pose values must be finite")
    return replace(
        pose,
        torso_lean_rad=_clamp(pose.torso_lean_rad, *SWING_TORSO_LEAN_LIMITS_RAD),
        hip_angle_rad=_clamp(pose.hip_angle_rad, *SWING_HIP_LIMITS_RAD),
        knee_angle_rad=_clamp(pose.knee_angle_rad, *SWING_KNEE_LIMITS_RAD),
        shoulder_angle_rad=_clamp(pose.shoulder_angle_rad, *SWING_SHOULDER_LIMITS_RAD),
        elbow_angle_rad=_clamp(pose.elbow_angle_rad, *SWING_ELBOW_LIMITS_RAD),
    )


@dataclass(frozen=True)
class SwingSetState:
    """Dynamic state for the swingset rollout."""

    pose: SwingPose
    swing_angular_velocity_rad_s: float = 0.0

    @classmethod
    def rest(cls) -> SwingSetState:
        return cls(pose=SwingPose(), swing_angular_velocity_rad_s=0.0)


@dataclass(frozen=True)
class SwingSetSnapshot:
    """Kinematic geometry derived from a swingset pose."""

    points: dict[str, FloatArray]
    chain_nodes: FloatArray
    center_of_mass_m: FloatArray
    hand_chain_error_m: float
    seat_chain_error_m: float


@dataclass(frozen=True)
class SwingRolloutMetrics:
    """Summary metrics returned after policy rollout."""

    max_abs_swing_angle_rad: float
    max_height_gain_m: float
    final_energy_proxy_j: float
    mean_hand_chain_error_m: float
    mean_seat_chain_error_m: float


@dataclass(frozen=True)
class SwingRollout:
    """Time history for a swingset policy simulation."""

    states: tuple[SwingSetState, ...]
    swing_angles_rad: FloatArray
    controls: FloatArray
    snapshots: tuple[SwingSetSnapshot, ...]
    metrics: SwingRolloutMetrics


@dataclass(frozen=True)
class CyclicPolicyParameters:
    """Parameters for a cyclic rider pumping policy."""

    frequency_hz: float
    hip_rate_amplitude_rad_s: float
    torso_rate_amplitude_rad_s: float
    knee_rate_ratio: float
    phase_rad: float

    def __post_init__(self) -> None:
        _require_positive("frequency_hz", self.frequency_hz)
        if self.hip_rate_amplitude_rad_s < 0.0:
            raise ValueError("hip_rate_amplitude_rad_s must be non-negative")
        if self.torso_rate_amplitude_rad_s < 0.0:
            raise ValueError("torso_rate_amplitude_rad_s must be non-negative")
        if self.knee_rate_ratio < 0.0:
            raise ValueError("knee_rate_ratio must be non-negative")


@dataclass(frozen=True)
class CyclicPolicySearchSpace:
    """Tunable grid for cyclic swingset policy search."""

    frequency_hz_min: float = 0.45
    frequency_hz_max: float = 0.75
    frequency_samples: int = 4
    hip_rate_min_rad_s: float = 0.5
    hip_rate_max_rad_s: float = 1.3
    hip_rate_samples: int = 3
    torso_rate_min_rad_s: float = 0.3
    torso_rate_max_rad_s: float = 1.1
    torso_rate_samples: int = 3
    knee_ratio_min: float = 0.25
    knee_ratio_max: float = 0.65
    knee_ratio_samples: int = 3
    phase_samples: int = 3

    def __post_init__(self) -> None:
        _require_range("frequency_hz", self.frequency_hz_min, self.frequency_hz_max)
        _require_range("hip_rate_rad_s", self.hip_rate_min_rad_s, self.hip_rate_max_rad_s)
        _require_range("torso_rate_rad_s", self.torso_rate_min_rad_s, self.torso_rate_max_rad_s)
        _require_range("knee_ratio", self.knee_ratio_min, self.knee_ratio_max)
        for name, value in (
            ("frequency_samples", self.frequency_samples),
            ("hip_rate_samples", self.hip_rate_samples),
            ("torso_rate_samples", self.torso_rate_samples),
            ("knee_ratio_samples", self.knee_ratio_samples),
            ("phase_samples", self.phase_samples),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1")


@dataclass(frozen=True)
class CyclicPolicyBounds:
    """Per-parameter search bounds for the iterative cyclic-policy optimizer.

    Each field is a ``(min, max)`` pair over the same parameters as
    :class:`CyclicPolicyParameters`. Defaults match the grid search ranges.

    Preconditions:
        Rate/frequency/ratio bounds are positive with ``max >= min``; the phase
        lower bound is non-negative with ``max >= min``.
    """

    frequency_hz: tuple[float, float] = (0.45, 0.75)
    hip_rate_rad_s: tuple[float, float] = (0.5, 1.3)
    torso_rate_rad_s: tuple[float, float] = (0.3, 1.1)
    knee_ratio: tuple[float, float] = (0.25, 0.65)
    phase_rad: tuple[float, float] = (0.0, np.pi)

    def __post_init__(self) -> None:
        _require_range("frequency_hz", *self.frequency_hz)
        _require_range("hip_rate_rad_s", *self.hip_rate_rad_s)
        _require_range("torso_rate_rad_s", *self.torso_rate_rad_s)
        _require_range("knee_ratio", *self.knee_ratio)
        phase_lower, phase_upper = self.phase_rad
        if phase_lower < 0.0:
            raise ValueError("phase_rad_min must be non-negative")
        if phase_upper < phase_lower:
            raise ValueError("phase_rad_max must be greater than or equal to phase_rad_min")

    def as_list(self) -> list[tuple[float, float]]:
        """Return bounds ordered to match the optimizer parameter vector."""
        return [
            self.frequency_hz,
            self.hip_rate_rad_s,
            self.torso_rate_rad_s,
            self.knee_ratio,
            self.phase_rad,
        ]


@dataclass(frozen=True)
class CyclicPolicyTraceSample:
    """One candidate evaluation from cyclic policy search."""

    iteration: int
    score_m: float
    best_score_m: float
    parameters: CyclicPolicyParameters
    best_parameters: CyclicPolicyParameters


@dataclass(frozen=True)
class CyclicPolicySearchResult:
    """Best cyclic policy and rollout found by deterministic search."""

    parameters: CyclicPolicyParameters
    rollout: SwingRollout
    objective_height_m: float
    evaluated_candidates: int
    optimized_cycles: float | None
    trace: tuple[CyclicPolicyTraceSample, ...] = ()


def build_swingset_snapshot(
    config: SwingSetConfig,
    pose: SwingPose,
) -> SwingSetSnapshot:
    pose = constrain_swing_pose(pose)
    chain_config = config.chain_config()
    if config.chain_segments == 1:
        chain_state = ChainState.stationary(chain_config, pose.swing_angle_rad)
    else:
        flexible_angles = pose.swing_angle_rad + initial_catenary_angles(
            config.chain_segments,
            0.04,
        )
        chain_state = ChainState(
            flexible_angles,
            np.zeros(config.chain_segments, dtype=np.float64),
        )
    chain_nodes = chain_state.node_positions(chain_config)
    points = _body_points(config, pose, chain_nodes)
    center = _center_of_mass(config, points)
    grip_point = chain_nodes[max(config.chain_segments - 2, 0)]
    seat_point = chain_nodes[-1]
    hand_error = float(np.linalg.norm(points["hand"] - grip_point))
    seat_error = float(np.linalg.norm(points["seat"] - seat_point))
    return SwingSetSnapshot(
        points=points,
        chain_nodes=chain_nodes,
        center_of_mass_m=center,
        hand_chain_error_m=hand_error,
        seat_chain_error_m=seat_error,
    )


def _body_points(
    config: SwingSetConfig,
    pose: SwingPose,
    chain_nodes: FloatArray,
) -> dict[str, FloatArray]:
    seat = chain_nodes[-1]
    grip_point = chain_nodes[max(config.chain_segments - 2, 0)]
    hand = grip_point.copy()
    thigh_angle = pose.swing_angle_rad + pose.hip_angle_rad
    thigh_vector = _segment_vector(config.thigh.length_m, thigh_angle)
    hip = seat - config.seat_placement_thigh_fraction * thigh_vector
    knee = hip + thigh_vector
    shank_angle = thigh_angle + pose.knee_angle_rad
    foot = knee + _segment_vector(config.shank.length_m, shank_angle)
    torso_angle = pose.swing_angle_rad + pose.torso_lean_rad
    torso_vector = _segment_vector(config.torso.length_m, torso_angle)
    shoulder = hip - torso_vector
    waist = hip - 0.45 * torso_vector
    elbow = _arm_elbow_point(config, shoulder, hand, pose.elbow_angle_rad)
    return {
        "seat": seat,
        "hand": hand,
        "elbow": elbow,
        "shoulder": shoulder,
        "waist": waist,
        "hip": hip,
        "knee": knee,
        "foot": foot,
    }


def _arm_elbow_point(
    config: SwingSetConfig,
    shoulder: FloatArray,
    hand: FloatArray,
    elbow_bias_rad: float,
) -> FloatArray:
    upper_length = config.upper_arm.length_m
    forearm_length = config.forearm.length_m
    delta = hand - shoulder
    distance = float(np.linalg.norm(delta))
    unit = delta / distance if distance > 1e-9 else np.asarray([0.0, 1.0], dtype=np.float64)
    minimum_reach = abs(upper_length - forearm_length) + 1e-9
    maximum_reach = upper_length + forearm_length - 1e-9
    effective_distance = _clamp(distance, minimum_reach, maximum_reach)
    along = (upper_length**2 - forearm_length**2 + effective_distance**2) / (
        2.0 * effective_distance
    )
    height = np.sqrt(max(upper_length**2 - along**2, 0.0))
    normal = np.asarray([-unit[1], unit[0]], dtype=np.float64)
    bias = _elbow_offset_bias(elbow_bias_rad)
    return shoulder + along * unit + bias * height * normal


def _elbow_offset_bias(elbow_bias_rad: float) -> float:
    """Map a requested elbow flexion to the perpendicular offset factor.

    The two-link IK places the elbow off the shoulder->hand line by
    ``bias * height * normal``. ``elbow_bias_rad`` (the rider's requested elbow
    flexion) is first clamped to the realistic swing ROM
    (``SWING_ELBOW_LIMITS_RAD``) and then mapped to a factor in
    ``[ELBOW_OFFSET_BIAS_MIN, 1.0]``: a neutral (0 rad) elbow sits at the full
    geometric ``+normal`` offset, while a fully flexed elbow tucks toward the
    shoulder->hand line. Previously this factor was hard-coded to ``1.0`` so the
    parameter had no effect (issue #489).

    Preconditions:
        ``elbow_bias_rad`` is finite.
    """

    clamped = constrain_swing_pose(SwingPose(elbow_angle_rad=elbow_bias_rad)).elbow_angle_rad
    lower, upper = SWING_ELBOW_LIMITS_RAD
    span = upper - lower
    if span <= 0.0:
        return 1.0
    flex_fraction = (clamped - lower) / span
    return 1.0 - (1.0 - ELBOW_OFFSET_BIAS_MIN) * flex_fraction


def _center_of_mass(
    config: SwingSetConfig,
    points: dict[str, FloatArray],
) -> FloatArray:
    seat = points["seat"]
    torso_mid = 0.5 * (points["hip"] + points["shoulder"])
    thigh_mid = 0.5 * (points["hip"] + points["knee"])
    shank_mid = 0.5 * (points["knee"] + points["foot"])
    upper_arm_mid = 0.5 * (points["shoulder"] + points["elbow"])
    forearm_mid = 0.5 * (points["elbow"] + points["hand"])
    weighted = (
        config.seat_mass_kg * seat
        + config.torso.mass_kg * torso_mid
        + 2.0 * config.thigh.mass_kg * thigh_mid
        + 2.0 * config.shank.mass_kg * shank_mid
        + 2.0 * config.upper_arm.mass_kg * upper_arm_mid
        + 2.0 * config.forearm.mass_kg * forearm_mid
    )
    total_mass = config.seat_mass_kg + config.rider_mass_kg
    return weighted / total_mass


def _step_joint_with_endstop(
    value: float,
    rate_rad_s: float,
    dt_s: float,
    limits: tuple[float, float],
) -> float:
    lower, upper = limits
    current = _clamp(value, lower, upper)
    width = upper - lower
    margin = max(width * JOINT_ENDSTOP_MARGIN_FRACTION, 1e-6)
    damped_rate = rate_rad_s
    if damped_rate > 0.0 and current > upper - margin:
        damped_rate *= ((upper - current) / margin) ** 2
    elif damped_rate < 0.0 and current < lower + margin:
        damped_rate *= ((current - lower) / margin) ** 2
    return _clamp(current + damped_rate * dt_s, lower, upper)


def _step_pose(pose: SwingPose, action: SwingControlAction, dt_s: float) -> SwingPose:
    bounded = constrain_swing_pose(pose)
    return replace(
        bounded,
        torso_lean_rad=_step_joint_with_endstop(
            bounded.torso_lean_rad,
            action.torso_lean_rate_rad_s,
            dt_s,
            SWING_TORSO_LEAN_LIMITS_RAD,
        ),
        hip_angle_rad=_step_joint_with_endstop(
            bounded.hip_angle_rad,
            action.hip_rate_rad_s,
            dt_s,
            SWING_HIP_LIMITS_RAD,
        ),
        knee_angle_rad=_step_joint_with_endstop(
            bounded.knee_angle_rad,
            action.knee_rate_rad_s,
            dt_s,
            SWING_KNEE_LIMITS_RAD,
        ),
        shoulder_angle_rad=_step_joint_with_endstop(
            bounded.shoulder_angle_rad,
            action.shoulder_rate_rad_s,
            dt_s,
            SWING_SHOULDER_LIMITS_RAD,
        ),
        elbow_angle_rad=_step_joint_with_endstop(
            bounded.elbow_angle_rad,
            action.elbow_rate_rad_s,
            dt_s,
            SWING_ELBOW_LIMITS_RAD,
        ),
    )


def _swing_acceleration(
    config: SwingSetConfig,
    state: SwingSetState,
    action: SwingControlAction,
) -> float:
    length = config.chain_length_m
    gravity = -(config.gravity_m_s2 / length) * np.sin(state.pose.swing_angle_rad)
    damping = -config.damping * state.swing_angular_velocity_rad_s
    inertia = (config.seat_mass_kg + config.rider_mass_kg) * length**2
    control = action.external_torque_nm / inertia
    pump = config.pump_gain * _pumping_projection(state, action) / length
    return float(gravity + damping + control + pump)


def _pumping_projection(state: SwingSetState, action: SwingControlAction) -> float:
    direction = 1.0 if state.swing_angular_velocity_rad_s >= 0.0 else -1.0
    leg_drive = action.hip_rate_rad_s - 0.35 * action.knee_rate_rad_s
    torso_drive = -0.5 * action.torso_lean_rate_rad_s
    return direction * (leg_drive + torso_drive)


def step_swingset(
    config: SwingSetConfig,
    state: SwingSetState,
    action: SwingControlAction,
    dt_s: float,
) -> SwingSetState:
    """Advance the swingset by one semi-implicit Euler step.

    Preconditions:
        ``dt_s`` is positive.
    """

    _require_positive("dt_s", dt_s)
    clipped = action.clipped()
    acceleration = _swing_acceleration(config, state, clipped)
    next_velocity = state.swing_angular_velocity_rad_s + acceleration * dt_s
    next_swing = state.pose.swing_angle_rad + next_velocity * dt_s
    pose = replace(_step_pose(state.pose, clipped, dt_s), swing_angle_rad=next_swing)
    return SwingSetState(pose=pose, swing_angular_velocity_rad_s=next_velocity)


def heuristic_pumping_policy(
    state: SwingSetState,
    _time_s: float,
) -> SwingControlAction:
    direction = 1.0 if state.swing_angular_velocity_rad_s >= 0.0 else -1.0
    return SwingControlAction(
        torso_lean_rate_rad_s=-0.7 * direction,
        hip_rate_rad_s=0.9 * direction,
        knee_rate_rad_s=-0.4 * direction,
        shoulder_rate_rad_s=-0.15 * direction,
        elbow_rate_rad_s=0.2 * direction,
    )


def cyclic_pumping_policy(parameters: CyclicPolicyParameters) -> Policy:
    """Return a sinusoidal cyclic policy suitable for policy search."""

    def _policy(_state: SwingSetState, time_s: float) -> SwingControlAction:
        phase = 2.0 * np.pi * parameters.frequency_hz * time_s + parameters.phase_rad
        driver = float(np.sin(phase))
        return SwingControlAction(
            torso_lean_rate_rad_s=-parameters.torso_rate_amplitude_rad_s * driver,
            hip_rate_rad_s=parameters.hip_rate_amplitude_rad_s * driver,
            knee_rate_rad_s=(
                -parameters.knee_rate_ratio * parameters.hip_rate_amplitude_rad_s * driver
            ),
            shoulder_rate_rad_s=-0.1 * driver,
            elbow_rate_rad_s=0.12 * driver,
        )

    return _policy


def cyclic_policy_controls(
    parameters: CyclicPolicyParameters,
    *,
    steps: int,
    dt_s: float,
) -> FloatArray:
    """Build the full cyclic control matrix with vectorized trigonometry.

    Preconditions:
        ``steps`` is at least 1 and ``dt_s`` is positive.
    """

    if steps < 1:
        raise ValueError("steps must be at least 1")
    _require_positive("dt_s", dt_s)
    times = np.arange(steps, dtype=np.float64) * dt_s
    driver = np.sin(2.0 * np.pi * parameters.frequency_hz * times + parameters.phase_rad)
    return np.column_stack(
        (
            -parameters.torso_rate_amplitude_rad_s * driver,
            parameters.hip_rate_amplitude_rad_s * driver,
            -parameters.knee_rate_ratio * parameters.hip_rate_amplitude_rad_s * driver,
            -0.1 * driver,
            0.12 * driver,
        )
    ).astype(np.float64, copy=False)


def _control_action_from_vector(vector: FloatArray) -> SwingControlAction:
    control = np.asarray(vector, dtype=np.float64)
    if control.shape != (CONTROL_DIMENSION,) or not np.all(np.isfinite(control)):
        raise ValueError("control vector must contain 5 finite values")
    return SwingControlAction(
        torso_lean_rate_rad_s=float(control[0]),
        hip_rate_rad_s=float(control[1]),
        knee_rate_rad_s=float(control[2]),
        shoulder_rate_rad_s=float(control[3]),
        elbow_rate_rad_s=float(control[4]),
    )


def simulate_swingset_controls(
    config: SwingSetConfig,
    initial_state: SwingSetState,
    controls: FloatArray,
    dt_s: float,
) -> SwingRollout:
    """Roll out a precomputed control matrix for the trainable swingset model.

    Preconditions:
        ``controls`` has shape ``(N, 5)`` with ``N >= 1`` and ``dt_s`` is positive.
    """

    control_array = np.asarray(controls, dtype=np.float64)
    if (
        control_array.ndim != 2
        or control_array.shape[0] < 1
        or control_array.shape[1] != CONTROL_DIMENSION
        or not np.all(np.isfinite(control_array))
    ):
        raise ValueError("controls must have shape (N >= 1, 5) and contain finite values")
    _require_positive("dt_s", dt_s)
    states = [replace(initial_state, pose=constrain_swing_pose(initial_state.pose))]
    snapshots = [build_swingset_snapshot(config, initial_state.pose)]
    for control in control_array:
        action = _control_action_from_vector(control).clipped()
        states.append(step_swingset(config, states[-1], action, dt_s))
        snapshots.append(build_swingset_snapshot(config, states[-1].pose))
    angles = np.asarray([state.pose.swing_angle_rad for state in states])
    metrics = _rollout_metrics(config, states, snapshots, angles)
    return SwingRollout(
        states=tuple(states),
        swing_angles_rad=angles,
        controls=control_array,
        snapshots=tuple(snapshots),
        metrics=metrics,
    )


def simulate_swingset(
    config: SwingSetConfig,
    initial_state: SwingSetState,
    steps: int,
    dt_s: float,
    policy: Policy,
) -> SwingRollout:
    """Roll out a control policy for a trainable swingset model.

    Preconditions:
        ``steps`` and ``dt_s`` are positive.
    """

    if steps < 1:
        raise ValueError("steps must be at least 1")
    _require_positive("dt_s", dt_s)
    states = [replace(initial_state, pose=constrain_swing_pose(initial_state.pose))]
    controls: list[FloatArray] = []
    snapshots = [build_swingset_snapshot(config, initial_state.pose)]
    for step_index in range(steps):
        action = policy(states[-1], step_index * dt_s).clipped()
        controls.append(action.vector())
        states.append(step_swingset(config, states[-1], action, dt_s))
        snapshots.append(build_swingset_snapshot(config, states[-1].pose))
    angles = np.asarray([state.pose.swing_angle_rad for state in states])
    control_array = np.vstack(controls).reshape(steps, CONTROL_DIMENSION)
    metrics = _rollout_metrics(config, states, snapshots, angles)
    return SwingRollout(
        states=tuple(states),
        swing_angles_rad=angles,
        controls=control_array,
        snapshots=tuple(snapshots),
        metrics=metrics,
    )


def optimize_cyclic_policy(
    config: SwingSetConfig,
    initial_state: SwingSetState | None = None,
    *,
    steps: int = DEFAULT_POLICY_STEPS,
    dt_s: float = DEFAULT_POLICY_DT_S,
    cycles: float | None = None,
    search_space: CyclicPolicySearchSpace | None = None,
    progress_callback: ProgressCallback | None = None,
) -> CyclicPolicySearchResult:
    """Search deterministic cyclic policies and maximize swing height.

    Preconditions:
        ``steps`` and ``dt_s`` are positive. ``cycles`` is positive when supplied.
    """

    if steps < 1:
        raise ValueError("steps must be at least 1")
    _require_positive("dt_s", dt_s)
    if cycles is not None:
        _require_positive("cycles", cycles)
    start = initial_state or SwingSetState(pose=SwingPose(swing_angle_rad=0.06))
    candidates = _cyclic_policy_candidates(search_space or CyclicPolicySearchSpace())
    best_params = candidates[0]
    best_rollout: SwingRollout | None = None
    best_score = -np.inf
    trace: list[CyclicPolicyTraceSample] = []
    for index, parameters in enumerate(candidates, start=1):
        candidate_steps = _steps_for_candidate(steps, dt_s, parameters, cycles)
        rollout = simulate_swingset_controls(
            config,
            start,
            cyclic_policy_controls(parameters, steps=candidate_steps, dt_s=dt_s),
            dt_s,
        )
        score = rollout.metrics.max_height_gain_m
        if score > best_score:
            best_params = parameters
            best_rollout = rollout
            best_score = score
        if best_rollout is None:  # pragma: no cover - defensive guard for malformed searches.
            raise RuntimeError("Policy search did not evaluate a rollout")
        trace.append(
            CyclicPolicyTraceSample(
                iteration=index,
                score_m=score,
                best_score_m=best_score,
                parameters=parameters,
                best_parameters=best_params,
            )
        )
        if progress_callback is not None:
            progress_callback(index, len(candidates), best_score, best_params)
    if best_rollout is None:  # pragma: no cover - defensive guard for malformed searches.
        raise RuntimeError("Policy search did not evaluate a rollout")
    return CyclicPolicySearchResult(
        best_params,
        best_rollout,
        best_score,
        len(candidates),
        cycles,
        tuple(trace),
    )


def _params_from_vector(vector: FloatArray, bounds: CyclicPolicyBounds) -> CyclicPolicyParameters:
    """Build clamped policy parameters from an optimizer vector.

    Clamping matters because the local-refinement stage (Nelder-Mead) is not
    bound-constrained and may probe slightly outside the box.
    """
    limits = bounds.as_list()
    clamped = [
        _clamp(float(value), low, high) for value, (low, high) in zip(vector, limits, strict=True)
    ]
    return CyclicPolicyParameters(
        frequency_hz=clamped[0],
        hip_rate_amplitude_rad_s=clamped[1],
        torso_rate_amplitude_rad_s=clamped[2],
        knee_rate_ratio=clamped[3],
        phase_rad=clamped[4],
    )


def optimize_cyclic_policy_iterative(
    config: SwingSetConfig,
    initial_state: SwingSetState | None = None,
    *,
    steps: int = DEFAULT_POLICY_STEPS,
    dt_s: float = DEFAULT_POLICY_DT_S,
    cycles: float | None = None,
    bounds: CyclicPolicyBounds | None = None,
    budget: int = DEFAULT_OPTIMIZER_BUDGET,
    seed: int = DEFAULT_OPTIMIZER_SEED,
    local_refine: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> CyclicPolicySearchResult:
    """Optimize the cyclic pumping policy to maximize swing height.

    Runs a seeded ``differential_evolution`` global search followed by an
    optional bound-clamped Nelder-Mead refinement, both sharing one hard
    evaluation ``budget``. Deterministic: identical ``(config, seed, budget,
    bounds, steps, dt_s, cycles)`` yields identical results.

    Preconditions:
        ``steps >= 1``; ``dt_s > 0``; ``cycles > 0`` when supplied;
        ``1 <= budget <= MAX_OPTIMIZER_BUDGET``.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")
    _require_positive("dt_s", dt_s)
    if cycles is not None:
        _require_positive("cycles", cycles)
    if not 1 <= budget <= MAX_OPTIMIZER_BUDGET:
        raise ValueError(f"budget must be in [1, {MAX_OPTIMIZER_BUDGET}]")

    bounds = bounds or CyclicPolicyBounds()
    start = initial_state or SwingSetState(pose=SwingPose(swing_angle_rad=0.06))
    bound_list = bounds.as_list()
    optimizer_bounds = Bounds(
        np.asarray([lower for lower, _upper in bound_list], dtype=np.float64),
        np.asarray([upper for _lower, upper in bound_list], dtype=np.float64),
    )

    eval_count = 0
    best_score = -np.inf
    best_params: CyclicPolicyParameters | None = None
    best_rollout: SwingRollout | None = None
    trace: list[CyclicPolicyTraceSample] = []

    def _objective(vector: FloatArray) -> float:
        nonlocal eval_count, best_score, best_params, best_rollout
        if eval_count >= budget:
            return float(np.inf)  # Hard budget reached: stop spending evaluations.
        params = _params_from_vector(vector, bounds)
        candidate_steps = _steps_for_candidate(steps, dt_s, params, cycles)
        rollout = simulate_swingset_controls(
            config,
            start,
            cyclic_policy_controls(params, steps=candidate_steps, dt_s=dt_s),
            dt_s,
        )
        score = rollout.metrics.max_height_gain_m
        eval_count += 1
        if score > best_score:
            best_score = score
            best_params = params
            best_rollout = rollout
        assert best_params is not None  # set on the first (finite-score) evaluation.
        trace.append(
            CyclicPolicyTraceSample(
                iteration=eval_count,
                score_m=score,
                best_score_m=best_score,
                parameters=params,
                best_parameters=best_params,
            )
        )
        if progress_callback is not None:
            progress_callback(eval_count, budget, best_score, best_params)
        return -score

    # Reserve ~30% of the budget for local refinement so it actually runs; the
    # objective's hard-budget guard still caps total evaluations either way.
    de_budget = int(budget * 0.7) if local_refine else budget
    maxiter = max(1, de_budget // (_OPTIMIZER_POPSIZE * CONTROL_DIMENSION) - 1)
    differential_evolution(
        _objective,
        optimizer_bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=_OPTIMIZER_POPSIZE,
        polish=False,
        init="sobol",
        tol=0.0,
        mutation=(0.5, 1.0),
        recombination=0.7,
        updating="deferred",
    )

    if local_refine and best_params is not None and eval_count < budget:
        start_vector = np.asarray(
            [
                best_params.frequency_hz,
                best_params.hip_rate_amplitude_rad_s,
                best_params.torso_rate_amplitude_rad_s,
                best_params.knee_rate_ratio,
                best_params.phase_rad,
            ],
            dtype=np.float64,
        )
        minimize(
            _objective,
            start_vector,
            method="Nelder-Mead",
            options={"maxfev": budget - eval_count, "xatol": 1e-4, "fatol": 1e-6},
        )

    if best_rollout is None or best_params is None:  # pragma: no cover - budget>=1 guarantees one.
        raise RuntimeError("Iterative policy search did not evaluate a rollout")
    return CyclicPolicySearchResult(
        best_params,
        best_rollout,
        best_score,
        eval_count,
        cycles,
        tuple(trace),
    )


def estimate_swingset_joint_torques(
    config: SwingSetConfig,
    rollout: SwingRollout,
    dt_s: float,
) -> FloatArray:
    """Estimate policy joint torques from controller rates.

    Preconditions:
        ``dt_s`` is positive and ``rollout.controls`` is shaped ``(N, 5)``.
    """

    _require_positive("dt_s", dt_s)
    controls = rollout.controls
    if controls.ndim != 2 or controls.shape[1] != CONTROL_DIMENSION:
        raise ValueError("rollout.controls must have shape (N, 5)")
    if controls.shape[0] == 0:
        return np.zeros((0, CONTROL_DIMENSION), dtype=np.float64)
    inertias = _policy_joint_inertias(config)
    accelerations = (
        np.gradient(controls, dt_s, axis=0) if controls.shape[0] > 1 else np.zeros_like(controls)
    )
    damping = 0.08 * inertias * controls
    return accelerations * inertias + damping


def _policy_joint_inertias(config: SwingSetConfig) -> FloatArray:
    torso = config.torso.mass_kg * config.torso.length_m**2 / 3.0
    hip = 2.0 * (
        config.thigh.mass_kg * config.thigh.length_m**2 / 3.0
        + config.shank.mass_kg * (config.thigh.length_m + 0.5 * config.shank.length_m) ** 2
    )
    knee = 2.0 * config.shank.mass_kg * config.shank.length_m**2 / 3.0
    shoulder = 2.0 * (
        config.upper_arm.mass_kg * config.upper_arm.length_m**2 / 3.0
        + config.forearm.mass_kg * (config.upper_arm.length_m + 0.5 * config.forearm.length_m) ** 2
    )
    elbow = 2.0 * config.forearm.mass_kg * config.forearm.length_m**2 / 3.0
    return np.asarray([torso, hip, knee, shoulder, elbow], dtype=np.float64)


def _steps_for_candidate(
    default_steps: int,
    dt_s: float,
    parameters: CyclicPolicyParameters,
    cycles: float | None,
) -> int:
    if cycles is None:
        return default_steps
    return max(1, round(cycles / parameters.frequency_hz / dt_s))


def _linspace(lower: float, upper: float, samples: int) -> FloatArray:
    if samples == 1:
        return np.asarray([lower], dtype=np.float64)
    return np.linspace(lower, upper, samples, dtype=np.float64)


def _cyclic_policy_candidates(
    search_space: CyclicPolicySearchSpace,
) -> tuple[CyclicPolicyParameters, ...]:
    candidates: list[CyclicPolicyParameters] = []
    phase_values = (
        np.asarray([0.0], dtype=np.float64)
        if search_space.phase_samples == 1
        else np.linspace(0.0, np.pi, search_space.phase_samples, dtype=np.float64)
    )
    for frequency_hz in _linspace(
        search_space.frequency_hz_min,
        search_space.frequency_hz_max,
        search_space.frequency_samples,
    ):
        for hip_rate in _linspace(
            search_space.hip_rate_min_rad_s,
            search_space.hip_rate_max_rad_s,
            search_space.hip_rate_samples,
        ):
            for torso_rate in _linspace(
                search_space.torso_rate_min_rad_s,
                search_space.torso_rate_max_rad_s,
                search_space.torso_rate_samples,
            ):
                for knee_ratio in _linspace(
                    search_space.knee_ratio_min,
                    search_space.knee_ratio_max,
                    search_space.knee_ratio_samples,
                ):
                    for phase in phase_values:
                        candidates.append(
                            CyclicPolicyParameters(
                                frequency_hz=float(frequency_hz),
                                hip_rate_amplitude_rad_s=float(hip_rate),
                                torso_rate_amplitude_rad_s=float(torso_rate),
                                knee_rate_ratio=float(knee_ratio),
                                phase_rad=float(phase),
                            )
                        )
    return tuple(candidates)


def _rollout_metrics(
    config: SwingSetConfig,
    states: list[SwingSetState],
    snapshots: list[SwingSetSnapshot],
    angles: FloatArray,
) -> SwingRolloutMetrics:
    final_velocity = states[-1].swing_angular_velocity_rad_s
    total_mass = config.seat_mass_kg + config.rider_mass_kg
    inertia = total_mass * config.chain_length_m**2
    energy = 0.5 * inertia * final_velocity**2
    errors = np.asarray([snapshot.hand_chain_error_m for snapshot in snapshots])
    seat_errors = np.asarray([snapshot.seat_chain_error_m for snapshot in snapshots])
    max_abs_angle = float(np.max(np.abs(angles)))
    return SwingRolloutMetrics(
        max_abs_swing_angle_rad=max_abs_angle,
        max_height_gain_m=float(config.chain_length_m * (1.0 - np.cos(max_abs_angle))),
        final_energy_proxy_j=float(energy),
        mean_hand_chain_error_m=float(np.mean(errors)),
        mean_seat_chain_error_m=float(np.mean(seat_errors)),
    )
