"""Matched club-torque allocation and phenomenological transmission tests.

The constrained model answers a narrow mechanical question: for the same
state and the same club angular-acceleration contribution, how do a proximal
joint-torque pattern and a direct bilateral wrist moment differ?  It does not
identify muscle activation or scapular motion from the resulting hand forces.

The transmission model is deliberately separate.  It operationalizes
``slack`` as a declared rotational dead zone followed by series stiffness and
first-order force development.  This is a falsifiable phenomenological test,
not a claim that a biological wrist is literally a backlash element.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from scripts.research.proximal_distal_energy.two_arm_closed_loop import (
    TwoArmControl,
    TwoArmParams,
    contact_wrench,
    drift_control_attribution,
    kinematics,
)

FloatArray = npt.NDArray[np.float64]
AllocationChannel = Literal["proximal", "wrist"]


@dataclass(frozen=True, slots=True)
class MatchedAllocationResult:
    """One same-state allocation satisfying a declared club task."""

    channel: AllocationChannel
    control: TwoArmControl
    target_control_angular_acceleration_rad_s2: float
    control_angular_acceleration_rad_s2: float
    net_control_moment_nm: float
    direct_wrist_moment_nm: float
    grip_force_couple_nm: float
    hand_force_rms_n: float
    hand_force_resultant_n: float
    joint_torque_norm_nm: float
    control_contact_force_on_club_n: FloatArray


@dataclass(frozen=True, slots=True)
class MatchedAllocationSweep:
    """Geometry-by-allocation surface for one fixed club moment task."""

    club_angles_rad: FloatArray
    wrist_fractions: FloatArray
    net_control_moment_nm: FloatArray
    direct_wrist_moment_nm: FloatArray
    grip_force_couple_nm: FloatArray
    hand_force_rms_n: FloatArray
    hand_force_resultant_n: FloatArray
    joint_torque_norm_nm: FloatArray


@dataclass(frozen=True, slots=True)
class TransmissionChannel:
    """Series transmission with dead zone, stiffness, and force-rise time."""

    stiffness_nm_rad: float
    dead_zone_rad: float
    time_constant_s: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.stiffness_nm_rad) or self.stiffness_nm_rad <= 0.0:
            raise ValueError("stiffness_nm_rad must be finite and positive")
        if not np.isfinite(self.dead_zone_rad) or self.dead_zone_rad < 0.0:
            raise ValueError("dead_zone_rad must be finite and non-negative")
        if not np.isfinite(self.time_constant_s) or self.time_constant_s <= 0.0:
            raise ValueError("time_constant_s must be finite and positive")


@dataclass(frozen=True, slots=True)
class RoleReversalProgram:
    """Pre- and post-transition desired torque in arm and wrist channels."""

    name: str
    arm_pre_nm: float
    wrist_pre_nm: float
    arm_post_nm: float
    wrist_post_nm: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.arm_pre_nm, self.wrist_pre_nm, self.arm_post_nm, self.wrist_post_nm]
        )
        if not self.name or not np.all(np.isfinite(values)):
            raise ValueError("program name and torque values must be finite")

    @property
    def pre_net_torque_nm(self) -> float:
        return self.arm_pre_nm + self.wrist_pre_nm

    @property
    def post_net_torque_nm(self) -> float:
        return self.arm_post_nm + self.wrist_post_nm

    @classmethod
    def persistent_direction(cls) -> RoleReversalProgram:
        """Persistent arm drive opposed by persistent wrist resistance."""
        return cls(
            name="persistent_arm_drive",
            arm_pre_nm=10.0,
            wrist_pre_nm=-4.0,
            arm_post_nm=16.0,
            wrist_post_nm=-6.0,
        )

    @classmethod
    def opposite_role_reversal(cls) -> RoleReversalProgram:
        """Wrist-led preparation followed by arm-led delivery."""
        return cls(
            name="wrist_to_arm_role_reversal",
            arm_pre_nm=-4.0,
            wrist_pre_nm=10.0,
            arm_post_nm=16.0,
            wrist_post_nm=-6.0,
        )


@dataclass(frozen=True, slots=True)
class RoleReversalTrace:
    """Torque-continuity evidence for one two-channel transition."""

    time_s: FloatArray
    desired_arm_torque_nm: FloatArray
    desired_wrist_torque_nm: FloatArray
    transmitted_arm_torque_nm: FloatArray
    transmitted_wrist_torque_nm: FloatArray
    desired_net_torque_nm: FloatArray
    transmitted_net_torque_nm: FloatArray
    arm_zero_transmission_duration_s: float
    wrist_zero_transmission_duration_s: float
    net_torque_error_impulse_nms: float
    arm_reversal_delay_s: float
    wrist_reversal_delay_s: float


def _finite_state(name: str, value: object) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (7,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain seven finite values")
    return array


def _channel_basis(channel: AllocationChannel) -> tuple[TwoArmControl, ...]:
    """Return independent coordinates for a minimum-norm allocation."""
    if channel == "proximal":
        names = (
            "right_shoulder_nm",
            "right_elbow_nm",
            "left_shoulder_nm",
            "left_elbow_nm",
        )
    elif channel == "wrist":
        names = ("right_wrist_nm", "left_wrist_nm")
    else:
        raise ValueError("channel must be 'proximal' or 'wrist'")
    return tuple(TwoArmControl(**{name: 1.0}) for name in names)


def _allocation_result(
    q: FloatArray,
    qdot: FloatArray,
    control: TwoArmControl,
    target_control_angular_acceleration_rad_s2: float,
    channel: AllocationChannel,
    params: TwoArmParams,
) -> MatchedAllocationResult:
    split = drift_control_attribution(q, qdot, control, params)
    contacts = split.control.contact_force_on_club_n
    points = kinematics(q, params)
    wrench = contact_wrench(
        contacts[0],
        contacts[1],
        points["right_grip"],
        points["left_grip"],
        points["club_center"],
        np.zeros(2),
        np.zeros(2),
    )
    direct = control.right_wrist_nm + control.left_wrist_nm
    net = params.club_inertia_kg_m2 * split.control.qddot[6]
    force_rms = float(np.sqrt(np.mean(np.sum(contacts**2, axis=1))))
    values = np.asarray(list(control.__dict__.values()), dtype=float)
    return MatchedAllocationResult(
        channel=channel,
        control=control,
        target_control_angular_acceleration_rad_s2=(
            target_control_angular_acceleration_rad_s2
        ),
        control_angular_acceleration_rad_s2=float(split.control.qddot[6]),
        net_control_moment_nm=float(net),
        direct_wrist_moment_nm=float(direct),
        grip_force_couple_nm=float(wrench.moment_about_center_nm),
        hand_force_rms_n=force_rms,
        hand_force_resultant_n=float(np.linalg.norm(wrench.resultant_force_n)),
        joint_torque_norm_nm=float(np.linalg.norm(values)),
        control_contact_force_on_club_n=contacts.copy(),
    )


def allocate_matched_angular_acceleration(
    q: object,
    qdot: object,
    target_control_angular_acceleration_rad_s2: float,
    channel: AllocationChannel,
    params: TwoArmParams,
) -> MatchedAllocationResult:
    """Solve the minimum-norm actuator allocation for a same-state task."""
    state = _finite_state("q", q)
    velocity = _finite_state("qdot", qdot)
    target = float(target_control_angular_acceleration_rad_s2)
    if not np.isfinite(target):
        raise ValueError("target angular acceleration must be finite")
    basis = _channel_basis(channel)
    authority = np.asarray(
        [
            drift_control_attribution(state, velocity, control, params).control.qddot[6]
            for control in basis
        ],
        dtype=float,
    )
    authority_norm_squared = float(authority @ authority)
    if authority_norm_squared <= 1e-24:
        raise ValueError(f"{channel} channel has no angular authority at this state")
    weights = target * authority / authority_norm_squared
    fields = {
        name: float(
            sum(
                weight * getattr(control, name)
                for weight, control in zip(weights, basis, strict=True)
            )
        )
        for name in TwoArmControl.zero().__dict__
    }
    control = TwoArmControl(**fields)
    return _allocation_result(state, velocity, control, target, channel, params)


def matched_allocation_sweep(
    club_angles_rad: object,
    wrist_fractions: object,
    target_net_control_moment_nm: float,
    params: TwoArmParams | None = None,
) -> MatchedAllocationSweep:
    """Sweep convex mixtures of independently task-normalized channel controls."""
    physical = params or TwoArmParams.publication_default()
    angles = np.asarray(club_angles_rad, dtype=float).reshape(-1)
    fractions = np.asarray(wrist_fractions, dtype=float).reshape(-1)
    target_moment = float(target_net_control_moment_nm)
    if angles.size == 0 or not np.all(np.isfinite(angles)):
        raise ValueError("club_angles_rad must be a non-empty finite array")
    if (
        fractions.size == 0
        or not np.all(np.isfinite(fractions))
        or np.any((fractions < 0.0) | (fractions > 1.0))
    ):
        raise ValueError("wrist_fractions must lie in [0, 1]")
    if not np.isfinite(target_moment):
        raise ValueError("target_net_control_moment_nm must be finite")
    shape = (angles.size, fractions.size)
    net = np.empty(shape)
    direct = np.empty(shape)
    couple = np.empty(shape)
    hand_rms = np.empty(shape)
    resultant = np.empty(shape)
    effort = np.empty(shape)
    target_accel = target_moment / physical.club_inertia_kg_m2
    for row, angle in enumerate(angles):
        state = physical.consistent_configuration(np.array([0.0, -0.50]), float(angle))
        velocity = np.zeros(7)
        proximal = allocate_matched_angular_acceleration(
            state, velocity, target_accel, "proximal", physical
        )
        wrist = allocate_matched_angular_acceleration(
            state, velocity, target_accel, "wrist", physical
        )
        for column, fraction in enumerate(fractions):
            fields = {
                name: (1.0 - fraction) * getattr(proximal.control, name)
                + fraction * getattr(wrist.control, name)
                for name in proximal.control.__dict__
            }
            result = _allocation_result(
                state,
                velocity,
                TwoArmControl(**fields),
                target_accel,
                "wrist" if fraction >= 0.5 else "proximal",
                physical,
            )
            net[row, column] = result.net_control_moment_nm
            direct[row, column] = result.direct_wrist_moment_nm
            couple[row, column] = result.grip_force_couple_nm
            hand_rms[row, column] = result.hand_force_rms_n
            resultant[row, column] = result.hand_force_resultant_n
            effort[row, column] = result.joint_torque_norm_nm
    return MatchedAllocationSweep(
        club_angles_rad=angles,
        wrist_fractions=fractions,
        net_control_moment_nm=net,
        direct_wrist_moment_nm=direct,
        grip_force_couple_nm=couple,
        hand_force_rms_n=hand_rms,
        hand_force_resultant_n=resultant,
        joint_torque_norm_nm=effort,
    )


def _engaged_torque(deflection: float, channel: TransmissionChannel) -> float:
    magnitude = abs(deflection) - channel.dead_zone_rad
    if magnitude <= 0.0:
        return 0.0
    return channel.stiffness_nm_rad * np.sign(deflection) * magnitude


def _command_deflection(torque_nm: float, channel: TransmissionChannel) -> float:
    if torque_nm == 0.0:
        return 0.0
    return np.sign(torque_nm) * (
        channel.dead_zone_rad + abs(torque_nm) / channel.stiffness_nm_rad
    )


def _reversal_delay(
    time: FloatArray, transmitted: FloatArray, post_target: float
) -> float:
    if post_target == 0.0:
        return 0.0
    threshold = 0.1 * abs(post_target)
    indices = np.flatnonzero(
        (np.sign(transmitted) == np.sign(post_target))
        & (np.abs(transmitted) >= threshold)
    )
    return float(time[indices[0]]) if indices.size else float(time[-1])


def evaluate_role_reversal(
    program: RoleReversalProgram,
    *,
    arm_channel: TransmissionChannel,
    wrist_channel: TransmissionChannel,
    duration_s: float,
    step_s: float,
    initialize_at_preload: bool,
) -> RoleReversalTrace:
    """Integrate two transmission channels after a declared role transition."""
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    if not np.isfinite(step_s) or step_s <= 0.0 or step_s > duration_s:
        raise ValueError("step_s must be positive and no larger than duration_s")
    intervals = int(round(duration_s / step_s))
    if not np.isclose(intervals * step_s, duration_s, atol=1e-12, rtol=0.0):
        raise ValueError("duration_s must be an integer multiple of step_s")
    time = np.arange(intervals + 1, dtype=float) * step_s
    desired_arm = np.full_like(time, program.arm_post_nm)
    desired_wrist = np.full_like(time, program.wrist_post_nm)
    arm_deflection = np.empty_like(time)
    wrist_deflection = np.empty_like(time)
    arm_torque = np.empty_like(time)
    wrist_torque = np.empty_like(time)
    if initialize_at_preload:
        arm_deflection[0] = _command_deflection(program.arm_pre_nm, arm_channel)
        wrist_deflection[0] = _command_deflection(program.wrist_pre_nm, wrist_channel)
    else:
        arm_deflection[0] = wrist_deflection[0] = 0.0
    arm_torque[0] = _engaged_torque(arm_deflection[0], arm_channel)
    wrist_torque[0] = _engaged_torque(wrist_deflection[0], wrist_channel)
    arm_target_deflection = _command_deflection(program.arm_post_nm, arm_channel)
    wrist_target_deflection = _command_deflection(program.wrist_post_nm, wrist_channel)
    for index in range(intervals):
        arm_deflection[index + 1] = (
            arm_deflection[index]
            + step_s
            * (arm_target_deflection - arm_deflection[index])
            / arm_channel.time_constant_s
        )
        wrist_deflection[index + 1] = (
            wrist_deflection[index]
            + step_s
            * (wrist_target_deflection - wrist_deflection[index])
            / wrist_channel.time_constant_s
        )
        arm_torque[index + 1] = _engaged_torque(arm_deflection[index + 1], arm_channel)
        wrist_torque[index + 1] = _engaged_torque(
            wrist_deflection[index + 1], wrist_channel
        )
    desired_net = desired_arm + desired_wrist
    transmitted_net = arm_torque + wrist_torque
    error_impulse = float(np.trapezoid(np.abs(desired_net - transmitted_net), time))
    tolerance = 1e-12
    return RoleReversalTrace(
        time_s=time,
        desired_arm_torque_nm=desired_arm,
        desired_wrist_torque_nm=desired_wrist,
        transmitted_arm_torque_nm=arm_torque,
        transmitted_wrist_torque_nm=wrist_torque,
        desired_net_torque_nm=desired_net,
        transmitted_net_torque_nm=transmitted_net,
        arm_zero_transmission_duration_s=float(
            np.count_nonzero(np.abs(arm_torque) <= tolerance) * step_s
        ),
        wrist_zero_transmission_duration_s=float(
            np.count_nonzero(np.abs(wrist_torque) <= tolerance) * step_s
        ),
        net_torque_error_impulse_nms=error_impulse,
        arm_reversal_delay_s=_reversal_delay(time, arm_torque, program.arm_post_nm),
        wrist_reversal_delay_s=_reversal_delay(
            time, wrist_torque, program.wrist_post_nm
        ),
    )


__all__ = [
    "MatchedAllocationResult",
    "MatchedAllocationSweep",
    "RoleReversalProgram",
    "RoleReversalTrace",
    "TransmissionChannel",
    "allocate_matched_angular_acceleration",
    "evaluate_role_reversal",
    "matched_allocation_sweep",
]
