"""Distributed passive grip fibers for articulated-contact falsification.

The fibers are engineering comparators with state-registered free lengths.
They are not finger anatomy, measured pressure, tissue, or intentional action.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_slack_contact import (
    AttachmentLawConfig,
    AttachmentLawKind,
    evaluate_attachment_law,
)
from scripts.research.proximal_distal_energy.spatial_full_body import (
    SpatialModel,
    forward_kinematics,
    point_contact_jacobians,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DistributedGripConfig:
    """Geometry, constitutive properties, and friction for two distributed grips."""

    station_count_per_hand: int = 3
    station_width_m: float = 0.03
    total_stiffness_n_m: float = 1800.0
    total_damping_n_s_m: float = 18.0
    tangential_damping_n_s_m: float = 18.0
    friction_coefficient: float = 0.0
    slip_velocity_tolerance_m_s: float = 1.0e-4
    slack_distance_m: float = 0.0
    closure_zero_tolerance_m: float = 1.0e-8

    def __post_init__(self) -> None:
        count = self.station_count_per_hand
        if not isinstance(count, int) or count <= 0 or count % 2 == 0:
            raise ValueError("station_count_per_hand must be a positive odd integer")
        if not np.isfinite(self.station_width_m) or self.station_width_m < 0.0:
            raise ValueError("station_width_m must be finite and nonnegative")
        if count > 1 and self.station_width_m <= 0.0:
            raise ValueError("multi-station grips require positive station_width_m")
        for name in (
            "total_stiffness_n_m",
            "total_damping_n_s_m",
            "tangential_damping_n_s_m",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not np.isfinite(self.friction_coefficient)
            or self.friction_coefficient < 0.0
        ):
            raise ValueError("friction_coefficient must be finite and nonnegative")
        if (
            not np.isfinite(self.slip_velocity_tolerance_m_s)
            or self.slip_velocity_tolerance_m_s <= 0.0
        ):
            raise ValueError("slip_velocity_tolerance_m_s must be finite and positive")
        if not np.isfinite(self.slack_distance_m) or self.slack_distance_m < 0.0:
            raise ValueError("slack_distance_m must be finite and nonnegative")
        if (
            not np.isfinite(self.closure_zero_tolerance_m)
            or self.closure_zero_tolerance_m <= 0.0
        ):
            raise ValueError("closure_zero_tolerance_m must be finite and positive")

    @property
    def station_offsets_m(self) -> FloatArray:
        """Return symmetric grip-axis offsets without changing total width."""

        if self.station_count_per_hand == 1:
            return np.zeros(1)
        return np.linspace(
            -0.5 * self.station_width_m,
            0.5 * self.station_width_m,
            self.station_count_per_hand,
        )

    @property
    def station_law(self) -> AttachmentLawConfig:
        """Return equal station shares of the declared total law."""

        count = self.station_count_per_hand
        kind = (
            AttachmentLawKind.DEAD_ZONE_TENSION
            if self.slack_distance_m > 0.0
            else AttachmentLawKind.TENSION_ONLY
        )
        return AttachmentLawConfig(
            kind=kind,
            stiffness=self.total_stiffness_n_m / count,
            damping=self.total_damping_n_s_m / count,
            slack_distance_m=self.slack_distance_m,
        )


@dataclass(frozen=True, slots=True)
class DistributedGripSnapshot:
    """Generalized load and complete per-station passive and friction ledger."""

    generalized_contact_force: FloatArray
    force_on_club_n: FloatArray
    active_station: NDArray[np.bool_]
    net_club_force_n: FloatArray
    force_couple_vector_nm: FloatArray
    maximum_station_force_n: float
    maximum_extension_m: float
    active_station_count: int
    load_concentration: float
    action_reaction_residual_n: float
    coincident_couple_residual_nm: float
    reversed_couple_sign_residual_nm: float
    virtual_power_residual_w: float
    storage_power_w: float
    dissipation_power_w: float
    strain_energy_j: float
    normal_force_on_club_n: FloatArray | None = None
    tangential_force_on_club_n: FloatArray | None = None
    slipping_station: NDArray[np.bool_] | None = None
    station_extension_m: FloatArray | None = None
    station_signed_gap_m: FloatArray | None = None
    normal_power_w: float = 0.0
    tangential_power_w: float = 0.0
    normal_dissipation_power_w: float = 0.0
    tangential_dissipation_power_w: float = 0.0
    maximum_tangential_force_n: float = 0.0
    maximum_sliding_speed_m_s: float = 0.0
    coulomb_limit_utilization: float = 0.0


def distributed_contact_kinematics(
    model: SpatialModel,
    q: FloatArray,
    *,
    grip_span_m: float,
    hand_contact_local_x_m: float,
    config: DistributedGripConfig,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return hand/grip points and Jacobians for every declared fiber."""

    position = np.asarray(q, dtype=float)
    if position.shape != (model.nq,):
        raise ValueError("q must match the articulated model dimension")
    if not np.isfinite(grip_span_m) or grip_span_m <= 0.0:
        raise ValueError("grip_span_m must be finite and positive")
    if not np.isfinite(hand_contact_local_x_m) or hand_contact_local_x_m <= 0.0:
        raise ValueError("hand_contact_local_x_m must be finite and positive")
    kin = forward_kinematics(model, position)
    hand_points, hand_jacobians = [], []
    grip_points, grip_jacobians = [], []
    hand_joints = (model.lead_hand_joint, model.trail_hand_joint)
    centers = (grip_span_m / 2.0, -grip_span_m / 2.0)
    for hand_joint, center in zip(hand_joints, centers, strict=True):
        hand_row, hand_jacobian_row = [], []
        grip_row, grip_jacobian_row = [], []
        for offset in config.station_offsets_m:
            hand_local = np.array([hand_contact_local_x_m, offset, 0.0])
            grip_local = np.array([0.0, center + offset, -0.03])
            hand_point, hand_jacobian, _ = point_contact_jacobians(
                model, kin, hand_joint, hand_local
            )
            grip_point, grip_jacobian, _ = point_contact_jacobians(
                model, kin, model.club_frame_joint, grip_local
            )
            hand_row.append(hand_point)
            hand_jacobian_row.append(hand_jacobian)
            grip_row.append(grip_point)
            grip_jacobian_row.append(grip_jacobian)
        hand_points.append(hand_row)
        hand_jacobians.append(hand_jacobian_row)
        grip_points.append(grip_row)
        grip_jacobians.append(grip_jacobian_row)
    return tuple(
        np.asarray(value)
        for value in (hand_points, hand_jacobians, grip_points, grip_jacobians)
    )  # type: ignore[return-value]


def distributed_reference_lengths(
    model: SpatialModel,
    q: FloatArray,
    *,
    grip_span_m: float,
    hand_contact_local_x_m: float,
    config: DistributedGripConfig,
) -> FloatArray:
    """Register zero-preload fiber lengths at one closed configuration."""

    hand, _, grip, _ = distributed_contact_kinematics(
        model,
        q,
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=hand_contact_local_x_m,
        config=config,
    )
    lengths = np.linalg.norm(hand - grip, axis=2)
    lengths[lengths <= config.closure_zero_tolerance_m] = 0.0
    return lengths


def _validate_reference_lengths(
    value: FloatArray, config: DistributedGripConfig
) -> FloatArray:
    lengths = np.asarray(value, dtype=float)
    expected = (2, config.station_count_per_hand)
    if lengths.shape != expected or np.any(~np.isfinite(lengths)):
        raise ValueError(f"reference_lengths_m must have finite shape {expected}")
    if np.any(lengths < 0.0):
        raise ValueError("reference_lengths_m must be nonnegative")
    return lengths


def evaluate_distributed_grip(
    model: SpatialModel,
    q: FloatArray,
    qd: FloatArray,
    *,
    grip_span_m: float,
    hand_contact_local_x_m: float,
    reference_lengths_m: FloatArray,
    config: DistributedGripConfig,
) -> DistributedGripSnapshot:
    """Project distributed equal-and-opposite fiber forces into coordinates."""

    position, velocity = np.asarray(q, dtype=float), np.asarray(qd, dtype=float)
    if position.shape != (model.nq,) or velocity.shape != (model.nq,):
        raise ValueError("q and qd must match the articulated model dimension")
    references = _validate_reference_lengths(reference_lengths_m, config)
    hand, hand_jac, grip, grip_jac = distributed_contact_kinematics(
        model,
        position,
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=hand_contact_local_x_m,
        config=config,
    )
    return evaluate_distributed_grip_kinematics(
        hand,
        hand_jac,
        grip,
        grip_jac,
        velocity,
        reference_lengths_m=references,
        config=config,
    )


def _compute_station_friction(
    disp: FloatArray,
    rel_velocity: FloatArray,
    f_norm: FloatArray,
    active: bool,
    config: DistributedGripConfig,
    station_c_t: float,
) -> tuple[FloatArray, float, float, float, bool]:
    f_tan = np.zeros(3)
    f_norm_mag = float(np.linalg.norm(f_norm))
    v_tan_mag = 0.0
    utilization = 0.0
    is_slipping = False
    tan_diss = 0.0

    if active and f_norm_mag > 0.0:
        dist = float(np.linalg.norm(disp))
        normal_dir = disp / dist if dist > 0.0 else np.zeros(3)
        v_n_scalar = float(normal_dir @ rel_velocity)
        v_tan = rel_velocity - v_n_scalar * normal_dir
        v_tan_mag = float(np.linalg.norm(v_tan))

        if config.friction_coefficient > 0.0 and v_tan_mag > 0.0:
            t_dir = v_tan / v_tan_mag
            f_tan_trial = station_c_t * v_tan_mag
            f_tan_cone = config.friction_coefficient * f_norm_mag
            if f_tan_cone > 0.0:
                utilization = min(1.0, f_tan_trial / f_tan_cone)
            if f_tan_trial >= f_tan_cone:
                f_tan_mag = f_tan_cone
                is_slipping = True
            else:
                f_tan_mag = f_tan_trial
                if v_tan_mag > config.slip_velocity_tolerance_m_s:
                    is_slipping = True
            f_tan = f_tan_mag * t_dir
            tan_diss = -float(f_tan @ v_tan)

    return f_tan, v_tan_mag, utilization, tan_diss, is_slipping


@dataclass(slots=True)
class _KinematicsBuffers:
    forces: FloatArray
    normal_forces: FloatArray
    tangential_forces: FloatArray
    active: NDArray[np.bool_]
    slipping: NDArray[np.bool_]
    extensions: FloatArray
    signed_gaps: FloatArray
    tangential_norms: FloatArray
    sliding_speeds: FloatArray
    coulomb_utilization: FloatArray
    generalized: FloatArray
    physical_power: float = 0.0
    storage: float = 0.0
    dissipation: float = 0.0
    strain: float = 0.0
    normal_power: float = 0.0
    tangential_power: float = 0.0
    normal_dissipation: float = 0.0
    tangential_dissipation: float = 0.0
    action_residual: float = 0.0


def _build_distributed_snapshot(
    grip: FloatArray,
    velocity: FloatArray,
    buf: _KinematicsBuffers,
) -> DistributedGripSnapshot:
    norms = np.linalg.norm(buf.forces, axis=2)
    total_norm = float(np.sum(norms))
    midpoint = np.mean(grip.reshape(-1, 3), axis=0)
    couple = np.sum(np.cross(grip - midpoint, buf.forces), axis=(0, 1))
    coincident_couple = np.sum(np.cross(np.zeros_like(grip), buf.forces), axis=(0, 1))
    reversed_couple = np.sum(np.cross(-(grip - midpoint), buf.forces), axis=(0, 1))
    net_force = np.sum(buf.forces, axis=(0, 1))
    return DistributedGripSnapshot(
        generalized_contact_force=buf.generalized,
        force_on_club_n=buf.forces,
        active_station=buf.active,
        net_club_force_n=net_force,
        force_couple_vector_nm=couple,
        maximum_station_force_n=float(np.max(norms)),
        maximum_extension_m=float(np.max(buf.extensions)),
        active_station_count=int(np.count_nonzero(buf.active)),
        load_concentration=(float(np.max(norms)) / total_norm if total_norm else 0.0),
        action_reaction_residual_n=buf.action_residual,
        coincident_couple_residual_nm=float(np.linalg.norm(coincident_couple)),
        reversed_couple_sign_residual_nm=float(
            np.linalg.norm(reversed_couple + couple)
        ),
        virtual_power_residual_w=float(
            abs(buf.generalized @ velocity - buf.physical_power)
        ),
        storage_power_w=float(buf.storage),
        dissipation_power_w=float(buf.dissipation),
        strain_energy_j=float(buf.strain),
        normal_force_on_club_n=buf.normal_forces,
        tangential_force_on_club_n=buf.tangential_forces,
        slipping_station=buf.slipping,
        station_extension_m=buf.extensions,
        station_signed_gap_m=buf.signed_gaps,
        normal_power_w=float(buf.normal_power),
        tangential_power_w=float(buf.tangential_power),
        normal_dissipation_power_w=float(buf.normal_dissipation),
        tangential_dissipation_power_w=float(buf.tangential_dissipation),
        maximum_tangential_force_n=float(np.max(buf.tangential_norms)),
        maximum_sliding_speed_m_s=float(np.max(buf.sliding_speeds)),
        coulomb_limit_utilization=float(np.max(buf.coulomb_utilization)),
    )


def _evaluate_all_stations(
    hand: FloatArray,
    hand_jac: FloatArray,
    grip: FloatArray,
    grip_jac: FloatArray,
    velocity: FloatArray,
    references: FloatArray,
    config: DistributedGripConfig,
    buf: _KinematicsBuffers,
) -> None:
    law = config.station_law
    station_c_t = (
        config.tangential_damping_n_s_m / config.station_count_per_hand
        if config.station_count_per_hand > 0
        else 0.0
    )
    for h_idx in range(2):
        for s_idx in range(config.station_count_per_hand):
            v_h = hand_jac[h_idx, s_idx] @ velocity
            v_g = grip_jac[h_idx, s_idx] @ velocity
            v_rel = v_h - v_g
            disp = hand[h_idx, s_idx] - grip[h_idx, s_idx]

            snap = evaluate_attachment_law(
                displacement_m=disp,
                relative_velocity_m_s=v_rel,
                config=law,
                reference_length_m=references[h_idx, s_idx],
            )
            f_norm = snap.force_on_club_n
            buf.normal_forces[h_idx, s_idx] = f_norm
            buf.active[h_idx, s_idx] = snap.active
            buf.extensions[h_idx, s_idx] = snap.extension_m
            buf.signed_gaps[h_idx, s_idx] = float(np.linalg.norm(disp)) - (
                references[h_idx, s_idx] + config.slack_distance_m
            )
            buf.storage += snap.storage_power_w
            normal_diss = snap.dissipation_power_w
            buf.normal_dissipation += normal_diss
            buf.normal_power += float(snap.interface_power_w)
            buf.strain += snap.strain_energy_j

            f_tan, v_tan_mag, util, tan_diss, is_slip = _compute_station_friction(
                disp, v_rel, f_norm, snap.active, config, station_c_t
            )
            buf.sliding_speeds[h_idx, s_idx] = v_tan_mag
            buf.coulomb_utilization[h_idx, s_idx] = util
            buf.slipping[h_idx, s_idx] = is_slip
            buf.tangential_dissipation += tan_diss
            buf.tangential_power += tan_diss
            buf.tangential_forces[h_idx, s_idx] = f_tan
            buf.tangential_norms[h_idx, s_idx] = float(np.linalg.norm(f_tan))

            f_total = f_norm + f_tan
            buf.forces[h_idx, s_idx] = f_total
            buf.generalized += (
                grip_jac[h_idx, s_idx].T @ f_total - hand_jac[h_idx, s_idx].T @ f_total
            )
            buf.physical_power += float(f_total @ v_g - f_total @ v_h)
            buf.dissipation += normal_diss + (-float(f_tan @ v_rel))


def evaluate_distributed_grip_kinematics(
    hand: FloatArray,
    hand_jac: FloatArray,
    grip: FloatArray,
    grip_jac: FloatArray,
    generalized_velocity: FloatArray,
    *,
    reference_lengths_m: FloatArray,
    config: DistributedGripConfig,
) -> DistributedGripSnapshot:
    """Evaluate the fiber law from declared points and common-coordinate Jacobians."""
    hand = np.asarray(hand, dtype=float)
    grip = np.asarray(grip, dtype=float)
    hand_jac = np.asarray(hand_jac, dtype=float)
    grip_jac = np.asarray(grip_jac, dtype=float)
    velocity = np.asarray(generalized_velocity, dtype=float)
    references = _validate_reference_lengths(reference_lengths_m, config)
    point_shape = (2, config.station_count_per_hand, 3)
    if hand.shape != point_shape or grip.shape != point_shape:
        raise ValueError(f"hand and grip points must have shape {point_shape}")
    jacobian_shape = (*point_shape, velocity.size)
    if hand_jac.shape != jacobian_shape or grip_jac.shape != jacobian_shape:
        raise ValueError(f"contact Jacobians must have shape {jacobian_shape}")
    arrays = (hand, grip, hand_jac, grip_jac, velocity)
    if any(np.any(~np.isfinite(value)) for value in arrays):
        raise ValueError("contact kinematics and velocity must be finite")

    buf = _KinematicsBuffers(
        forces=np.zeros_like(hand),
        normal_forces=np.zeros_like(hand),
        tangential_forces=np.zeros_like(hand),
        active=np.zeros(references.shape, dtype=bool),
        slipping=np.zeros(references.shape, dtype=bool),
        extensions=np.zeros(references.shape),
        signed_gaps=np.zeros(references.shape),
        tangential_norms=np.zeros(references.shape),
        sliding_speeds=np.zeros(references.shape),
        coulomb_utilization=np.zeros(references.shape),
        generalized=np.zeros(velocity.size),
    )

    _evaluate_all_stations(
        hand, hand_jac, grip, grip_jac, velocity, references, config, buf
    )

    return _build_distributed_snapshot(grip, velocity, buf)


__all__ = [
    "DistributedGripConfig",
    "DistributedGripSnapshot",
    "distributed_contact_kinematics",
    "distributed_reference_lengths",
    "evaluate_distributed_grip",
    "evaluate_distributed_grip_kinematics",
]
