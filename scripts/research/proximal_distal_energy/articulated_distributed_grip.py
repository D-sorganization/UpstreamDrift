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
    """Geometry and total constitutive properties for two distributed grips."""

    station_count_per_hand: int = 3
    station_width_m: float = 0.03
    total_stiffness_n_m: float = 1800.0
    total_damping_n_s_m: float = 18.0
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
        for name in ("total_stiffness_n_m", "total_damping_n_s_m"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
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
    """Generalized load and complete per-station passive ledger."""

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
    forces = np.zeros_like(hand)
    active = np.zeros(references.shape, dtype=bool)
    generalized = np.zeros(model.nq)
    physical_power = storage = dissipation = strain = 0.0
    extensions = np.zeros(references.shape)
    action_residual = 0.0
    law = config.station_law
    for hand_index in range(2):
        for station_index in range(config.station_count_per_hand):
            hand_velocity = hand_jac[hand_index, station_index] @ velocity
            grip_velocity = grip_jac[hand_index, station_index] @ velocity
            snapshot = evaluate_attachment_law(
                displacement_m=hand[hand_index, station_index]
                - grip[hand_index, station_index],
                relative_velocity_m_s=hand_velocity - grip_velocity,
                config=law,
                reference_length_m=references[hand_index, station_index],
            )
            force = snapshot.force_on_club_n
            forces[hand_index, station_index] = force
            active[hand_index, station_index] = snapshot.active
            extensions[hand_index, station_index] = snapshot.extension_m
            generalized += grip_jac[hand_index, station_index].T @ force
            generalized += hand_jac[hand_index, station_index].T @ (-force)
            physical_power += float(force @ grip_velocity - force @ hand_velocity)
            storage += snapshot.storage_power_w
            dissipation += snapshot.dissipation_power_w
            strain += snapshot.strain_energy_j
            action_residual = max(
                action_residual,
                float(np.linalg.norm(force + snapshot.force_on_hand_n)),
            )
    norms = np.linalg.norm(forces, axis=2)
    total_norm = float(np.sum(norms))
    midpoint = np.mean(grip.reshape(-1, 3), axis=0)
    couple = np.sum(np.cross(grip - midpoint, forces), axis=(0, 1))
    coincident_couple = np.sum(np.cross(np.zeros_like(grip), forces), axis=(0, 1))
    reversed_couple = np.sum(np.cross(-(grip - midpoint), forces), axis=(0, 1))
    net_force = np.sum(forces, axis=(0, 1))
    return DistributedGripSnapshot(
        generalized_contact_force=generalized,
        force_on_club_n=forces,
        active_station=active,
        net_club_force_n=net_force,
        force_couple_vector_nm=couple,
        maximum_station_force_n=float(np.max(norms)),
        maximum_extension_m=float(np.max(extensions)),
        active_station_count=int(np.count_nonzero(active)),
        load_concentration=(float(np.max(norms)) / total_norm if total_norm else 0.0),
        action_reaction_residual_n=action_residual,
        coincident_couple_residual_nm=float(np.linalg.norm(coincident_couple)),
        reversed_couple_sign_residual_nm=float(
            np.linalg.norm(reversed_couple + couple)
        ),
        virtual_power_residual_w=float(abs(generalized @ velocity - physical_power)),
        storage_power_w=float(storage),
        dissipation_power_w=float(dissipation),
        strain_energy_j=float(strain),
    )


__all__ = [
    "DistributedGripConfig",
    "DistributedGripSnapshot",
    "distributed_contact_kinematics",
    "distributed_reference_lengths",
    "evaluate_distributed_grip",
]
