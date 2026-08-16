"""Typed passive attachment laws for articulated grip-contact falsification.

The laws in this module are synthetic constitutive comparators. They do not
identify biological tissue, intentional hand action, or a coaching strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_contact_projection import (
    contact_kinematics,
)
from scripts.research.proximal_distal_energy.spatial_full_body import SpatialModel

FloatArray = NDArray[np.float64]


class AttachmentLawKind(str, Enum):
    """Supported constitutive classes with distinct open-interface behavior."""

    BILATERAL = "bilateral"
    TENSION_ONLY = "tension_only"
    DEAD_ZONE_TENSION = "dead_zone_tension"


@dataclass(frozen=True, slots=True)
class AttachmentLawConfig:
    """Parameters for one passive attachment law."""

    kind: AttachmentLawKind
    stiffness: float = 1800.0
    damping: float = 18.0
    slack_distance_m: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, AttachmentLawKind):
            raise TypeError("kind must be an AttachmentLawKind")
        if not np.isfinite(self.stiffness) or self.stiffness <= 0.0:
            raise ValueError("stiffness must be finite and positive")
        if not np.isfinite(self.damping) or self.damping < 0.0:
            raise ValueError("damping must be finite and nonnegative")
        if not np.isfinite(self.slack_distance_m) or self.slack_distance_m < 0.0:
            raise ValueError("slack_distance_m must be finite and nonnegative")
        if (
            self.kind is AttachmentLawKind.DEAD_ZONE_TENSION
            and self.slack_distance_m <= 0.0
        ):
            raise ValueError("dead-zone slack_distance_m must be positive")
        if (
            self.kind is not AttachmentLawKind.DEAD_ZONE_TENSION
            and self.slack_distance_m != 0.0
        ):
            raise ValueError("bilateral and tension-only laws require zero slack")


@dataclass(frozen=True, slots=True)
class AttachmentLawSnapshot:
    """Force, power, storage, and active-set state for one interface."""

    force_on_club_n: FloatArray
    force_on_hand_n: FloatArray
    storage_power_w: float
    dissipation_power_w: float
    interface_power_w: float
    strain_energy_j: float
    extension_m: float
    active: bool


@dataclass(frozen=True, slots=True)
class SlackProjectionSnapshot:
    """Projected two-hand generalized load and constitutive ledger."""

    generalized_contact_force: FloatArray
    maximum_contact_force_n: float
    maximum_attachment_separation_m: float
    active_interface_count: int
    virtual_power_residual_w: float
    storage_power_w: float
    dissipation_power_w: float
    strain_energy_j: float


def _finite_vector(value: FloatArray, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be one finite 3-vector")
    return array


def _bilateral_snapshot(
    displacement: FloatArray,
    relative_velocity: FloatArray,
    config: AttachmentLawConfig,
) -> AttachmentLawSnapshot:
    force = config.stiffness * displacement + config.damping * relative_velocity
    storage_power = -float(config.stiffness * displacement @ relative_velocity)
    dissipation = -float(config.damping * relative_velocity @ relative_velocity)
    return AttachmentLawSnapshot(
        force_on_club_n=force,
        force_on_hand_n=-force,
        storage_power_w=storage_power,
        dissipation_power_w=dissipation,
        interface_power_w=-float(force @ relative_velocity),
        strain_energy_j=0.5 * config.stiffness * float(displacement @ displacement),
        extension_m=float(np.linalg.norm(displacement)),
        active=True,
    )


def _tension_snapshot(
    displacement: FloatArray,
    relative_velocity: FloatArray,
    config: AttachmentLawConfig,
    reference_length_m: float,
) -> AttachmentLawSnapshot:
    distance = float(np.linalg.norm(displacement))
    free_length = reference_length_m + config.slack_distance_m
    extension = max(0.0, distance - free_length)
    if extension == 0.0 or distance <= np.finfo(float).eps:
        zero = np.zeros(3)
        return AttachmentLawSnapshot(zero, zero, 0.0, 0.0, 0.0, 0.0, 0.0, False)
    direction = displacement / distance
    extension_rate = float(direction @ relative_velocity)
    loading_rate = max(0.0, extension_rate)
    magnitude = config.stiffness * extension + config.damping * loading_rate
    force = magnitude * direction
    storage_power = -config.stiffness * extension * extension_rate
    dissipation = -config.damping * loading_rate * extension_rate
    return AttachmentLawSnapshot(
        force_on_club_n=force,
        force_on_hand_n=-force,
        storage_power_w=float(storage_power),
        dissipation_power_w=float(dissipation),
        interface_power_w=-float(force @ relative_velocity),
        strain_energy_j=0.5 * config.stiffness * extension * extension,
        extension_m=extension,
        active=True,
    )


def evaluate_attachment_law(
    *,
    displacement_m: FloatArray,
    relative_velocity_m_s: FloatArray,
    config: AttachmentLawConfig,
    reference_length_m: float = 0.0,
) -> AttachmentLawSnapshot:
    """Evaluate one declared passive law without hidden force while open."""

    if not isinstance(config, AttachmentLawConfig):
        raise TypeError("config must be an AttachmentLawConfig")
    displacement = _finite_vector(displacement_m, "displacement_m")
    relative_velocity = _finite_vector(relative_velocity_m_s, "relative_velocity_m_s")
    if not np.isfinite(reference_length_m) or reference_length_m < 0.0:
        raise ValueError("reference_length_m must be finite and nonnegative")
    if config.kind is AttachmentLawKind.BILATERAL:
        return _bilateral_snapshot(displacement, relative_velocity, config)
    return _tension_snapshot(
        displacement, relative_velocity, config, reference_length_m
    )


def evaluate_slack_projection(
    model: SpatialModel,
    q: FloatArray,
    qd: FloatArray,
    *,
    grip_span_m: float,
    hand_contact_local_x_m: float,
    law: AttachmentLawConfig,
) -> SlackProjectionSnapshot:
    """Project two typed attachment interfaces into articulated coordinates."""

    position, velocity = np.asarray(q, dtype=float), np.asarray(qd, dtype=float)
    if position.shape != (model.nq,) or velocity.shape != (model.nq,):
        raise ValueError("q and qd must match the articulated model dimension")
    hand_points, hand_jacobians, grip_points, grip_jacobians = contact_kinematics(
        model,
        position,
        velocity,
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=hand_contact_local_x_m,
    )
    generalized = np.zeros(model.nq)
    force_norms, active = [], 0
    physical_power = storage_power = dissipation = strain_energy = 0.0
    for index in range(2):
        hand_velocity = hand_jacobians[index] @ velocity
        grip_velocity = grip_jacobians[index] @ velocity
        snapshot = evaluate_attachment_law(
            displacement_m=hand_points[index] - grip_points[index],
            relative_velocity_m_s=hand_velocity - grip_velocity,
            config=law,
        )
        generalized += grip_jacobians[index].T @ snapshot.force_on_club_n
        generalized += hand_jacobians[index].T @ snapshot.force_on_hand_n
        physical_power += float(
            snapshot.force_on_club_n @ grip_velocity
            + snapshot.force_on_hand_n @ hand_velocity
        )
        storage_power += snapshot.storage_power_w
        dissipation += snapshot.dissipation_power_w
        strain_energy += snapshot.strain_energy_j
        force_norms.append(float(np.linalg.norm(snapshot.force_on_club_n)))
        active += int(snapshot.active)
    return SlackProjectionSnapshot(
        generalized_contact_force=generalized,
        maximum_contact_force_n=max(force_norms),
        maximum_attachment_separation_m=float(
            np.max(np.linalg.norm(hand_points - grip_points, axis=1))
        ),
        active_interface_count=active,
        virtual_power_residual_w=float(abs(generalized @ velocity - physical_power)),
        storage_power_w=float(storage_power),
        dissipation_power_w=float(dissipation),
        strain_energy_j=float(strain_energy),
    )


__all__ = [
    "AttachmentLawConfig",
    "AttachmentLawKind",
    "AttachmentLawSnapshot",
    "SlackProjectionSnapshot",
    "evaluate_attachment_law",
    "evaluate_slack_projection",
]
