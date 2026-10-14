"""Constitutive laws and momentum closure auditors for spatial carriage contact.

Governs internal angular-momentum balance and rigid-motion objectivity for the
spatial carriage Kelvin-Voigt interface (Issue #11195).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import subprocess
from typing import Any, Callable, NamedTuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

DOCUMENTED_ARCHIVE_SHA256 = (
    "0abc856f275678d04009d1950e4779d7e98ef5d8c98398a8d6f1653d6a1fc61d"
)
CURRENT_ARCHIVE_SHA256 = (
    "29b955b16aaba2e348060248d50fe954dd54a351e143055c8e29e2b7af51694a"
)
DOCUMENTED_PEAK_MOMENTS = {
    "mujoco": 0.0826086564194171,
    "pinocchio": 0.08273899228765469,
}


class CentralContactResult(NamedTuple):
    """Result of evaluating a central-force compliant contact interface."""

    force_on_club: FloatArray
    force_on_hand: FloatArray
    storage_power: float
    dissipated_power: float

    @property
    def torque_on_club(self) -> FloatArray:
        """Central force exerts zero contact torque on the club."""
        return np.zeros(3, dtype=np.float64)

    @property
    def torque_on_hand(self) -> FloatArray:
        """Central force exerts zero contact torque on the hand."""
        return np.zeros(3, dtype=np.float64)


class ClosedWrenchResult(NamedTuple):
    """Result of evaluating a closed-wrench compliant contact interface."""

    force_on_club: FloatArray
    force_on_hand: FloatArray
    torque_on_club: FloatArray
    torque_on_hand: FloatArray
    storage_power: float
    dissipated_power: float


@dataclass(frozen=True)
class ObjectivityAuditResult:
    """Diagnostic outcome of rigid-motion frame indifference verification."""

    is_objective: bool
    spurious_damping_force: FloatArray
    spurious_damping_norm: float
    spurious_dissipation: float

    def __bool__(self) -> bool:
        return self.is_objective


@dataclass(frozen=True)
class ArchivedContactMomentAudit:
    """Diagnostic audit of archived spatial carriage contact-pair moments."""

    sha256: str
    documented_sha256: str
    peak_moments: dict[str, float]
    post_killswitch_peak_moments: dict[str, float]
    max_endpoint_gap: float
    mujoco_peak: float
    pinocchio_peak: float


def central_force_compliant_contact(
    *,
    hand_position: FloatArray,
    hand_velocity: FloatArray,
    club_point_position: FloatArray,
    club_point_velocity: FloatArray,
    stiffness: float,
    damping: float,
) -> CentralContactResult:
    """Evaluate central-force compliant contact with line-of-action damping.

    Projects the viscous damping velocity onto the unit line-of-action vector
    d_hat = (h - p) / ||h - p||:
        F = [k ||d|| + c (v_h - v_p) . d_hat] d_hat
    so that (p - h) x F == 0 identically for all separated endpoints p != h,
    closing internal angular-momentum balance and preserving rigid-body
    objectivity under superimposed rigid rotation (v_rel . d_hat == 0).
    """
    vectors = [hand_position, hand_velocity, club_point_position, club_point_velocity]
    if any(np.asarray(v).shape != (3,) for v in vectors):
        raise ValueError("contact state values must be 3-vectors")
    if stiffness <= 0.0 or damping < 0.0:
        raise ValueError("contact stiffness must be positive and damping nonnegative")

    displacement = np.asarray(hand_position, dtype=np.float64) - np.asarray(
        club_point_position, dtype=np.float64
    )
    relative_velocity = np.asarray(hand_velocity, dtype=np.float64) - np.asarray(
        club_point_velocity, dtype=np.float64
    )
    distance = float(np.linalg.norm(displacement))
    if distance > 1e-12:
        direction = displacement / distance
        radial_velocity = float(np.dot(relative_velocity, direction))
    else:
        direction = np.zeros(3, dtype=np.float64)
        radial_velocity = 0.0

    force_magnitude = stiffness * distance + damping * radial_velocity
    force_on_club = force_magnitude * direction
    force_on_hand = -force_on_club

    storage_power = -float(stiffness * distance * radial_velocity)
    dissipated_power = -float(damping * (radial_velocity**2))

    return CentralContactResult(
        force_on_club=force_on_club,
        force_on_hand=force_on_hand,
        storage_power=storage_power,
        dissipated_power=dissipated_power,
    )


def closed_wrench_compliant_contact(
    *,
    hand_position: FloatArray,
    hand_velocity: FloatArray,
    club_point_position: FloatArray,
    club_point_velocity: FloatArray,
    stiffness: float,
    damping: float,
    hand_angular_velocity: FloatArray | None = None,
    club_angular_velocity: FloatArray | None = None,
) -> ClosedWrenchResult:
    """Evaluate directional spring-damper contact with compensating couples.

    Pairs the directional spring-damper force F = k(h - p) + c(v_h - v_p)
    with the compensating internal reaction couple:
        tau_p = tau_h = 1/2 (h - p) x F
    so the total wrench pair (F, tau_p) at p and (-F, tau_h) at h has zero
    net internal moment about any common origin:
        (p x F + tau_p) + (h x (-F) + tau_h) == 0.
    """
    vectors = [hand_position, hand_velocity, club_point_position, club_point_velocity]
    if any(np.asarray(v).shape != (3,) for v in vectors):
        raise ValueError("contact state values must be 3-vectors")
    if stiffness <= 0.0 or damping < 0.0:
        raise ValueError("contact stiffness must be positive and damping nonnegative")

    displacement = np.asarray(hand_position, dtype=np.float64) - np.asarray(
        club_point_position, dtype=np.float64
    )
    relative_velocity = np.asarray(hand_velocity, dtype=np.float64) - np.asarray(
        club_point_velocity, dtype=np.float64
    )

    force_on_club = stiffness * displacement + damping * relative_velocity
    force_on_hand = -force_on_club

    compensating_couple = 0.5 * np.cross(displacement, force_on_club)
    torque_on_club = compensating_couple
    torque_on_hand = compensating_couple.copy()

    storage_power = -float(stiffness * np.dot(displacement, relative_velocity))
    dissipated_power = -float(damping * np.dot(relative_velocity, relative_velocity))

    return ClosedWrenchResult(
        force_on_club=force_on_club,
        force_on_hand=force_on_hand,
        torque_on_club=torque_on_club,
        torque_on_hand=torque_on_hand,
        storage_power=storage_power,
        dissipated_power=dissipated_power,
    )


def evaluate_contact_pair_moment_residual(
    *,
    hand_position: FloatArray,
    club_point_position: FloatArray,
    force_on_club: FloatArray,
    force_on_hand: FloatArray | None = None,
    torque_on_club: FloatArray | None = None,
    torque_on_hand: FloatArray | None = None,
    reference_point: FloatArray | None = None,
) -> FloatArray:
    """Evaluate net contact-pair moment residual about reference_point.

    Total internal moment is:
        M = (p - ref) x F_club + tau_club + (h - ref) x F_hand + tau_hand.
    When F_hand == -F_club, this simplifies to (p - h) x F_club + tau_club + tau_hand,
    which is independent of the reference origin.
    """
    p = np.asarray(club_point_position, dtype=np.float64)
    h = np.asarray(hand_position, dtype=np.float64)
    f_c = np.asarray(force_on_club, dtype=np.float64)
    f_h = -f_c if force_on_hand is None else np.asarray(force_on_hand, dtype=np.float64)
    t_c = (
        np.zeros(3, dtype=np.float64)
        if torque_on_club is None
        else np.asarray(torque_on_club, dtype=np.float64)
    )
    t_h = (
        np.zeros(3, dtype=np.float64)
        if torque_on_hand is None
        else np.asarray(torque_on_hand, dtype=np.float64)
    )
    ref = (
        np.zeros(3, dtype=np.float64)
        if reference_point is None
        else np.asarray(reference_point, dtype=np.float64)
    )

    moment_club = np.cross(p - ref, f_c) + t_c
    moment_hand = np.cross(h - ref, f_h) + t_h
    return moment_club + moment_hand


def verify_contact_law_objectivity(
    *,
    contact_law: Callable[..., Any],
    hand_position: FloatArray,
    club_point_position: FloatArray,
    angular_velocity: FloatArray,
    stiffness: float,
    damping: float,
    tolerance: float = 1e-12,
) -> ObjectivityAuditResult:
    """Verify rigid-motion objectivity under superimposed rigid rotation.

    Under pure rigid rotation with angular velocity omega:
        v_h - v_p = omega x (h - p).
    An objective law produces zero spurious damping force and zero dissipation.
    """
    h = np.asarray(hand_position, dtype=np.float64)
    p = np.asarray(club_point_position, dtype=np.float64)
    omega = np.asarray(angular_velocity, dtype=np.float64)

    v_p = np.zeros(3, dtype=np.float64)
    v_h = v_p + np.cross(omega, h - p)

    res_dynamic = contact_law(
        hand_position=h,
        hand_velocity=v_h,
        club_point_position=p,
        club_point_velocity=v_p,
        stiffness=stiffness,
        damping=damping,
    )
    res_static = contact_law(
        hand_position=h,
        hand_velocity=v_p,
        club_point_position=p,
        club_point_velocity=v_p,
        stiffness=stiffness,
        damping=damping,
    )

    f_dyn = (
        res_dynamic.force_on_club
        if hasattr(res_dynamic, "force_on_club")
        else np.asarray(res_dynamic[0], dtype=np.float64)
    )
    f_stat = (
        res_static.force_on_club
        if hasattr(res_static, "force_on_club")
        else np.asarray(res_static[0], dtype=np.float64)
    )
    spurious_damping_force = f_dyn - f_stat
    spurious_damping_norm = float(np.linalg.norm(spurious_damping_force))

    dissipated_power = (
        res_dynamic.dissipated_power
        if hasattr(res_dynamic, "dissipated_power")
        else float(res_dynamic[3])
    )
    spurious_dissipation = abs(float(dissipated_power))

    is_objective = (
        spurious_damping_norm <= tolerance and spurious_dissipation <= tolerance
    )
    return ObjectivityAuditResult(
        is_objective=is_objective,
        spurious_damping_force=spurious_damping_force,
        spurious_damping_norm=spurious_damping_norm,
        spurious_dissipation=spurious_dissipation,
    )


def audit_archived_spatial_contact_moments(
    npz_path: str | Path | None = None,
    *,
    git_revision: str | None = None,
) -> ArchivedContactMomentAudit:
    """Audit archived spatial carriage contact-pair moments from NPZ evidence.

    Reproduces the exact archived peak moments from
    docs/research/proximal_distal_energy_transfer/data/spatial_forward_contact_study.npz:
        MuJoCo peak: 0.0826086564194171 N m
        Pinocchio peak: 0.08273899228765469 N m
    """
    if git_revision is not None:
        cmd = [
            "git",
            "show",
            f"{git_revision}:docs/research/proximal_distal_energy_transfer/data/spatial_forward_contact_study.npz",
        ]
        raw = subprocess.check_output(cmd)
        digest = hashlib.sha256(raw).hexdigest()
        data = np.load(io.BytesIO(raw), allow_pickle=False)
    else:
        if npz_path is None:
            repo_root = Path(__file__).resolve().parents[3]
            npz_path = (
                repo_root
                / "docs"
                / "research"
                / "proximal_distal_energy_transfer"
                / "data"
                / "spatial_forward_contact_study.npz"
            )
        path = Path(npz_path)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        data = np.load(io.BytesIO(raw), allow_pickle=False)

    peak_moments: dict[str, float] = {}
    post_killswitch_peak_moments: dict[str, float] = {}
    max_endpoint_gaps: list[float] = []

    prefixes = [
        "mujoco_baseline",
        "mujoco_killswitch",
        "pinocchio_baseline",
        "pinocchio_killswitch",
    ]
    for prefix in prefixes:
        contact_points = data[f"{prefix}_contact_points"]
        hand_positions = data[f"{prefix}_hand_positions"]
        contact_forces = data[f"{prefix}_contact_forces"]
        time_array = data[f"{prefix}_time"]

        pair_moments = np.cross(contact_points - hand_positions, contact_forces)
        net_contact_moment = pair_moments.sum(axis=1)
        peak = float(np.linalg.norm(net_contact_moment, axis=1).max())
        peak_moments[prefix] = peak

        post_mask = time_array >= 0.180
        post_peak = float(np.linalg.norm(net_contact_moment[post_mask], axis=1).max())
        post_killswitch_peak_moments[prefix] = post_peak

        gap = float(np.linalg.norm(contact_points - hand_positions, axis=-1).max())
        max_endpoint_gaps.append(gap)

    peak_moments["mujoco"] = peak_moments["mujoco_baseline"]
    peak_moments["pinocchio"] = peak_moments["pinocchio_baseline"]

    return ArchivedContactMomentAudit(
        sha256=digest,
        documented_sha256=DOCUMENTED_ARCHIVE_SHA256,
        peak_moments=peak_moments,
        post_killswitch_peak_moments=post_killswitch_peak_moments,
        max_endpoint_gap=max(max_endpoint_gaps),
        mujoco_peak=peak_moments["mujoco_baseline"],
        pinocchio_peak=peak_moments["pinocchio_baseline"],
    )


__all__ = [
    "CURRENT_ARCHIVE_SHA256",
    "DOCUMENTED_ARCHIVE_SHA256",
    "DOCUMENTED_PEAK_MOMENTS",
    "ArchivedContactMomentAudit",
    "CentralContactResult",
    "ClosedWrenchResult",
    "ObjectivityAuditResult",
    "audit_archived_spatial_contact_moments",
    "central_force_compliant_contact",
    "closed_wrench_compliant_contact",
    "evaluate_contact_pair_moment_residual",
    "verify_contact_law_objectivity",
]
