"""Shared sphere-on-cylinder pad contact law (issue #11739, OSV-7 phase 3).

The ``contact`` grip evaluated analytically: spherical pads fixed on a hand
frame press on the club's cylindrical grip.  Each pad is a sphere against the
tangent plane of the cylinder at its nearest point, so the existing shared
law of :mod:`src.shared.python.motion_matching.contact_law` (Hunt-Crossley
normal force, regularised Coulomb friction with a Stribeck blend) is reused
unchanged, plus its spin (pivot) friction moment for the contact patch.

Used directly by the Pinocchio engine (no native contact) and as the reference
for the static checks of the MuJoCo and Drake native contacts.

Conventions.  ``hand`` and ``club`` are :class:`BushingState` grip frames in the
world (``R`` world <- frame, origin velocity, angular velocity).  Forces and
moments are ON THE CLUB; the moment is about the club grip-frame origin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from src.shared.python.grip_contact.bushing_law import BushingState, cross3
from src.shared.python.grip_contact.interface import GripInterface
from src.shared.python.grip_contact.pad_layout import PadLayout, matched_pad_parameters
from src.shared.python.motion_matching.contact_law import (
    ContactParameters,
    GroundPlane,
    sphere_ground_contact,
    torsional_friction_moment,
)

__all__ = [
    "GripCylinder",
    "PadContactModel",
    "PadWrench",
    "build_pad_model",
    "grip_cylinder",
    "pad_wrench",
]


@dataclass(frozen=True)
class GripCylinder:
    """The grip cylinder on the club: radius and axial range.

    ``axial_range_m`` is in coordinates along the grip axis measured from the
    *right* grip frame origin; ``left_axial_m`` is the left grip frame's
    coordinate on that axis.
    """

    radius_m: float
    axial_range_m: tuple[float, float]
    left_axial_m: float

    def __post_init__(self) -> None:
        lo, hi = self.axial_range_m
        if self.radius_m <= 0.0 or not hi > lo:
            raise ValueError("cylinder needs a positive radius and hi > lo")

    def origin_axial_m(self, side: str) -> float:
        """Axial coordinate of ``side``'s grip frame origin."""
        return self.left_axial_m if side == "L" else 0.0


def grip_cylinder(
    interface: GripInterface, layout: PadLayout, margin_m: float = 0.08
) -> GripCylinder:
    """Cylinder through both grip points, extended ``margin_m`` beyond them.

    Raises:
        ValueError: if the two grip origins are not ``2 * grip_radius`` apart
            across a common grip axis (the layout would not sit on the shaft).
    """
    rot = np.asarray(interface.right.rotation, dtype=float)
    offset = rot.T @ (
        np.asarray(interface.left.position_m) - np.asarray(interface.right.position_m)
    )
    across = np.array([0.0, 0.0, 2.0 * layout.grip_radius_m])
    if not np.allclose(offset[1:], across[1:], atol=1e-6):
        raise ValueError(
            f"grip origins must be {across[2]:.4f} m apart across the grip "
            f"(got {offset[1]:.4f}, {offset[2]:.4f})"
        )
    left = float(offset[0])
    return GripCylinder(
        layout.grip_radius_m,
        (min(0.0, left) - margin_m, max(0.0, left) + margin_m),
        left,
    )


@dataclass(frozen=True)
class PadContactModel:
    """Pad layout, shared contact-law parameters and the patch for spin friction."""

    layout: PadLayout
    law: ContactParameters
    cylinder: GripCylinder
    patch_radius_m: float
    spin_transition_rad_s: float = 1.0

    def __post_init__(self) -> None:
        if self.patch_radius_m < 0.0 or self.spin_transition_rad_s <= 0.0:
            raise ValueError("patch radius >= 0 and spin transition > 0 required")


#: Friction transition speed [m/s] of the regularised Coulomb law.
FRICTION_TRANSITION_M_S = 1.0e-3


def build_pad_model(
    interface: GripInterface,
    squeeze_per_hand_n: float,
    layout: PadLayout | None = None,
    friction_transition_m_s: float = FRICTION_TRANSITION_M_S,
) -> PadContactModel:
    """Pad model matched to ``interface.bushing`` with the requested squeeze.

    The contact patch radius for spin friction is the Hertz-like half width
    ``sqrt(2 r_eff delta0)`` of a pad sphere pressed ``delta0`` into the
    cylinder, ``1 / r_eff = 1 / r_pad + 1 / r_grip``.
    """
    base = layout or PadLayout()
    matched = matched_pad_parameters(interface.bushing, base, squeeze_per_hand_n)
    layout = replace(base, preload_penetration_m=matched.preload_penetration_m)
    material = interface.contact_material
    law = ContactParameters(
        stiffness_n_m=matched.stiffness_n_m,
        dissipation_s_m=matched.dissipation_s_m,
        static_friction=material.static_friction,
        dynamic_friction=material.dynamic_friction,
        viscous_friction=material.viscous_friction,
        transition_velocity_m_s=friction_transition_m_s,
    )
    r_eff = 1.0 / (1.0 / layout.pad_radius_m + 1.0 / layout.grip_radius_m)
    patch = math.sqrt(2.0 * r_eff * layout.preload_penetration_m)
    return PadContactModel(layout, law, grip_cylinder(interface, layout), patch)


@dataclass(frozen=True)
class PadWrench:
    """Wrench of one hand's pads on the club (world axes, about the grip origin)."""

    force_n: np.ndarray
    moment_nm: np.ndarray
    normal_force_n: np.ndarray  # per pad, nonnegative
    tangential_force_n: np.ndarray  # per pad, magnitude


def pad_wrench(
    model: PadContactModel,
    side: str,
    hand: BushingState,
    club: BushingState,
    positions_grip_frame: np.ndarray | None = None,
) -> PadWrench:
    """Contact wrench of ``side``'s pads on the club grip.

    Postconditions: per-pad normal force is nonnegative; the force is the sum
    of the pad forces and the moment their moment about the club grip origin.
    """
    pads = (
        model.layout.positions_grip_frame(side)
        if positions_grip_frame is None
        else positions_grip_frame
    )
    axis = club.rotation[:, 0]
    axis_point = club.position_m + club.rotation @ model.layout.axis_offset_grip_frame(
        side
    )
    origin_s = model.cylinder.origin_axial_m(side)
    lo, hi = model.cylinder.axial_range_m
    force = np.zeros(3)
    moment = np.zeros(3)
    normal = np.zeros(len(pads))
    tangent = np.zeros(len(pads))
    for i, a in enumerate(pads):
        arm = hand.rotation @ a
        centre = hand.position_m + arm
        rel = centre - axis_point
        s = float(axis @ rel)
        if not lo <= origin_s + s <= hi:
            continue
        rho = rel - s * axis
        dist = math.sqrt(float(rho @ rho))
        if dist < 1e-9:
            continue
        n = rho / dist
        surface = axis_point + s * axis + model.cylinder.radius_m * n
        v_pad = hand.velocity_m_s + cross3(hand.omega_rad_s, arm)
        v_club = club.velocity_m_s + cross3(club.omega_rad_s, surface - club.position_m)
        plane = GroundPlane((float(n[0]), float(n[1]), float(n[2])), float(n @ surface))
        sample = sphere_ground_contact(
            centre, v_pad - v_club, model.layout.pad_radius_m, plane, model.law
        )
        if sample.penetration_m <= 0.0:
            continue
        on_pad = sample.normal_force_n + sample.friction_force_n
        spin = torsional_friction_moment(
            sample.normal_force_n,
            hand.omega_rad_s - club.omega_rad_s,
            plane,
            patch_radius_m=model.patch_radius_m,
            friction=model.law.static_friction,
            transition_rad_s=model.spin_transition_rad_s,
        )
        f_club = -on_pad
        force += f_club
        moment += cross3(sample.contact_point_m - club.position_m, f_club) - spin
        normal[i] = float(np.linalg.norm(sample.normal_force_n))
        tangent[i] = float(np.linalg.norm(sample.friction_force_n))
    return PadWrench(force, moment, normal, tangent)
