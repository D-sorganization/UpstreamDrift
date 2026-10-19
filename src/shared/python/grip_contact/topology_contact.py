"""Engine-agnostic grip contact law driven by ContactTopology (HRG-GS-G6, #13304).

Evaluates contact between anatomical hand sites defined by :class:`ContactTopology`
and the club grip cylinder via the shared G2 kernels (:func:`sphere_capsule` and
:func:`capsule_capsule`).

Conventions:
- Sites ride on the hand frame (mapped from the shared butt-origin grip frame into
  each hand's grip frame).
- The club grip capsule rides on the club frame.
- Hand-hand pairs are never physical forces in prescribed motion (reported only
  via hand-hand allocation, HRG-GS-G3).
- Wrenches on the club are expressed in world axes, with moments about the club
  grip-frame origin, matching :mod:`src.shared.python.grip_contact.pad_contact`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.grip_contact.bushing_law import BushingState
from src.shared.python.grip_contact.contact_topology import (
    GRIP,
    PHYSICAL_HAND_GRIP_MODES,
    ContactPair,
    ContactSite,
    ContactTopology,
    contact_topology,
)
from src.shared.python.grip_contact.interface import GripInterface
from src.shared.python.grip_contact.pad_contact import (
    FRICTION_TRANSITION_M_S,
    GripCylinder,
    PadContactModel,
    build_pad_model,
)
from src.shared.python.grip_contact.pair_contact import (
    ContactCapsule,
    ContactSphere,
    PairWrench,
    capsule_capsule,
    sphere_capsule,
)
from src.shared.python.motion_matching.contact_law import ContactParameters

if TYPE_CHECKING:
    from src.shared.python.motion_matching.grip_contract import GripSpec

__all__ = [
    "TopologyContactModel",
    "TopologySideWrench",
    "build_topology_model",
    "hand_x_origin",
    "map_site_to_hand_frame",
    "site_primitive_in_hand_frame",
    "topology_pair_wrench",
    "topology_side_wrench",
]

SIDES = ("L", "R")
HAND_ROLES = ("lead", "trail")


def _normalize_side(side: str) -> tuple[str, str]:
    """Return ``(hand_role, side_char)`` from 'L', 'R', 'lead', or 'trail'."""
    if not isinstance(side, str):
        raise TypeError(f"side must be a string, got {type(side).__name__}")
    if side in ("L", "lead"):
        return "lead", "L"
    if side in ("R", "trail"):
        return "trail", "R"
    raise ValueError(f"side must be one of ('L', 'R', 'lead', 'trail'), got {side!r}")


def _validate_bushing_state(state: BushingState, name: str) -> None:
    """DbC check for BushingState proper rotation and finite values."""
    if not isinstance(state, BushingState):
        raise TypeError(f"{name} must be a BushingState, got {type(state).__name__}")
    rot = np.asarray(state.rotation, dtype=float)
    if rot.shape != (3, 3) or not np.isfinite(rot).all():
        raise ValueError(f"{name}.rotation must be a finite 3x3 matrix")
    rtr = rot.T @ rot
    if (
        abs(rtr[0, 0] - 1.0) > 1e-5
        or abs(rtr[1, 1] - 1.0) > 1e-5
        or abs(rtr[2, 2] - 1.0) > 1e-5
        or abs(rtr[0, 1]) > 1e-5
        or abs(rtr[0, 2]) > 1e-5
        or abs(rtr[1, 2]) > 1e-5
    ):
        raise ValueError(f"{name}.rotation must be a proper rotation matrix")
    det = (
        rot[0, 0] * (rot[1, 1] * rot[2, 2] - rot[1, 2] * rot[2, 1])
        - rot[0, 1] * (rot[1, 0] * rot[2, 2] - rot[1, 2] * rot[2, 0])
        + rot[0, 2] * (rot[1, 0] * rot[2, 1] - rot[1, 1] * rot[2, 0])
    )
    if abs(det - 1.0) > 1e-5:
        raise ValueError(f"{name}.rotation must be a proper rotation matrix")
    for field_name in ("position_m", "velocity_m_s", "omega_rad_s"):
        val = np.asarray(getattr(state, field_name), dtype=float)
        if val.shape != (3,) or not np.isfinite(val).all():
            raise ValueError(f"{name}.{field_name} must be a finite 3-vector")


def hand_x_origin(topology: ContactTopology, hand: str) -> float:
    """Compute axial origin coordinate (m) for ``hand`` in ``topology``.

    Preconditions:
        ``hand`` in ``('lead', 'trail')``.
    Postconditions:
        Returns midpoint of palm proximal and palm distal pad sites along x.
    """
    require(hand in HAND_ROLES, f"hand must be one of {HAND_ROLES}, got {hand!r}")
    p_prox = topology.site(f"{hand}_palm_proximal").centre_m[0]
    p_dist = topology.site(f"{hand}_palm_distal").centre_m[0]
    return float(0.5 * (p_prox + p_dist))


def map_site_to_hand_frame(
    topology: ContactTopology,
    site: ContactSite | str,
    side: str,
    x_origin: float | None = None,
) -> ContactSite:
    """Map a :class:`ContactSite` into the hand grip frame of ``side``.

    Preconditions:
        ``topology`` is a :class:`ContactTopology`;
        ``site`` is a ContactSite or name of a site in ``topology``;
        ``site.hand`` matches ``side``;
        ``side`` in ``('L', 'R', 'lead', 'trail')``.
    Postconditions:
        Returns a new ContactSite with ``centre_m`` translated into the hand frame.
    """
    if not isinstance(topology, ContactTopology):
        raise TypeError(
            f"topology must be ContactTopology, got {type(topology).__name__}"
        )
    if isinstance(site, str):
        target_site = topology.site(site)
    elif isinstance(site, ContactSite):
        target_site = site
    else:
        raise TypeError(f"site must be ContactSite or str, got {type(site).__name__}")
    role, side_char = _normalize_side(side)
    if target_site.hand != role:
        raise ValueError(
            f"site {target_site.name!r} belongs to hand {target_site.hand!r}, "
            f"cannot map to side {side!r} ({role!r})"
        )
    if x_origin is None:
        x_origin = hand_x_origin(topology, role)

    # In PadLayout convention:
    # Right origin has shaft axis at +r along z; left origin has shaft axis at -r along z.
    axis_offset = np.array(
        [
            0.0,
            0.0,
            -topology.grip_radius_m if side_char == "L" else topology.grip_radius_m,
        ]
    )
    cx, cy, cz = target_site.centre_m
    p_hand = axis_offset + np.array([cx - x_origin, cy, cz])
    return replace(
        target_site,
        centre_m=(float(p_hand[0]), float(p_hand[1]), float(p_hand[2])),
    )


def site_primitive_in_hand_frame(
    topology: ContactTopology,
    site: ContactSite | str,
    side: str,
    x_origin: float | None = None,
) -> ContactSphere | ContactCapsule:
    """Convert ``site`` to a :class:`ContactSphere` or :class:`ContactCapsule` in hand frame."""
    mapped = map_site_to_hand_frame(topology, site, side, x_origin)
    if mapped.primitive == "sphere":
        return ContactSphere(
            center_m=np.asarray(mapped.centre_m, dtype=float),
            radius_m=mapped.radius_m,
        )
    if mapped.primitive == "capsule":
        p0, p1 = mapped.endpoints_m
        return ContactCapsule(
            p0_m=np.asarray(p0, dtype=float),
            p1_m=np.asarray(p1, dtype=float),
            radius_m=mapped.radius_m,
        )
    raise ValueError(f"unknown site primitive {mapped.primitive!r}")


@dataclass(frozen=True)
class TopologyContactModel:
    """ContactTopology plus contact-law parameters and grip cylinder geometry."""

    topology: ContactTopology
    law: ContactParameters
    cylinder: GripCylinder
    patch_radius_m: float
    spin_transition_rad_s: float = 1.0
    _grip_capsules: dict[str, ContactCapsule] = field(
        init=False, repr=False, compare=False
    )
    _physical_pairs: dict[
        str, tuple[tuple[ContactPair, ContactSphere | ContactCapsule], ...]
    ] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.topology, ContactTopology):
            raise TypeError(
                f"topology must be ContactTopology, got {type(self.topology).__name__}"
            )
        if not isinstance(self.law, ContactParameters):
            raise TypeError(
                f"law must be ContactParameters, got {type(self.law).__name__}"
            )
        if not isinstance(self.cylinder, GripCylinder):
            raise TypeError(
                f"cylinder must be GripCylinder, got {type(self.cylinder).__name__}"
            )
        if not math.isfinite(self.patch_radius_m) or self.patch_radius_m < 0.0:
            raise ValueError(
                f"patch_radius_m must be finite and >= 0, got {self.patch_radius_m}"
            )
        if (
            not math.isfinite(self.spin_transition_rad_s)
            or self.spin_transition_rad_s <= 0.0
        ):
            raise ValueError(
                f"spin_transition_rad_s must be positive and finite, "
                f"got {self.spin_transition_rad_s}"
            )

        capsules = {s: self._make_grip_capsule(s) for s in ("L", "R")}
        object.__setattr__(self, "_grip_capsules", capsules)

        pairs_dict: dict[
            str, tuple[tuple[ContactPair, ContactSphere | ContactCapsule], ...]
        ] = {}
        for role, side_char in (("lead", "L"), ("trail", "R")):
            p_list = []
            for p in self.topology.pairs:
                if p.is_hand_grip and self.topology.site(p.site_a).hand == role:
                    site = self.topology.site(p.site_a)
                    prim = site_primitive_in_hand_frame(self.topology, site, side_char)
                    p_list.append((p, prim))
            pairs_dict[side_char] = tuple(p_list)
        object.__setattr__(self, "_physical_pairs", pairs_dict)

    @classmethod
    def from_pad_model(
        cls, pads: PadContactModel, topology: ContactTopology
    ) -> TopologyContactModel:
        """Construct TopologyContactModel by reusing PadContactModel's parameters."""
        if not isinstance(pads, PadContactModel):
            raise TypeError(f"pads must be PadContactModel, got {type(pads).__name__}")
        return cls(
            topology=topology,
            law=pads.law,
            cylinder=pads.cylinder,
            patch_radius_m=pads.patch_radius_m,
            spin_transition_rad_s=pads.spin_transition_rad_s,
        )

    def physical_grip_pairs(self, side: str) -> tuple[ContactPair, ...]:
        """Ordered physical pad-grip pairs acting on ``side``."""
        _, side_char = _normalize_side(side)
        return tuple(p for p, _ in self._physical_pairs[side_char])

    def _make_grip_capsule(self, side: str) -> ContactCapsule:
        _, side_char = _normalize_side(side)
        axis_offset = np.array(
            [
                0.0,
                0.0,
                -self.cylinder.radius_m if side_char == "L" else self.cylinder.radius_m,
            ]
        )
        origin_s = self.cylinder.origin_axial_m(side_char)
        s_lo = self.cylinder.axial_range_m[0] - origin_s
        s_hi = self.cylinder.axial_range_m[1] - origin_s
        p0 = axis_offset + np.array([s_lo, 0.0, 0.0])
        p1 = axis_offset + np.array([s_hi, 0.0, 0.0])
        return ContactCapsule(p0_m=p0, p1_m=p1, radius_m=self.cylinder.radius_m)

    def grip_capsule(self, side: str) -> ContactCapsule:
        """Grip cylinder segment modeled as a ContactCapsule in ``side``'s club frame."""
        _, side_char = _normalize_side(side)
        return self._grip_capsules[side_char]


@dataclass(frozen=True)
class TopologySideWrench:
    """Wrench of one hand's physical contact sites on the club grip.

    Conventions follow :class:`src.shared.python.grip_contact.pad_contact.PadWrench`:
    - ``force_n``: 3-vector resultant force on the club in world axes.
    - ``moment_nm``: 3-vector resultant moment on the club about the club frame origin.
    - ``pair_wrenches``: per-pair :class:`PairWrench` values.
    - ``normal_force_n``: array of per-pair normal force magnitudes.
    - ``tangential_force_n``: array of per-pair tangential force magnitudes.
    - ``total_normal_force_n``: scalar total normal force of the side.
    """

    force_n: np.ndarray
    moment_nm: np.ndarray
    pair_wrenches: dict[str, PairWrench]
    normal_force_n: np.ndarray
    tangential_force_n: np.ndarray
    total_normal_force_n: float


def topology_pair_wrench(
    model: TopologyContactModel,
    pair: ContactPair | str,
    hand_frame: BushingState,
    club_frame: BushingState,
    mode: str = "contact",
) -> PairWrench:
    """Evaluate physical contact wrench for a single hand-grip pair.

    Preconditions:
        ``model`` is a :class:`TopologyContactModel`;
        ``pair`` is a ContactPair in ``model.topology``;
        ``hand_frame`` and ``club_frame`` are valid :class:`BushingState` instances;
        ``pair`` must be a physical hand-grip pair in ``mode`` (raises ``ValueError``
        if a hand-hand pair is passed).
    Postconditions:
        Returns :class:`PairWrench` satisfying action/reaction by construction.
    """
    if not isinstance(model, TopologyContactModel):
        raise TypeError(
            f"model must be TopologyContactModel, got {type(model).__name__}"
        )
    target_pair = model.topology.pair(pair) if isinstance(pair, str) else pair
    if not isinstance(target_pair, ContactPair):
        raise TypeError(
            f"pair must be ContactPair or str, got {type(target_pair).__name__}"
        )
    if target_pair.is_hand_hand or target_pair.site_b != GRIP:
        raise ValueError(
            f"pair {target_pair.name!r} is a hand-hand pair; hand-hand contact is "
            f"allocated, not a physical force in prescribed motion"
        )
    if mode not in target_pair.physical_modes:
        raise ValueError(
            f"pair {target_pair.name!r} is not physical in mode {mode!r} "
            f"(physical modes: {target_pair.physical_modes})"
        )
    _validate_bushing_state(hand_frame, "hand_frame")
    _validate_bushing_state(club_frame, "club_frame")

    site = model.topology.site(target_pair.site_a)
    side_char = "L" if site.hand == "lead" else "R"
    prim_a = site_primitive_in_hand_frame(model.topology, site, side_char)
    prim_b = model.grip_capsule(side_char)

    if isinstance(prim_a, ContactSphere):
        return sphere_capsule(
            prim_a,
            prim_b,
            hand_frame,
            club_frame,
            model.law,
            patch_radius_m=model.patch_radius_m,
            spin_transition_rad_s=model.spin_transition_rad_s,
        )
    if isinstance(prim_a, ContactCapsule):
        return capsule_capsule(
            prim_a,
            prim_b,
            hand_frame,
            club_frame,
            model.law,
            patch_radius_m=model.patch_radius_m,
            spin_transition_rad_s=model.spin_transition_rad_s,
        )
    raise ValueError(f"unsupported site primitive {prim_a!r}")


def topology_side_wrench(
    model: TopologyContactModel,
    side: str,
    hand_frame: BushingState,
    club_frame: BushingState,
    mode: str = "contact",
) -> TopologySideWrench:
    """Evaluate contact wrench of ``side``'s physical pads on the club grip.

    Preconditions:
        ``model`` is a :class:`TopologyContactModel`;
        ``side`` in ``('L', 'R', 'lead', 'trail')``;
        ``hand_frame`` and ``club_frame`` are valid :class:`BushingState` instances;
        ``mode`` is a valid execution mode (default ``'contact'``).
    Postconditions:
        Resultant force and moment on the club match the sum of individual pair
        wrenches on the club about the club frame origin.
    """
    if not isinstance(model, TopologyContactModel):
        raise TypeError(
            f"model must be TopologyContactModel, got {type(model).__name__}"
        )
    role, side_char = _normalize_side(side)
    _validate_bushing_state(hand_frame, "hand_frame")
    _validate_bushing_state(club_frame, "club_frame")

    pairs_with_prims = model._physical_pairs[side_char]
    grip_cap = model._grip_capsules[side_char]
    law = model.law
    patch_r = model.patch_radius_m
    spin_tr = model.spin_transition_rad_s

    force = np.zeros(3)
    moment = np.zeros(3)
    pair_wrenches: dict[str, PairWrench] = {}
    normal_forces: list[float] = []
    tangential_forces: list[float] = []

    for pair, prim_a in pairs_with_prims:
        if mode not in pair.physical_modes:
            continue
        if isinstance(prim_a, ContactSphere):
            pw = sphere_capsule(
                prim_a,
                grip_cap,
                hand_frame,
                club_frame,
                law,
                patch_radius_m=patch_r,
                spin_transition_rad_s=spin_tr,
            )
        elif isinstance(prim_a, ContactCapsule):
            pw = capsule_capsule(
                prim_a,
                grip_cap,
                hand_frame,
                club_frame,
                law,
                patch_radius_m=patch_r,
                spin_transition_rad_s=spin_tr,
            )
        else:
            raise ValueError(f"unsupported site primitive {prim_a!r}")

        pair_wrenches[pair.name] = pw
        force += pw.force_b_n
        moment += pw.moment_b_nm
        normal_forces.append(pw.normal_force_mag_n)
        tangential_forces.append(pw.tangential_force_mag_n)

    norm_arr = np.array(normal_forces, dtype=float)
    tang_arr = np.array(tangential_forces, dtype=float)
    total_norm = float(np.sum(norm_arr)) if len(norm_arr) > 0 else 0.0

    return TopologySideWrench(
        force_n=force,
        moment_nm=moment,
        pair_wrenches=pair_wrenches,
        normal_force_n=norm_arr,
        tangential_force_n=tang_arr,
        total_normal_force_n=total_norm,
    )


def build_topology_model(
    interface: GripInterface,
    topology: ContactTopology | None = None,
    squeeze_per_hand_n: float = 1000.0,
    spec: GripSpec | None = None,
    friction_transition_m_s: float = FRICTION_TRANSITION_M_S,
) -> TopologyContactModel:
    """Build a :class:`TopologyContactModel` matched to ``interface`` and ``topology``."""
    if not isinstance(interface, GripInterface):
        raise TypeError(
            f"interface must be GripInterface, got {type(interface).__name__}"
        )
    topo = topology if topology is not None else contact_topology(spec)
    pads = build_pad_model(
        interface,
        squeeze_per_hand_n,
        friction_transition_m_s=friction_transition_m_s,
    )
    return TopologyContactModel.from_pad_model(pads, topo)
