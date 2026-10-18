"""Engine-agnostic grip-style contact topology (HRG-GS-G1, #13299).

Defines the single canonical :class:`GripStyle` enum and constructs the
per-style :class:`ContactTopology` (:class:`ContactSite` and
:class:`ContactPair` tables) from a :class:`GripSpec` and hand anthropometry.

Grip frame convention (shared with :mod:`src.shared.python.grip_contact.pad_layout`):
``+x`` runs along the club grip axis from butt to clubhead; ``y`` and ``z``
span the cross-section perpendicular to the shaft so ``hypot(y, z)`` is the
radial distance from the shaft centreline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Final

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.grip_contact.pad_layout import GRIP_RADIUS_M
from src.shared.python.grip_contact.pair_contact import closest_points_segments

if TYPE_CHECKING:
    from src.shared.python.motion_matching.grip_contract import GripSpec

#: Sentinel partner identifier for a club-grip cylinder contact pair.
GRIP: Final[str] = "grip"


#: Default neutral-pose preload penetration (m), within the 0.5 mm tolerance.
DEFAULT_TOPOLOGY_PRELOAD_M: Final[float] = 2.5e-4

#: Reference adult hand length L0 (m) from distal wrist crease to middle fingertip.
DEFAULT_HAND_LENGTH_M: Final[float] = 0.19

#: Valid anthropometric range for hand length L (m).
HAND_LENGTH_RANGE_M: Final[tuple[float, float]] = (0.14, 0.24)

#: Base site radii (m) at the reference hand length L0 = 0.19 m.
BASE_LITTLE_FINGER_RADIUS_M: Final[float] = 0.0075
BASE_INDEX_FINGER_RADIUS_M: Final[float] = 0.0080
BASE_THUMB_RADIUS_M: Final[float] = 0.0095
BASE_HEEL_PAD_RADIUS_M: Final[float] = 0.0120
BASE_PALM_PAD_RADIUS_M: Final[float] = 0.0080
BASE_FINGER_PAD_RADIUS_M: Final[float] = 0.0080

#: Allowed primitive shapes for a contact site and for a contact pair.
SITE_PRIMITIVES: Final[tuple[str, ...]] = ("sphere", "capsule")
PAIR_PRIMITIVES: Final[tuple[str, ...]] = ("sphere", "capsule", "cylinder")
HAND_ROLES: Final[tuple[str, ...]] = ("lead", "trail")

#: Execution modes where hand-grip and hand-hand pairs are physical vs allocated.
PHYSICAL_HAND_GRIP_MODES: Final[tuple[str, ...]] = (
    "contact",
    "contact_v1",
    "separated_v2",
)
ALLOCATED_HAND_GRIP_MODES: Final[tuple[str, ...]] = (
    "weld",
    "closure",
    "bushing",
)
PHYSICAL_HAND_HAND_MODES: Final[tuple[str, ...]] = ("separated_v2",)
ALLOCATED_HAND_HAND_MODES: Final[tuple[str, ...]] = (
    "weld",
    "closure",
    "bushing",
    "contact",
    "contact_v1",
)


class GripStyle(str, Enum):
    """Classical two-handed golf grip inter-hand linking styles."""

    OVERLAP = "overlap"
    INTERLOCK = "interlock"
    TEN_FINGER = "ten_finger"


def _validate_vec3(
    vec: tuple[float, float, float], label: str
) -> tuple[float, float, float]:
    if not isinstance(vec, tuple | list) or len(vec) != 3:
        raise ValueError(f"{label} must be a 3-tuple of finite floats")
    coords = tuple(float(c) for c in vec)
    if not all(math.isfinite(c) for c in coords):
        raise ValueError(f"{label} must contain only finite floats")
    return (coords[0], coords[1], coords[2])


@dataclass(frozen=True)
class HandAnatomy:
    """Hand anthropometry scaling contact site radii linearly with hand length L.

    Preconditions:
        ``hand_length_m`` is a finite real number in ``[0.14, 0.24]`` m.
    Postconditions:
        Little finger, index, thumb, and heel-pad radii equal their reference
        values scaled by ``hand_length_m / 0.19``.
    """

    hand_length_m: float = DEFAULT_HAND_LENGTH_M

    def __post_init__(self) -> None:
        if isinstance(self.hand_length_m, bool) or not isinstance(
            self.hand_length_m, int | float
        ):
            raise TypeError("hand_length_m must be a finite real number")
        lo, hi = HAND_LENGTH_RANGE_M
        require(
            math.isfinite(self.hand_length_m) and lo <= self.hand_length_m <= hi,
            f"hand_length_m must be in [{lo}, {hi}] m, got {self.hand_length_m}",
        )
        object.__setattr__(self, "hand_length_m", float(self.hand_length_m))

    @property
    def scale(self) -> float:
        """Dimensionless anatomical scale factor ``L / 0.19``."""
        return self.hand_length_m / DEFAULT_HAND_LENGTH_M

    @property
    def little_finger_radius_m(self) -> float:
        """Trail little-finger capsule radius ``0.0075 * L / 0.19`` (m)."""
        return BASE_LITTLE_FINGER_RADIUS_M * self.scale

    @property
    def index_finger_radius_m(self) -> float:
        """Index/middle finger capsule radius ``0.008 * L / 0.19`` (m)."""
        return BASE_INDEX_FINGER_RADIUS_M * self.scale

    @property
    def thumb_radius_m(self) -> float:
        """Lead thumb pad capsule radius ``0.0095 * L / 0.19`` (m)."""
        return BASE_THUMB_RADIUS_M * self.scale

    @property
    def heel_pad_radius_m(self) -> float:
        """Lead heel-pad sphere radius ``0.012 * L / 0.19`` (m)."""
        return BASE_HEEL_PAD_RADIUS_M * self.scale

    @property
    def palm_pad_radius_m(self) -> float:
        """Palm pad sphere radius ``0.008 * L / 0.19`` (m)."""
        return BASE_PALM_PAD_RADIUS_M * self.scale

    @property
    def finger_pad_radius_m(self) -> float:
        """Finger pad sphere radius ``0.008 * L / 0.19`` (m)."""
        return BASE_FINGER_PAD_RADIUS_M * self.scale


@dataclass(frozen=True)
class ContactSite:
    """Single anatomical contact primitive expressed in the shared grip frame.

    Preconditions:
        ``hand`` in ``("lead", "trail")``; ``primitive`` in
        ``("sphere", "capsule")``; ``radius_m > 0``; ``axis_m`` has unit norm.
    """

    name: str
    hand: str
    segment: str
    primitive: str
    centre_m: tuple[float, float, float]
    radius_m: float
    axis_m: tuple[float, float, float] = (1.0, 0.0, 0.0)
    half_length_m: float = 0.0

    def __post_init__(self) -> None:
        require(
            isinstance(self.name, str) and bool(self.name.strip()),
            "name must be a non-empty string",
        )
        require(
            self.hand in HAND_ROLES,
            f"hand must be one of {HAND_ROLES}, got {self.hand!r}",
        )
        require(
            isinstance(self.segment, str) and bool(self.segment.strip()),
            "segment must be a non-empty string",
        )
        require(
            self.primitive in SITE_PRIMITIVES,
            f"primitive must be one of {SITE_PRIMITIVES}, got {self.primitive!r}",
        )
        centre = _validate_vec3(self.centre_m, "centre_m")
        axis = _validate_vec3(self.axis_m, "axis_m")
        norm = math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)
        require(norm > 1e-12, "axis_m must be non-zero")
        unit_axis = (axis[0] / norm, axis[1] / norm, axis[2] / norm)
        require(
            not isinstance(self.radius_m, bool)
            and isinstance(self.radius_m, int | float)
            and math.isfinite(self.radius_m)
            and self.radius_m > 0.0,
            f"radius_m must be a positive finite float, got {self.radius_m!r}",
        )
        require(
            not isinstance(self.half_length_m, bool)
            and isinstance(self.half_length_m, int | float)
            and math.isfinite(self.half_length_m)
            and self.half_length_m >= 0.0,
            f"half_length_m must be a non-negative finite float, got {self.half_length_m!r}",
        )
        if self.primitive == "sphere":
            require(
                self.half_length_m == 0.0,
                "sphere ContactSite must have half_length_m == 0.0",
            )
        else:
            require(
                self.half_length_m > 0.0,
                "capsule ContactSite must have half_length_m > 0.0",
            )
        object.__setattr__(self, "centre_m", centre)
        object.__setattr__(self, "axis_m", unit_axis)
        object.__setattr__(self, "radius_m", float(self.radius_m))
        object.__setattr__(self, "half_length_m", float(self.half_length_m))

    @property
    def centre(self) -> tuple[float, float, float]:
        """Alias for :attr:`centre_m`."""
        return self.centre_m

    @property
    def axis(self) -> tuple[float, float, float]:
        """Alias for :attr:`axis_m`."""
        return self.axis_m

    @property
    def radius(self) -> float:
        """Alias for :attr:`radius_m`."""
        return self.radius_m

    @property
    def endpoints_m(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Capsule medial-axis segment endpoints ``(p0, p1)`` in the grip frame."""
        cx, cy, cz = self.centre_m
        ax, ay, az = self.axis_m
        hl = self.half_length_m
        return (
            (cx - hl * ax, cy - hl * ay, cz - hl * az),
            (cx + hl * ax, cy + hl * ay, cz + hl * az),
        )


@dataclass(frozen=True)
class ContactPair:
    """Interacting contact pair between two hand sites or a hand site and ``GRIP``.

    Preconditions:
        ``site_a != site_b``; ``primitive_pair`` is a 2-tuple of valid primitive
        names; ``physical_modes`` and ``allocated_modes`` are disjoint.
    """

    name: str
    site_a: str
    site_b: str
    primitive_pair: tuple[str, str]
    physical_modes: tuple[str, ...] = PHYSICAL_HAND_GRIP_MODES
    allocated_modes: tuple[str, ...] = ALLOCATED_HAND_GRIP_MODES

    def __post_init__(self) -> None:
        for field_name, val in (
            ("name", self.name),
            ("site_a", self.site_a),
            ("site_b", self.site_b),
        ):
            require(
                isinstance(val, str) and bool(val.strip()),
                f"{field_name} must be a non-empty string",
            )
        require(self.site_a != self.site_b, "site_a and site_b must be distinct")
        require(
            isinstance(self.primitive_pair, tuple)
            and len(self.primitive_pair) == 2
            and self.primitive_pair[0] in SITE_PRIMITIVES
            and self.primitive_pair[1] in PAIR_PRIMITIVES,
            f"invalid primitive_pair {self.primitive_pair!r}",
        )
        if self.site_b == GRIP:
            require(
                self.primitive_pair[1] == "cylinder",
                "GRIP partner requires primitive_pair[1] == 'cylinder'",
            )
        overlap = set(self.physical_modes) & set(self.allocated_modes)
        require(
            not overlap,
            f"physical_modes and allocated_modes must be disjoint, shared: {sorted(overlap)}",
        )

    @property
    def is_hand_hand(self) -> bool:
        """True when this pair links two hand sites rather than the club grip."""
        return self.site_b != GRIP

    @property
    def is_hand_grip(self) -> bool:
        """True when this pair links a hand site to the club grip cylinder."""
        return self.site_b == GRIP

    @property
    def allocated_label(self) -> str:
        """Canonical non-physical allocation channel label (HRG-GS-G3)."""
        return f"external:{self.name}_allocated"


@dataclass(frozen=True)
class ContactTopology:
    """Complete grip contact site and pair table for a :class:`GripSpec`."""

    style: GripStyle
    trigger_finger: bool
    grip_radius_m: float
    hand_length_m: float
    sites: tuple[ContactSite, ...]
    pairs: tuple[ContactPair, ...]

    def site(self, name: str) -> ContactSite:
        """Return the :class:`ContactSite` named ``name`` or raise ``KeyError``."""
        for s in self.sites:
            if s.name == name:
                return s
        raise KeyError(f"unknown contact site {name!r}")

    def pair(self, name: str) -> ContactPair:
        """Return the :class:`ContactPair` named ``name`` or raise ``KeyError``."""
        for p in self.pairs:
            if p.name == name:
                return p
        raise KeyError(f"unknown contact pair {name!r}")

    def sites_for_hand(self, hand: str) -> tuple[ContactSite, ...]:
        """All contact sites belonging to ``hand`` (``'lead'`` or ``'trail'``)."""
        require(hand in HAND_ROLES, f"hand must be one of {HAND_ROLES}, got {hand!r}")
        return tuple(s for s in self.sites if s.hand == hand)

    def hand_grip_pairs(self) -> tuple[ContactPair, ...]:
        """All pairs acting between a hand site and ``GRIP``."""
        return tuple(p for p in self.pairs if p.is_hand_grip)

    def hand_hand_pairs(self) -> tuple[ContactPair, ...]:
        """All pairs acting between two hand sites."""
        return tuple(p for p in self.pairs if p.is_hand_hand)

    def site_names(self) -> tuple[str, ...]:
        """Ordered tuple of all site names in this topology."""
        return tuple(s.name for s in self.sites)

    def pair_names(self) -> tuple[str, ...]:
        """Ordered tuple of all pair names in this topology."""
        return tuple(p.name for p in self.pairs)


def _cyl_point(x: float, rho: float, theta_rad: float) -> tuple[float, float, float]:
    return (
        float(x),
        float(rho * math.cos(theta_rad)),
        float(rho * math.sin(theta_rad)),
    )


def site_grip_surface_gap_m(
    site: ContactSite, grip_radius_m: float = GRIP_RADIUS_M
) -> float:
    """Signed distance (m) between ``site``'s outer surface and the grip cylinder.

    Negative values indicate preload penetration; zero indicates exact tangency.
    """
    radial_centre = math.hypot(site.centre_m[1], site.centre_m[2])
    return (radial_centre - float(grip_radius_m)) - site.radius_m


def _segment_distance_m(site_a: ContactSite, site_b: ContactSite) -> float:
    """Minimum Euclidean distance (m) between the medial axes of two sites."""
    a0, a1 = (np.asarray(p, dtype=float) for p in site_a.endpoints_m)
    b0, b1 = (np.asarray(p, dtype=float) for p in site_b.endpoints_m)
    pa, pb = closest_points_segments(a0, a1, b0, b1)
    return float(np.linalg.norm(pa - pb))


def pair_surface_gap_m(topology: ContactTopology, pair: ContactPair) -> float:
    """Signed surface-to-surface gap (m) for ``pair`` in ``topology``.

    Negative values indicate preload penetration; zero indicates exact contact.
    """
    site_a = topology.site(pair.site_a)
    if pair.site_b == GRIP:
        return site_grip_surface_gap_m(site_a, topology.grip_radius_m)
    site_b = topology.site(pair.site_b)
    dist = _segment_distance_m(site_a, site_b)
    return dist - (site_a.radius_m + site_b.radius_m)


def _coerce_anatomy(anatomy: HandAnatomy | float | None) -> HandAnatomy:
    if anatomy is None:
        return HandAnatomy()
    if isinstance(anatomy, HandAnatomy):
        return anatomy
    if isinstance(anatomy, bool) or not isinstance(anatomy, int | float):
        raise TypeError(
            "anatomy must be a HandAnatomy, a hand length in metres, or None"
        )
    return HandAnatomy(hand_length_m=float(anatomy))


def _four_hand_pad_sites(
    hand: str,
    x_origin: float,
    alpha_rad: float,
    anatomy: HandAnatomy,
    grip_radius_m: float,
    preload_m: float,
) -> tuple[ContactSite, ...]:
    r_palm = anatomy.palm_pad_radius_m
    r_finger = anatomy.finger_pad_radius_m
    rho_palm = grip_radius_m + r_palm - preload_m
    rho_finger = grip_radius_m + r_finger - preload_m
    return (
        ContactSite(
            name=f"{hand}_palm_proximal",
            hand=hand,
            segment="palm_proximal",
            primitive="sphere",
            centre_m=_cyl_point(x_origin - 0.015, rho_palm, alpha_rad + math.pi),
            radius_m=r_palm,
        ),
        ContactSite(
            name=f"{hand}_palm_distal",
            hand=hand,
            segment="palm_distal",
            primitive="sphere",
            centre_m=_cyl_point(x_origin + 0.015, rho_palm, alpha_rad + math.pi),
            radius_m=r_palm,
        ),
        ContactSite(
            name=f"{hand}_finger_proximal",
            hand=hand,
            segment="finger_proximal",
            primitive="sphere",
            centre_m=_cyl_point(x_origin - 0.012, rho_finger, alpha_rad),
            radius_m=r_finger,
        ),
        ContactSite(
            name=f"{hand}_finger_distal",
            hand=hand,
            segment="finger_distal",
            primitive="sphere",
            centre_m=_cyl_point(x_origin + 0.012, rho_finger, alpha_rad),
            radius_m=r_finger,
        ),
    )


def _overlap_finger_sites(
    x_lead: float,
    alpha_lead: float,
    anatomy: HandAnatomy,
    grip_radius_m: float,
    preload_m: float,
) -> tuple[ContactSite, ContactSite, ContactSite]:
    """Return sites for overlap grip style."""
    r_idx = anatomy.index_finger_radius_m
    r_mid = anatomy.index_finger_radius_m
    r_lit = anatomy.little_finger_radius_m
    hl = 0.008 * anatomy.scale
    rho_idx_grip = grip_radius_m + r_idx - preload_m
    rho_mid_grip = grip_radius_m + r_mid - preload_m
    x_mid = x_lead + 0.022
    x_idx = x_lead + 0.036
    rho_lit = rho_idx_grip + (r_idx + r_lit - preload_m)
    lead_middle = ContactSite(
        name="lead_middle",
        hand="lead",
        segment="middle_finger",
        primitive="capsule",
        centre_m=_cyl_point(x_mid, rho_mid_grip, alpha_lead - 0.15),
        radius_m=r_mid,
        half_length_m=hl,
    )
    lead_index = ContactSite(
        name="lead_index",
        hand="lead",
        segment="index_finger",
        primitive="capsule",
        centre_m=_cyl_point(x_idx, rho_idx_grip, alpha_lead),
        radius_m=r_idx,
        half_length_m=hl,
    )
    trail_little = ContactSite(
        name="trail_little_finger",
        hand="trail",
        segment="little_finger",
        primitive="capsule",
        centre_m=_cyl_point(x_idx, rho_lit, alpha_lead),
        radius_m=r_lit,
        half_length_m=hl,
    )
    return lead_middle, lead_index, trail_little


def _interlock_finger_sites(
    x_lead: float,
    alpha_lead: float,
    anatomy: HandAnatomy,
    grip_radius_m: float,
    preload_m: float,
) -> tuple[ContactSite, ContactSite, ContactSite]:
    """Return sites for interlock grip style."""
    r_idx = anatomy.index_finger_radius_m
    r_mid = anatomy.index_finger_radius_m
    r_lit = anatomy.little_finger_radius_m
    hl = 0.008 * anatomy.scale
    rho_idx_grip = grip_radius_m + r_idx - preload_m
    rho_mid_grip = grip_radius_m + r_mid - preload_m
    rho_lit_grip = grip_radius_m + r_lit - preload_m
    x_slot = x_lead + 0.032
    target_mid_lit = r_mid + r_lit - preload_m
    cos_dtheta = (rho_mid_grip**2 + rho_lit_grip**2 - target_mid_lit**2) / (
        2.0 * rho_mid_grip * rho_lit_grip
    )
    dtheta = math.acos(max(-1.0, min(1.0, cos_dtheta)))
    rho_idx_wrap = rho_lit_grip + (r_lit + r_idx - preload_m)
    lead_middle = ContactSite(
        name="lead_middle",
        hand="lead",
        segment="middle_finger",
        primitive="capsule",
        centre_m=_cyl_point(x_slot, rho_mid_grip, alpha_lead - dtheta),
        radius_m=r_mid,
        half_length_m=hl,
    )
    trail_little = ContactSite(
        name="trail_little_finger",
        hand="trail",
        segment="little_finger",
        primitive="capsule",
        centre_m=_cyl_point(x_slot, rho_lit_grip, alpha_lead),
        radius_m=r_lit,
        half_length_m=hl,
    )
    lead_index = ContactSite(
        name="lead_index",
        hand="lead",
        segment="index_finger",
        primitive="capsule",
        centre_m=_cyl_point(x_slot, rho_idx_wrap, alpha_lead),
        radius_m=r_idx,
        half_length_m=hl,
    )
    return lead_middle, lead_index, trail_little


def _ten_finger_sites(
    x_lead: float,
    alpha_lead: float,
    anatomy: HandAnatomy,
    grip_radius_m: float,
    preload_m: float,
) -> tuple[ContactSite, ContactSite, ContactSite]:
    """Return sites for ten-finger grip style."""
    r_idx = anatomy.index_finger_radius_m
    r_mid = anatomy.index_finger_radius_m
    r_lit = anatomy.little_finger_radius_m
    hl = 0.008 * anatomy.scale
    rho_idx_grip = grip_radius_m + r_idx - preload_m
    rho_mid_grip = grip_radius_m + r_mid - preload_m
    rho_lit_grip = grip_radius_m + r_lit - preload_m
    x_mid = x_lead + 0.022
    x_idx = x_lead + 0.036
    target_idx_lit = r_idx + r_lit - preload_m
    radial_diff = rho_idx_grip - rho_lit_grip
    dx_abut = math.sqrt(max(0.0, target_idx_lit**2 - radial_diff**2))
    x_lit = x_idx + dx_abut
    lead_middle = ContactSite(
        name="lead_middle",
        hand="lead",
        segment="middle_finger",
        primitive="capsule",
        centre_m=_cyl_point(x_mid, rho_mid_grip, alpha_lead - 0.15),
        radius_m=r_mid,
        half_length_m=hl,
    )
    lead_index = ContactSite(
        name="lead_index",
        hand="lead",
        segment="index_finger",
        primitive="capsule",
        centre_m=_cyl_point(x_idx, rho_idx_grip, alpha_lead),
        radius_m=r_idx,
        half_length_m=hl,
    )
    trail_little = ContactSite(
        name="trail_little_finger",
        hand="trail",
        segment="little_finger",
        primitive="capsule",
        centre_m=_cyl_point(x_lit, rho_lit_grip, alpha_lead),
        radius_m=r_lit,
        half_length_m=hl,
    )
    return lead_middle, lead_index, trail_little


def _interhand_finger_sites(
    style: GripStyle,
    x_lead: float,
    alpha_lead: float,
    anatomy: HandAnatomy,
    grip_radius_m: float,
    preload_m: float,
) -> tuple[ContactSite, ContactSite, ContactSite]:
    """Return ``(lead_middle, lead_index, trail_little_finger)`` for ``style``."""
    if style is GripStyle.OVERLAP:
        return _overlap_finger_sites(
            x_lead, alpha_lead, anatomy, grip_radius_m, preload_m
        )
    if style is GripStyle.INTERLOCK:
        return _interlock_finger_sites(
            x_lead, alpha_lead, anatomy, grip_radius_m, preload_m
        )
    return _ten_finger_sites(x_lead, alpha_lead, anatomy, grip_radius_m, preload_m)


def _build_pairs(style: GripStyle, trigger_finger: bool) -> tuple[ContactPair, ...]:
    pad_sites = [
        f"{hand}_{segment}_{station}"
        for hand in HAND_ROLES
        for segment in ("palm", "finger")
        for station in ("proximal", "distal")
    ]
    pairs: list[ContactPair] = [
        ContactPair(
            name=f"{site}_grip",
            site_a=site,
            site_b=GRIP,
            primitive_pair=("sphere", "cylinder"),
        )
        for site in pad_sites
    ]
    pairs += [
        ContactPair(
            name="lead_heel_pad_butt",
            site_a="lead_heel_pad",
            site_b=GRIP,
            primitive_pair=("sphere", "cylinder"),
        ),
        ContactPair(
            name="trail_palm_lead_thumb",
            site_a="trail_palm_pocket",
            site_b="lead_thumb",
            primitive_pair=("sphere", "capsule"),
            physical_modes=PHYSICAL_HAND_HAND_MODES,
            allocated_modes=ALLOCATED_HAND_HAND_MODES,
        ),
    ]
    if style is GripStyle.OVERLAP:
        pairs.append(
            ContactPair(
                name="overlap_little_finger_lead_index",
                site_a="trail_little_finger",
                site_b="lead_index",
                primitive_pair=("capsule", "capsule"),
                physical_modes=PHYSICAL_HAND_HAND_MODES,
                allocated_modes=ALLOCATED_HAND_HAND_MODES,
            )
        )
    elif style is GripStyle.INTERLOCK:
        pairs.extend(
            (
                ContactPair(
                    name="interlock_little_finger_lead_middle",
                    site_a="trail_little_finger",
                    site_b="lead_middle",
                    primitive_pair=("capsule", "capsule"),
                    physical_modes=PHYSICAL_HAND_HAND_MODES,
                    allocated_modes=ALLOCATED_HAND_HAND_MODES,
                ),
                ContactPair(
                    name="interlock_lead_index_wrap",
                    site_a="lead_index",
                    site_b="trail_little_finger",
                    primitive_pair=("capsule", "capsule"),
                    physical_modes=PHYSICAL_HAND_HAND_MODES,
                    allocated_modes=ALLOCATED_HAND_HAND_MODES,
                ),
            )
        )
    else:
        pairs.append(
            ContactPair(
                name="ten_finger_little_finger_grip",
                site_a="trail_little_finger",
                site_b=GRIP,
                primitive_pair=("capsule", "cylinder"),
            )
        )
    if trigger_finger:
        pairs.append(
            ContactPair(
                name="trail_index_trigger_grip",
                site_a="trail_index_trigger",
                site_b=GRIP,
                primitive_pair=("capsule", "cylinder"),
            )
        )
    return tuple(pairs)


def _lead_heel_site(
    x_lead: float,
    alpha_lead: float,
    hand_anatomy: HandAnatomy,
    r_grip: float,
    preload: float,
) -> ContactSite:
    """Return the lead heel pad ContactSite."""
    r_heel = hand_anatomy.heel_pad_radius_m
    return ContactSite(
        name="lead_heel_pad",
        hand="lead",
        segment="heel_pad",
        primitive="sphere",
        centre_m=_cyl_point(
            x_lead - 0.020, r_grip + r_heel - preload, alpha_lead + math.pi
        ),
        radius_m=r_heel,
    )


def _thumb_palm_sites(
    x_lead: float,
    alpha_lead: float,
    thumb_offset_deg: float,
    hand_anatomy: HandAnatomy,
    r_grip: float,
    preload: float,
) -> tuple[ContactSite, ContactSite]:
    """Return (lead_thumb, trail_palm_pocket) ContactSites."""
    r_thumb = hand_anatomy.thumb_radius_m
    r_pocket = hand_anatomy.palm_pad_radius_m
    theta_thumb = alpha_lead + math.radians(thumb_offset_deg)
    rho_thumb = r_grip + r_thumb - preload
    rho_pocket = rho_thumb + (r_thumb + r_pocket - preload)
    x_thumb = x_lead + 0.045
    lead_thumb = ContactSite(
        name="lead_thumb",
        hand="lead",
        segment="thumb",
        primitive="capsule",
        centre_m=_cyl_point(x_thumb, rho_thumb, theta_thumb),
        radius_m=r_thumb,
        half_length_m=0.010 * hand_anatomy.scale,
    )
    trail_palm_pocket = ContactSite(
        name="trail_palm_pocket",
        hand="trail",
        segment="palm_thenar",
        primitive="sphere",
        centre_m=_cyl_point(x_thumb, rho_pocket, theta_thumb),
        radius_m=r_pocket,
    )
    return lead_thumb, trail_palm_pocket


def contact_topology(
    spec: GripSpec | None = None,
    anatomy: HandAnatomy | float | None = None,
    *,
    grip_radius_m: float = GRIP_RADIUS_M,
) -> ContactTopology:
    """Build the engine-agnostic :class:`ContactTopology` for ``spec`` and ``anatomy``.

    Preconditions:
        ``spec`` is a :class:`GripSpec` (or ``None`` for a default spec);
        ``anatomy`` is a :class:`HandAnatomy`, a hand length ``L`` in
        ``[0.14, 0.24]`` m, or ``None`` (default ``0.19`` m);
        ``grip_radius_m > 0``.
    Postconditions:
        Every returned site lies on or within one pad radius of the grip
        surface or its partner site at the neutral pose (surface gap/penetration
        magnitude ``<= 0.5`` mm), and the pair table matches ``spec.style`` and
        ``spec.trigger_finger``.
    """
    from src.shared.python.motion_matching.grip_contract import GripSpec

    if spec is None:
        spec = GripSpec()
    elif not isinstance(spec, GripSpec):
        raise TypeError(f"spec must be a GripSpec, got {type(spec).__name__}")
    hand_anatomy = _coerce_anatomy(anatomy)
    require(
        not isinstance(grip_radius_m, bool)
        and isinstance(grip_radius_m, int | float)
        and math.isfinite(grip_radius_m)
        and grip_radius_m > 0.0,
        f"grip_radius_m must be a positive finite float, got {grip_radius_m!r}",
    )
    r_grip = float(grip_radius_m)
    preload = DEFAULT_TOPOLOGY_PRELOAD_M

    x_lead = float(spec.lead_below_butt_m)
    extra_ten_finger = 0.012 if spec.style is GripStyle.TEN_FINGER else 0.0
    x_trail = x_lead + float(spec.hand_spacing_m) + extra_ten_finger
    alpha_lead = math.radians(spec.lead.alpha_deg)
    alpha_trail = math.radians(spec.trail.alpha_deg)

    lead_pads = _four_hand_pad_sites(
        "lead", x_lead, alpha_lead, hand_anatomy, r_grip, preload
    )
    trail_pads = _four_hand_pad_sites(
        "trail", x_trail, alpha_trail, hand_anatomy, r_grip, preload
    )
    lead_heel = _lead_heel_site(x_lead, alpha_lead, hand_anatomy, r_grip, preload)
    lead_thumb, trail_palm_pocket = _thumb_palm_sites(
        x_lead,
        alpha_lead,
        spec.lead_thumb_offset_deg,
        hand_anatomy,
        r_grip,
        preload,
    )
    lead_middle, lead_index, trail_little = _interhand_finger_sites(
        spec.style, x_lead, alpha_lead, hand_anatomy, r_grip, preload
    )

    sites: list[ContactSite] = [
        *lead_pads,
        *trail_pads,
        lead_heel,
        lead_thumb,
        trail_palm_pocket,
        lead_middle,
        lead_index,
        trail_little,
    ]
    if spec.trigger_finger:
        r_trig = hand_anatomy.index_finger_radius_m
        sites.append(
            ContactSite(
                name="trail_index_trigger",
                hand="trail",
                segment="index_trigger",
                primitive="capsule",
                centre_m=_cyl_point(
                    x_trail + 0.032, r_grip + r_trig - preload, alpha_trail
                ),
                radius_m=r_trig,
                half_length_m=0.008 * hand_anatomy.scale,
            )
        )

    pairs = _build_pairs(spec.style, spec.trigger_finger)
    return ContactTopology(
        style=spec.style,
        trigger_finger=spec.trigger_finger,
        grip_radius_m=r_grip,
        hand_length_m=hand_anatomy.hand_length_m,
        sites=tuple(sites),
        pairs=pairs,
    )
