"""Engine-agnostic grip contact law driven by ContactTopology (HRG-GS-G6, #13304; HRG-GS-G11, #13309).

Evaluates contact between anatomical hand sites defined by :class:`ContactTopology`
and the club grip cylinder (or between separated lead and trail hands in
``mode="separated_v2"``) via the shared G2 kernels (:func:`sphere_capsule` and
:func:`capsule_capsule`).

Conventions:
- Sites ride on the hand frame (mapped from the shared butt-origin grip frame into
  each hand's grip frame).
- The club grip capsule rides on the club frame.
- In default ``mode="contact"``, hand-hand pairs are never physical forces in
  prescribed motion (reported only via hand-hand allocation, HRG-GS-G3).
- In opt-in ``mode="separated_v2"`` (HRG-GS-G11, #13309), ``LHand`` and ``RHand``
  are separate rigid bodies retained on the club by hand-on-club retention, and
  hand-hand pairs (``trail_palm_lead_thumb``, ``overlap_little_finger_lead_index``,
  ``interlock_little_finger_lead_middle``, ``interlock_lead_index_wrap``) are
  evaluated with the shared :mod:`pair_contact` kernels and applied as equal and
  opposite wrenches on the lead and trail hands.
- Identifiability note: finger contact is not observable from motion-capture
  markers because standard capture marker sets carry no finger markers. Only the
  net club wrench is dynamically identified; individual pair forces depend on the
  hand-on-club retention and contact parameters.
- Wrenches on the club are expressed in world axes, with moments about the club
  grip-frame origin, matching :mod:`src.shared.python.grip_contact.pad_contact`.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.core.handedness import (
    DEFAULT_HANDEDNESS,
    Handedness,
    lead_side,
    trail_side,
)
from src.shared.python.grip_contact.batched_pair_contact import (
    PairRow,
    SideBatch,
    side_rows_wrench,
)
from src.shared.python.grip_contact.bushing_law import (
    BushingState,
    cross3,
    rotation_errors,
)
from src.shared.python.grip_contact.contact_topology import (
    GRIP,
    PHYSICAL_HAND_GRIP_MODES,
    PHYSICAL_HAND_HAND_MODES,
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
    capsule_sphere,
    closest_point_segment,
    sphere_capsule,
)
from src.shared.python.motion_matching.contact_law import ContactParameters

if TYPE_CHECKING:
    from src.shared.python.motion_matching.grip_contract import GripSpec

#: Identifiability note required by HRG-GS-G11 (#13309).
HAND_HAND_IDENTIFIABILITY_NOTE = (
    "Identifiability limit (HRG-GS-G11, #13309): finger contact is not observable "
    "from motion-capture markers because standard capture marker sets carry no "
    "finger markers. Only the net club wrench (net force and net moment about the "
    "club centre of mass) is dynamically identified by club kinematics; individual "
    "hand-on-club and hand-hand pair forces depend on the hand-on-club retention "
    "and contact parameters."
)

#: Chatter status reported when a contact pair's normal force never varies over a run.
CONSTANT_PRELOAD_CHATTER_STATUS = "constant (preload only)"

#: Minimum normal-force peak-to-peak range (N) required for a pair to be considered dynamically varying.
DEFAULT_FORCE_VARIATION_TOL_N = 1.0e-6

__all__ = [
    "CONSTANT_PRELOAD_CHATTER_STATUS",
    "DEFAULT_FORCE_VARIATION_TOL_N",
    "HAND_HAND_IDENTIFIABILITY_NOTE",
    "PairChatterSummary",
    "PairTraceRecorder",
    "TopologyContactModel",
    "TopologyHandHandWrench",
    "TopologySideWrench",
    "aggregate_side_wrench",
    "build_topology_model",
    "contact_transition_fraction",
    "evaluate_pair_chatter",
    "hand_x_origin",
    "map_site_to_hand_frame",
    "site_primitive_in_hand_frame",
    "topology_hand_hand_pair_wrench",
    "topology_hand_hand_wrenches",
    "topology_pair_wrench",
    "topology_side_wrench",
    "topology_side_wrench_reference",
]


SIDES = ("L", "R")
HAND_ROLES = ("lead", "trail")


def _normalize_side(
    side: str, handedness: Handedness = DEFAULT_HANDEDNESS
) -> tuple[str, str]:
    """Return ``(hand_role, side_char)`` from 'L', 'R', 'lead', or 'trail'."""
    if not isinstance(side, str):
        raise TypeError(f"side must be a string, got {type(side).__name__}")
    if side == "lead":
        return "lead", lead_side(handedness)
    if side == "trail":
        return "trail", trail_side(handedness)
    if side == "L":
        return ("lead" if lead_side(handedness) == "L" else "trail"), "L"
    if side == "R":
        return ("lead" if lead_side(handedness) == "R" else "trail"), "R"
    raise ValueError(f"side must be one of ('L', 'R', 'lead', 'trail'), got {side!r}")


def _validate_bushing_state(state: BushingState, name: str) -> None:
    """DbC check for BushingState proper rotation and finite values.

    Runs once per side evaluation on the integrator hot path, so it works on
    Python floats: ``|R^T R - I|`` entrywise <= 1e-5 and ``|det R - 1|`` <= 1e-5.
    """
    if not isinstance(state, BushingState):
        raise TypeError(f"{name} must be a BushingState, got {type(state).__name__}")
    rot = np.asarray(state.rotation, dtype=float)
    if rot.shape != (3, 3):
        raise ValueError(f"{name}.rotation must be a finite 3x3 matrix")
    rows = rot.tolist()
    if not all(map(math.isfinite, rows[0] + rows[1] + rows[2])):
        raise ValueError(f"{name}.rotation must be a finite 3x3 matrix")
    gram_error, det = rotation_errors(rows)
    if gram_error > 1e-5 or abs(det - 1.0) > 1e-5:
        raise ValueError(f"{name}.rotation must be a proper rotation matrix")
    for field_name in ("position_m", "velocity_m_s", "omega_rad_s"):
        val = np.asarray(getattr(state, field_name), dtype=float)
        if val.shape != (3,) or not all(map(math.isfinite, val.tolist())):
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
    handedness: Handedness = DEFAULT_HANDEDNESS,
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
    role, side_char = _normalize_side(side, handedness=handedness)
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
    if handedness == "left":
        cz = -cz
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


def _build_side_batch(
    pairs: tuple[tuple[ContactPair, ContactSphere | ContactCapsule], ...],
    grip: ContactCapsule,
    mode: str,
) -> SideBatch:
    """Stack the hand-site primitives of ``pairs`` physical in ``mode``, in order."""
    prims = [prim for p, prim in pairs if mode in p.physical_modes]
    starts = [q.center_m if isinstance(q, ContactSphere) else q.p0_m for q in prims]
    ends = [q.p1_m for q in prims if isinstance(q, ContactCapsule)]
    end_rows: list[int] = []
    for q in prims:
        capsule = isinstance(q, ContactCapsule)
        end_rows.append(len(prims) + sum(r >= 0 for r in end_rows) if capsule else -1)
    return SideBatch(
        pair_names=tuple(p.name for p, _ in pairs if mode in p.physical_modes),
        hand_points_m=np.array(starts + ends, dtype=float).reshape(-1, 1, 3),
        capsule_ends=tuple(end_rows),
        site_radii_m=tuple(float(q.radius_m) for q in prims),
        grip_points_m=np.array([grip.p0_m, grip.p1_m], dtype=float).reshape(2, 1, 3),
        grip_radius_m=float(grip.radius_m),
    )


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
    _batches: dict[tuple[str, str], SideBatch] = field(
        init=False, repr=False, compare=False
    )

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

        topo = self.topology
        capsules = {s: self._make_grip_capsule(s) for s in ("L", "R")}
        object.__setattr__(self, "_grip_capsules", capsules)

        pairs_dict: dict[
            str, tuple[tuple[ContactPair, ContactSphere | ContactCapsule], ...]
        ] = {}
        for role, side_char in (("lead", "L"), ("trail", "R")):
            p_list = []
            for p in topo.pairs:
                site = topo.site(p.site_a)
                if p.is_hand_grip and site.hand == role:
                    prim = site_primitive_in_hand_frame(topo, site, side_char)
                    p_list.append((p, prim))
            pairs_dict[side_char] = tuple(p_list)
        object.__setattr__(self, "_physical_pairs", pairs_dict)
        object.__setattr__(self, "_batches", {})

    @property
    def parameters(self) -> ContactParameters:
        """Alias for ``law`` (:class:`ContactParameters`)."""
        return self.law

    @property
    def grip_radius_m(self) -> float:
        """Radius (m) of the club grip cylinder."""
        return self.cylinder.radius_m

    def origin_axial_m(self, side: str) -> float:
        """Axial coordinate (m) of ``side``'s grip frame origin."""
        return self.cylinder.origin_axial_m(side)

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

    def physical_pair_primitives(
        self, side: str
    ) -> tuple[tuple[ContactPair, ContactSphere | ContactCapsule], ...]:
        """Ordered (pair, primitive) tuples for physical pad-grip pairs on ``side``."""
        _, side_char = _normalize_side(side)
        return self._physical_pairs[side_char]

    def side_batch(self, side: str, mode: str = "contact") -> SideBatch:
        """Stacked hand-site primitives of ``side``'s pairs physical in ``mode``.

        Built once per ``(side, mode)`` and cached; the topology is immutable.
        """
        _, side_char = _normalize_side(side)
        key = (side_char, mode)
        if key not in self._batches:
            self._batches[key] = _build_side_batch(
                self._physical_pairs[side_char], self._grip_capsules[side_char], mode
            )
        return self._batches[key]

    def _make_grip_capsule(self, side: str) -> ContactCapsule:
        _, side_char = _normalize_side(side)
        cyl = self.cylinder
        axis_offset = np.array(
            [
                0.0,
                0.0,
                -cyl.radius_m if side_char == "L" else cyl.radius_m,
            ]
        )
        origin_s = cyl.origin_axial_m(side_char)
        s_lo = cyl.axial_range_m[0] - origin_s
        s_hi = cyl.axial_range_m[1] - origin_s
        p0 = axis_offset + np.array([s_lo, 0.0, 0.0])
        p1 = axis_offset + np.array([s_hi, 0.0, 0.0])
        return ContactCapsule(p0_m=p0, p1_m=p1, radius_m=cyl.radius_m)

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
    pair_wrenches: Mapping[str, PairWrench]
    normal_force_n: np.ndarray
    tangential_force_n: np.ndarray
    total_normal_force_n: float


@dataclass(frozen=True)
class TopologyHandHandWrench:
    """Equal-and-opposite physical hand-hand contact wrenches (HRG-GS-G11, #13309).

    - ``force_on_lead_n``: resultant hand-hand contact force on the lead ('L') hand
      in world axes (N).
    - ``moment_on_lead_nm``: resultant hand-hand contact moment on the lead ('L')
      hand about ``lead_frame.position_m`` in world axes (N*m).
    - ``force_on_trail_n``: resultant hand-hand contact force on the trail ('R')
      hand in world axes (N), equal and opposite to ``force_on_lead_n``.
    - ``moment_on_trail_nm``: resultant hand-hand contact moment on the trail ('R')
      hand about ``trail_frame.position_m`` in world axes (N*m).
    - ``pairs``: per-pair :class:`PairWrench` mapping keyed by pair name.
    """

    force_on_lead_n: np.ndarray
    moment_on_lead_nm: np.ndarray
    force_on_trail_n: np.ndarray
    moment_on_trail_nm: np.ndarray
    pairs: Mapping[str, PairWrench]


class _LazyPairWrenches(Mapping[str, PairWrench]):
    """Per-pair :class:`PairWrench` views of a batched side result, built on access.

    The integrator right-hand side only needs the resultants and force
    magnitudes, so the validated per-pair objects are built only when a caller
    (a trace recorder, a test) asks for them.
    """

    def __init__(
        self,
        names: tuple[str, ...],
        rows: tuple[PairRow, ...],
        hand_frame: BushingState,
        club_frame: BushingState,
    ) -> None:
        self._rows = dict(zip(names, rows, strict=True))
        self._hand_frame = hand_frame
        self._club_frame = club_frame

    def __getitem__(self, name: str) -> PairWrench:
        row = self._rows[name]
        force_b = np.array(row.force_n)
        force_a = -force_b
        lever_a = np.array(row.contact_point_m) - self._hand_frame.position_m
        moment_a = cross3(lever_a, force_a) - np.array(row.spin_nm)
        return PairWrench(
            force_a_n=force_a,
            moment_a_nm=moment_a,
            force_b_n=force_b,
            moment_b_nm=np.array(row.moment_nm),
            contact_point_m=np.array(row.contact_point_m),
            normal=np.array(row.normal),
            penetration_m=row.penetration_m,
            point_a_m=self._hand_frame.position_m,
            point_b_m=self._club_frame.position_m,
            normal_force_mag_n=row.normal_force_mag_n,
            tangential_force_mag_n=row.tangential_force_mag_n,
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)


def _evaluate_site_on_grip(
    model: TopologyContactModel,
    primitive: ContactSphere | ContactCapsule,
    grip_capsule: ContactCapsule,
    hand_frame: BushingState,
    club_frame: BushingState,
) -> PairWrench:
    """Evaluate contact wrench between a hand site primitive and club grip capsule."""
    if isinstance(primitive, ContactSphere):
        return sphere_capsule(
            primitive,
            grip_capsule,
            hand_frame,
            club_frame,
            model.law,
            patch_radius_m=model.patch_radius_m,
            spin_transition_rad_s=model.spin_transition_rad_s,
        )
    if isinstance(primitive, ContactCapsule):
        return capsule_capsule(
            primitive,
            grip_capsule,
            hand_frame,
            club_frame,
            model.law,
            patch_radius_m=model.patch_radius_m,
            spin_transition_rad_s=model.spin_transition_rad_s,
        )
    raise ValueError(f"unsupported site primitive {primitive!r}")


def topology_pair_wrench(
    model: TopologyContactModel,
    pair: ContactPair | str,
    hand_frame: BushingState,
    club_frame: BushingState,
    mode: str = "contact",
) -> PairWrench:
    """Evaluate physical contact wrench for a single hand-grip pair (DbC validated)."""
    if not isinstance(model, TopologyContactModel):
        raise TypeError(
            f"model must be TopologyContactModel, got {type(model).__name__}"
        )
    topo = model.topology
    target_pair = topo.pair(pair) if isinstance(pair, str) else pair
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

    site = topo.site(target_pair.site_a)
    side_char = "L" if site.hand == "lead" else "R"
    prim_a = site_primitive_in_hand_frame(topo, site, side_char)
    prim_b = model.grip_capsule(side_char)
    return _evaluate_site_on_grip(model, prim_a, prim_b, hand_frame, club_frame)


def topology_hand_hand_pair_wrench(
    model: TopologyContactModel,
    pair: ContactPair | str,
    frame_a: BushingState,
    frame_b: BushingState,
    *,
    mode: str = "separated_v2",
    handedness: Handedness = DEFAULT_HANDEDNESS,
) -> PairWrench:
    """Evaluate physical contact wrench for a hand-hand pair on separate phalanx/hand bodies.

    Preconditions:
        ``model`` is a :class:`TopologyContactModel`;
        ``pair`` is a hand-hand :class:`ContactPair` in ``model.topology``;
        ``mode`` is in ``pair.physical_modes`` (e.g. ``'separated_v2'``);
        ``frame_a`` and ``frame_b`` are valid :class:`BushingState` instances.
    Postconditions:
        Returns :class:`PairWrench` evaluated via the G2 kernels
        (:func:`sphere_capsule`, :func:`capsule_sphere`, :func:`capsule_capsule`)
        satisfying exact action-reaction.
    """
    if not isinstance(model, TopologyContactModel):
        raise TypeError(
            f"model must be TopologyContactModel, got {type(model).__name__}"
        )
    topo = model.topology
    target_pair = topo.pair(pair) if isinstance(pair, str) else pair
    if not isinstance(target_pair, ContactPair):
        raise TypeError(
            f"pair must be ContactPair or str, got {type(target_pair).__name__}"
        )
    if not target_pair.is_hand_hand:
        raise ValueError(f"pair {target_pair.name!r} is not a hand-hand pair")
    if mode not in target_pair.physical_modes:
        raise ValueError(
            f"hand-hand pair {target_pair.name!r} is not physical in mode {mode!r} "
            f"(physical modes: {target_pair.physical_modes})"
        )
    _validate_bushing_state(frame_a, "frame_a")
    _validate_bushing_state(frame_b, "frame_b")

    site_a = topo.site(target_pair.site_a)
    site_b = topo.site(target_pair.site_b)
    l_side = lead_side(handedness)
    t_side = trail_side(handedness)
    side_a = l_side if site_a.hand == "lead" else t_side
    side_b = l_side if site_b.hand == "lead" else t_side

    cyl = model.cylinder
    x_lead = hand_x_origin(topo, "lead")
    x_trail = x_lead - cyl.left_axial_m if l_side == "L" else x_lead + cyl.left_axial_m
    x_orig_a = x_lead if site_a.hand == "lead" else x_trail
    x_orig_b = x_lead if site_b.hand == "lead" else x_trail

    prim_a = site_primitive_in_hand_frame(topo, site_a, side_a, x_origin=x_orig_a)
    prim_b = site_primitive_in_hand_frame(topo, site_b, side_b, x_origin=x_orig_b)
    if isinstance(prim_a, ContactSphere) and isinstance(prim_b, ContactCapsule):
        return sphere_capsule(
            prim_a,
            prim_b,
            frame_a,
            frame_b,
            model.law,
            patch_radius_m=model.patch_radius_m,
            spin_transition_rad_s=model.spin_transition_rad_s,
        )
    if isinstance(prim_a, ContactCapsule) and isinstance(prim_b, ContactSphere):
        return capsule_sphere(
            prim_a,
            prim_b,
            frame_a,
            frame_b,
            model.law,
            patch_radius_m=model.patch_radius_m,
            spin_transition_rad_s=model.spin_transition_rad_s,
        )
    if isinstance(prim_a, ContactCapsule) and isinstance(prim_b, ContactCapsule):
        return capsule_capsule(
            prim_a,
            prim_b,
            frame_a,
            frame_b,
            model.law,
            patch_radius_m=model.patch_radius_m,
            spin_transition_rad_s=model.spin_transition_rad_s,
        )
    raise ValueError(
        f"unsupported hand-hand primitive combination for {target_pair.name!r}"
    )


def topology_hand_hand_wrenches(
    model: TopologyContactModel,
    lead_frame: BushingState,
    trail_frame: BushingState,
    mode: str = "separated_v2",
    *,
    handedness: Handedness = DEFAULT_HANDEDNESS,
) -> TopologyHandHandWrench:
    """Evaluate physical hand-hand contact pairs and equal-and-opposite hand wrenches."""
    if not isinstance(model, TopologyContactModel):
        raise TypeError(
            f"model must be TopologyContactModel, got {type(model).__name__}"
        )
    if mode not in PHYSICAL_HAND_HAND_MODES:
        raise ValueError(
            f"hand-hand pairs are not physical in mode {mode!r}; "
            f"expected one of {PHYSICAL_HAND_HAND_MODES}"
        )
    _validate_bushing_state(lead_frame, "lead_frame")
    _validate_bushing_state(trail_frame, "trail_frame")

    topo = model.topology
    frames = {"lead": lead_frame, "trail": trail_frame}
    forces = {"lead": np.zeros(3, dtype=float), "trail": np.zeros(3, dtype=float)}
    moments = {"lead": np.zeros(3, dtype=float), "trail": np.zeros(3, dtype=float)}
    pair_map: dict[str, PairWrench] = {}

    for pair in topo.hand_hand_pairs():
        if mode not in pair.physical_modes:
            continue
        role_a = topo.site(pair.site_a).hand
        role_b = topo.site(pair.site_b).hand
        pw = topology_hand_hand_pair_wrench(
            model,
            pair,
            frames[role_a],
            frames[role_b],
            mode=mode,
            handedness=handedness,
        )
        pair_map[pair.name] = pw
        forces[role_a] += pw.force_a_n
        moments[role_a] += pw.moment_a_nm
        forces[role_b] += pw.force_b_n
        moments[role_b] += pw.moment_b_nm

    return TopologyHandHandWrench(
        force_on_lead_n=forces["lead"],
        moment_on_lead_nm=moments["lead"],
        force_on_trail_n=forces["trail"],
        moment_on_trail_nm=moments["trail"],
        pairs=pair_map,
    )


@dataclass(frozen=True)
class PairChatterSummary:
    """Honest per-pair chatter summary distinguishing dynamic contact from constant preload."""

    min_normal_force_n: float
    peak_normal_force_n: float
    normal_force_range_n: float
    transition_count: int
    step_count: int
    transition_fraction: float
    is_constant: bool
    passes_chatter: bool
    status: str

    def to_dict(self) -> dict[str, float | int | bool | str | None]:
        """Return JSON-safe dictionary representation (NaN transition_fraction becomes None)."""
        return {
            "min_normal_force_n": self.min_normal_force_n,
            "peak_normal_force_n": self.peak_normal_force_n,
            "normal_force_range_n": self.normal_force_range_n,
            "transition_count": self.transition_count,
            "step_count": self.step_count,
            "transition_fraction": (
                None
                if math.isnan(self.transition_fraction)
                else self.transition_fraction
            ),
            "is_constant": self.is_constant,
            "passes_chatter": self.passes_chatter,
            "chatter_status": self.status,
        }


def contact_transition_fraction(
    normal_force_n: np.ndarray,
    threshold_n: float = 0.0,
    *,
    variation_tol_n: float = DEFAULT_FORCE_VARIATION_TOL_N,
) -> float:
    """Fraction of adjacent time-step transitions where contact activates or separates.

    Returns ``NaN`` when ``ptp(normal_force_n) <= variation_tol_n`` (static preload).
    """
    arr = np.asarray(normal_force_n, dtype=float)
    if arr.ndim != 1 or arr.size < 2 or not np.all(np.isfinite(arr)):
        raise ValueError("normal_force_n must be a finite 1-D array with >= 2 samples")
    if not math.isfinite(threshold_n) or threshold_n < 0.0:
        raise ValueError(f"threshold_n must be finite and >= 0, got {threshold_n}")
    if not math.isfinite(variation_tol_n) or variation_tol_n < 0.0:
        raise ValueError(
            f"variation_tol_n must be finite and >= 0, got {variation_tol_n}"
        )
    if float(np.ptp(arr)) <= variation_tol_n:
        return float("nan")
    active = arr > threshold_n
    transitions = int(np.sum(active[1:] != active[:-1]))
    return float(transitions) / float(arr.size - 1)


def evaluate_pair_chatter(
    normal_force_n: np.ndarray,
    *,
    chatter_tolerance: float = 0.01,
    threshold_n: float = 0.0,
    variation_tol_n: float = DEFAULT_FORCE_VARIATION_TOL_N,
) -> PairChatterSummary:
    """Evaluate honest chatter status for a single contact pair's normal-force trace."""
    if (
        not math.isfinite(chatter_tolerance)
        or chatter_tolerance <= 0.0
        or chatter_tolerance > 1.0
    ):
        raise ValueError(
            f"chatter_tolerance must be in (0, 1], got {chatter_tolerance}"
        )
    frac = contact_transition_fraction(
        normal_force_n, threshold_n=threshold_n, variation_tol_n=variation_tol_n
    )
    arr = np.asarray(normal_force_n, dtype=float)
    n_steps = int(arr.size - 1)
    f_min = float(np.min(arr))
    f_max = float(np.max(arr))
    f_range = float(f_max - f_min)
    if math.isnan(frac):
        return PairChatterSummary(
            min_normal_force_n=f_min,
            peak_normal_force_n=f_max,
            normal_force_range_n=f_range,
            transition_count=0,
            step_count=n_steps,
            transition_fraction=float("nan"),
            is_constant=True,
            passes_chatter=False,
            status=CONSTANT_PRELOAD_CHATTER_STATUS,
        )
    active = arr > threshold_n
    transitions = int(np.count_nonzero(active[1:] != active[:-1]))
    passes = bool(frac < chatter_tolerance)
    return PairChatterSummary(
        min_normal_force_n=f_min,
        peak_normal_force_n=f_max,
        normal_force_range_n=f_range,
        transition_count=transitions,
        step_count=n_steps,
        transition_fraction=frac,
        is_constant=False,
        passes_chatter=passes,
        status="pass" if passes else "fail",
    )


def _check_side_inputs(
    model: TopologyContactModel,
    side: str,
    hand_frame: BushingState,
    club_frame: BushingState,
) -> None:
    """DbC of a side evaluation: model type, side name and both frame states."""
    if not isinstance(model, TopologyContactModel):
        raise TypeError(
            f"model must be TopologyContactModel, got {type(model).__name__}"
        )
    _normalize_side(side)
    _validate_bushing_state(hand_frame, "hand_frame")
    _validate_bushing_state(club_frame, "club_frame")


def topology_side_wrench(
    model: TopologyContactModel,
    side: str,
    hand_frame: BushingState,
    club_frame: BushingState,
    mode: str = "contact",
) -> TopologySideWrench:
    """Contact wrench of ``side``'s physical sites on the grip, in one batched pass."""
    _check_side_inputs(model, side, hand_frame, club_frame)
    batch = model.side_batch(side, mode)
    if not batch.pair_names:
        return TopologySideWrench(
            force_n=np.zeros(3),
            moment_nm=np.zeros(3),
            pair_wrenches={},
            normal_force_n=np.zeros(0),
            tangential_force_n=np.zeros(0),
            total_normal_force_n=0.0,
        )
    side_rows = side_rows_wrench(
        batch,
        hand_frame,
        club_frame,
        model.law,
        model.patch_radius_m,
        model.spin_transition_rad_s,
    )
    rows = side_rows.rows
    normal = [row.normal_force_mag_n for row in rows]
    # One array for every output keeps the numpy call count per side constant.
    flat = np.array(
        [
            *side_rows.force_n,
            *side_rows.moment_nm,
            *normal,
            *(row.tangential_force_mag_n for row in rows),
        ]
    )
    count = len(rows)
    return TopologySideWrench(
        force_n=flat[0:3],
        moment_nm=flat[3:6],
        pair_wrenches=_LazyPairWrenches(batch.pair_names, rows, hand_frame, club_frame),
        normal_force_n=flat[6 : 6 + count],
        tangential_force_n=flat[6 + count :],
        total_normal_force_n=sum(normal),
    )


def topology_side_wrench_reference(
    model: TopologyContactModel,
    side: str,
    hand_frame: BushingState,
    club_frame: BushingState,
    mode: str = "contact",
) -> TopologySideWrench:
    """Per-pair reference: one shared G2 kernel call per physical pair."""
    _check_side_inputs(model, side, hand_frame, club_frame)
    pairs_with_prims = model.physical_pair_primitives(side)
    grip_cap = model.grip_capsule(side)
    effective_mode = "contact" if mode == "separated_v2" else mode

    pair_wrenches: dict[str, PairWrench] = {}
    for pair, prim_a in pairs_with_prims:
        if effective_mode not in pair.physical_modes:
            continue
        pair_wrenches[pair.name] = _evaluate_site_on_grip(
            model, prim_a, grip_cap, hand_frame, club_frame
        )
    return aggregate_side_wrench(pair_wrenches)


def aggregate_side_wrench(
    pair_wrenches: Mapping[str, PairWrench],
) -> TopologySideWrench:
    """Sum per-pair wrenches into one :class:`TopologySideWrench`."""
    force = np.zeros(3)
    moment = np.zeros(3)
    normal_forces: list[float] = []
    tangential_forces: list[float] = []
    for pw in pair_wrenches.values():
        force += pw.force_b_n
        moment += pw.moment_b_nm
        normal_forces.append(pw.normal_force_mag_n)
        tangential_forces.append(pw.tangential_force_mag_n)
    norm_arr = np.array(normal_forces, dtype=float)
    return TopologySideWrench(
        force_n=force,
        moment_nm=moment,
        pair_wrenches=dict(pair_wrenches),
        normal_force_n=norm_arr,
        tangential_force_n=np.array(tangential_forces, dtype=float),
        total_normal_force_n=float(np.sum(norm_arr)),
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


def _contact_point_in_club_frame(
    c_in_club: np.ndarray,
    radius_m: float,
    grip_cap: ContactCapsule,
) -> np.ndarray:
    """Project site center ``c_in_club`` onto ``grip_cap`` in the club grip frame."""
    p_axis, _ = closest_point_segment(c_in_club, grip_cap.p0_m, grip_cap.p1_m)
    diff = p_axis - c_in_club
    dist = float(np.linalg.norm(diff))
    normal = diff / dist if dist > 1e-12 else np.array([0.0, 0.0, 1.0])
    return c_in_club + radius_m * normal


class PairTraceRecorder:
    """Engine-agnostic per-step pair trace and state recorder (HRG-GS-G6, #13304; HRG-GS-G11, #13309)."""

    def __init__(self, model: TopologyContactModel, mode: str = "contact") -> None:
        """Construct recorder from TopologyContactModel."""
        if not isinstance(model, TopologyContactModel):
            raise TypeError(
                f"model must be TopologyContactModel, got {type(model).__name__}"
            )
        if mode not in ("contact", *PHYSICAL_HAND_HAND_MODES):
            raise ValueError(
                f"mode must be 'contact' or in {PHYSICAL_HAND_HAND_MODES}, got {mode!r}"
            )
        self._model = model
        self._mode = mode
        self._all_pair_names: set[str] = {p.name for p in model.topology.pairs}
        self._hand_hand_pair_names: set[str] = {
            p.name for p in model.topology.hand_hand_pairs()
        }
        self._pair_normal: dict[str, list[float]] = {
            p.name: [] for p in model.topology.pairs
        }
        self._pair_tangential: dict[str, list[float]] = {
            p.name: [] for p in model.topology.pairs
        }
        self._pair_drift: dict[str, list[float]] = {
            p.name: [] for p in model.topology.pairs
        }
        self._pair_penetration: dict[str, list[float]] = {
            p.name: [] for p in model.topology.pairs
        }
        self._nominal_site_info: dict[
            str, tuple[str, np.ndarray, float, ContactCapsule, np.ndarray]
        ] = {}
        self._nominal_hh_contact_in_lead: dict[str, np.ndarray] = {}
        for side in SIDES:
            gcap = model.grip_capsule(side)
            for pair, prim_a in model.physical_pair_primitives(side):
                c_hand = (
                    prim_a.center_m
                    if isinstance(prim_a, ContactSphere)
                    else 0.5 * (prim_a.p0_m + prim_a.p1_m)
                )
                p_nom = _contact_point_in_club_frame(c_hand, prim_a.radius_m, gcap)
                self._nominal_site_info[pair.name] = (
                    side,
                    c_hand,
                    prim_a.radius_m,
                    gcap,
                    p_nom,
                )
        if mode in PHYSICAL_HAND_HAND_MODES:
            self._channel_reasons: dict[str, str] = {}
        else:
            self._channel_reasons = {
                p.name: (
                    "hand-hand pairs are allocated, not physical forces under prescribed hand motion"
                )
                for p in model.topology.pairs
                if p.is_hand_hand
            }
        self._hand_states: dict[str, list[Any]] = {s: [] for s in SIDES}
        self._club_states: dict[str, list[Any]] = {s: [] for s in SIDES}

    def _pair_contact_drift_m(
        self,
        pname: str,
        hand_frames: Mapping[str, Any],
        club_frames: Mapping[str, Any],
    ) -> float:
        """Compute Euclidean contact-point drift (m) in the club grip frame."""
        side, c_hand, r_site, gcap, p_nom = self._nominal_site_info[pname]
        hf = hand_frames[side]
        cf = club_frames[side]
        h_rot = np.asarray(hf.rotation, dtype=float)
        h_pos = np.asarray(hf.position_m, dtype=float)
        c_rot = np.asarray(cf.rotation, dtype=float)
        c_pos = np.asarray(cf.position_m, dtype=float)
        c_in_club = c_rot.T @ (h_pos + h_rot @ c_hand - c_pos)
        p_cur = _contact_point_in_club_frame(c_in_club, r_site, gcap)
        return float(np.linalg.norm(p_cur - p_nom))

    def _hh_contact_drift_m(self, pname: str, pw: PairWrench, lead_frame: Any) -> float:
        """Compute contact-point drift (m) of a hand-hand pair in the lead hand frame."""
        l_rot = np.asarray(lead_frame.rotation, dtype=float)
        l_pos = np.asarray(lead_frame.position_m, dtype=float)
        p_in_lead = l_rot.T @ (pw.contact_point_m - l_pos)
        if pname not in self._nominal_hh_contact_in_lead:
            self._nominal_hh_contact_in_lead[pname] = p_in_lead.copy()
        return float(
            np.linalg.norm(p_in_lead - self._nominal_hh_contact_in_lead[pname])
        )

    def record(
        self,
        side_wrenches: Mapping[str, TopologySideWrench],
        hand_frames: Mapping[str, Any],
        club_frames: Mapping[str, Any],
        hand_hand: TopologyHandHandWrench | None = None,
    ) -> None:
        """Append one sample of pair forces, contact-point drift, and frame states.

        Preconditions:
            Every pair in ``model.topology`` must be recorded exactly once.
        Raises:
            ValueError: On an unknown pair name, duplicate record, or missing pair.
        """
        recorded_pairs: set[str] = set()

        for sw in side_wrenches.values():
            for pname, pw in sw.pair_wrenches.items():
                if pname not in self._all_pair_names:
                    raise ValueError(f"unknown pair name: {pname!r}")
                if pname in recorded_pairs:
                    raise ValueError(
                        f"pair {pname!r} recorded more than once in sample"
                    )
                recorded_pairs.add(pname)
                self._pair_normal[pname].append(float(pw.normal_force_mag_n))
                self._pair_tangential[pname].append(float(pw.tangential_force_mag_n))
                self._pair_drift[pname].append(
                    self._pair_contact_drift_m(pname, hand_frames, club_frames)
                )
                self._pair_penetration[pname].append(float(pw.penetration_m))

        if self._mode in PHYSICAL_HAND_HAND_MODES:
            hh = hand_hand or topology_hand_hand_wrenches(
                self._model, hand_frames["L"], hand_frames["R"], mode=self._mode
            )
            for pname in self._hand_hand_pair_names:
                if pname in recorded_pairs:
                    raise ValueError(
                        f"hand-hand pair {pname!r} was already recorded in side wrenches"
                    )
                if pname not in hh.pairs:
                    raise ValueError(f"missing hand-hand pair {pname!r} in evaluation")
                pw = hh.pairs[pname]
                recorded_pairs.add(pname)
                self._pair_normal[pname].append(float(pw.normal_force_mag_n))
                self._pair_tangential[pname].append(float(pw.tangential_force_mag_n))
                self._pair_drift[pname].append(
                    self._hh_contact_drift_m(pname, pw, hand_frames["L"])
                )
                self._pair_penetration[pname].append(float(pw.penetration_m))
        else:
            for pname in self._hand_hand_pair_names:
                if pname in recorded_pairs:
                    raise ValueError(
                        f"hand-hand pair {pname!r} was physically recorded in side wrenches"
                    )
                recorded_pairs.add(pname)
                self._pair_normal[pname].append(float("nan"))
                self._pair_tangential[pname].append(float("nan"))
                self._pair_drift[pname].append(float("nan"))
                self._pair_penetration[pname].append(float("nan"))

        if len(recorded_pairs) != len(self._all_pair_names):
            missing = self._all_pair_names - recorded_pairs
            raise ValueError(f"missing pairs in sample: {sorted(missing)}")

        for s in SIDES:
            self._hand_states[s].append(hand_frames[s])
            self._club_states[s].append(club_frames[s])

    def pair_penetration_m(self) -> dict[str, np.ndarray]:
        """Return per-pair penetration depth traces (m) as 1-D numpy arrays."""
        return {
            p: np.asarray(vals, dtype=float)
            for p, vals in self._pair_penetration.items()
        }

    def result(self) -> dict[str, Any]:
        """Return dict matching ContactRun fields with numpy arrays for pair traces."""
        return {
            "pair_normal_force_n": {
                p: np.asarray(vals, dtype=float)
                for p, vals in self._pair_normal.items()
            },
            "pair_tangential_force_n": {
                p: np.asarray(vals, dtype=float)
                for p, vals in self._pair_tangential.items()
            },
            "pair_contact_point_drift_m": {
                p: np.asarray(vals, dtype=float) for p, vals in self._pair_drift.items()
            },
            "channel_reasons": dict(self._channel_reasons),
            "hand_states": {s: list(states) for s, states in self._hand_states.items()},
            "club_states": {s: list(states) for s, states in self._club_states.items()},
        }
