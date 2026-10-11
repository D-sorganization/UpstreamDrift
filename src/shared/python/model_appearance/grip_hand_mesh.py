"""Rigid curled-finger hand mesh wrapped on the grip (OSV-2 slice 2, #11728).

A parametric finger-segment generator: a palm, four fingers of three phalanges
and a two-segment thumb, each a closed capsule (or, for the palm, a rounded
curved block), curled around the grip cylinder of the given radius in the
placement of :mod:`grip_pose`. The result is visual only: it is never used
for collision, mass or inertia, and no engine attaches it yet.

Hand-grip frame
---------------
The mesh is returned in the hand-grip frame, which is rigid in the hand body:
origin on the grip axis at the hand centre (``GripPose.distance_below_butt_m``
of that hand), ``z`` along the grip axis from the butt toward the head, ``x``
the :mod:`grip_pose` azimuth 0 (``-n``) and ``y`` azimuth 90 deg (``-t``). The
grip cylinder is therefore ``x**2 + y**2 = R**2``. Placing it in an engine is
the next slice: a translation of ``distance_below_butt_m`` down the shaft and
the club-frame rotation of the azimuth convention.

Wrapping
--------
Each phalanx of a finger is a straight capsule whose axis is tangent to the
circle of radius ``d = R + c + r`` about the grip axis (``r`` the finger
radius, ``c`` the pad clearance), so every phalanx pad touches the clearance
surface ``R + c`` at its tangent point and nowhere comes closer. Consecutive
tangent lines meet at the joints; the proximal phalanx's tangent point is its
midpoint, and each later tangent point is set so the joints coincide, which
puts most of the flexion at the proximal interphalangeal joint. The lead
fingers curl toward decreasing azimuth and the trail fingers toward
increasing azimuth: the two hands are mirror images about their V azimuths.
The thumb lies along the shaft (toward the head) at the :mod:`grip_pose`
thumb azimuth.

Linked fingers (overlap, interlock) take the trail little finger off the grip
and over (or between) the lead index and middle fingers; it is then flagged
``on_grip=False`` and excluded from the grip clearance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.geometry import (
    Mesh,
    closed_grid_mesh,
    concat_meshes,
)
from src.shared.python.model_appearance.grip_pose import (
    DEFAULT_GRIP_POSE,
    GripPose,
    GripStyle,
    Hand,
)

Array = NDArray[np.float64]

#: Fingers on the grip, from the head end of the hand to the butt end.
FINGER_NAMES: tuple[str, ...] = ("index", "middle", "ring", "little")
PHALANX_NAMES: tuple[str, ...] = ("proximal", "middle", "distal")
#: Acceptance band of the finger-pad-to-grip distance (#11728 Scope item 5).
MAX_PAD_CLEARANCE_M = 0.003
#: Admissible grip radius (a junior grip to an oversize putter grip).
GRIP_RADIUS_RANGE_M: tuple[float, float] = (0.008, 0.03)


@dataclass(frozen=True)
class HandMeshParams:
    """Nominal adult hand dimensions (metres, degrees) for the generator.

    Finger radii and phalanx lengths are mid-range adult values; they are a
    visual parameterisation, not an anthropometric fit. ``scale`` scales every
    length and radius (not the clearances).
    """

    finger_radius_m: dict[str, float] = field(
        default_factory=lambda: {
            "index": 0.0080,
            "middle": 0.0082,
            "ring": 0.0078,
            "little": 0.0068,
        }
    )
    phalanx_lengths_m: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: {
            "index": (0.040, 0.024, 0.019),
            "middle": (0.045, 0.028, 0.020),
            "ring": (0.042, 0.027, 0.020),
            "little": (0.034, 0.020, 0.018),
        }
    )
    finger_gap_m: float = 0.0003
    pad_clearance_m: float = 0.001
    #: Azimuth of the knuckle (MCP) from the V, in the curl direction.
    mcp_from_v_deg: float = 110.0
    palm_thickness_m: float = 0.018
    palm_clearance_m: float = 0.0015
    thumb_radius_m: float = 0.0095
    #: Thumb segment ends along the grip axis, relative to the hand centre.
    thumb_stations_m: tuple[float, float, float] = (-0.012, 0.020, 0.046)
    scale: float = 1.0
    sides: int = 16

    def __post_init__(self) -> None:
        for name in ("finger_gap_m", "pad_clearance_m", "palm_clearance_m"):
            value = getattr(self, name)
            if not (math.isfinite(value) and 0.0 <= value <= MAX_PAD_CLEARANCE_M):
                raise ValueError(f"{name} must be in [0, 0.003] m, got {value!r}")
        if not (math.isfinite(self.scale) and 0.5 <= self.scale <= 1.5):
            raise ValueError(f"scale must be in [0.5, 1.5], got {self.scale!r}")
        if self.sides < 8:
            raise ValueError(f"sides must be >= 8, got {self.sides}")
        if set(self.finger_radius_m) != set(FINGER_NAMES) or set(
            self.phalanx_lengths_m
        ) != set(FINGER_NAMES):
            raise ValueError(f"finger tables must cover exactly {FINGER_NAMES}")
        for finger, lengths in self.phalanx_lengths_m.items():
            proximal, middle, distal = lengths
            # Joint coincidence needs the middle phalanx to reach past the
            # proximal half-length and the distal to reach past what remains.
            if not (0.0 < proximal / 2.0 < middle < proximal / 2.0 + distal):
                raise ValueError(f"phalanx lengths of {finger} cannot wrap: {lengths}")


DEFAULT_HAND_MESH_PARAMS = HandMeshParams()


@dataclass(frozen=True)
class HandComponent:
    """One closed shell of the hand mesh (hand-grip frame)."""

    name: str
    finger: str  # FINGER_NAMES, "thumb" or "palm"
    mesh: Mesh
    on_grip: bool


@dataclass(frozen=True)
class GripHandMesh:
    """A baked rigid hand in grip pose, with its placement metadata.

    Postconditions: ``mesh`` is the union of the component shells (each
    closed and outward wound); no on-grip vertex is inside the grip cylinder.
    """

    hand: Hand
    style: GripStyle
    grip_radius_m: float
    distance_below_butt_m: float
    v_azimuth_deg: float
    thumb_azimuth_deg: float
    curl_sign: int  # -1 lead (decreasing azimuth), +1 trail
    components: tuple[HandComponent, ...]
    mesh: Mesh
    min_pad_clearance_m: float

    def on_grip_depths_below_butt_m(self) -> dict[str, float]:
        """Axial centre of each on-grip finger, as distance below the butt."""
        depths = {}
        for part in self.components:
            if part.on_grip and part.name.endswith("_proximal"):
                if part.finger in FINGER_NAMES:
                    z = float(part.mesh.vertices[:, 2].mean())
                    depths[part.finger] = self.distance_below_butt_m + z
        return depths

    def on_grip_extent_below_butt_m(self) -> tuple[float, float]:
        """(nearest, farthest) distance below the butt of the on-grip shells."""
        z = np.concatenate(
            [p.mesh.vertices[:, 2] for p in self.components if p.on_grip]
        )
        return (
            self.distance_below_butt_m + float(z.min()),
            self.distance_below_butt_m + float(z.max()),
        )


# ------------------------------------------------------------------ primitives


def _radial(theta: float) -> Array:
    return np.array([math.cos(theta), math.sin(theta), 0.0])


def _tangential(theta: float) -> Array:
    return np.array([-math.sin(theta), math.cos(theta), 0.0])


def capsule_mesh(
    start: Array,
    end: Array,
    radius: float,
    *,
    inward: Array,
    stations: tuple[float, ...] = (),
    sides: int = 16,
    cap_rings: int = 4,
) -> Mesh:
    """Closed capsule from ``start`` to ``end`` with one vertex column on ``inward``.

    ``inward`` (perpendicular to the axis) fixes the ring orientation so a
    vertex column lies on the side nearest the grip; ``stations`` are extra
    axial ring fractions in (0, 1) (the pad tangent point). Preconditions:
    positive radius and length, ``inward`` perpendicular to the axis.
    """
    p0, p1 = np.asarray(start, float), np.asarray(end, float)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if not (radius > 0.0 and length > 1e-9):
        raise ValueError("capsule needs a positive radius and length")
    a = axis / length
    u = np.asarray(inward, float)
    if abs(float(np.dot(u, a))) > 1e-9:
        raise ValueError("inward must be perpendicular to the capsule axis")
    u = u / np.linalg.norm(u)
    v = np.cross(a, u)
    phi = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    ring = np.outer(np.cos(phi), u) + np.outer(np.sin(phi), v)
    lat = np.linspace(0.0, 0.5 * math.pi, cap_rings + 1)[1:]
    fractions = sorted({0.0, 1.0, *(s for s in stations if 0.0 < s < 1.0)})
    grid = [p0 - radius * math.cos(t) * a + radius * math.sin(t) * ring for t in lat]
    grid += [p0 + f * axis + radius * ring for f in fractions[1:-1]]
    grid += [
        p1 + radius * math.cos(t) * a + radius * math.sin(t) * ring for t in lat[::-1]
    ]
    return closed_grid_mesh(np.asarray(grid), p0 - radius * a, p1 + radius * a)


def _rounded_block(
    inner_radius: float,
    thickness: float,
    theta: tuple[float, float],
    z: tuple[float, float],
    sides: int,
) -> Mesh:
    """Rounded curved block: a superellipse section swept about the grip axis.

    The section spans radii ``[inner_radius, inner_radius + thickness]`` and
    axial ``z``; it is swept from ``theta[0]`` to ``theta[1]`` (radians) and
    its ends shrink to rounded poles. Postcondition: no vertex is nearer the
    grip axis than ``inner_radius``.
    """
    r_c, h_r = inner_radius + 0.5 * thickness, 0.5 * thickness
    z_c, h_z = 0.5 * (z[0] + z[1]), 0.5 * (z[1] - z[0])
    phi = np.linspace(0.0, 2.0 * math.pi, 2 * sides, endpoint=False)
    c, s = np.cos(phi), np.sin(phi)
    sec_r = np.sign(c) * np.abs(c) ** 0.5  # superellipse, exponent 4
    sec_z = np.sign(s) * np.abs(s) ** 0.5
    lo, hi = min(theta), max(theta)
    n_mid = max(8, math.ceil(math.degrees(hi - lo) / 4.0))
    end_scale = (0.55, 0.85, 0.97)
    pad = 0.5 * h_r / r_c  # angular reach of the rounded ends
    sweep = [(lo - pad * (1.0 - f), f) for f in end_scale]
    sweep += [(float(t), 1.0) for t in np.linspace(lo, hi, n_mid)]
    sweep += [(hi + pad * (1.0 - f), f) for f in end_scale[::-1]]
    grid = []
    for t, f in sweep:
        r = r_c + f * h_r * sec_r
        grid.append(
            np.column_stack([r * math.cos(t), r * math.sin(t), z_c + f * h_z * sec_z])
        )
    first, last = lo - pad, hi + pad
    top = np.array([r_c * math.cos(first), r_c * math.sin(first), z_c])
    bottom = np.array([r_c * math.cos(last), r_c * math.sin(last), z_c])
    return closed_grid_mesh(np.asarray(grid), top, bottom)


def _finger(
    name: str,
    lengths: tuple[float, float, float],
    radius: float,
    tangent_radius: float,
    mcp_azimuth: float,
    z: float,
    curl: int,
    sides: int,
) -> list[tuple[str, Mesh]]:
    """Three phalanx capsules whose axes are tangent to ``tangent_radius``.

    Postcondition: each phalanx touches the circle at one tangent point, so
    its nearest approach to the grip axis is ``tangent_radius - radius``.
    """
    d = tangent_radius
    before = lengths[0] / 2.0  # proximal tangent point at its midpoint
    theta = mcp_azimuth + curl * math.atan2(before, d)
    lift = np.array([0.0, 0.0, z])
    parts = []
    for bone, length in zip(PHALANX_NAMES, lengths, strict=True):
        after = length - before
        e, t = _radial(theta), curl * _tangential(theta)
        start, end = d * e - before * t + lift, d * e + after * t + lift
        mesh = capsule_mesh(
            start, end, radius, inward=-e, stations=(before / length,), sides=sides
        )
        parts.append((f"{name}_{bone}", mesh))
        theta += curl * 2.0 * math.atan2(after, d)
        before = after
    return parts


def _stack_z(params: HandMeshParams) -> dict[str, float]:
    """Axial centre of each finger: index toward the head, little toward the butt."""
    radii = {f: params.scale * params.finger_radius_m[f] for f in FINGER_NAMES}
    width = 2.0 * sum(radii.values()) + params.finger_gap_m * (len(FINGER_NAMES) - 1)
    centres, edge = {}, -0.5 * width
    for finger in reversed(FINGER_NAMES):
        centres[finger] = edge + radii[finger]
        edge += 2.0 * radii[finger] + params.finger_gap_m
    return centres


def _check_inputs(
    hand: object, grip_radius_m: object, pose: object, style: object
) -> None:
    if not isinstance(hand, Hand):
        raise TypeError(f"hand must be a Hand, got {hand!r}")
    if isinstance(grip_radius_m, bool) or not isinstance(grip_radius_m, (int, float)):
        raise TypeError(f"grip_radius_m must be a number, got {grip_radius_m!r}")
    lo, hi = GRIP_RADIUS_RANGE_M
    if not (math.isfinite(grip_radius_m) and lo <= grip_radius_m <= hi):
        raise ValueError(
            f"grip_radius_m must be finite and in [{lo}, {hi}] m, got {grip_radius_m}"
        )
    if not isinstance(pose, GripPose):
        raise TypeError(f"pose must be a GripPose, got {type(pose).__name__}")
    if style is not None and not isinstance(style, GripStyle):
        raise TypeError(f"style must be a GripStyle or None, got {style!r}")


def build_grip_hand_mesh(
    hand: Hand,
    *,
    grip_radius_m: float,
    pose: GripPose = DEFAULT_GRIP_POSE,
    style: GripStyle | None = None,
    params: HandMeshParams = DEFAULT_HAND_MESH_PARAMS,
) -> GripHandMesh:
    """Bake one hand, fingers curled on a grip of radius ``grip_radius_m``.

    ``style`` overrides ``pose.style`` when given. Preconditions: ``hand`` a
    :class:`Hand`, ``grip_radius_m`` a finite number in
    ``GRIP_RADIUS_RANGE_M`` (``TypeError``/``ValueError`` otherwise), ``pose``
    a :class:`GripPose` and ``style`` a :class:`GripStyle` or ``None``.
    Postconditions: every shell is closed with outward winding; every on-grip
    shell clears the grip cylinder by ``[0, MAX_PAD_CLEARANCE_M]`` at its
    nearest vertex; the trail little finger is off the grip for linked styles.
    """
    _check_inputs(hand, grip_radius_m, pose, style)
    if not isinstance(params, HandMeshParams):
        raise TypeError(f"params must be HandMeshParams, got {params!r}")
    if style is not None:
        pose = replace(pose, style=style)
    grip_style = pose.style
    placement = pose.placement(hand)
    radius = float(grip_radius_m)
    curl = -1 if hand is Hand.LEAD else 1
    v = math.radians(placement.v_azimuth_deg)
    mcp = v + curl * math.radians(params.mcp_from_v_deg)
    stack = _stack_z(params)
    clearance, k = params.pad_clearance_m, params.scale

    parts: list[HandComponent] = []
    for finger in FINGER_NAMES:
        r = k * params.finger_radius_m[finger]
        p = params.phalanx_lengths_m[finger]
        lengths = (k * p[0], k * p[1], k * p[2])

        d, z, on_grip = radius + clearance + r, stack[finger], True
        if hand is Hand.TRAIL and finger == "little" and pose.linked_fingers():
            z, d = _linked_little(pose, params, radius, r)
            on_grip = False
        for name, mesh in _finger(finger, lengths, r, d, mcp, z, curl, params.sides):
            parts.append(HandComponent(name, finger, mesh, on_grip))
    z_lo = min(stack.values()) - k * params.finger_radius_m["little"] * 0.6
    z_hi = max(stack.values()) + k * params.finger_radius_m["index"] * 0.6
    palm = _rounded_block(
        radius + params.palm_clearance_m,
        k * params.palm_thickness_m,
        (v, mcp),
        (z_lo, z_hi),
        params.sides,
    )
    parts.append(HandComponent("palm", "palm", palm, True))
    parts += _thumb(radius, math.radians(placement.thumb_offset_deg), params)

    on_grip_vertices = np.vstack([p.mesh.vertices for p in parts if p.on_grip])
    gap = float(np.min(np.hypot(on_grip_vertices[:, 0], on_grip_vertices[:, 1])))
    gap -= radius
    if not 0.0 <= gap <= MAX_PAD_CLEARANCE_M:
        raise AssertionError(f"pad clearance {gap} m outside [0, 0.003] m")
    return GripHandMesh(
        hand=hand,
        style=grip_style,
        grip_radius_m=radius,
        distance_below_butt_m=placement.distance_below_butt_m,
        v_azimuth_deg=placement.v_azimuth_deg,
        thumb_azimuth_deg=placement.thumb_offset_deg,
        curl_sign=curl,
        components=tuple(parts),
        mesh=concat_meshes(*(p.mesh for p in parts)),
        min_pad_clearance_m=gap,
    )


def _linked_little(
    pose: GripPose,
    params: HandMeshParams,
    grip_radius: float,
    little_radius: float,
) -> tuple[float, float]:
    """Axial position and tangent radius of a linked trail little finger.

    It lies over the lead index/middle gap: on top of the lead finger pads for
    overlap, half-sunk between them for interlock (visual approximation).
    """
    stack = _stack_z(params)
    gap_depth = pose.distance_below_butt_m(Hand.LEAD)
    gap_depth += 0.5 * (stack["index"] + stack["middle"])
    z = gap_depth - pose.distance_below_butt_m(Hand.TRAIL)
    lead_r = params.scale * max(params.finger_radius_m[f] for f in ("index", "middle"))
    pads = grip_radius + params.pad_clearance_m
    if pose.style is GripStyle.OVERLAP:
        return z, pads + 2.0 * lead_r + params.pad_clearance_m + little_radius
    return z, pads + lead_r + little_radius


def _thumb(
    grip_radius: float, azimuth: float, params: HandMeshParams
) -> list[HandComponent]:
    """Two thumb capsules along the shaft (toward the head) at ``azimuth``."""
    r = params.scale * params.thumb_radius_m
    e = _radial(azimuth)
    centre = (grip_radius + params.pad_clearance_m + r) * e
    z0, z1, z2 = (params.scale * s for s in params.thumb_stations_m)
    parts = []
    for bone, (a, b) in (("proximal", (z0, z1)), ("distal", (z1, z2))):
        mesh = capsule_mesh(
            centre + np.array([0.0, 0.0, a]),
            centre + np.array([0.0, 0.0, b]),
            r,
            inward=-e,
            sides=params.sides,
        )
        parts.append(HandComponent(f"thumb_{bone}", "thumb", mesh, True))
    return parts
