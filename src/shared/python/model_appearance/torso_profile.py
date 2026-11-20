"""Trunk solids lofted through a profile of cross-sections (#15279).

A spindle between two joints reads as an egg, and two of them stacked read as
a snowman. A torso is closer to a column whose width and depth change slowly:
waist, ribs, chest, the shoulder line and the slope up to the neck. A
:class:`TorsoProfile` lists those cross-sections as fractions of the body's
own half width and half depth, so the solid still scales with the subject.

Visual only: nothing here carries mass. The mass of the trunk is the spec's
solids (see ``full_body_v2_geometry.human_mass_ledger``).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator

from src.shared.python.model_appearance.geometry import Mesh, closed_grid_mesh

Array = NDArray[np.float64]


@dataclass(frozen=True)
class TorsoStation:
    """One cross-section of a trunk solid.

    ``along`` is the fraction of the way from the body's lower joint plane
    (0) to its upper end (1); values outside 0..1 tuck the solid into its
    neighbours. ``width`` and ``depth`` scale the body's half width and half
    depth. ``slant`` is how far the front of the section drops along the
    axis, as a fraction of the segment length (0 at the back, which stays; it
    needs a :attr:`TorsoFrame.front_axis`): a slanted collar.
    """

    along: float
    width: float
    depth: float
    slant: float = 0.0
    #: Superellipse exponent here; 0 keeps the profile's (rounder at a collar).
    squareness: float = 0.0
    #: Whole section shifted to the front, as a fraction of the segment length
    #: (needs a :attr:`TorsoFrame.front_axis`): centres the collar on the neck.
    forward: float = 0.0


@dataclass(frozen=True)
class TorsoProfile:
    """Cross-sections of a trunk solid, lowest first.

    ``squareness`` is the superellipse exponent of every section: 2 is an
    ellipse, larger values flatten the front, back and flanks.
    """

    stations: tuple[TorsoStation, ...]
    squareness: float = 2.0
    rings: int = 18
    sides: int = 28
    #: Extra rings from ``fine_from`` up to the top (a finely sampled dome).
    fine_from: float = 1.0
    fine_rings: int = 0

    def __post_init__(self) -> None:
        values = [(s.along, s.width, s.depth) for s in self.stations]
        if any(s.squareness != 0.0 and s.squareness < 2.0 for s in self.stations):
            raise ValueError("torso station squareness must be 0 or at least 2")
        if any(not math.isfinite(s.slant) or s.slant < 0.0 for s in self.stations):
            raise ValueError("torso profile slants must be finite and not negative")
        if len(values) < 3 or not np.isfinite(np.asarray(values)).all():
            raise ValueError("torso profile needs at least three finite stations")
        if any(b[0] <= a[0] for a, b in zip(values, values[1:], strict=False)):
            raise ValueError("torso profile stations must rise in 'along'")
        if any(width <= 0.0 or depth <= 0.0 for _, width, depth in values):
            raise ValueError("torso profile widths and depths must be positive")
        if not (math.isfinite(self.squareness) and self.squareness >= 2.0):
            raise ValueError("torso profile squareness must be at least 2")
        if self.rings < 4 or self.sides < 8:
            raise ValueError("torso profile needs rings >= 4 and sides >= 8")

    def clipped(self, lowest: float) -> TorsoProfile:
        """This profile with nothing below ``lowest``; its section there is kept.

        A trunk solid that stands on the pelvis starts at its joint plane; one
        that stands on another trunk solid tucks into it.
        """
        kept = [s for s in self.stations if s.along > lowest]
        if len(kept) == len(self.stations):
            return self
        width, depth = self.scales_at([lowest])
        first = TorsoStation(lowest, float(width[0]), float(depth[0]))
        return dataclasses.replace(self, stations=(first, *kept))

    def scales_at(
        self, along: Sequence[float] | NDArray[np.floating]
    ) -> tuple[Array, Array]:
        """Width and depth scales at ``along``, smooth between the stations."""
        knots = [s.along for s in self.stations]
        width = PchipInterpolator(knots, [s.width for s in self.stations])
        depth = PchipInterpolator(knots, [s.depth for s in self.stations])
        at = np.clip(np.asarray(along, dtype=float), knots[0], knots[-1])
        return np.asarray(width(at)), np.asarray(depth(at))

    def squareness_at(self, along: Sequence[float] | NDArray[np.floating]) -> Array:
        """Superellipse exponent at ``along``, smooth between the stations."""
        knots = [s.along for s in self.stations]
        values = [s.squareness or self.squareness for s in self.stations]
        at = np.clip(np.asarray(along, float), knots[0], knots[-1])
        return np.asarray(PchipInterpolator(knots, values)(at))

    def forwards_at(self, along: Sequence[float] | NDArray[np.floating]) -> Array:
        """Forward shift (fraction of the length) at ``along``, smooth between."""
        knots = [s.along for s in self.stations]
        shift = PchipInterpolator(knots, [s.forward for s in self.stations])
        return np.asarray(shift(np.clip(np.asarray(along, float), knots[0], knots[-1])))

    def slants_at(self, along: Sequence[float] | NDArray[np.floating]) -> Array:
        """Front drop (fraction of the length) at ``along``, smooth between."""
        knots = [s.along for s in self.stations]
        slant = PchipInterpolator(knots, [s.slant for s in self.stations])
        return np.asarray(slant(np.clip(np.asarray(along, float), knots[0], knots[-1])))


@dataclass(frozen=True)
class TorsoAcceptance:
    """Acceptance bands of the look of the trunk (#15299).

    Breadths are (low, high) fractions of stature, each widened by
    ``norm_tolerance``: ANSUR II male means (Gordon et al. 2014, mean stature
    1.756 m) scaled by stature, which for a 1.82 m man are a chest breadth of
    330 to 350 mm, bideltoid 500 to 520, waist 310 to 330 and hip 345 to 360.
    ``min_chest_over_hip``: the chest is about as wide as the hips.
    ``min_chest_depth`` is a floor, NOT the norm (250 to 265 mm, 0.137 to
    0.146 of stature): at address the lead arm lies across the chest front, so
    a chest of the norm depth swallows it; ``max_forearm_in_chest`` (forearm
    radii a forearm may sink into the shirt at address) guards that.

    The back silhouette is read every ``back_step`` over ``back_span`` (head
    lengths about the neck point: mid-thorax to mid-neck). ``max_back_bump``:
    its largest distance behind its own chord, over the chord (a hump);
    ``max_back_jump``: its largest forward jump within ``jump_window`` (a shelf
    under the neck); ``max_neck_ahead_of_back``: how far the back of the neck
    base is in front of the upper back's own line through ``back_tangent``.
    """

    chest_breadth: tuple[float, float] = (0.181, 0.192)
    bideltoid_breadth: tuple[float, float] = (0.275, 0.286)
    waist_breadth: tuple[float, float] = (0.170, 0.181)
    hip_breadth: tuple[float, float] = (0.190, 0.198)
    norm_tolerance: float = 0.06
    min_chest_over_hip: float = 0.97
    min_chest_depth: float = 0.118
    # 1.0 before the skinned-hand forearm (#15288), which alone reads 1.0004 on
    # the unchanged torso; the deeper chest here reads 1.154 (49.9 mm).
    max_forearm_in_chest: float = 1.2
    back_span: tuple[float, float] = (-0.9, 0.08)
    back_step: float = 0.02
    max_back_bump: float = 0.08
    jump_window: float = 0.08
    max_back_jump: float = 0.11
    back_tangent: tuple[float, float] = (-0.36, -0.16)
    max_neck_ahead_of_back: float = 0.06

    def __post_init__(self) -> None:
        bands = (
            self.chest_breadth,
            self.bideltoid_breadth,
            self.waist_breadth,
            self.hip_breadth,
            self.back_span,
            self.back_tangent,
        )
        if any(not low < high for low, high in bands):
            raise ValueError("torso acceptance bands must rise from low to high")
        limits = (
            self.norm_tolerance,
            self.min_chest_over_hip,
            self.min_chest_depth,
            self.max_forearm_in_chest,
            self.back_step,
            self.max_back_bump,
            self.jump_window,
            self.max_back_jump,
            self.max_neck_ahead_of_back,
        )
        if not (np.isfinite(limits).all() and min(limits) > 0.0):
            raise ValueError("torso acceptance limits must be positive and finite")


TORSO_ACCEPTANCE = TorsoAcceptance()


@dataclass(frozen=True)
class ShoulderSlope:
    """The top of the thorax, from the neck to the point of the shoulder (#15299).

    The shoulder joint is not at the same height under the neck in every
    subject, so the slope is built on the subject's own shoulder line instead
    of a fixed share of the thorax. The point of the shoulder (acromion) is
    ``acromion_below_neck_m`` under the neck point at ``reference_stature_m``,
    held to ``rise_m`` (low, high) above the joint centre. The thorax runs
    straight from the collar to its own width there: ``max_point_width`` (of
    its half width), or less where the line would be flatter than
    ``min_slope_deg``. Under the point it rounds over ``corner`` ((share of
    the length below the point, wider by), highest first) into the chest,
    ``chest_below`` under the point. ``squareness`` is the exponent of those
    sections, from the corner to the collar (2 is an ellipse): rounder than
    the chest, so the upper back has no edges. ``point_lift_m``
    is how far the collar's slant drops the slope: the thorax point is set that
    much higher. The point stays ``collar_clear`` (share of the collar-to-joint
    height) under the collar. Visual only.
    """

    acromion_below_neck_m: float = 0.062
    reference_stature_m: float = 1.82
    rise_m: tuple[float, float] = (0.0, 0.035)
    point_lift_m: float = 0.015
    max_point_width: float = 1.04
    min_slope_deg: float = 19.0
    corner: tuple[tuple[float, float], ...] = (
        (0.012, 0.07),
        (0.025, 0.125),
        (0.08, 0.185),
    )
    chest_below: float = 0.16
    slope_knots: int = 3
    squareness: tuple[float, float] = (2.35, 2.0)
    collar_clear: float = 0.25

    def __post_init__(self) -> None:
        low, high = self.rise_m
        if min(self.acromion_below_neck_m, self.reference_stature_m) <= 0.0:
            raise ValueError("shoulder slope lengths must be positive")
        if not (math.isfinite(high) and high >= low >= 0.0 <= self.point_lift_m):
            raise ValueError("shoulder rise must be 0 <= low <= high, lift >= 0")
        if not 0.0 < self.collar_clear < 1.0:
            raise ValueError("the point of the shoulder must stay under the collar")
        if not (self.max_point_width > 0.0 and 0.0 < self.min_slope_deg < 90.0):
            raise ValueError("shoulder slope needs a width > 0, an angle in (0, 90)")
        below = [share for share, _ in self.corner]
        if below != sorted(below) or not (below and self.chest_below > below[-1]):
            raise ValueError("shoulder corner must deepen down to the chest")
        if self.slope_knots < 1 or min(self.squareness) < 2.0:
            raise ValueError("shoulder slope needs knots >= 1 and squareness >= 2")

    def rise(self, drop_m: float, stature_m: float | None = None) -> float:
        """Top of the shoulder above its joint, for a joint ``drop_m`` under the neck."""
        scale = (stature_m or self.reference_stature_m) / self.reference_stature_m
        if not (math.isfinite(drop_m) and drop_m >= 0.0 and scale > 0.0):
            raise ValueError("shoulder drop must be >= 0 and the stature positive")
        low, high = (scale * limit for limit in self.rise_m)
        return min(high, max(low, drop_m - scale * self.acromion_below_neck_m))

    def lift(self, stature_m: float | None = None) -> float:
        """Thorax point above the top of the shoulder (m): the collar slant's drop."""
        return (
            self.point_lift_m
            * (stature_m or self.reference_stature_m)
            / (self.reference_stature_m)
        )

    def stations(
        self,
        base: Sequence[TorsoStation],
        shoulder: float,
        rise: float,
        slenderness: float,
    ) -> tuple[TorsoStation, ...]:
        """``base`` with its widths above the chest rebuilt on the shoulder line.

        ``shoulder`` is the joints' share of the thorax length, ``rise`` the
        point of the shoulder above it (same unit) and ``slenderness`` the
        length over the half width. Depth, slant and forward shift stay those
        of ``base`` at the same height, and the corner is never narrower than
        ``base``. Postcondition: widths never grow
        upwards from the chest; the last station is the collar of ``base``.
        """
        collar = base[-1]
        if not (0.0 < shoulder < collar.along and rise >= 0.0 and slenderness > 0.0):
            raise ValueError("shoulder line must lie inside the thorax profile")
        clear = self.collar_clear * (collar.along - shoulder)
        point = min(shoulder + rise, collar.along - clear)
        run = (collar.along - point) * slenderness
        reach = run / math.tan(math.radians(self.min_slope_deg))
        width = max(collar.width, min(self.max_point_width, collar.width + reach))
        chest = max(station.width for station in base)
        at = [s.along for s in base]
        knots = [(point - self.chest_below, chest)]
        for below, wider in self.corner[::-1]:  # never narrower than ``base``
            was = float(np.interp(point - below, at, [s.width for s in base]))
            knots.append((point - below, min(chest, max(was, width + wider))))
        for i in range(self.slope_knots + 1):
            u = i / self.slope_knots
            knots.append(
                (point + u * (collar.along - point), width + u * (collar.width - width))
            )
        rows = [s for s in base if s.along < knots[0][0]]
        corner = point - self.corner[-1][0]
        for along, wide in knots:
            depth, slant, forward = (
                float(np.interp(along, at, [getattr(s, name) for s in base]))
                for name in ("depth", "slant", "forward")
            )
            square = 0.0  # the profile's own squareness, below the corner
            if along >= corner:
                u = (along - corner) / (collar.along - corner)
                square = self.squareness[0] + u * (
                    self.squareness[1] - self.squareness[0]
                )
            rows.append(TorsoStation(along, wide, depth, slant, square, forward))
        return tuple(rows)


SHOULDER_SLOPE = ShoulderSlope()


@dataclass(frozen=True)
class TorsoFrame:
    """Where a trunk solid sits: its axis and its two half extents (metres)."""

    start: Array
    end: Array
    width_axis: Array
    half_width: float
    half_depth: float
    #: Which way is front (any length); ``None`` ignores every station slant.
    front_axis: Array | None = None


def _unit_section(squareness: float, sides: int) -> tuple[Array, Array]:
    """Width and depth coordinates of a unit superellipse, ``sides`` points."""
    angle = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    power = 2.0 / squareness
    cos, sin = np.cos(angle), np.sin(angle)
    return (
        np.sign(cos) * np.abs(cos) ** power,
        np.sign(sin) * np.abs(sin) ** power,
    )


def _front_drop(unit_d: Array, front_sign: float) -> Array:
    """0 at the back, 1 at the front of a section (smoothstep between)."""
    towards_front = np.clip(0.5 * (1.0 + front_sign * unit_d), 0.0, 1.0)
    return np.asarray(towards_front**2 * (3.0 - 2.0 * towards_front))


def torso_mesh(
    frame: TorsoFrame,
    profile: TorsoProfile,
    *,
    thickness: float = 1.0,
    cap_inset_m: float = 0.0,
) -> Mesh:
    """The closed solid of ``profile`` on ``frame``.

    Preconditions: finite frame with positive length and half extents; the
    width axis is not along the segment; ``thickness > 0`` (a garment is the
    same solid, thicker). Postconditions: closed, positive volume; at a
    station the half width is ``thickness * half_width * station.width`` along
    the width axis and likewise in depth; the solid is mirror-symmetric about
    the plane through the axis normal to the width axis. ``cap_inset_m >= 0``
    lowers the top ring and pole along the axis, so a skin solid's flat top
    sits under its garment's (no coplanar z-fighting at an open neckline).
    """
    start, end = np.asarray(frame.start, float), np.asarray(frame.end, float)
    axis = end - start
    length = float(np.linalg.norm(axis))
    extents = (frame.half_width, frame.half_depth, thickness, length)
    if not (np.isfinite(extents).all() and min(extents) > 0.0 and cap_inset_m >= 0.0):
        raise ValueError("torso frame needs positive finite length and extents")
    axis = axis / length
    width = np.asarray(frame.width_axis, float)
    width = width - (width @ axis) * axis
    if not np.linalg.norm(width) > 1e-9:
        raise ValueError("torso width axis must not lie along the segment")
    width = width / np.linalg.norm(width)
    depth = np.cross(axis, width)
    front_sign = 0.0
    if frame.front_axis is not None:
        front_sign = float(np.sign(depth @ np.asarray(frame.front_axis, float)))
    lowest, highest = profile.stations[0].along, profile.stations[-1].along
    along = np.linspace(lowest, highest, profile.rings)
    if profile.fine_rings:
        fine = np.linspace(max(profile.fine_from, lowest), highest, profile.fine_rings)
        along = np.unique(np.concatenate([along, fine]))
    scale_w, scale_d = profile.scales_at(along)
    sections = [_unit_section(q, profile.sides) for q in profile.squareness_at(along)]
    slants = profile.slants_at(along) * length
    shifts = np.outer(profile.forwards_at(along) * length, depth * front_sign)
    rows = [
        start
        + t * length * axis
        + thickness * frame.half_width * w * np.outer(uw, width)
        + thickness * frame.half_depth * d * np.outer(ud, depth)
        - np.outer(_front_drop(ud, front_sign) * q, axis)
        + shift
        for t, w, d, q, shift, (uw, ud) in zip(
            along, scale_w, scale_d, slants, shifts, sections, strict=True
        )
    ]
    rows[-1] = rows[-1] - cap_inset_m * axis
    return closed_grid_mesh(
        np.asarray(rows),
        start + lowest * length * axis,
        start
        + highest * length * axis
        - (cap_inset_m + 0.5 * slants[-1]) * axis
        + shifts[-1],
    )


__all__ = [
    "SHOULDER_SLOPE",
    "TORSO_ACCEPTANCE",
    "ShoulderSlope",
    "TorsoAcceptance",
    "TorsoFrame",
    "TorsoProfile",
    "TorsoStation",
    "torso_mesh",
]
