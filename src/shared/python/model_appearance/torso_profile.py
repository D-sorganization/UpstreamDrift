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


__all__ = ["TorsoFrame", "TorsoProfile", "TorsoStation", "torso_mesh"]
