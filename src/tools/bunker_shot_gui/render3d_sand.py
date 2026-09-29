"""Drawing a sand volume in the matplotlib 3-D scene (issue #8729, epic #8699).

Headless: matplotlib and no GUI toolkit, like every other renderer in this
package, so the same frame is produced in a test, by a batch sweep, or
inside the Qt workbench.

Why sheets rather than a smooth cloud
-------------------------------------

F1 solves one plane-strain section. :mod:`.sandvolume` sweeps it across
the declared effective width, and the sweep is an **extrusion** -- every
sheet is the same solve repeated, because plane strain has no heel-to-toe
direction to vary along.

Blending those sheets into a smooth continuum would draw exactly the
picture the model does not support: a solved volume with across-width
structure. So they are drawn as discrete, separated sheets with air
between them. The repetition is meant to be visible. A viewer who notices
that the third sheet is identical to the first has read the model
correctly, and the in-frame label says the same thing in words.

Why only moving sand is painted
-------------------------------

A bunker bed is mostly still. Painting every occupied cell fills the box
with a uniform slab and hides the one thing the view exists to show,
which is where the sand is *going*. Cells below a floor on the injected
ramp are therefore left out of the collection entirely -- not drawn
transparent, dropped -- so the frame is the moving material and the draw
cost falls with it.

That floor is a fraction of the **shared** ramp, never of this frame's own
peak. A per-frame floor would make every frame look equally busy and
would be issue #8728's defect wearing a different hat.

Why the artists are built once
------------------------------

Rebuilding a 3-D collection per frame costs about a quarter of a second;
mutating one costs under a millisecond. Both artists here are created
empty and updated through public setters only -- ``set_verts`` and
``set_facecolors`` on the sheets, ``set_segments`` on the arrows -- so an
animation cannot break on a matplotlib upgrade in a picture that still
looks plausible.
"""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy.typing import NDArray

from bunkershot3d.fields.schema import FieldQuantity

from .sandvolume import ArrowLattice, SandVolume, SandVolumeScale

__all__ = [
    "PAINT_FLOOR",
    "SandVolumeArtists",
    "arrow_segments_mm",
    "sheet_quads_mm",
]

_MM_PER_M = 1e3

PAINT_FLOOR = 0.05
"""Normalised value below which a cell is not drawn at all.

A fraction of the *injected* ramp, so the same physical speed is painted
or dropped in every frame and in every compared design."""

_ALPHA_LOW = 0.20
_ALPHA_HIGH = 0.92
"""Opacity ramp across the painted range.

Slow sand is nearly clear and fast sand is nearly solid, so depth reads
without the fastest material being hidden behind the slowest."""

_CELL_FILL = 0.86
"""Fraction of a lattice cell each quad spans.

Leaving a hairline of air between quads is what stops a painted region
reading as one poured solid."""

_ARROW_COLUMNS = 14
_ARROW_ROWS = 9
_ARROW_SPAN_FRACTION = 0.055
"""Longest arrow, as a fraction of the lattice's along-track span.

Fixed, and scaled by the injected ramp's peak rather than the frame's, so
an arrow of a given length means the same speed in every frame and in
every compared design."""

_ARROW_HEAD_FRACTION = 0.34
_ARROW_HEAD_ANGLE_RAD = 0.42
_ARROW_COLOUR = "#ffffff"
_ARROW_EDGE = "#101010"
_ARROW_WIDTH = 1.0
_ARROW_EDGE_WIDTH = 2.6
"""Arrows are drawn twice: a dark wide stroke, then a light narrow one.

A single colour is unreadable here whatever it is. The ramp runs from
near-black slow sand to bright ejecta and the panel behind it is pale, so
a light arrow vanishes on the plume and a dark one vanishes in the bed.
The first render of a captured field showed no arrows at all for exactly
this reason. An outlined stroke reads on all three."""

_ARROW_FLOOR = 0.04
_ARROW_LIFT_MM = 0.6
"""How far in front of the nearest sheet the arrows sit.

mplot3d sorts whole collections rather than individual faces, so arrows
sharing a sheet's plane are as likely to be drawn behind it as in front.
Lifting them clear toward the eye is what makes them reliably visible."""

_SHEET_EDGE = "#00000000"
_MIN_SPAN_M = 1e-6


class SandVolumeArtists:
    """The sand field's artists in one 3-D panel, built once and mutated.

    Attributes are read-only properties; every per-frame change goes
    through :meth:`update`.
    """

    def __init__(
        self,
        axes: Axes3D,
        volume: SandVolume,
        scale: SandVolumeScale,
        *,
        quantity: FieldQuantity = FieldQuantity.VELOCITY,
    ) -> None:
        """Add the empty sand artists to a 3-D axes.

        Args:
            axes: The 3-D axes to draw into.
            volume: The extruded sand field.
            scale: The fixed colour ramp, from
                :func:`~.sandvolume.sand_volume_scale`. Injected, never
                inferred here: two designs each normalised to their own
                peak are issue #8728's defect.
            quantity: Which channel to paint.

        Raises:
            ValueError: If the quantity is not one this view paints.
        """
        self._axes = axes
        self._volume = volume
        self._scale = scale
        self._quantity = quantity
        self._colormap = colormaps[scale.colormap_name(quantity)]
        self._frame = 0
        self._n_painted = 0
        self._painted_values: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        self._painted_across_m: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        self._arrow_segments: NDArray[np.float64] = np.zeros((0, 2, 3))

        self._quads_mm, self._across_mm = sheet_quads_mm(volume)
        self._sheets = Poly3DCollection(
            np.zeros((0, 4, 3), dtype=np.float64),
            edgecolor=_SHEET_EDGE,
            linewidths=0.0,
            # ``zsort="average"`` rather than "max": the sheets interleave
            # in depth with the head, and sorting on a quad's furthest
            # corner puts whole sheets in front of material they are
            # behind.
            zsort="average",
            # ``FieldQuantity.label`` already reads "sand speed [m/s]".
            label=f"{quantity.label}, {volume.fidelity.label}",
        )
        # ``autolim=False``: the world box is injected from SceneScale and
        # autoscale is off, so a collection that re-ranged the axes would
        # reintroduce exactly the per-design reframing issue #8728 removed.
        # An empty collection cannot be auto-limited at all.
        axes.add_collection3d(self._sheets, autolim=False)

        # Drawn twice for legibility; see _ARROW_EDGE_WIDTH.
        self._arrow_edge = Line3DCollection(
            [],
            colors=_ARROW_EDGE,
            linewidths=_ARROW_EDGE_WIDTH,
            zorder=6.0,
        )
        axes.add_collection3d(self._arrow_edge, autolim=False)
        self._arrows = Line3DCollection(
            [],
            colors=_ARROW_COLOUR,
            linewidths=_ARROW_WIDTH,
            zorder=7.0,
            label="sand flow direction (in the solved plane only)",
        )
        axes.add_collection3d(self._arrows, autolim=False)
        self._eye = np.array([0.0, -1.0, 0.0], dtype=np.float64)

    # ----------------------------------------------------------- accessors

    @property
    def quantity(self) -> FieldQuantity:
        """Which channel is painted."""
        return self._quantity

    @property
    def scale(self) -> SandVolumeScale:
        """The fixed ramp in force."""
        return self._scale

    @property
    def volume(self) -> SandVolume:
        """The field being drawn."""
        return self._volume

    @property
    def n_painted(self) -> int:
        """Quads drawn in the current frame."""
        return self._n_painted

    @property
    def painted_values(self) -> NDArray[np.float64]:
        """The normalised values behind the current frame's colours."""
        return self._painted_values

    @property
    def painted_across_m(self) -> NDArray[np.float64]:
        """World ``y`` of every quad drawn in the current frame [m]."""
        return self._painted_across_m

    @property
    def n_arrows(self) -> int:
        """Direction arrows drawn in the current frame."""
        return int(self._arrow_segments.shape[0])

    @property
    def arrow_segments(self) -> NDArray[np.float64]:
        """``(n, 2, 3)`` arrow shafts of the current frame, in millimetres."""
        return self._arrow_segments

    def set_eye_direction(self, eye: NDArray[np.float64]) -> None:
        """Point the arrow sheet at the camera.

        The arrows live on one sheet, not all five: the same vectors drawn
        five times over is true to the extrusion and unreadable. Which
        sheet has to follow the eye, or the four in front of it hide the
        one carrying the direction.

        Args:
            eye: Unit vector from the subject toward the eye, world axes.
        """
        self._eye = np.asarray(eye, dtype=np.float64).reshape(3)
        self._update_arrows(self._frame)

    def viewing_note(self) -> str:
        """How this camera is cutting the extrusion, in words.

        Delegated to the volume, which both backends ask, so the fallback
        and the VTK upgrade cannot drift into qualifying the same picture
        differently.

        Returns:
            One line for the caption.
        """
        return self._volume.viewing_note(self._eye)

    def legend_label(self) -> str:
        """What the colour ramp means, with its fixed limits.

        Returns:
            A line naming the channel, its unit and the ramp's own
            range -- which is the shared range, so two designs read
            against the same numbers.
        """
        low, high = self._scale.limits(self._quantity)
        # ``FieldQuantity.label`` already reads "sand speed [m/s]"; prefixing
        # it again produced "sand sand speed [m/s]" in the first rendered
        # frame, which is the sort of thing only looking at the picture finds.
        return (
            f"{self._quantity.label} ramp {low:.3g} to {high:.3g}, "
            "fixed across frames and designs"
        )

    # -------------------------------------------------------------- drawing

    def update(self, frame: int) -> None:
        """Show one field frame.

        Args:
            frame: The field frame index -- the sand's own clock, not the
                shot's. :class:`~.slices.CursorMap` is what maps one onto
                the other.

        Raises:
            ValueError: If the frame is outside the field's record.
        """
        volume = self._volume
        values = volume.channel(self._quantity, frame)
        self._frame = int(frame)
        normalised = self._scale.normalise(self._quantity, values)
        # ``nan`` is air. It must not survive into a comparison, which
        # would paint it as the ramp's floor -- still sand where there is
        # no sand at all.
        painted = np.nan_to_num(normalised, nan=-1.0)
        keep_all = np.tile((painted >= PAINT_FLOOR).ravel(), volume.n_sheets)

        self._sheets.set_verts(self._quads_mm[keep_all])
        strength = np.tile(painted.ravel(), volume.n_sheets)[keep_all]
        colours = self._colormap(strength)
        colours[:, 3] = _ALPHA_LOW + (_ALPHA_HIGH - _ALPHA_LOW) * strength
        self._sheets.set_facecolors(colours)
        self._n_painted = int(keep_all.sum())
        self._painted_values = strength
        self._painted_across_m = self._across_mm[keep_all] / _MM_PER_M

        self._update_arrows(frame)

    def _update_arrows(self, frame: int) -> None:
        """Redraw the direction arrows on the sheet nearest the eye.

        Drawn on one sheet only. An arrow field repeated on every sheet
        is the same vectors five times over -- true to the extrusion, and
        unreadable -- so one sheet carries them and the label says the
        flow is in the solved plane.
        """
        volume = self._volume
        lattice = volume.arrows(frame, n_along=_ARROW_COLUMNS, n_up=_ARROW_ROWS)
        segments = arrow_segments_mm(
            lattice,
            across_mm=self._arrow_across_mm(),
            span_mm=self._arrow_span_mm(),
            peak_m_s=self._scale.speed_m_s[1],
        )
        listed = list(segments)
        self._arrows.set_segments(listed)
        self._arrow_edge.set_segments(listed)
        # Only the shafts are reported: the barbs are decoration, and a
        # test that measured them would be measuring the arrowhead angle.
        self._arrow_segments = segments[::3] if segments.size else segments

    def _arrow_across_mm(self) -> float:
        """World ``y`` of the arrow sheet: the one nearest the eye [mm]."""
        across = self._volume.across_m * _MM_PER_M
        toward_viewer = float(self._eye[1]) >= 0.0
        nearest = float(across.max()) if toward_viewer else float(across.min())
        lift = _ARROW_LIFT_MM if toward_viewer else -_ARROW_LIFT_MM
        return nearest + lift

    def _arrow_span_mm(self) -> float:
        """The longest arrow this lattice draws, in millimetres."""
        along = self._volume.along_m
        span = float(np.ptp(along)) if along.size > 1 else _MIN_SPAN_M
        return max(span, _MIN_SPAN_M) * _MM_PER_M * _ARROW_SPAN_FRACTION


def sheet_quads_mm(
    volume: SandVolume,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Every sheet's quads, in millimetres, and the ``y`` of each.

    Shared by both backends: the sheets a PyVista frame draws are the
    same quads at the same coordinates the matplotlib frame draws, so the
    upgrade cannot quietly show a different volume from the fallback.

    The lattice never moves, so this is built once. Each quad is centred
    on a lattice node and spans a fraction of a cell, leaving a hairline
    of air between neighbours: a painted region with no gaps in it reads
    as one poured solid rather than as sand.

    Returns:
        ``(quads, across_mm)`` with shapes ``(ny * nx * nz, 4, 3)`` and
        ``(ny * nx * nz,)``.
    """
    along = volume.along_m * _MM_PER_M
    up = volume.up_m * _MM_PER_M
    half_x = _half_cell(along)
    half_z = _half_cell(up)
    grid_x, grid_z = np.meshgrid(along, up, indexing="ij")
    centres_x = grid_x.ravel()
    centres_z = grid_z.ravel()
    corners = np.array(
        [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)], dtype=np.float64
    )
    quad_x = centres_x[:, None] + corners[None, :, 0] * half_x
    quad_z = centres_z[:, None] + corners[None, :, 1] * half_z

    sheets: list[NDArray[np.float64]] = []
    across: list[NDArray[np.float64]] = []
    for offset_m in volume.across_m:
        offset_mm = float(offset_m) * _MM_PER_M
        quad = np.stack([quad_x, np.full_like(quad_x, offset_mm), quad_z], axis=-1)
        sheets.append(quad)
        across.append(np.full(quad.shape[0], offset_mm, dtype=np.float64))
    return np.concatenate(sheets, axis=0), np.concatenate(across, axis=0)


def _half_cell(axis_mm: NDArray[np.float64]) -> float:
    """Half a lattice cell on one axis, in millimetres."""
    if axis_mm.size < 2:
        return _MIN_SPAN_M * _MM_PER_M
    return 0.5 * _CELL_FILL * float(np.abs(np.diff(axis_mm)).min())


def arrow_segments_mm(
    lattice: ArrowLattice,
    *,
    across_mm: float,
    span_mm: float,
    peak_m_s: float,
) -> NDArray[np.float64]:
    """Arrow polylines for one frame, as ``(3n, 2, 3)`` segments in mm.

    Three segments per arrow -- a shaft and two barbs -- rather than a
    quiver primitive, so both backends draw the *same* arrows from the
    same numbers: matplotlib can mutate a
    :class:`~mpl_toolkits.mplot3d.art3d.Line3DCollection` through the
    public ``set_segments`` where a 3-D quiver has no equivalent, and
    PyVista takes line segments directly.

    The length is set by the *injected* peak, so an arrow of a given
    length means the same speed in every frame and every compared design.
    """
    speed = lattice.speed_m_s
    live = lattice.occupied & (speed > _ARROW_FLOOR * max(peak_m_s, _MIN_SPAN_M))
    if not np.any(live):
        return np.zeros((0, 2, 3), dtype=np.float64)

    grid_x, grid_z = np.meshgrid(
        lattice.along_m * _MM_PER_M, lattice.up_m * _MM_PER_M, indexing="ij"
    )
    base_x = grid_x[live]
    base_z = grid_z[live]
    length = span_mm * np.clip(speed[live] / max(peak_m_s, _MIN_SPAN_M), 0.0, 1.0)
    magnitude = np.maximum(speed[live], _MIN_SPAN_M)
    unit_x = lattice.velocity_along_m_s[live] / magnitude
    unit_z = lattice.velocity_up_m_s[live] / magnitude
    tip_x = base_x + unit_x * length
    tip_z = base_z + unit_z * length

    shafts = _segment_stack(base_x, base_z, tip_x, tip_z, across_mm)
    head = length * _ARROW_HEAD_FRACTION
    angle = _ARROW_HEAD_ANGLE_RAD
    barbs = []
    for sign in (1.0, -1.0):
        cos, sin = np.cos(sign * angle), np.sin(sign * angle)
        back_x = -(unit_x * cos - unit_z * sin) * head
        back_z = -(unit_x * sin + unit_z * cos) * head
        barbs.append(
            _segment_stack(tip_x, tip_z, tip_x + back_x, tip_z + back_z, across_mm)
        )
    # Interleaved shaft, barb, barb so ``segments[::3]`` is the shafts.
    stacked = np.stack([shafts, barbs[0], barbs[1]], axis=1)
    return stacked.reshape(-1, 2, 3)


def _segment_stack(
    from_x: NDArray[np.float64],
    from_z: NDArray[np.float64],
    to_x: NDArray[np.float64],
    to_z: NDArray[np.float64],
    across_mm: float,
) -> NDArray[np.float64]:
    """``(n, 2, 3)`` segments at one constant world ``y``."""
    across = np.full_like(from_x, across_mm)
    start = np.stack([from_x, across, from_z], axis=-1)
    end = np.stack([to_x, across, to_z], axis=-1)
    return np.stack([start, end], axis=1)
