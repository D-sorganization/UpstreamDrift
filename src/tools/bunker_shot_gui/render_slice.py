"""Drawing a cut through the sand (issue #8711).

Headless: matplotlib and no GUI toolkit, so the same figure comes out of a
test, a batch sweep, or the Qt workbench.

Three panels, one cut
---------------------

Speed, density and shear rate share the cut's axes and its cursor.  They
are three panels rather than three overlays on one because three scalar
fields stacked on a single axes is a picture nobody can read a number
off, and the point of this view is reading numbers off it.

The velocity panel carries **both** magnitude and direction: the colour
is the speed, the arrows are the flow.  Sand pushed ahead of the sole and
sand riding up the face can carry identical speeds, so a magnitude-only
heatmap would hide the distinction the whole view exists to make.  Arrow
*length* is scaled by the same injected :class:`~.slices.SliceScale` as
the colour, so an arrow means the same speed in every frame and in every
design being compared.

What the stamp has to say
-------------------------

More than the other views, because a field picture is the most persuasive
thing this package produces and the least validated:

* status and tier, through the shared :func:`~.render.stamp_axes`;
* what this cut *is* -- solved, extruded or projected -- since a
  plane-strain tier repeats one solution at every heel-to-toe station;
* whether a through-cut velocity exists at all;
* how far outside the published corpus the query sits.
  ``MAX_VALIDATED_SPEED_M_S`` is 1.44 m/s, so a 25 m/s shot is outside it
  from its first sample and stays there;
* how the shared cursor maps onto this field's own frames, because an F1
  field is a strided march of a declared approach and not the shot's
  clock.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.quiver import Quiver
from matplotlib.text import Text
from numpy.typing import NDArray

from bunkershot3d.fields.schema import SandFieldSeries

from .render import stamp_axes
from .slices import (
    DENSITY_COLORMAP,
    SHEAR_COLORMAP,
    SPEED_COLORMAP,
    CursorMap,
    CuttingPlane,
    SliceSample,
    SliceScale,
    body_focus_bounds_m,
    sample_plane,
    slice_scale,
)

__all__ = [
    "SliceArtists",
    "draw_slice_frame",
    "slice_still",
]

_MM_PER_M = 1.0e3
_ARROW_COLUMNS = 22
_ARROW_ROWS = 15
_ARROW_SCALE = 26.0
"""Arrows per figure width at the scale's peak speed.

Fixed rather than derived from the frame, for the same reason the colour
limits are: an arrow whose length meant something different each frame
would be a per-frame autoscale with a different geometry."""

_SAMPLES_ALONG = 180
_SAMPLES_UP = 140

_STAMP_FONT_PT = 5.0
_STAMP_WRAP_CHARS = 104
_STAMP_HEADROOM = 0.42
"""Extra vertical range above the cut, as a fraction of its own span.

The stamp is inside the axes on purpose -- a caption can be cropped off a
screenshot and the picture is far more persuasive than the caption. But a
stamp that covered the flow it qualifies would get the same treatment for
a better reason, so the panel is given air to put it in."""


def _ramp(name: str) -> mpl.colors.Colormap:
    """A colormap that draws ``nan`` as nothing at all.

    Empty sand is ``nan`` throughout this package, and a colormap that
    painted it the low end of the ramp would draw stationary sand where
    the solve had no sand.
    """
    return mpl.colormaps[name].with_extremes(bad=(0.0, 0.0, 0.0, 0.0))


@dataclass(frozen=True)
class _Panel:
    """One quantity's mesh and its axes."""

    axes: Axes
    mesh: QuadMesh
    outline: Line2D


class SliceArtists:
    """The artists of one cut, mutated frame by frame.

    Built once and updated in place, like every other animated view in
    this package: the limits, the colour bars and the axes are set from
    the injected scale at construction and never touched again, so a
    frame that looks faster *is* faster.

    Attributes are read-only properties; the only mutation entry point is
    :meth:`update`.
    """

    def __init__(
        self,
        figure: Figure,
        series: SandFieldSeries,
        plane: CuttingPlane,
        scale: SliceScale,
        *,
        cursor: CursorMap | None = None,
        n_along: int = _SAMPLES_ALONG,
        n_up: int = _SAMPLES_UP,
        bounds_m: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ) -> None:
        """Lay out the panels and draw the opening frame.

        Args:
            figure: The figure to draw into; cleared first.
            series: The sand field.
            plane: The cut.
            scale: Colour limits, covering every frame and every design
                being compared. Injected, never inferred.
            cursor: How an external transport's frame index maps onto
                this field's frames. Defaults to the identity map over
                this field's own frames.
            n_along: Samples along the cut.
            n_up: Samples up the cut.
            bounds_m: ``(along, up)`` window of the cut to draw. Defaults
                to the intruder's travel plus a margin, because an F1 bed
                runs in and out far beyond the impact zone and a cut over
                the whole of it puts the interesting part in the middle
                third. The axes stay labelled in millimetres, so the
                window is visible rather than hidden.
        """
        figure.clear()
        self._series = series
        self._plane = plane
        self._scale = scale
        self._n_along = int(n_along)
        self._n_up = int(n_up)
        self._bounds = (
            bounds_m if bounds_m is not None else body_focus_bounds_m(series, plane)
        )
        self._cursor = cursor or CursorMap(
            n_transport=series.n_frames, n_field=series.n_frames
        )
        self._frame = 0

        sample = self._sample(0)
        self._panels: dict[str, _Panel] = {}
        rows = 2 + (1 if sample.shear_rate_1_s is not None else 0)
        axes_list = figure.subplots(rows, 1, sharex=True, sharey=True)
        panes = list(np.atleast_1d(axes_list))

        self._panels["speed"] = self._add_panel(
            panes[0],
            sample,
            sample.speed_m_s,
            SPEED_COLORMAP,
            scale.speed_m_s,
            f"sand speed [{scale.speed_unit}]",
        )
        self._quiver = self._add_quiver(panes[0], sample)
        self._panels["density"] = self._add_panel(
            panes[1],
            sample,
            sample.masked_density_kg_m3,
            DENSITY_COLORMAP,
            scale.density_kg_m3,
            f"sand density [{scale.density_unit}]",
        )
        if sample.shear_rate_1_s is not None and scale.shear_rate_1_s is not None:
            self._panels["shear"] = self._add_panel(
                panes[2],
                sample,
                sample.masked_shear_rate_1_s,
                SHEAR_COLORMAP,
                scale.shear_rate_1_s,
                f"shear rate [{scale.shear_unit}]",
            )
        panes[-1].set_xlabel(
            f"along the cut, leading edge -> trailing [mm] ({plane.name})"
        )
        for pane in panes:
            pane.set_ylabel("above the free surface [mm]")
            pane.set_autoscale_on(False)

        # Headroom above the sand so the stamp sits over air rather than over
        # the flow it qualifies. Set once, like every other limit here.
        lower, upper = panes[0].get_ylim()
        panes[0].set_ylim(lower, upper + _STAMP_HEADROOM * (upper - lower))
        self._stamp: Text = stamp_axes(
            panes[0],
            series.provenance.envelope_status,
            series.provenance.fidelity_tier,
            extra=self._extra(sample),
        )
        self._stamp.set_fontsize(_STAMP_FONT_PT)
        figure.tight_layout()

    # ------------------------------------------------------------- reading

    @property
    def frame_index(self) -> int:
        """Which field frame is currently drawn."""
        return self._frame

    @property
    def n_frames(self) -> int:
        """Frames in the field."""
        return self._series.n_frames

    @property
    def n_panels(self) -> int:
        """How many quantity panels were drawn."""
        return len(self._panels)

    @property
    def scale(self) -> SliceScale:
        """The injected colour scale."""
        return self._scale

    @property
    def cursor(self) -> CursorMap:
        """The transport-to-field frame mapping."""
        return self._cursor

    @property
    def plane(self) -> CuttingPlane:
        """The cut being drawn."""
        return self._plane

    # ------------------------------------------------------------ mutation

    def update(self, frame: int) -> None:
        """Redraw at one field frame.

        Args:
            frame: Field frame index.

        Raises:
            BunkerShot3DValueError: If the frame is outside the field.
        """
        self._series.require_frame(frame)
        self._frame = int(frame)
        sample = self._sample(self._frame)
        self._panels["speed"].mesh.set_array(_mesh_values(sample.speed_m_s))
        self._panels["density"].mesh.set_array(
            _mesh_values(sample.masked_density_kg_m3)
        )
        shear = sample.masked_shear_rate_1_s
        if "shear" in self._panels and shear is not None:
            self._panels["shear"].mesh.set_array(_mesh_values(shear))
        along, up = _arrow_components(sample, self._quiver_index)
        self._quiver.set_UVC(along, up)
        for panel in self._panels.values():
            _set_outline(panel.outline, sample)
        self._stamp.set_text(self._extra_with_stamp(sample))

    def follow_transport(self, transport_frame: int) -> int:
        """Redraw at whatever field frame a transport frame maps onto.

        Args:
            transport_frame: Index in the shared transport's record.

        Returns:
            The field frame that was drawn.
        """
        field_frame = self._cursor.field_frame(transport_frame)
        self.update(field_frame)
        return field_frame

    # ------------------------------------------------------------ internals

    def _sample(self, frame: int) -> SliceSample:
        """Resample the cut at one frame."""
        along_bounds, up_bounds = (None, None) if self._bounds is None else self._bounds
        return sample_plane(
            self._series,
            frame,
            self._plane,
            n_along=self._n_along,
            n_up=self._n_up,
            along_bounds_m=along_bounds,
            up_bounds_m=up_bounds,
        )

    def _add_panel(
        self,
        axes: Axes,
        sample: SliceSample,
        values: NDArray[np.float64] | None,
        colormap: str,
        limits: tuple[float, float],
        label: str,
    ) -> _Panel:
        """One quantity's mesh, colour bar and body outline."""
        if values is None:  # pragma: no cover - callers check first
            raise ValueError(f"{label} has no values to draw")
        mesh = axes.pcolormesh(
            sample.along_m * _MM_PER_M,
            sample.up_m * _MM_PER_M,
            _mesh_grid(values),
            cmap=_ramp(colormap),
            shading="nearest",
        )
        # Set once, from the injected scale. Nothing here autoscales.
        mesh.set_clim(*limits)
        parent = axes.get_figure()
        if parent is None:  # pragma: no cover - the axes was just created here
            raise ValueError("a slice panel must belong to a figure")
        bar = parent.colorbar(mesh, ax=axes, pad=0.02)
        bar.set_label(label, fontsize=7)
        bar.ax.tick_params(labelsize=6)
        (outline,) = axes.plot(
            [], [], color="#111111", linewidth=1.2, zorder=4.0, label="club section"
        )
        _set_outline(outline, sample)
        axes.tick_params(labelsize=7)
        return _Panel(axes=axes, mesh=mesh, outline=outline)

    def _add_quiver(self, axes: Axes, sample: SliceSample) -> Quiver:
        """Direction arrows on the speed panel, at a fixed length scale."""
        self._quiver_index = _arrow_index(sample, _ARROW_COLUMNS, _ARROW_ROWS)
        rows, columns = self._quiver_index
        along, up = _arrow_components(sample, self._quiver_index)
        peak = max(self._scale.speed_m_s[1], 1.0e-9)
        return axes.quiver(
            sample.along_m[rows] * _MM_PER_M,
            sample.up_m[columns] * _MM_PER_M,
            along,
            up,
            angles="xy",
            scale=peak * _ARROW_SCALE,
            scale_units="width",
            width=0.0026,
            color="#f5f5f5",
            edgecolor="#222222",
            linewidth=0.25,
            zorder=3.0,
        )

    def _extra(self, sample: SliceSample) -> str:
        """The lines this view has to add to the shared validity stamp.

        Kept to four, and short. A stamp that covered the data it
        qualifies would be answered by cropping it, which is the failure
        mode putting it inside the axes was meant to prevent. The full
        wording of every claim lives on the objects' own ``describe``
        methods and in the provenance sidecar.
        """
        provenance = self._series.provenance
        occupancy = self._series.occupancy
        lines = [
            f"{sample.plane.describe()} | {sample.fidelity.label}",
            sample.through_plane_note,
            f"{provenance.speed_headline()}; declared approach, not a swing",
            (
                f"frame {sample.frame + 1}/{self.n_frames}, "
                f"t = {sample.time_s * 1e3:.4g} ms, "
                f"peak sand {sample.peak_speed_m_s:.4g} {self._scale.speed_unit}; "
                f"{self._cursor.describe()}"
            ),
            occupancy.describe(),
        ]
        packing = occupancy.packing_note(sample.density_kg_m3)
        if packing:
            lines.append(packing)
        # Wrapped as a block rather than per line: a stamp that ran off the
        # side of the panel would be as unreadable as one that had been
        # cropped, which is the thing drawing it inside the axes prevents.
        return "\n".join(_wrapped(line) for line in lines)

    def _extra_with_stamp(self, sample: SliceSample) -> str:
        """The whole stamp text, since only the extra part changes."""
        from .render import validity_stamp

        provenance = self._series.provenance
        head = validity_stamp(provenance.envelope_status, provenance.fidelity_tier)
        return f"{head}\n{self._extra(sample)}"


def _wrapped(line: str, width: int = _STAMP_WRAP_CHARS) -> str:
    """Fold one long stamp line so it stays inside the panel it stamps."""
    return "\n".join(textwrap.wrap(line, width=width)) or line


def _mesh_grid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """``(ns, nh)`` slice values as the ``(nh, ns)`` mesh wants them."""
    return np.asarray(values).T


def _mesh_values(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """The flat array a ``QuadMesh`` update takes."""
    return _mesh_grid(values).ravel()


def _arrow_index(
    sample: SliceSample, columns: int, rows: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Indices of the sample points that carry an arrow."""
    along = np.unique(
        np.linspace(0, sample.along_m.size - 1, min(columns, sample.along_m.size))
        .round()
        .astype(np.int64)
    )
    up = np.unique(
        np.linspace(0, sample.up_m.size - 1, min(rows, sample.up_m.size))
        .round()
        .astype(np.int64)
    )
    return along, up


def _arrow_components(
    sample: SliceSample, index: tuple[NDArray[np.int64], NDArray[np.int64]]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Arrow components at the subsampled points, zero where there is no sand.

    Zero rather than ``nan``: a zero-length arrow draws nothing, which is
    the right picture for an empty cell, whereas ``nan`` makes the whole
    quiver refuse to update.
    """
    along_index, up_index = index
    grid = np.ix_(along_index, up_index)
    occupied = sample.occupied[grid]
    along = np.where(occupied, np.nan_to_num(sample.velocity_along_m_s[grid]), 0.0)
    up = np.where(occupied, np.nan_to_num(sample.velocity_up_m_s[grid]), 0.0)
    return along.T, up.T


def _set_outline(line: Line2D, sample: SliceSample) -> None:
    """Draw the intruder section as a closed loop, or nothing."""
    outline = sample.body_outline_m
    if outline is None:
        line.set_data([], [])
        return
    closed = np.vstack([outline, outline[:1]])
    line.set_data(closed[:, 0] * _MM_PER_M, closed[:, 1] * _MM_PER_M)


def draw_slice_frame(
    figure: Figure,
    series: SandFieldSeries,
    plane: CuttingPlane,
    *,
    frame: int = 0,
    scale: SliceScale | None = None,
    cursor: CursorMap | None = None,
    bounds_m: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> SliceArtists:
    """Draw one cut into a figure and return its artists.

    Args:
        figure: Destination figure; cleared.
        series: The sand field.
        plane: The cut.
        scale: Colour limits. Defaults to this field's own coverage,
            which is right for a single field and **wrong** for a
            comparison -- pass a merged scale there.
        cursor: Transport-to-field frame mapping.
        bounds_m: Window of the cut to draw.
        frame: Opening field frame.

    Returns:
        The artists, for per-frame updates.
    """
    artists = SliceArtists(
        figure,
        series,
        plane,
        scale or slice_scale([series]),
        cursor=cursor,
        bounds_m=bounds_m,
    )
    artists.update(frame)
    return artists


def slice_still(
    series: SandFieldSeries,
    plane: CuttingPlane,
    *,
    frame: int | None = None,
    scale: SliceScale | None = None,
    figsize: tuple[float, float] = (8.5, 9.0),
) -> Figure:
    """A single figure of one cut, for a report or a test.

    Args:
        series: The sand field.
        plane: The cut.
        frame: Field frame. Defaults to the frame carrying the fastest
            reportable sand, which is the one worth a still.
        scale: Colour limits; defaults to this field's own coverage.
        figsize: Figure size in inches.

    Returns:
        The figure.
    """
    chosen = frame
    if chosen is None:
        speeds = series.occupied_speed_m_s()
        # Zero-filled before the reduction rather than masked after it: a
        # nanmax over an all-nan row is correct but warns, and frame 0 of
        # every capture is the undisturbed bed.
        peaks = np.nan_to_num(speeds, nan=0.0).max(axis=1)
        chosen = int(np.argmax(peaks))
    figure = Figure(figsize=figsize)
    draw_slice_frame(figure, series, plane, frame=chosen, scale=scale)
    return figure
