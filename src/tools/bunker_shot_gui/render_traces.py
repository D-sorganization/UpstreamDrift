"""Drawing the linked trace panel (issue #8708, epic #8699).

Headless. One stacked panel per :class:`~.traces.TraceGroup`, a shared time
axis, and one cursor across all of them -- the same cursor the 3-D scene and
the sole load field are on, so a force peak can be traced to a moment *and*
to a location.

The band, drawn
---------------

The validity band is shaded *behind* the traces, one span per regime, rather
than stamped once in a corner. That is not decoration. A shot sits inside
3D-RFT's stated limits during the free-flight lead-in and leaves them when
the sole engages at speed, and a single badge invites the reader to apply
the worst verdict evenly across a record where it does not apply evenly. The
shading says which stretch of the curve is which, and the legend says what
each shade means -- a stripe nobody can decode is worse than no stripe.

Nothing auto-scales
-------------------

Every panel's y-limits are set once, from the whole trace, and
``autoscale`` is switched off. A y-axis that re-ranged while scrubbing would
make the eye read noise as signal, which is the same failure issue #8728
removed from the sole load field's colour ramp.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text

from bunkershot3d.solvers import EnvelopeStatus

from .render import stamp_axes
from .report import status_colour
from .traces import ScalarTrace, ShotTraces, TraceGroup, ValidityBand

__all__ = [
    "TracePanelArtists",
    "draw_trace_panel",
    "trace_panel_still",
]

_TRACE_COLOURS: tuple[str, ...] = ("#2f6f9f", "#9b1c1c", "#1e7a4a", "#7a4f9b")
"""Series colours, cycled within a panel. Fixed, so a colour means a series."""

_CURSOR_COLOUR = "#3a3a3a"
_BAND_ALPHA = 0.18
_BAND_ZORDER = 0.2
"""Below every line, so the shading qualifies the curves without hiding them."""

_TRACE_ZORDER = 2.0
_PADDING = 0.06
"""Slack on a panel's y-range, as a fraction of its own span."""


def _limits(traces: tuple[ScalarTrace, ...]) -> tuple[float, float]:
    """Return the fixed y-range covering every trace on one panel.

    Args:
        traces: The panel's traces.

    Returns:
        ``(low, high)``, never degenerate -- a flat trace still needs a
        visible axis rather than a zero-height one.
    """
    values = np.concatenate([trace.values for trace in traces])
    low, high = float(values.min()), float(values.max())
    span = high - low
    if span <= 0.0:
        span = max(abs(high), 1.0)
    pad = span * _PADDING
    return (low - pad, high + pad)


def _draw_band(axes: Axes, band: ValidityBand) -> None:
    """Shade one panel with the verdict that applied over each stretch.

    Args:
        axes: The panel.
        band: The per-sample verdicts.
    """
    for span in band.spans():
        axes.axvspan(
            span.start_s * 1e3,
            span.end_s * 1e3,
            color=status_colour(span.status),
            alpha=_BAND_ALPHA,
            linewidth=0.0,
            zorder=_BAND_ZORDER,
        )


def _band_legend(band: ValidityBand) -> str:
    """Return the line explaining what the shading means.

    Args:
        band: The per-sample verdicts.

    Returns:
        One line naming every regime the record passed through, so a reader
        can decode the shading without leaving the figure.
    """
    seen: list[EnvelopeStatus] = []
    for span in band.spans():
        if span.status not in seen:
            seen.append(span.status)
    regimes = "; ".join(f"{status.value.replace('_', ' ').upper()}" for status in seen)
    changed = (
        "the shot changes regime mid-record"
        if band.changes
        else "one regime throughout"
    )
    return f"validity envelope, shaded: {regimes} ({changed})"


class TracePanelArtists:
    """Panels built once for one shot, and the artists a scrub touches.

    Only the cursors and one readout move. Everything else -- the curves,
    the shaded validity spans, the labels, the fixed y-limits -- is built
    once, so scrubbing costs a redraw and not a rebuild.
    """

    def __init__(self, figure: Figure, traces: ShotTraces) -> None:
        """Build the panels for one shot.

        Args:
            figure: The figure to build into; cleared first.
            traces: The trace set.

        Raises:
            ValueError: If the set carries no traces to draw. An empty panel
                stack under a validity band would read as "nothing happened"
                rather than as "nothing was computed".
        """
        groups = traces.groups()
        if not groups:
            raise ValueError(
                "a trace panel needs at least one trace to draw; an empty "
                "stack reads as though the shot produced nothing"
            )
        self._traces = traces
        figure.clear()
        panels = figure.subplots(len(groups), 1, sharex=True, squeeze=False)
        self._axes: list[Axes] = [row[0] for row in panels]
        self._cursors: list[Line2D] = []
        for axes, group in zip(self._axes, groups, strict=True):
            self._cursors.append(self._build_panel(axes, group))
        self._axes[-1].set_xlabel(traces.time_axis_label, fontsize=7)
        self._readout: Text = self._axes[0].text(
            0.99,
            0.02,
            "",
            transform=self._axes[0].transAxes,
            ha="right",
            va="bottom",
            fontsize=6,
            color="#333333",
            zorder=9,
        )
        figure.tight_layout()

    @property
    def n_panels(self) -> int:
        """How many panels the stack owns."""
        return len(self._axes)

    def _build_panel(self, axes: Axes, group: TraceGroup) -> Line2D:
        """Build one panel and return its cursor.

        Args:
            axes: The panel.
            group: Which traces go on it.

        Returns:
            The cursor line, the only artist a scrub moves.
        """
        traces = self._traces
        members = traces.group(group)
        time_ms = traces.time_display
        _draw_band(axes, traces.band)
        for index, trace in enumerate(members):
            axes.plot(
                time_ms,
                trace.values,
                color=_TRACE_COLOURS[index % len(_TRACE_COLOURS)],
                linewidth=1.2,
                label=trace.name,
                zorder=_TRACE_ZORDER,
            )
        # One unit per panel by construction -- see TraceGroup -- so the
        # axis label is the group's heading and that shared unit.
        axes.set_ylabel(f"{group.label} [{members[0].unit}]", fontsize=7)
        axes.set_xlim(float(time_ms[0]), float(time_ms[-1]))
        axes.set_ylim(*_limits(members))
        axes.set_autoscale_on(False)
        axes.tick_params(labelsize=6)
        axes.legend(loc="upper left", fontsize=5, framealpha=0.6, ncols=len(members))
        cursor = axes.axvline(
            float(time_ms[0]),
            color=_CURSOR_COLOUR,
            linewidth=1.0,
            linestyle=":",
            zorder=_TRACE_ZORDER + 1.0,
        )
        return cursor

    def _check_frame(self, frame: int) -> int:
        """Validate a frame index against the record.

        Args:
            frame: The requested sample index.

        Returns:
            The index.

        Raises:
            ValueError: If it is outside the recorded shot. A clamped index
                would leave the cursor describing a different moment from
                the one the transport says it is showing.
        """
        if not 0 <= int(frame) < self._traces.n_frames:
            raise ValueError(
                f"frame {frame} is outside the recorded shot, which has "
                f"{self._traces.n_frames} samples"
            )
        return int(frame)

    def update(self, frame: int) -> None:
        """Move the shared cursor to one sample.

        Args:
            frame: The sample index.

        Raises:
            ValueError: If the index is outside the recorded shot.
        """
        index = self._check_frame(frame)
        traces = self._traces
        moment = float(traces.time_display[index])
        for cursor in self._cursors:
            cursor.set_xdata([moment, moment])
        status = traces.band.status_at(index)
        readings = "  ".join(
            f"{trace.name} {trace.value_at(index):.4g} {trace.unit}"
            for trace in traces.traces
        )
        self._readout.set_text(
            f"{moment:.2f} {traces.time_unit} - "
            f"{status.value.replace('_', ' ').upper()}\n{readings}"
        )


def draw_trace_panel(
    figure: Figure, traces: ShotTraces, *, frame: int = 0
) -> TracePanelArtists:
    """Draw the trace panel for one shot into an existing figure.

    The figure is cleared and rebuilt, so this is the right call for a still
    and the wrong one for scrubbing: hold the returned
    :class:`TracePanelArtists` and call :meth:`~TracePanelArtists.update`,
    which moves one line per panel and nothing else.

    Args:
        figure: The figure to draw into.
        traces: The trace set.
        frame: Which sample to put the cursor on.

    Returns:
        The built artists.

    Raises:
        ValueError: If the frame is outside the record, or the set carries
            no traces.
    """
    artists = TracePanelArtists(figure, traces)
    artists.update(frame)
    _stamp_panel(figure, traces)
    return artists


def _stamp_panel(figure: Figure, traces: ShotTraces) -> None:
    """Stamp the top panel with the verdict and what the shading means.

    Args:
        figure: The figure holding the panels.
        traces: The trace set, supplying the band.
    """
    band = traces.band
    stamp_axes(
        figure.axes[0],
        band.worst,
        traces.fidelity_tier,
        extra=_band_legend(band),
    )


def trace_panel_still(
    traces: ShotTraces,
    *,
    frame: int | None = None,
    figsize: tuple[float, float] = (8.0, 9.0),
) -> Figure:
    """Render the trace panel as a standalone figure.

    Args:
        traces: The trace set.
        frame: Which sample to put the cursor on; defaults to the moment the
            sand force peaked, which is the moment a designer looks for.
        figsize: Figure size in inches.

    Returns:
        The figure.

    Raises:
        ValueError: If the frame is outside the record.
    """
    chosen = _peak_force_frame(traces) if frame is None else frame
    figure = Figure(figsize=figsize)
    draw_trace_panel(figure, traces, frame=chosen)
    return figure


def _peak_force_frame(traces: ShotTraces) -> int:
    """Return the sample at which the sand force magnitude peaked.

    Args:
        traces: The trace set.

    Returns:
        The sample index; zero when the set carries no force traces.
    """
    forces = traces.group(TraceGroup.SAND_FORCE)
    if not forces:
        return 0
    magnitude = np.sqrt(sum(trace.values**2 for trace in forces))
    return int(np.argmax(magnitude))
