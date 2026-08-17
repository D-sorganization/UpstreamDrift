"""Drawing the cross-tier comparison (issue #8713, epic #8699).

Headless, like every other renderer in this package: matplotlib and no GUI
toolkit, so the same figure appears in a test, in a batch sweep and inside
the Qt workbench.

Four things this figure does that a plain overlay would not
-----------------------------------------------------------

**The F1 points are never joined.** F1 has no shot history -- each point is
its own march to one pose under a declared straight-line approach (issue
#8733) -- so a line through them would draw a trajectory that was never
computed. They are markers, the legend says why, and the method line in the
figure says it again in words.

**The gap is an object.** At every probe a connector is drawn *between* the
two tiers' values and labelled with the ratio, and stretches where a
quantity left the declared band are shaded. "Divergence highlighted, not
just plotted" is the issue's wording, and a reader should not have to
measure a distance by eye to find the result.

**The crossover has its own axes.** F0's inertial share and F1's
momentum-flux share are drawn against *speed* rather than against time,
because that is the variable they diverge in and because a greenside shot
never slows through the crossing -- a time axis would show two flat lines
and hide the sharpest single result in the epic.

**The licence is inside the frame.** Not a caption: a screenshot of a panel
keeps its contents and loses its surroundings, which is exactly how a
picture from two uncalibrated models ends up in a deck as though it had
been measured.

Nothing autoscales. Every panel's limits are set once from the whole
record and both tiers, for the reason issue #8728 fixed the sole field's
colour ramp: an axis that re-ranges while the cursor moves makes the eye
read noise as signal.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text

from .agreement import AgreementClass, ComparedQuantity
from .crosstier import CrossTierComparison
from .render import stamp_axes
from .report import status_colour

__all__ = [
    "PANEL_QUANTITIES",
    "CrossTierArtists",
    "cross_tier_still",
    "draw_cross_tier",
]

PANEL_QUANTITIES: tuple[ComparedQuantity, ...] = (
    ComparedQuantity.WRENCH,
    ComparedQuantity.SOLE_DEPTH,
    ComparedQuantity.DIVOT_SECTION,
    ComparedQuantity.SPEED_LOST,
)
"""The quantities that resolve in time, in the order they are stacked.

:attr:`~.agreement.ComparedQuantity.DIVOT_MASS` is absent on purpose: it is
its own section times one declared width and one density, so a fourth curve
of the same shape would look like a fourth measurement. It is in the
agreement table, where the number belongs."""

_F0_COLOUR = "#2f6f9f"
_F1_COLOUR = "#9b1c1c"
_CURSOR_COLOUR = "#3a3a3a"
_CONSISTENT_COLOUR = "#1e7a4a"
_DIVERGENT_COLOUR = "#b3441d"
_INCOMPARABLE_COLOUR = "#7a7a7a"

_AGREEMENT_COLOUR: dict[AgreementClass, str] = {
    AgreementClass.CONSISTENT: _CONSISTENT_COLOUR,
    AgreementClass.DIVERGENT: _DIVERGENT_COLOUR,
    AgreementClass.INCOMPARABLE: _INCOMPARABLE_COLOUR,
}

_BAND_ALPHA = 0.16
_VALIDITY_ALPHA = 0.10
_BAND_ZORDER = 0.2
_SERIES_ZORDER = 2.0
_PADDING = 0.10
_LABEL_SIZE = 6
_TICK_SIZE = 5
_TEXT_SIZE = 5.5


def _limits(values: list[np.ndarray]) -> tuple[float, float]:
    """The fixed y-range covering both tiers on one panel.

    Args:
        values: Every series the panel will draw, NaNs allowed.

    Returns:
        ``(low, high)``, never degenerate.
    """
    finite = np.concatenate(
        [np.asarray(item, dtype=np.float64).reshape(-1) for item in values]
    )
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (0.0, 1.0)
    low, high = float(finite.min()), float(finite.max())
    span = high - low
    if span <= 0.0:
        span = max(abs(high), 1.0)
    pad = span * _PADDING
    return (low - pad, high + pad)


class CrossTierArtists:
    """The comparison figure, built once, and the artists a scrub touches.

    Only the cursors and one readout move. Everything else -- the curves,
    the probe markers, the connectors, the shaded spans, the fixed limits,
    the licence -- is built once, so scrubbing costs a redraw rather than a
    rebuild.
    """

    def __init__(self, figure: Figure, comparison: CrossTierComparison) -> None:
        """Build the figure for one comparison.

        Args:
            figure: The figure to build into; cleared first.
            comparison: The comparison to draw.
        """
        self.comparison = comparison
        figure.clear()
        grid = figure.add_gridspec(
            len(PANEL_QUANTITIES),
            2,
            width_ratios=(3.0, 2.0),
            hspace=0.16,
            wspace=0.22,
        )
        self.panels: list[Axes] = []
        self.cursors: list[Line2D] = []
        self.f0_series: list[Line2D] = []
        self.f1_series: list[Line2D] = []
        self.connectors: list[Line2D] = []
        self.ratio_labels: list[Text] = []
        self._divergence_bands = 0

        shared: Axes | None = None
        for row, quantity in enumerate(PANEL_QUANTITIES):
            axes = figure.add_subplot(grid[row, 0], sharex=shared)
            shared = shared or axes
            self.panels.append(axes)
            self._build_panel(axes, quantity)
        self.panels[-1].set_xlabel("time [ms]", fontsize=_LABEL_SIZE)

        self.crossover_axes = figure.add_subplot(grid[0:2, 1])
        self.crossover_marked = False
        self.crossover_caption = self._build_crossover(self.crossover_axes)

        text_axes = figure.add_subplot(grid[2:, 1])
        text_axes.axis("off")
        self.agreement_text = self._build_agreement_table(text_axes)
        self.method_text = self._build_method_note(text_axes)
        self.licence_text = self._build_licence(text_axes)

        self.stamp = stamp_axes(
            self.panels[0],
            comparison.worst_status,
            comparison.tiers[1],
            extra=comparison.licence_stamp(),
        )
        self.readout: Text = self.panels[0].text(
            0.99,
            0.03,
            "",
            transform=self.panels[0].transAxes,
            ha="right",
            va="bottom",
            fontsize=_TEXT_SIZE,
            color="#333333",
            zorder=9,
        )

    # ------------------------------------------------------------ accessors

    @property
    def panel_quantities(self) -> tuple[ComparedQuantity, ...]:
        """Which quantity each stacked panel shows, top to bottom."""
        return PANEL_QUANTITIES

    @property
    def n_panels(self) -> int:
        """How many stacked panels the figure owns."""
        return len(self.panels)

    @property
    def n_connectors(self) -> int:
        """How many tier-to-tier connectors were drawn."""
        return len(self.connectors)

    @property
    def n_divergence_bands(self) -> int:
        """How many divergent stretches were shaded."""
        return self._divergence_bands

    # -------------------------------------------------------------- panels

    def _panel_series(
        self, quantity: ComparedQuantity
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """``(f0_curve, f1_curve)`` for one panel, in the reporting unit.

        Args:
            quantity: Which panel.

        Returns:
            F0's continuous record and, for the speed panel only, the
            speed history F1's force implies. ``None`` elsewhere, because
            F1 has no continuous history of anything else.
        """
        model = self.comparison
        scale = quantity.display_scale
        if quantity is ComparedQuantity.WRENCH:
            return (model.f0_force_magnitude_n * scale, None)
        if quantity is ComparedQuantity.SOLE_DEPTH:
            return (model.f0_sole_depth_m * scale, None)
        if quantity is ComparedQuantity.DIVOT_SECTION:
            return (model.f0_divot_section_area_m2 * scale, None)
        try:
            implied = model.f1_implied_speed_m_s()
        except ValueError:
            implied = None
        return (model.f0_speed_m_s, implied)

    def _probe_points(
        self, quantity: ComparedQuantity
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(time_ms, f0, f1)`` at every probe, in the reporting unit."""
        model = self.comparison
        ordered = sorted(model.shot_probes, key=lambda item: item.frame)
        times = np.array([probe.time_s * 1e3 for probe in ordered])
        if quantity is ComparedQuantity.SPEED_LOST:
            f0 = np.array([model.f0_speed_m_s[probe.frame] for probe in ordered])
            try:
                implied = model.f1_implied_speed_m_s()
            except ValueError:
                return (times, f0, np.full(times.size, np.nan))
            return (times, f0, np.array([implied[probe.frame] for probe in ordered]))
        scale = quantity.display_scale
        pairs = [probe.agreement(quantity) for probe in ordered]
        return (
            times,
            np.array([item.f0_value * scale for item in pairs]),
            np.array([item.f1_value * scale for item in pairs]),
        )

    def _build_panel(self, axes: Axes, quantity: ComparedQuantity) -> None:
        """Draw one stacked panel: both tiers, the gaps, and the shading."""
        model = self.comparison
        time_ms = model.time_s * 1e3
        self._shade_validity(axes)
        self._shade_divergence(axes, quantity)

        f0_curve, f1_curve = self._panel_series(quantity)
        (f0_line,) = axes.plot(
            time_ms,
            f0_curve,
            color=_F0_COLOUR,
            linewidth=1.2,
            label="F0 (3D-RFT), whole record",
            zorder=_SERIES_ZORDER,
        )
        self.f0_series.append(f0_line)
        series = [f0_curve]
        if f1_curve is not None:
            axes.plot(
                time_ms,
                f1_curve,
                color=_F1_COLOUR,
                linewidth=1.0,
                linestyle="--",
                label="F1-implied (one-way coupled)",
                zorder=_SERIES_ZORDER,
            )
            series.append(f1_curve)

        times, f0_points, f1_points = self._probe_points(quantity)
        (f1_line,) = axes.plot(
            times,
            f1_points,
            color=_F1_COLOUR,
            linestyle="None",
            marker="o",
            markersize=3.5,
            label="F1 (MPM), one march per probe -- not a history",
            zorder=_SERIES_ZORDER + 1.0,
        )
        self.f1_series.append(f1_line)
        series.append(f1_points)
        self._draw_gaps(axes, quantity, times, f0_points, f1_points)

        axes.set_ylabel(f"{quantity.label}\n[{quantity.unit}]", fontsize=_LABEL_SIZE)
        axes.set_xlim(float(time_ms[0]), float(time_ms[-1]))
        axes.set_ylim(*_limits(series))
        axes.set_autoscale_on(False)
        axes.tick_params(labelsize=_TICK_SIZE)
        axes.legend(loc="upper left", fontsize=4.5, framealpha=0.6)
        self.cursors.append(
            axes.axvline(
                float(time_ms[0]),
                color=_CURSOR_COLOUR,
                linewidth=1.0,
                linestyle=":",
                zorder=_SERIES_ZORDER + 2.0,
            )
        )

    def _shade_validity(self, axes: Axes) -> None:
        """Shade the record with the verdict that applied over each stretch."""
        for span in self.comparison.band.spans():
            axes.axvspan(
                span.start_s * 1e3,
                span.end_s * 1e3,
                color=status_colour(span.status),
                alpha=_VALIDITY_ALPHA,
                linewidth=0.0,
                zorder=_BAND_ZORDER,
            )

    def _shade_divergence(self, axes: Axes, quantity: ComparedQuantity) -> None:
        """Shade the stretches where this quantity left the declared band."""
        try:
            spans = self.comparison.divergence_spans(quantity)
        except ValueError:
            return
        for span in spans:
            axes.axvspan(
                span.start_s * 1e3,
                span.end_s * 1e3,
                color=_DIVERGENT_COLOUR,
                alpha=_BAND_ALPHA,
                linewidth=0.0,
                zorder=_BAND_ZORDER + 0.1,
            )
            self._divergence_bands += 1

    def _draw_gaps(
        self,
        axes: Axes,
        quantity: ComparedQuantity,
        times: np.ndarray,
        f0_points: np.ndarray,
        f1_points: np.ndarray,
    ) -> None:
        """Draw the disagreement itself: a connector per probe, labelled."""
        ordered = sorted(self.comparison.shot_probes, key=lambda item: item.frame)
        for index, probe in enumerate(ordered):
            low = float(f0_points[index])
            high = float(f1_points[index])
            if not (np.isfinite(low) and np.isfinite(high)):
                continue
            if quantity is ComparedQuantity.SPEED_LOST:
                agreement = self.comparison.agreement(ComparedQuantity.SPEED_LOST)
            else:
                agreement = probe.agreement(quantity)
            colour = _AGREEMENT_COLOUR[agreement.agreement]
            (connector,) = axes.plot(
                [times[index], times[index]],
                [low, high],
                color=colour,
                linewidth=1.4,
                solid_capstyle="butt",
                zorder=_SERIES_ZORDER + 0.5,
            )
            self.connectors.append(connector)
            if not agreement.diverged or not np.isfinite(agreement.ratio):
                continue
            self.ratio_labels.append(
                axes.annotate(
                    f"{agreement.ratio:.3g}x",
                    xy=(times[index], 0.5 * (low + high)),
                    xytext=(3, 0),
                    textcoords="offset points",
                    fontsize=_TEXT_SIZE,
                    color=colour,
                    va="center",
                    zorder=_SERIES_ZORDER + 3.0,
                )
            )

    # ------------------------------------------------------------ crossover

    def _build_crossover(self, axes: Axes) -> Text:
        """Draw both inertial shares against speed and mark the crossing."""
        model = self.comparison
        probes = sorted(model.crossover_probes, key=lambda item: item.speed_m_s)
        speeds = np.array([probe.speed_m_s for probe in probes])
        axes.plot(
            speeds,
            [probe.f0_inertial_fraction for probe in probes],
            color=_F0_COLOUR,
            marker="o",
            markersize=3.0,
            linewidth=1.2,
            label="F0 inertial share",
        )
        axes.plot(
            speeds,
            [probe.f1_flux_fraction for probe in probes],
            color=_F1_COLOUR,
            marker="s",
            markersize=3.0,
            linewidth=1.2,
            label="F1 momentum-flux share",
        )
        crossing = model.crossover()
        if crossing is not None:
            axes.axvline(
                crossing.speed_m_s,
                color=_DIVERGENT_COLOUR,
                linewidth=1.0,
                linestyle="--",
            )
            axes.plot(
                [crossing.speed_m_s],
                [crossing.shared_share],
                marker="*",
                markersize=9.0,
                color=_DIVERGENT_COLOUR,
                linestyle="None",
            )
            self.crossover_marked = True
        axes.set_xlabel("intrusion speed [m/s]", fontsize=_LABEL_SIZE)
        axes.set_ylabel("share of resultant [-]", fontsize=_LABEL_SIZE)
        axes.set_ylim(0.0, 1.05)
        axes.set_autoscale_on(False)
        axes.tick_params(labelsize=_TICK_SIZE)
        axes.legend(loc="lower right", fontsize=4.5, framealpha=0.6)
        source = (
            "declared speed sweep at the deepest recorded pose"
            if model.sweep_probes
            else "probes along the shot"
        )
        return axes.text(
            0.0,
            -0.34,
            f"{source}\n{_wrap(model.crossover_summary(), 74)}",
            transform=axes.transAxes,
            ha="left",
            va="top",
            fontsize=_TEXT_SIZE,
            color="#333333",
        )

    # ---------------------------------------------------------------- text

    def _build_agreement_table(self, axes: Axes) -> Text:
        """Write the per-quantity agreement summary into the figure."""
        rows = "\n".join(
            _wrap(item.summary(), 74, indent="    ")
            for item in self.comparison.agreements()
        )
        return axes.text(
            0.0,
            1.0,
            f"Agreement, quantity by quantity\n{rows}",
            transform=axes.transAxes,
            ha="left",
            va="top",
            fontsize=_TEXT_SIZE,
            color="#222222",
            family="monospace",
        )

    def _build_method_note(self, axes: Axes) -> Text:
        """State how the F1 points were produced, in the figure."""
        model = self.comparison
        caveats = [
            probe.divot_caveat() for probe in model.shot_probes if probe.divot_caveat()
        ]
        note = (
            f"Method: {model.n_probes} probe(s) of a {model.n_frames}-sample F0 "
            "record. F1 has no shot history yet (#8733), so each F1 point is a "
            "separate march to that pose under a declared straight-line "
            "constant-speed approach. The points are therefore not joined."
        )
        if caveats:
            note = f"{note} {caveats[0]}"
        return axes.text(
            0.0,
            0.46,
            _wrap(note, 76),
            transform=axes.transAxes,
            ha="left",
            va="top",
            fontsize=_TEXT_SIZE,
            color="#444444",
        )

    def _build_licence(self, axes: Axes) -> Text:
        """Write the licence statement into the figure, not under it."""
        return axes.text(
            0.0,
            0.22,
            _wrap(self.comparison.licence(), 76),
            transform=axes.transAxes,
            ha="left",
            va="top",
            fontsize=_TEXT_SIZE,
            color="#7a1f1f",
        )

    # ------------------------------------------------------------ scrubbing

    def update(self, frame: int) -> None:
        """Move the shared cursor to one sample of the F0 record.

        Args:
            frame: The sample index.

        Raises:
            ValueError: If the index is outside the record. A clamped index
                would leave the cursor describing a different moment from
                the one the transport says it is showing.
        """
        model = self.comparison
        if not 0 <= int(frame) < model.n_frames:
            raise ValueError(
                f"frame {frame} is outside the recorded shot, which has "
                f"{model.n_frames} samples"
            )
        index = int(frame)
        moment = float(model.time_s[index]) * 1e3
        for cursor in self.cursors:
            cursor.set_xdata([moment, moment])
        status = model.band.status_at(index)
        self.readout.set_text(
            f"{moment:.2f} ms - {status.value.replace('_', ' ').upper()} - "
            f"|F0| {float(model.f0_force_magnitude_n[index]):.4g} N, "
            f"{float(model.f0_speed_m_s[index]):.4g} m/s"
        )


def _wrap(text: str, width: int, *, indent: str = "") -> str:
    """Wrap a paragraph to a fixed column, preserving its own line breaks.

    Args:
        text: The paragraph.
        width: Column to wrap at.
        indent: Prefix for continuation lines.

    Returns:
        The wrapped text.
    """
    lines: list[str] = []
    for paragraph in text.splitlines():
        current = ""
        for word in paragraph.split():
            candidate = f"{current} {word}".strip()
            if len(candidate) > width and current:
                lines.append(current)
                current = f"{indent}{word}"
            else:
                current = candidate
        lines.append(current)
    return "\n".join(lines)


def draw_cross_tier(
    figure: Figure, comparison: CrossTierComparison, *, frame: int | None = None
) -> CrossTierArtists:
    """Draw the comparison into an existing figure.

    The figure is cleared and rebuilt, so this is the right call for a
    still and the wrong one for scrubbing: hold the returned
    :class:`CrossTierArtists` and call :meth:`~CrossTierArtists.update`,
    which moves one line per panel and nothing else.

    Args:
        figure: The figure to draw into.
        comparison: The comparison.
        frame: Which sample to put the cursor on; defaults to the probe
            F0's own force peaked at, which is the moment the shot-level
            agreement is quoted at.

    Returns:
        The built artists.

    Raises:
        ValueError: If the frame is outside the record.
    """
    artists = CrossTierArtists(figure, comparison)
    artists.update(comparison.peak_probe.frame if frame is None else frame)
    return artists


def cross_tier_still(
    comparison: CrossTierComparison,
    *,
    frame: int | None = None,
    figsize: tuple[float, float] = (11.0, 8.0),
) -> Figure:
    """Render the comparison as a standalone figure.

    Args:
        comparison: The comparison.
        frame: Which sample to put the cursor on.
        figsize: Figure size in inches.

    Returns:
        The figure.

    Raises:
        ValueError: If the frame is outside the record.
    """
    figure = Figure(figsize=figsize)
    draw_cross_tier(figure, comparison, frame=frame)
    return figure
