"""Drawing the bunker ball flight, landing and green rollout (issue #15497).

Headless. Two linked panels from one :class:`~.flight.BunkerFlight`: a side
view (distance against height) and an overhead view (distance against
offline), each carrying the landing and rest markers.

Fail closed
-----------

ADR-0032: a refused or invalid shot has no flight (``None``). It draws a
banner with the reason and **no** line -- never a placeholder arc.

The assumption travels with the picture
---------------------------------------

The launch is uncalibrated and the rollout is a flat 10 ft stimp green with
simplified bounce and spin. :attr:`BunkerFlight.assumption` is stamped
*inside* the axes so a cropped screenshot cannot shed it. It wraps to the
figure width, and the legend lives below the plots, so a narrow (phone-width)
canvas shows neither a clipped label nor a banner over the legend.
"""

from __future__ import annotations

from enum import Enum

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .flight import BunkerFlight

__all__ = ["FlightUnits", "flight_figure", "flight_readout"]

_YARD_M = 0.9144
_FLIGHT_COLOUR = "#2f6f9f"
_ROLL_COLOUR = "#1e7a4a"
_LANDING_COLOUR = "#9b1c1c"
_BANNER_COLOUR = "#7a4f9b"


class FlightUnits(Enum):
    """Display units for distances; the data stays SI."""

    METRES = ("m", 1.0)
    YARDS = ("yd", 1.0 / _YARD_M)

    @property
    def symbol(self) -> str:
        """Unit symbol for axis labels."""
        return self.value[0]

    @property
    def scale(self) -> float:
        """Multiplier from metres."""
        return self.value[1]


def _banner(axes: Axes, text: str) -> None:
    """Stamp a message inside ``axes`` (survives cropping)."""
    axes.text(
        0.02,
        0.98,
        text,
        transform=axes.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        wrap=True,
        color="white",
        bbox={"facecolor": _BANNER_COLOUR, "edgecolor": "none", "boxstyle": "round"},
        zorder=10,
    )


def _draw_panel(
    axes: Axes,
    flight: BunkerFlight,
    k: float,
    unit: str,
    *,
    vertical: int,
    ylabel: str,
) -> None:
    """Draw one projection; ``vertical`` is the coordinate index on y."""
    fx = [p[0] * k for p in flight.flight_m]
    fy = [p[vertical] * k for p in flight.flight_m]
    rx = [p[0] * k for p in flight.rollout_m]
    ry = [p[vertical] * k for p in flight.rollout_m]
    axes.plot(fx, fy, color=_FLIGHT_COLOUR, label="flight")
    axes.plot(rx, ry, color=_ROLL_COLOUR, linestyle="--", label="rollout")
    axes.plot([fx[-1]], [fy[-1]], "o", color=_LANDING_COLOUR, label="landing")
    axes.plot([rx[-1]], [ry[-1]], "s", color=_ROLL_COLOUR, label="rest")
    axes.set_xlabel(f"downrange [{unit}]")
    axes.set_ylabel(f"{ylabel} [{unit}]")
    axes.margins(y=0.35)  # headroom so the in-axes banner clears the apex
    axes.set_aspect("equal", adjustable="datalim")
    axes.grid(True, alpha=0.3)


def flight_figure(
    flight: BunkerFlight | None,
    *,
    units: FlightUnits = FlightUnits.METRES,
    reason: str = "no ball flight available",
    figure: Figure | None = None,
) -> Figure:
    """Draw the linked side and overhead panels.

    Args:
        flight: The trace, or ``None`` for a refused / invalid shot.
        units: Display units.
        reason: Banner text when ``flight`` is ``None``.
        figure: Figure to reuse (cleared first); a new one when omitted.

    Returns:
        The figure with two axes: side view, then overhead view.
    """
    fig = figure if figure is not None else Figure(figsize=(7.0, 6.0))
    fig.clear()
    side = fig.add_subplot(2, 1, 1)
    top = fig.add_subplot(2, 1, 2)
    # Fixed margins keep the y labels on-canvas and leave a strip under the
    # plots for the figure legend, so a narrow window never clips or overlaps.
    fig.subplots_adjust(left=0.2, right=0.96, top=0.93, bottom=0.14, hspace=0.6)
    side.set_title("Ball flight (side)", fontsize=9)
    top.set_title("Landing and rollout (overhead)", fontsize=9)
    if flight is None:
        for axes in (side, top):
            axes.set_xticks([])
            axes.set_yticks([])
        _banner(side, reason)
        return fig
    k, unit = units.scale, units.symbol
    _draw_panel(side, flight, k, unit, vertical=2, ylabel="height")
    _draw_panel(top, flight, k, unit, vertical=1, ylabel="offline")
    handles, labels = side.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        fontsize=7,
        frameon=False,
    )
    _banner(side, flight.assumption)
    _banner(top, flight.assumption)
    return fig


def flight_readout(
    flight: BunkerFlight | None,
    units: FlightUnits,
    *,
    reason: str = "no flight trace",
) -> str:
    """One-paragraph text under the canvas; never a number for no flight."""
    if flight is None:
        return f"no flight: {reason}"
    k, unit = units.scale, units.symbol
    rest = flight.rollout_m[-1]
    total = (rest[0] ** 2 + rest[1] ** 2) ** 0.5 * k
    return (
        f"carry {flight.carry_m * k:.1f} {unit}, "
        f"carry + rollout {total:.1f} {unit}. {flight.assumption}"
    )
