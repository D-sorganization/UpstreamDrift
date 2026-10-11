"""Render one conventional overhead panel from verified saved trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes


@dataclass(frozen=True)
class FlightPanelLayout:
    """Shared target, viewport and style for a comparable flight-panel grid."""

    target: float
    xhalf: float
    ymax: float
    rows: int
    supplemental: bool
    colors: dict[str, str]


def plot_flight_panel(
    ax: Axes,
    label: str,
    paths: list[np.ndarray],
    index: int,
    nominal: np.ndarray | None,
    layout: FlightPanelLayout,
) -> None:
    """Show the replay cloud and a distinct frozen zero-error reference."""
    color = layout.colors[label.split(" · ")[0]]
    for path in paths:
        ax.plot(
            path[:, 0],
            path[:, 1],
            color=color,
            lw=0.65 if layout.supplemental else 1.0,
            alpha=0.16 if layout.supplemental else 0.55,
            rasterized=layout.supplemental,
        )
        if not layout.supplemental:
            ax.plot(path[-1, 0], path[-1, 1], "o", color=color, ms=2.4)
    if layout.supplemental and nominal is not None:
        ax.plot(nominal[:, 0], nominal[:, 1], "--", color="#1b2733", lw=1.55)
    ax.plot(0, layout.target, "+", color="#c51f30", ms=10, mew=1.7)
    ax.axvline(0, color="#536577", lw=0.6)
    ax.set_xlim(-layout.xhalf, layout.xhalf)
    ax.set_ylim(0, layout.ymax)
    ax.set_title(label, fontsize=13)
    if index // 3 == layout.rows - 1:
        ax.set_xlabel("Lateral Position (m; Right Positive)", fontsize=11)
    if index % 3 == 0:
        ax.set_ylabel("Downrange Distance (m)", fontsize=11)
    ax.grid(alpha=0.16)
