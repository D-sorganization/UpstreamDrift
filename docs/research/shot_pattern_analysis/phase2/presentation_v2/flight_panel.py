"""Render one conventional overhead panel from verified saved trajectories."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes


def plot_flight_panel(
    ax: Axes,
    label: str,
    paths: list[np.ndarray],
    target: float,
    xhalf: float,
    ymax: float,
    index: int,
    rows: int,
    supplemental: bool,
    colors: dict[str, str],
    nominal: np.ndarray | None,
) -> None:
    """Show the replay cloud and a distinct frozen zero-error reference."""
    color = colors[label.split(" · ")[0]]
    for path in paths:
        ax.plot(
            path[:, 0],
            path[:, 1],
            color=color,
            lw=0.65 if supplemental else 1.0,
            alpha=0.16 if supplemental else 0.55,
            rasterized=supplemental,
        )
        if not supplemental:
            ax.plot(path[-1, 0], path[-1, 1], "o", color=color, ms=2.4)
    if supplemental and nominal is not None:
        ax.plot(nominal[:, 0], nominal[:, 1], "--", color="#1b2733", lw=1.55)
    ax.plot(0, target, "+", color="#c51f30", ms=10, mew=1.7)
    ax.axvline(0, color="#536577", lw=0.6)
    ax.set_xlim(-xhalf, xhalf)
    ax.set_ylim(0, ymax)
    ax.set_title(label, fontsize=13)
    if index // 3 == rows - 1:
        ax.set_xlabel("Lateral Position (m; Right Positive)", fontsize=11)
    if index % 3 == 0:
        ax.set_ylabel("Downrange Distance (m)", fontsize=11)
    ax.grid(alpha=0.16)
