"""Screw Theory Visualization Tools.

This module provides agnostic plotting capabilities for testing and rendering
screw axes out-of-band using `matplotlib`. The 3D render logic is completely
abstracted away from the physics engine representations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.shared.python.screw_theory.kinematics import ScrewAxis, compute_screw_endpoints


def plot_screw_axis_3d(
    ax: Any,  # type: ignore[misc]
    screw: ScrewAxis,
    length: float = 0.5,
    color: str = "blue",
    label: str | None = None,
) -> None:
    """Plot screw axis in 3D matplotlib axes.

    Helper function for visualizing screw axes. Requires `ax` to be a
    matplotlib 3D Axes3D subplot projection object.

    Args:
        ax: Matplotlib 3D axes
        screw: Screw axis to plot
        length: Length of axis to draw [m]
        color: Color for the axis
        label: Label for legend
    """
    if not (screw is not None):
        raise ValueError("screw must be provided")

    start, end = compute_screw_endpoints(screw, length)

    # Draw axis as line
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
        color=color,
        linewidth=3,
        label=label,
    )

    # Draw arrow at end showing torque/twist direction
    arrow_length = length * 0.1
    arrow = end - start

    # safeguard division by zero
    arrow_norm_scale = float(np.linalg.norm(arrow))
    if arrow_norm_scale > 1e-10:
        arrow_norm = arrow / arrow_norm_scale
        ax.quiver(
            end[0],
            end[1],
            end[2],
            arrow_norm[0],
            arrow_norm[1],
            arrow_norm[2],
            length=arrow_length,
            color=color,
            arrow_length_ratio=0.3,
        )

    # Annotate pitch if not singular
    if not screw.is_singular and abs(screw.pitch) < 10:
        mid = (start + end) / 2.0
        ax.text(
            mid[0],
            mid[1],
            mid[2],
            f"h={screw.pitch:.3f}",
            fontsize=8,
        )
