"""View 1: trajectory overlay plot.

Mirror of ``plot_trajectory_overlay.m`` (VISUALIZATION_SPEC View 1).
Shows clubhead trace and butt trace for sim vs target side-by-side. The
function is matplotlib-backed and returns the ``Figure`` so callers can
embed it (e.g. in a PyQt6 dock or save it to disk).

Matplotlib is imported lazily so importing this module on a headless host
without a backend installed doesn't break the package surface.

Public API:
    plot_trajectory_overlay -- return a :class:`matplotlib.figure.Figure`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .sim_out import SimOut
from .target import ClubTarget

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

__all__ = ["plot_trajectory_overlay"]


def plot_trajectory_overlay(
    target: ClubTarget,
    sim: SimOut,
    *,
    title: str | None = None,
) -> Figure:
    """Two-panel overlay of measured vs simulated clubhead/butt traces.

    Args:
        target: Measured :class:`ClubTarget`.
        sim:    Simulated :class:`SimOut` on the same timegrid.
        title:  Optional figure title.

    Returns:
        Matplotlib :class:`Figure`. Caller is responsible for ``plt.show()``
        / ``fig.savefig()``.

    Raises:
        ValueError: If shapes don't match.
        ImportError: If matplotlib is not installed.
    """
    if target.butt.shape != sim.butt.shape:
        raise ValueError(
            f"butt shape mismatch: target {target.butt.shape} vs sim {sim.butt.shape}"
        )
    if target.clubhead.shape != sim.clubhead.shape:
        raise ValueError(
            f"clubhead shape mismatch: target {target.clubhead.shape} vs sim {sim.clubhead.shape}"
        )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(target.butt[:, 0], target.butt[:, 1], "k-", label="butt (meas)")
    axes[0].plot(sim.butt[:, 0], sim.butt[:, 1], "r--", label="butt (sim)")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].set_title("Butt trace (XY)")
    axes[0].axis("equal")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        target.clubhead[:, 0], target.clubhead[:, 1], "k-", label="clubhead (meas)"
    )
    axes[1].plot(sim.clubhead[:, 0], sim.clubhead[:, 1], "r--", label="clubhead (sim)")
    k = int(target.impact_idx) - 1
    if 0 <= k < target.clubhead.shape[0]:
        axes[1].plot(
            target.clubhead[k, 0],
            target.clubhead[k, 1],
            "go",
            markersize=10,
            label="impact",
        )
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Y (m)")
    axes[1].set_title("Clubhead trace (XY)")
    axes[1].axis("equal")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    diff_butt = target.butt - sim.butt
    rmse_butt = float(np.sqrt(np.vdot(diff_butt, diff_butt) / diff_butt.shape[0]))
    diff_ch = target.clubhead - sim.clubhead
    rmse_ch = float(np.sqrt(np.vdot(diff_ch, diff_ch) / diff_ch.shape[0]))
    logger.debug(
        "trajectory_overlay: butt RMSE %.4g m, clubhead RMSE %.4g m", rmse_butt, rmse_ch
    )
    return fig
