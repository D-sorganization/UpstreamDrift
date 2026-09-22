"""View 3: scalar fit-quality card.

Mirror of ``plot_fit_quality_card.m`` (VISUALIZATION_SPEC View 3). Renders
a one-glance summary card with the headline scalars: total cost,
per-term breakdown, impact-frame error, and provenance metadata.

Public API:
    fit_quality_summary    -- compute the headline scalars (no plot).
    plot_fit_quality_card  -- return a :class:`matplotlib.figure.Figure`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._geodesic import quaternion_geodesic_angles
from .cost import CostBreakdown
from .sim_out import SimOut
from .target import ClubTarget

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

__all__ = ["FitQualityScalars", "fit_quality_summary", "plot_fit_quality_card"]


@dataclass(frozen=True)
class FitQualityScalars:
    """Headline scalars for the fit-quality card."""

    rmse_butt_m: float
    rmse_clubhead_m: float
    rmse_orientation_deg: float
    impact_butt_err_m: float
    impact_clubhead_err_m: float
    cost_total: float
    cost_position: float
    cost_orientation: float
    cost_impact_anchor: float
    cost_regularizer: float


def fit_quality_summary(
    target: ClubTarget,
    sim: SimOut,
    breakdown: CostBreakdown,
) -> FitQualityScalars:
    """Compute the scalar metrics shown on the quality card."""
    n = target.time.shape[0]
    if sim.butt.shape != (n, 3) or sim.clubhead.shape != (n, 3):
        raise ValueError(
            "sim butt/clubhead must match target time length and have 3 columns"
        )
    if sim.club_quat.shape != (n, 4):
        raise ValueError("sim.club_quat must have shape (N, 4) matching target.time")

    diff_butt = target.butt - sim.butt
    rmse_butt = float(np.sqrt(np.vdot(diff_butt, diff_butt) / diff_butt.shape[0]))
    diff_ch = target.clubhead - sim.clubhead
    rmse_ch = float(np.sqrt(np.vdot(diff_ch, diff_ch) / diff_ch.shape[0]))
    angles = quaternion_geodesic_angles(target.club_quat, sim.club_quat)
    rmse_ang_deg = float(np.degrees(np.sqrt(np.mean(angles**2))))

    k = int(target.impact_idx) - 1
    if not (0 <= k < n):
        raise ValueError(f"impact_idx {target.impact_idx} out of range for N={n}")
    impact_butt = float(np.linalg.norm(target.butt[k] - sim.butt[k]))
    impact_ch = float(np.linalg.norm(target.clubhead[k] - sim.clubhead[k]))

    return FitQualityScalars(
        rmse_butt_m=rmse_butt,
        rmse_clubhead_m=rmse_ch,
        rmse_orientation_deg=rmse_ang_deg,
        impact_butt_err_m=impact_butt,
        impact_clubhead_err_m=impact_ch,
        cost_total=float(breakdown.total),
        cost_position=float(breakdown.position),
        cost_orientation=float(breakdown.orientation),
        cost_impact_anchor=float(breakdown.impact_anchor),
        cost_regularizer=float(breakdown.regularizer),
    )


def plot_fit_quality_card(
    target: ClubTarget,
    sim: SimOut,
    breakdown: CostBreakdown,
    *,
    title: str | None = None,
) -> Figure:
    """Render a one-glance scalar quality card.

    Args:
        target:    The :class:`ClubTarget`.
        sim:       The :class:`SimOut` to score.
        breakdown: The matching :class:`CostBreakdown`.
        title:     Optional figure title; defaults to the trial id.

    Returns:
        Matplotlib :class:`Figure` containing a single text axes with the
        rendered scalar table.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    import matplotlib.pyplot as plt

    s = fit_quality_summary(target, sim, breakdown)
    src = target.source
    lines = [
        f"Trial:    {src.subject_id} / {src.trial_id}  ({src.format})",
        f"Source:   {src.filename}",
        "",
        f"Total cost J = {s.cost_total:.6g}",
        f"  position    {s.cost_position:.6g}",
        f"  orientation {s.cost_orientation:.6g}",
        f"  impact      {s.cost_impact_anchor:.6g}",
        f"  regularizer {s.cost_regularizer:.6g}",
        "",
        f"RMSE butt        {s.rmse_butt_m * 1000:.3f} mm",
        f"RMSE clubhead    {s.rmse_clubhead_m * 1000:.3f} mm",
        f"RMSE orientation {s.rmse_orientation_deg:.3f} deg",
        "",
        f"Impact butt err     {s.impact_butt_err_m * 1000:.3f} mm",
        f"Impact clubhead err {s.impact_clubhead_err_m * 1000:.3f} mm",
    ]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        family="monospace",
        fontsize=11,
        va="top",
        ha="left",
        transform=ax.transAxes,
    )
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    return fig
