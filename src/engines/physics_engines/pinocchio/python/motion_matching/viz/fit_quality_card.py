"""View 3 — Fit quality summary card.

Single-figure summary safe to drop into a PR or status update. Renders
RMSEs, work, wall clock, and provenance text in a tidy text block, suitable
for ``.png`` export at 200 DPI.
"""

from __future__ import annotations

from pathlib import Path

from .._types import ClubTargetLike, FitResult
from ._style import AXES_FONTSIZE, DPI_PNG, TITLE_FONTSIZE


def plot_fit_quality_card(
    target: ClubTargetLike,
    result: FitResult,
    *,
    out_path: Path | None = None,
):
    """Render View 3 — single-figure summary card.

    Args:
        target: Measured trajectory; only its ``impact_idx`` is consulted.
        result: Pinocchio fit result; supplies all numbers.
        out_path: Optional PNG output path (200 DPI).

    Returns:
        The :class:`matplotlib.figure.Figure` instance.

    Raises:
        ImportError: matplotlib is not installed.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI_PNG // 2)
    ax.set_axis_off()

    title = f"Swing fit summary — {result.trial_id}"
    ax.set_title(title, fontsize=TITLE_FONTSIZE + 1, loc="left")

    # The impact frame is part of the target spec (CLUB_IK_SPEC.md).
    impact_idx = int(getattr(target, "impact_idx", -1))

    lines = [
        f"Solver:                 {result.solver}",
        f"Iterations:             {result.n_iterations}",
        f"Wall clock:             {result.wall_clock_s:.2f} s",
        "",
        f"Final RMSE — clubhead position:   {result.clubhead_rmse_mm:.2f} mm",
        f"Final RMSE — butt position:       {result.grip_rmse_mm:.2f} mm",
        f"Final mean orientation error:     {result.orientation_rmse_deg:.3f} deg",
        "",
        f"Total work (regularised): {result.total_work_J:.2f} J",
        f"Impact frame index:       {impact_idx}",
        "",
        f"Hash:   {result.commit}",
    ]
    text = "\n".join(lines)
    ax.text(
        0.02,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=AXES_FONTSIZE,
        family="monospace",
        verticalalignment="top",
    )

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=DPI_PNG, bbox_inches="tight")

    return fig
