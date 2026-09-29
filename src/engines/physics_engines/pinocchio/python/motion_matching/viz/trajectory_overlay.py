"""View 1 — Trajectory overlay (measured vs simulated club skeleton).

Side-by-side 3D plots: left = measured, right = simulated, with a faint
clubhead trace, all rendered with matplotlib for static export. Live
animation and the Meshcat 3D viewer are handled by
:mod:`.meshcat_overlay` per VISUALIZATION_SPEC.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .._types import ClubTargetLike, FitResult
from ._style import (
    AXES_FONTSIZE,
    COLOR_MEASURED,
    COLOR_SIMULATED,
    DPI_PNG,
    TITLE_FONTSIZE,
)


def _draw_skeleton(
    ax, butt: np.ndarray, head: np.ndarray, colour: str, label: str
) -> None:
    """Draw the butt→clubhead line at the impact frame plus the head trace."""
    n = butt.shape[0]
    if n == 0:
        return
    impact = min(n - 1, max(0, n // 2))
    # Faint trace of the clubhead path.
    ax.plot(
        head[:, 0],
        head[:, 1],
        head[:, 2],
        color=colour,
        alpha=0.25,
        linewidth=1.0,
    )
    # Single-frame skeleton line at impact.
    ax.plot(
        [butt[impact, 0], head[impact, 0]],
        [butt[impact, 1], head[impact, 1]],
        [butt[impact, 2], head[impact, 2]],
        color=colour,
        linewidth=2.5,
        label=label,
    )
    ax.scatter(
        [butt[impact, 0], head[impact, 0]],
        [butt[impact, 1], head[impact, 1]],
        [butt[impact, 2], head[impact, 2]],
        color=colour,
        s=30,
    )


def plot_trajectory_overlay(
    target: ClubTargetLike,
    result: FitResult,
    *,
    out_path: Path | None = None,
):
    """Render View 1 — measured (left) vs simulated (right) skeletons.

    Args:
        target: Measured trajectory.
        result: Pinocchio fit result; provides ``butt_sim`` / ``clubhead_sim``.
        out_path: If given, write a PNG at 200 DPI to this path. The parent
            directory must already exist.

    Returns:
        The :class:`matplotlib.figure.Figure` instance, so callers can
        further customise or close it.

    Raises:
        ImportError: matplotlib is not installed.
        ValueError: target arrays are inconsistent in length.
    """
    import matplotlib.pyplot as plt  # local import keeps headless modules cheap

    if target.butt.shape[0] != target.clubhead.shape[0]:
        raise ValueError(
            f"target.butt and target.clubhead disagree on length: "
            f"{target.butt.shape[0]} vs {target.clubhead.shape[0]}"
        )

    fig = plt.figure(figsize=(12, 5), dpi=DPI_PNG // 2)
    ax_m = fig.add_subplot(1, 2, 1, projection="3d")
    ax_s = fig.add_subplot(1, 2, 2, projection="3d")

    _draw_skeleton(ax_m, target.butt, target.clubhead, COLOR_MEASURED, "measured")
    _draw_skeleton(
        ax_s, result.butt_sim, result.clubhead_sim, COLOR_SIMULATED, "simulated"
    )

    for ax, title in ((ax_m, "Measured"), (ax_s, "Simulated")):
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
        ax.set_xlabel("x (m)", fontsize=AXES_FONTSIZE)
        ax.set_ylabel("y (m)", fontsize=AXES_FONTSIZE)
        ax.set_zlabel("z (m)", fontsize=AXES_FONTSIZE)
        ax.legend(loc="upper right", fontsize=AXES_FONTSIZE - 1)

    fig.suptitle(f"Trajectory overlay — {result.trial_id}", fontsize=TITLE_FONTSIZE + 1)
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=DPI_PNG, bbox_inches="tight")

    return fig
