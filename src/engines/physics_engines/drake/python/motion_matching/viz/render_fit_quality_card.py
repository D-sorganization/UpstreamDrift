"""Fit quality summary card (View 3 of VISUALIZATION_SPEC.md).

A single 2D matplotlib figure with the headline RMSE metrics laid out
as a card -- safe to drop into a PR or a status update without further
formatting. The card mirrors the ASCII mock in VISUALIZATION_SPEC.md:

* swing id / solver / iteration count / wall clock,
* clubhead RMSE, butt RMSE, mean orientation error,
* impact-frame clubhead speed (sim vs measured),
* commit hash + branch footer.

The card is engine-agnostic: it consumes the same
:class:`DrakeFitResult` + canonical :class:`ClubTarget` as the other
two views, so any parity engine can adopt it once it produces a
``FitResult``-shaped bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.shared.python.motion_matching.club_target import ClubTarget

    from .render_trajectory_overlay import DrakeFitResult


M_TO_MM = 1000.0
MS_TO_MPH = 2.23694


__all__ = [
    "FitQualityMetrics",
    "compute_fit_quality_metrics",
    "render_fit_quality_card",
]


@dataclass(frozen=True)
class FitQualityMetrics:
    """Headline metrics rendered onto the summary card.

    Attributes:
        clubhead_rmse_mm: RMS clubhead position error (mm).
        butt_rmse_mm:     RMS butt position error (mm).
        mean_orient_err_deg: Mean geodesic orientation error (deg).
        impact_speed_sim_mph: Simulated clubhead speed at impact (mph).
        impact_speed_meas_mph: Measured clubhead speed at impact (mph).
    """

    clubhead_rmse_mm: float
    butt_rmse_mm: float
    mean_orient_err_deg: float
    impact_speed_sim_mph: float
    impact_speed_meas_mph: float


def compute_fit_quality_metrics(
    fit: DrakeFitResult, target: ClubTarget
) -> FitQualityMetrics:
    """Compute the headline metrics rendered on the summary card.

    Public so the same numbers can be logged or written to JSON without
    re-rendering the matplotlib card.
    """
    if fit.time.shape[0] != target.time.shape[0]:
        msg = (
            "fit and target must share the canonical timegrid; got "
            f"{fit.time.shape[0]} vs {target.time.shape[0]} samples"
        )
        raise ValueError(msg)
    head_err_m = np.linalg.norm(
        np.asarray(fit.clubhead, dtype=float)
        - np.asarray(target.clubhead, dtype=float),
        axis=1,
    )
    butt_err_m = np.linalg.norm(
        np.asarray(fit.grip, dtype=float) - np.asarray(target.butt, dtype=float),
        axis=1,
    )
    orient_err_deg = _orientation_error_deg(
        np.asarray(fit.club_quat, dtype=float),
        np.asarray(target.club_quat, dtype=float),
    )
    impact_idx = max(0, min(int(target.impact_idx) - 1, fit.time.shape[0] - 1))
    sim_speed_mph = _impact_speed_mph(
        np.asarray(fit.clubhead, dtype=float), fit.time, impact_idx
    )
    meas_speed_mph = _impact_speed_mph(
        np.asarray(target.clubhead, dtype=float), fit.time, impact_idx
    )
    return FitQualityMetrics(
        clubhead_rmse_mm=float(np.sqrt(np.mean(head_err_m**2)) * M_TO_MM),
        butt_rmse_mm=float(np.sqrt(np.mean(butt_err_m**2)) * M_TO_MM),
        mean_orient_err_deg=float(np.mean(orient_err_deg)),
        impact_speed_sim_mph=float(sim_speed_mph),
        impact_speed_meas_mph=float(meas_speed_mph),
    )


def render_fit_quality_card(
    fit: DrakeFitResult,
    target: ClubTarget,
    out_dir: Path,
    *,
    title: str | None = None,
) -> Path:
    """Render the fit quality summary card.

    Args:
        fit: :class:`DrakeFitResult` from the optimiser.
        target: Canonical :class:`ClubTarget` consumed by the fit.
        out_dir: Directory to write into. Created if missing.
        title: Optional override; defaults to ``fit.swing_id``.

    Returns:
        :class:`pathlib.Path` of the PNG written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_fit_quality_metrics(fit, target)

    # Local imports keep matplotlib optional at module-import time.
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.figure as _mfig

    fig = _mfig.Figure(figsize=(7.5, 5.0), dpi=150)
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    header = title or fit.swing_id or "Fit quality summary"
    solver_line = (
        f"Solver: drake float-pathway   Iterations: {fit.n_iterations}   "
        f"Wall clock: {fit.wall_clock_s:.1f}s"
    )

    body_lines = [
        f"Final RMSE - clubhead position : {metrics.clubhead_rmse_mm:6.2f} mm",
        f"Final RMSE - butt position     : {metrics.butt_rmse_mm:6.2f} mm",
        f"Final mean orientation error   : {metrics.mean_orient_err_deg:6.3f} deg",
        (
            f"Clubhead speed at impact       : "
            f"{metrics.impact_speed_sim_mph:6.1f} mph "
            f"(meas: {metrics.impact_speed_meas_mph:.1f})"
        ),
        f"Final loss                     : {fit.final_loss:.6e}",
        f"Solver status                  : {fit.solver_status}",
    ]
    footer = f"Hash: {fit.commit_hash or '-'}     Branch: {fit.branch or '-'}"

    # Stack the lines vertically; explicit y coordinates keep the layout
    # deterministic across matplotlib versions.
    ax.text(
        0.02,
        0.95,
        header,
        fontsize=14,
        fontweight="bold",
        transform=ax.transAxes,
        verticalalignment="top",
    )
    ax.text(
        0.02,
        0.86,
        solver_line,
        fontsize=10,
        transform=ax.transAxes,
        verticalalignment="top",
    )
    y = 0.74
    for line in body_lines:
        ax.text(
            0.04,
            y,
            line,
            fontsize=11,
            family="monospace",
            transform=ax.transAxes,
            verticalalignment="top",
        )
        y -= 0.08
    ax.text(
        0.02,
        0.05,
        footer,
        fontsize=9,
        color="#555555",
        transform=ax.transAxes,
        verticalalignment="bottom",
    )

    # Card border for the at-a-glance look.
    ax.add_patch(_card_border())

    png_path = out_dir / "fit_quality_card.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    return png_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _card_border():  # type: ignore[no-untyped-def]
    """Return a matplotlib ``Rectangle`` patch for the card border."""
    from matplotlib.patches import Rectangle

    return Rectangle(
        (0.005, 0.005),
        0.99,
        0.99,
        transform=None,
        fill=False,
        linewidth=1.0,
        edgecolor="#cccccc",
    )


def _orientation_error_deg(
    q_sim: NDArray[np.float64], q_meas: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Per-frame geodesic orientation error in degrees."""
    n_sim = np.linalg.norm(q_sim, axis=1, keepdims=True)
    n_meas = np.linalg.norm(q_meas, axis=1, keepdims=True)
    a = q_sim / np.maximum(n_sim, 1.0e-12)
    b = q_meas / np.maximum(n_meas, 1.0e-12)
    dot = np.abs(np.sum(a * b, axis=1))
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def _impact_speed_mph(
    path: NDArray[np.float64], t: NDArray[np.float64], impact_idx: int
) -> float:
    """Linear speed (mph) at ``impact_idx`` of a (N, 3) path."""
    if path.shape[0] < 2:
        return 0.0
    v = np.gradient(path, t, axis=0)
    return float(np.linalg.norm(v[impact_idx]) * MS_TO_MPH)
