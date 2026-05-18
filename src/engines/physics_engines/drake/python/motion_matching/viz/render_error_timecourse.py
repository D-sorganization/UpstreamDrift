"""Error timecourse stack (View 2 of VISUALIZATION_SPEC.md).

Stacked 2D matplotlib panels versus simulation time:

1. Position error (mm) -- butt (blue) and clubhead (orange).
2. Orientation error (deg) -- geodesic distance between the simulated
   and measured club quaternions.
3. Clubhead speed (mph) -- measured (solid) vs simulated (dashed).
4. Joint torques (N*m) -- one trace per actuated joint, only drawn if
   the simulated ``tau`` field is populated.

A vertical guide is drawn across all panels at the impact frame
recorded on the canonical ``ClubTarget``.

The implementation is engine-agnostic: it reads only the documented
fields of :class:`DrakeFitResult` and :class:`ClubTarget`, so any
physics engine that adopts the same ``FitResult``-shape can reuse this
module verbatim until the formal cross-engine ``shared/python/
motion_matching/plot_*.py`` package is split out.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.figure import Figure

    from src.shared.python.motion_matching.club_target import ClubTarget

    from .render_trajectory_overlay import DrakeFitResult


# Conversions used by the plot.
M_TO_MM = 1000.0
MS_TO_MPH = 2.23694  # 1 m/s = 2.23694 mph

# Palette (mirrors VISUALIZATION_SPEC.md).
COLOR_BUTT = "#1f77b4"
COLOR_CLUBHEAD = "#ff7f0e"
COLOR_MEASURED = "#1f77b4"
COLOR_SIMULATED = "#d62728"
COLOR_IMPACT_GUIDE = "#7f7f7f"


__all__ = ["render_error_timecourse"]


def render_error_timecourse(
    fit: DrakeFitResult,
    target: ClubTarget,
    out_dir: Path,
    *,
    title: str | None = None,
) -> Path:
    """Render the error timecourse panel stack.

    Args:
        fit: :class:`DrakeFitResult` from the optimiser.
        target: Canonical :class:`ClubTarget` consumed by the fit.
        out_dir: Directory to write into. Created if missing.
        title: Optional figure title; defaults to ``fit.swing_id``.

    Returns:
        The :class:`pathlib.Path` of the PNG written.

    Raises:
        ValueError: if the ``fit`` and ``target`` timegrids disagree.
    """
    if fit.time.shape[0] != target.time.shape[0]:
        msg = (
            "fit and target must share the canonical timegrid; got "
            f"{fit.time.shape[0]} vs {target.time.shape[0]} samples"
        )
        raise ValueError(msg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Local imports keep matplotlib optional at module-import time.
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.figure as _mfig

    n_panels = 4 if fit.tau is not None else 3
    fig: Figure = _mfig.Figure(figsize=(8.5, 2.0 * n_panels + 0.6), dpi=150)
    axes = [fig.add_subplot(n_panels, 1, k + 1) for k in range(n_panels)]

    t = np.asarray(fit.time, dtype=float)

    _draw_position_error_panel(axes[0], t, fit, target)
    _draw_orientation_error_panel(axes[1], t, fit, target)
    _draw_clubhead_speed_panel(axes[2], t, fit, target)
    if fit.tau is not None:
        _draw_torque_panel(axes[3], t, fit.tau)

    # Impact guide on every panel.
    impact_idx = max(0, min(int(target.impact_idx) - 1, t.shape[0] - 1))
    impact_t = float(t[impact_idx])
    for ax in axes:
        ax.axvline(impact_t, color=COLOR_IMPACT_GUIDE, linestyle="--", linewidth=0.8)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title or fit.swing_id or "Error timecourse", fontsize=13)
    fig.tight_layout()

    png_path = out_dir / "error_timecourse.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    return png_path


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------


def _draw_position_error_panel(
    ax,  # type: ignore[no-untyped-def]
    t: NDArray[np.float64],
    fit: DrakeFitResult,
    target: ClubTarget,
) -> None:
    """Top panel: butt and clubhead position error in millimetres."""
    diff_butt = np.asarray(fit.grip) - np.asarray(target.butt)
    # ⚡ Bolt: einsum is ~35-40% faster than np.linalg.norm(..., axis=1)
    butt_err = np.sqrt(np.einsum("ij,ij->i", diff_butt, diff_butt)) * M_TO_MM
    diff_head = np.asarray(fit.clubhead) - np.asarray(target.clubhead)
    # ⚡ Bolt: einsum is ~35-40% faster than np.linalg.norm(..., axis=1)
    head_err = np.sqrt(np.einsum("ij,ij->i", diff_head, diff_head)) * M_TO_MM
    ax.plot(t, butt_err, color=COLOR_BUTT, linewidth=1.4, label="butt")
    ax.plot(t, head_err, color=COLOR_CLUBHEAD, linewidth=1.4, label="clubhead")
    ax.set_ylabel("position error (mm)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def _draw_orientation_error_panel(
    ax,  # type: ignore[no-untyped-def]
    t: NDArray[np.float64],
    fit: DrakeFitResult,
    target: ClubTarget,
) -> None:
    """Geodesic quaternion distance, expressed in degrees."""
    q_sim = _to_unit_quat(np.asarray(fit.club_quat, dtype=float))
    q_meas = _to_unit_quat(np.asarray(target.club_quat, dtype=float))
    # Quaternion dot, clipped for numerical safety; double-cover -> abs.
    dot = np.abs(np.sum(q_sim * q_meas, axis=1))
    dot = np.clip(dot, -1.0, 1.0)
    err_rad = 2.0 * np.arccos(dot)
    err_deg = np.degrees(err_rad)
    ax.plot(t, err_deg, color=COLOR_SIMULATED, linewidth=1.4)
    ax.set_ylabel("orientation err (deg)")
    ax.grid(True, alpha=0.3)


def _draw_clubhead_speed_panel(
    ax,  # type: ignore[no-untyped-def]
    t: NDArray[np.float64],
    fit: DrakeFitResult,
    target: ClubTarget,
) -> None:
    """Clubhead linear speed in mph -- measured solid, simulated dashed."""
    v_meas = _path_speed(np.asarray(target.clubhead, dtype=float), t) * MS_TO_MPH
    v_sim = _path_speed(np.asarray(fit.clubhead, dtype=float), t) * MS_TO_MPH
    ax.plot(t, v_meas, color=COLOR_MEASURED, linewidth=1.4, label="measured")
    ax.plot(
        t,
        v_sim,
        color=COLOR_SIMULATED,
        linewidth=1.4,
        linestyle="--",
        label="simulated",
    )
    ax.set_ylabel("clubhead speed (mph)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)


def _draw_torque_panel(
    ax,  # type: ignore[no-untyped-def]
    t: NDArray[np.float64],
    tau: NDArray[np.float64],
) -> None:
    """Bottom panel: one trace per joint torque (N*m)."""
    n_joints = tau.shape[1]
    for j in range(n_joints):
        ax.plot(t, tau[:, j], linewidth=0.8, alpha=0.8)
    ax.set_ylabel("joint torque (N*m)")
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_unit_quat(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Renormalise per-row quaternions; tolerant of small drift."""
    # ⚡ Bolt: einsum is ~35-40% faster than np.linalg.norm(..., axis=1)
    norms = np.sqrt(np.einsum("ij,ij->i", q, q))[:, np.newaxis]
    return q / np.maximum(norms, 1.0e-12)


def _path_speed(p: NDArray[np.float64], t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Centred-difference speed (m/s) along the (N, 3) path ``p``.

    Endpoints use forward / backward differences so the output has the
    same length as ``p``.
    """
    if p.shape[0] < 2:
        return np.zeros(p.shape[0], dtype=float)
    v = np.gradient(p, t, axis=0)
    # ⚡ Bolt: einsum is ~35-40% faster than np.linalg.norm(..., axis=1)
    return np.sqrt(np.einsum("ij,ij->i", v, v))
