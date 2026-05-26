"""View 2 — Error timecourse stacked panels.

Four panels stacked vs simulation time:

1. Position error (mm) — butt and clubhead.
2. Orientation error (deg) — geodesic distance between simulated and
   measured club orientation quaternions.
3. Clubhead speed (mph) — measured (solid) vs simulated (dashed), if speed
   is supplied on the result.
4. Joint torques (N·m) — one trace per joint, if torques are supplied.

A vertical line marks the impact frame across all panels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .._types import ClubTargetLike, FitResult
from ._style import (
    AXES_FONTSIZE,
    COLOR_ERROR,
    COLOR_MEASURED,
    COLOR_SIMULATED,
    DPI_PNG,
    TITLE_FONTSIZE,
)


def _quat_geodesic_deg(q_sim: np.ndarray, q_meas: np.ndarray) -> np.ndarray:
    """Per-frame geodesic angle (deg) between two unit-quaternion arrays.

    Both arrays are ``(N, 4)`` in wxyz order. Returns ``(N,)``.
    """
    n = min(q_sim.shape[0], q_meas.shape[0])
    if n == 0:
        return np.zeros((0,))
    a = q_sim[:n]
    b = q_meas[:n]
    # Robust against double-cover sign ambiguity.
    # ⚡ Bolt: einsum avoids temp arrays and is faster than np.sum(..., axis=1)
    dot = np.clip(np.abs(np.einsum("ij,ij->i", a, b)), -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def plot_error_timecourse(  # noqa: C901
    target: ClubTargetLike,
    result: FitResult,
    *,
    out_path: Path | None = None,
):
    """Render View 2 — stacked error timecourse.

    Args:
        target: Measured trajectory.
        result: Pinocchio fit result.
        out_path: Optional PNG output path (200 DPI).

    Returns:
        The :class:`matplotlib.figure.Figure` instance.

    Raises:
        ImportError: matplotlib is not installed.
    """
    import matplotlib.pyplot as plt

    n = min(
        target.time.shape[0],
        result.time.shape[0] if result.time.size else target.time.shape[0],
    )
    t = np.asarray(target.time[:n], dtype=float)

    # Panel 1 — position error (mm).
    # ⚡ Bolt: einsum is faster than np.linalg.norm(..., axis=1)
    if result.butt_sim.shape[0] >= n:
        butt_diff = result.butt_sim[:n] - target.butt[:n]
        butt_err_mm = np.sqrt(np.einsum("ij,ij->i", butt_diff, butt_diff)) * 1e3
    else:
        butt_err_mm = np.zeros((n,))
    if result.clubhead_sim.shape[0] >= n:
        head_diff = result.clubhead_sim[:n] - target.clubhead[:n]
        head_err_mm = np.sqrt(np.einsum("ij,ij->i", head_diff, head_diff)) * 1e3
    else:
        head_err_mm = np.zeros((n,))

    # Panel 2 — orientation error (deg).
    if result.club_quat_sim.shape[0] >= n:
        ori_err_deg = _quat_geodesic_deg(result.club_quat_sim[:n], target.club_quat[:n])
    else:
        ori_err_deg = np.zeros((n,))

    # Panel 3 — clubhead speed (mph).
    if result.clubhead_speed_mph is not None and result.clubhead_speed_mph.size >= n:
        speed_sim_mph = np.asarray(result.clubhead_speed_mph[:n], dtype=float)
    else:
        speed_sim_mph = None

    # Panel 4 — joint torques (N·m).
    if result.joint_torques is not None and result.joint_torques.size:
        torques = np.asarray(result.joint_torques, dtype=float)
        torques = torques[: min(torques.shape[0], n)]
    else:
        torques = None

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, dpi=DPI_PNG // 2)

    ax = axes[0]
    ax.plot(t, butt_err_mm, color=COLOR_MEASURED, label="butt")
    ax.plot(t, head_err_mm, color="#ff7f0e", label="clubhead")
    ax.set_ylabel("Position error (mm)", fontsize=AXES_FONTSIZE)
    ax.legend(loc="upper right", fontsize=AXES_FONTSIZE - 1)

    ax = axes[1]
    ax.plot(t, ori_err_deg, color=COLOR_ERROR)
    ax.set_ylabel("Orientation error (deg)", fontsize=AXES_FONTSIZE)

    ax = axes[2]
    if speed_sim_mph is not None:
        ax.plot(
            t, speed_sim_mph, color=COLOR_SIMULATED, linestyle="--", label="simulated"
        )
    # Numerical speed of the measured clubhead by central differences.
    if n >= 3:
        dt = np.gradient(t)
        grad_head = np.gradient(target.clubhead[:n], axis=0)
        v_meas = np.sqrt(np.einsum("ij,ij->i", grad_head, grad_head))
        # Avoid divide-by-zero at degenerate steps.
        v_meas = np.divide(v_meas, np.where(dt > 0, dt, 1.0))
        speed_meas_mph = v_meas * 2.23693629
        ax.plot(t, speed_meas_mph, color=COLOR_MEASURED, label="measured")
    ax.set_ylabel("Clubhead speed (mph)", fontsize=AXES_FONTSIZE)
    ax.legend(loc="upper right", fontsize=AXES_FONTSIZE - 1)

    ax = axes[3]
    if torques is not None and torques.ndim == 2 and torques.shape[1] > 0:
        for j in range(torques.shape[1]):
            ax.plot(t[: torques.shape[0]], torques[:, j], linewidth=0.8)
    ax.set_ylabel("Joint torques (N·m)", fontsize=AXES_FONTSIZE)
    ax.set_xlabel("time (s)", fontsize=AXES_FONTSIZE)

    # Impact line across all panels.
    impact_idx = int(getattr(target, "impact_idx", n - 1))
    impact_idx = max(0, min(impact_idx, n - 1))
    if t.size:
        for ax_i in axes:
            ax_i.axvline(
                float(t[impact_idx]), color=COLOR_ERROR, linestyle=":", linewidth=1
            )

    fig.suptitle(f"Error timecourse — {result.trial_id}", fontsize=TITLE_FONTSIZE + 1)
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=DPI_PNG, bbox_inches="tight")

    return fig
