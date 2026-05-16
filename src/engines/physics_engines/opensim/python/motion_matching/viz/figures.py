"""The three canonical visualisation figures (issue #4130).

Each public entry point takes the same first two arguments — a target
trajectory and a sim output — and returns a ``matplotlib.figure.Figure``.
The signatures are uniform across engines per VISUALIZATION_SPEC.md so a
cross-engine comparison plot can render any engine's results without
glue code.

Headless safety
---------------

These functions never call ``plt.show`` and rely on the
``matplotlib.figure.Figure`` API directly so callers are free to use any
backend (Agg in CI, Qt for interactive use). No ``matplotlib`` warnings
are emitted under pytest — every potentially-noisy operation (axes
sharing, missing optional series) is gated.
"""

from __future__ import annotations

from typing import Any, cast

import matplotlib

# Ensure a non-interactive backend is selected when this module is imported
# from a context that has not already configured one (e.g. pytest under
# headless CI). Setting the backend after pyplot import is a noop so we
# guard with ``get_backend()`` to avoid the warning.
if matplotlib.get_backend().lower() == "agg" or "pytest" in matplotlib.rcParams.get(
    "backend", ""
):
    pass  # already non-interactive
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402  # registers 3-D projection

from . import _adapters
from ._adapters import (
    AXES_FONTSIZE,
    COLOR_ERROR,
    COLOR_MEASURED,
    COLOR_SIMULATED,
    TITLE_FONTSIZE,
    quat_geodesic_deg,
)
from .native import render_with_opensim_visualizer

__all__ = [
    "plot_trajectory_overlay",
    "plot_error_timecourse",
    "plot_fit_quality_card",
    "render_with_opensim_visualizer",
]


# ---------------------------------------------------------------------------
# View 1 — Trajectory overlay
# ---------------------------------------------------------------------------


def _opensim_visualizer_available() -> bool:
    """Return ``True`` iff ``opensim.Visualizer`` can be imported."""
    try:
        import opensim  # noqa: F401
    except ImportError:  # pragma: no cover - depends on host install
        return False
    # The Visualizer class is part of the SimTK Java/OpenGL viewer.
    return hasattr(__import__("opensim"), "Visualizer")


def plot_trajectory_overlay(
    target: Any,
    sim_out: Any,
    *,
    use_opensim_visualizer: bool = False,
    model: Any = None,
    title: str | None = None,
) -> Figure:
    """View 1 — measured vs simulated club path in 3-D.

    By default a matplotlib 3-D figure is returned (headless-safe). When
    ``use_opensim_visualizer=True`` the function delegates to
    :func:`render_with_opensim_visualizer` and returns a placeholder
    matplotlib figure that records the visualizer was launched (so callers
    that always expect a ``Figure`` keep working).
    """
    if use_opensim_visualizer:
        # The interactive path. Always still returns a Figure so the
        # caller has something to attach to a report.
        render_with_opensim_visualizer(model=model, sim_out=sim_out)
        fig = plt.figure(figsize=(5.0, 1.5))
        fig.text(
            0.5,
            0.5,
            "OpenSim interactive Visualizer launched in a separate window.",
            ha="center",
            va="center",
            fontsize=AXES_FONTSIZE,
        )
        fig.suptitle(title or "Trajectory overlay", fontsize=TITLE_FONTSIZE)
        return fig

    return _plot_trajectory_overlay_matplotlib(target, sim_out, title=title)


def _plot_trajectory_overlay_matplotlib(
    target: Any,
    sim_out: Any,
    *,
    title: str | None,
) -> Figure:
    """Matplotlib 3-D fallback for View 1 (always headless-safe)."""
    t_meas = _adapters.normalise(target)
    t_sim = _adapters.normalise(sim_out)

    fig = plt.figure(figsize=(11.0, 5.0))
    ax_meas = cast(Axes3D, fig.add_subplot(1, 2, 1, projection="3d"))
    ax_sim = cast(Axes3D, fig.add_subplot(1, 2, 2, projection="3d"))

    _draw_club_path(ax_meas, t_meas, color=COLOR_MEASURED, label="measured")
    _draw_club_path(ax_sim, t_sim, color=COLOR_SIMULATED, label="simulated")

    # Tie axes limits so the eye sees absolute drift, not auto-rescaling.
    bounds = _shared_bounds(t_meas.clubhead, t_sim.clubhead)
    for ax in (ax_meas, ax_sim):
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_zlim(bounds[2])
        ax.set_xlabel("x (m)", fontsize=AXES_FONTSIZE)
        ax.set_ylabel("y (m)", fontsize=AXES_FONTSIZE)
        ax.set_zlabel("z (m)", fontsize=AXES_FONTSIZE)

    ax_meas.set_title("Measured", fontsize=TITLE_FONTSIZE)
    ax_sim.set_title("Simulated", fontsize=TITLE_FONTSIZE)

    fig.suptitle(title or "Trajectory overlay", fontsize=TITLE_FONTSIZE)
    fig.tight_layout()
    return fig


def _draw_club_path(
    ax: Any,
    series: _adapters._NormalisedSeries,
    *,
    color: str,
    label: str,
) -> None:
    """Render a clubhead path plus optional butt-clubhead skeleton."""
    head = series.clubhead
    ax.plot(head[:, 0], head[:, 1], head[:, 2], color=color, label=f"{label} clubhead")
    if series.butt is not None:
        butt = series.butt
        ax.plot(
            butt[:, 0],
            butt[:, 1],
            butt[:, 2],
            color=color,
            linestyle="--",
            alpha=0.7,
            label=f"{label} butt",
        )
        # A thin shaft connector at the impact frame, mid-frame, and start.
        sample_idxs = _shaft_sample_indices(series)
        for i in sample_idxs:
            ax.plot(
                [butt[i, 0], head[i, 0]],
                [butt[i, 1], head[i, 1]],
                [butt[i, 2], head[i, 2]],
                color=color,
                alpha=0.3,
                linewidth=1.0,
            )
    ax.legend(fontsize=AXES_FONTSIZE - 2, loc="best")


def _shaft_sample_indices(series: _adapters._NormalisedSeries) -> list[int]:
    """Pick a small set of frames at which to draw the club shaft."""
    n = series.time.size
    out: list[int] = [0, n // 2, n - 1]
    if series.impact_idx is not None and 0 <= int(series.impact_idx) - 1 < n:
        out.append(int(series.impact_idx) - 1)
    return sorted(set(out))


def _shared_bounds(
    a: np.ndarray,
    b: np.ndarray,
) -> list[tuple[float, float]]:
    """Return per-axis ``(min, max)`` bounds covering both arrays.

    Adds a 5% margin so the figure has visible padding.
    """
    stacked = np.vstack([a, b])
    bounds: list[tuple[float, float]] = []
    for axis in range(3):
        col = stacked[:, axis]
        lo = float(col.min())
        hi = float(col.max())
        span = hi - lo
        if span < 1e-9:
            span = 1.0
        margin = 0.05 * span
        bounds.append((lo - margin, hi + margin))
    return bounds


# ---------------------------------------------------------------------------
# View 2 — Error timecourse
# ---------------------------------------------------------------------------


def plot_error_timecourse(
    target: Any,
    sim_out: Any,
    *,
    title: str | None = None,
) -> Figure:
    """View 2 — stacked 2-D error plots versus simulation time.

    Panels (top to bottom):

    1. Position error in mm — butt (blue) and clubhead (orange).
    2. Orientation error in degrees — geodesic distance between
       ``club_quat`` rows. Skipped when either side lacks quaternions.
    3. Clubhead speed in mph — measured (solid) vs simulated (dashed).
    4. Joint torques in N·m — one trace per joint. Skipped when the
       sim output does not carry torques.

    A vertical dashed line marks the impact frame (when known) on every
    panel.
    """
    t_meas = _adapters.normalise(target)
    t_sim = _adapters.normalise(sim_out)

    panels = _enabled_panels(t_meas, t_sim)
    n_panels = len(panels)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(8.0, max(2.0, 1.6 * n_panels) + 0.5),
        sharex=True,
        squeeze=False,
    )
    axes_flat = list(axes[:, 0])

    for ax, panel_name in zip(axes_flat, panels, strict=True):
        _draw_panel(ax, panel_name, t_meas, t_sim)

    # Common impact line.
    impact_t = _impact_time(t_meas, t_sim)
    if impact_t is not None:
        for ax in axes_flat:
            ax.axvline(impact_t, color=COLOR_ERROR, linestyle=":", linewidth=0.8)

    axes_flat[-1].set_xlabel("time (s)", fontsize=AXES_FONTSIZE)
    fig.suptitle(title or "Error timecourse", fontsize=TITLE_FONTSIZE)
    fig.tight_layout()
    return fig


def _enabled_panels(
    target: _adapters._NormalisedSeries,
    sim: _adapters._NormalisedSeries,
) -> list[str]:
    """Decide which of the four panels to render based on inputs."""
    panels = ["position"]
    if target.club_quat is not None and sim.club_quat is not None:
        panels.append("orientation")
    panels.append("speed")
    if sim.joint_torques is not None:
        panels.append("torques")
    return panels


def _interp_to_target(
    t_target: np.ndarray,
    t_sim: np.ndarray,
    series: np.ndarray,
) -> np.ndarray:
    """Interpolate a sim-time series onto the target time grid."""
    if series.ndim == 1:
        return np.interp(t_target, t_sim, series)
    out = np.empty((t_target.size, series.shape[1]), dtype=float)
    for col in range(series.shape[1]):
        out[:, col] = np.interp(t_target, t_sim, series[:, col])
    return out


def _draw_panel(
    ax: Any,
    name: str,
    target: _adapters._NormalisedSeries,
    sim: _adapters._NormalisedSeries,
) -> None:
    if name == "position":
        _draw_position_error(ax, target, sim)
    elif name == "orientation":
        _draw_orientation_error(ax, target, sim)
    elif name == "speed":
        _draw_speed(ax, target, sim)
    elif name == "torques":
        _draw_torques(ax, sim)
    else:  # pragma: no cover - guarded by _enabled_panels
        raise ValueError(f"unknown panel: {name}")


def _draw_position_error(
    ax: Any,
    target: _adapters._NormalisedSeries,
    sim: _adapters._NormalisedSeries,
) -> None:
    sim_clubhead = _interp_to_target(target.time, sim.time, sim.clubhead)
    diff_head = target.clubhead - sim_clubhead
    # ⚡ Bolt: np.sqrt(np.einsum) avoids temporary allocations and is faster than np.linalg.norm(..., axis=1)
    head_err_mm = 1000.0 * np.sqrt(np.einsum("ij,ij->i", diff_head, diff_head))
    ax.plot(target.time, head_err_mm, color="#ff7f0e", label="clubhead")
    if target.butt is not None and sim.butt is not None:
        sim_butt = _interp_to_target(target.time, sim.time, sim.butt)
        diff_butt = target.butt - sim_butt
        # ⚡ Bolt: np.sqrt(np.einsum) avoids temporary allocations and is faster than np.linalg.norm(..., axis=1)
        butt_err_mm = 1000.0 * np.sqrt(np.einsum("ij,ij->i", diff_butt, diff_butt))
        ax.plot(target.time, butt_err_mm, color=COLOR_MEASURED, label="butt")
    ax.set_ylabel("Position\nerror (mm)", fontsize=AXES_FONTSIZE)
    ax.legend(fontsize=AXES_FONTSIZE - 2, loc="upper right")


def _draw_orientation_error(
    ax: Any,
    target: _adapters._NormalisedSeries,
    sim: _adapters._NormalisedSeries,
) -> None:
    assert target.club_quat is not None and sim.club_quat is not None
    sim_quat = _interp_to_target(target.time, sim.time, sim.club_quat)
    # Renormalise after interpolation.
    # ⚡ Bolt: np.sqrt(np.einsum) avoids temporary allocations and is faster than np.linalg.norm(..., axis=1)
    sim_quat = (
        sim_quat / np.sqrt(np.einsum("ij,ij->i", sim_quat, sim_quat))[:, np.newaxis]
    )
    err_deg = quat_geodesic_deg(target.club_quat, sim_quat)
    ax.plot(target.time, err_deg, color=COLOR_ERROR)
    ax.set_ylabel("Orientation\nerror (deg)", fontsize=AXES_FONTSIZE)


_MS_TO_MPH = 2.236936


def _draw_speed(
    ax: Any,
    target: _adapters._NormalisedSeries,
    sim: _adapters._NormalisedSeries,
) -> None:
    if target.clubhead_speed is not None:
        ax.plot(
            target.time,
            target.clubhead_speed * _MS_TO_MPH,
            color=COLOR_MEASURED,
            label="measured",
        )
    if sim.clubhead_speed is not None:
        ax.plot(
            sim.time,
            sim.clubhead_speed * _MS_TO_MPH,
            color=COLOR_SIMULATED,
            linestyle="--",
            label="simulated",
        )
    ax.set_ylabel("Clubhead\nspeed (mph)", fontsize=AXES_FONTSIZE)
    ax.legend(fontsize=AXES_FONTSIZE - 2, loc="upper left")


def _draw_torques(
    ax: Any,
    sim: _adapters._NormalisedSeries,
) -> None:
    assert sim.joint_torques is not None
    n_joints = sim.joint_torques.shape[1]
    cmap = plt.get_cmap("tab20")
    for j in range(n_joints):
        ax.plot(
            sim.time,
            sim.joint_torques[:, j],
            color=cmap(j % 20),
            linewidth=0.8,
        )
    ax.set_ylabel("Joint\ntorques (N·m)", fontsize=AXES_FONTSIZE)


def _impact_time(
    target: _adapters._NormalisedSeries,
    sim: _adapters._NormalisedSeries,
) -> float | None:
    for series in (target, sim):
        if series.impact_idx is None:
            continue
        idx = int(series.impact_idx) - 1  # 1-based per ClubTarget convention
        if 0 <= idx < series.time.size:
            return float(series.time[idx])
    return None


# ---------------------------------------------------------------------------
# View 3 — Fit quality card
# ---------------------------------------------------------------------------


def plot_fit_quality_card(
    fit_result: Any,
    *,
    title: str | None = None,
) -> Figure:
    """View 3 — single-figure summary card for PRs and status updates.

    The card renders whatever fields the supplied ``fit_result`` exposes
    so the function works with the canonical ``FitResult`` dataclass once
    it lands as well as with the simpler dict / namespace stand-ins used
    by the smoke tests.
    """
    rows = _summary_rows(fit_result)

    fig = plt.figure(figsize=(7.5, 5.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")

    header = _attr_str(fit_result, "swing_id") or "OpenSim fit"
    solver = _attr_str(fit_result, "solver") or _attr_str(fit_result, "solver_status")
    iterations = _attr_str(fit_result, "iterations")
    wall = _attr_str(fit_result, "wall_clock") or _attr_str(fit_result, "duration_s")

    lines: list[str] = [f"Swing: {header}"]
    if solver:
        lines.append(f"Solver: {solver}")
    if iterations or wall:
        meta_parts = []
        if iterations:
            meta_parts.append(f"Iterations: {iterations}")
        if wall:
            meta_parts.append(f"Wall clock: {wall}")
        lines.append("   ".join(meta_parts))
    lines.append("")  # blank separator

    for label, value in rows:
        lines.append(f"{label:<32}{value}")

    commit = _attr_str(fit_result, "commit") or _attr_str(fit_result, "hash")
    branch = _attr_str(fit_result, "branch")
    if commit or branch:
        lines.append("")
        provenance = []
        if commit:
            provenance.append(f"Hash: {commit}")
        if branch:
            provenance.append(f"Branch: {branch}")
        lines.append("   ".join(provenance))

    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        fontfamily="monospace",
        fontsize=AXES_FONTSIZE,
        va="top",
        ha="left",
    )
    fig.suptitle(title or "Fit quality summary", fontsize=TITLE_FONTSIZE)
    fig.tight_layout()
    return fig


def _summary_rows(fit_result: Any) -> list[tuple[str, str]]:
    """Return a list of ``(label, value)`` pairs for the card body."""
    rows: list[tuple[str, str]] = []

    rmse_clubhead = _adapters._attr(fit_result, "rmse_clubhead_mm")
    if rmse_clubhead is None:
        rmse_clubhead = _adapters._attr(fit_result, "final_rmse_clubhead_mm")
    rmse_butt = _adapters._attr(fit_result, "rmse_butt_mm")
    if rmse_butt is None:
        rmse_butt = _adapters._attr(fit_result, "final_rmse_butt_mm")
    orient_err = _adapters._attr(fit_result, "mean_orientation_error_deg")
    speed_at_impact = _adapters._attr(fit_result, "clubhead_speed_at_impact_mph")
    measured_speed_at_impact = _adapters._attr(
        fit_result, "measured_clubhead_speed_at_impact_mph"
    )
    work = _adapters._attr(fit_result, "total_work_J")
    peak_power = _adapters._attr(fit_result, "peak_joint_power_kW")

    if rmse_clubhead is not None:
        rows.append(
            ("Final RMSE — clubhead position:", f"{float(rmse_clubhead):.2f} mm")
        )
    if rmse_butt is not None:
        rows.append(("Final RMSE — butt position:", f"{float(rmse_butt):.2f} mm"))
    if orient_err is not None:
        rows.append(("Final mean orientation error:", f"{float(orient_err):.2f}°"))
    if speed_at_impact is not None:
        meas = (
            f" (meas: {float(measured_speed_at_impact):.0f})"
            if measured_speed_at_impact is not None
            else ""
        )
        rows.append(
            (
                "Final clubhead speed at impact:",
                f"{float(speed_at_impact):.0f} mph{meas}",
            )
        )
    if work is not None:
        rows.append(("Total work (regularized):", f"{float(work):.0f} J"))
    if peak_power is not None:
        rows.append(("Peak joint power:", f"{float(peak_power):.2f} kW"))

    if not rows:
        # Nothing usable came in; emit a single placeholder row so the
        # card is never empty (this also guarantees a non-trivial axis
        # for the smoke tests).
        rows.append(("Fit summary:", "no metrics provided"))
    return rows


def _attr_str(obj: Any, name: str) -> str | None:
    """Stringify ``obj.name`` if present (and not ``None``)."""
    val = _adapters._attr(obj, name)
    if val is None:
        return None
    return str(val)
