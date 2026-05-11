"""Compare two ``ClubTarget`` clubhead traces (measured vs simulated).

This module produces a numeric :class:`TraceCompareReport` plus four
matplotlib figures suitable for offline review:

1. 3D trajectory overlay
2. Per-axis time-series with shaded delta
3. Clubhead speed comparison (mph)
4. Setup-pose skeleton overlay at ``t = 0``

Both inputs must be :class:`ClubTarget` instances. Time alignment is performed
before metric computation per :attr:`TraceCompareOptions.time_alignment`.

All metric units are explicit in their field names: distances in millimetres,
speeds in miles per hour, time in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .._geodesic import quaternion_geodesic_angles
from ..club_target import ClubTarget
from ..loaders._quaternion import slerp

# Public unit conversion constants.
_MPS_TO_MPH = 2.236936292054402
_M_TO_MM = 1000.0

TimeAlignment = Literal["impact", "address", "none"]


@dataclass(frozen=True)
class TraceCompareOptions:
    """Comparison and rendering knobs for :func:`compare_clubhead_traces`.

    Attributes:
        time_alignment: ``"impact"`` (default), ``"address"``, or ``"none"``.
        address_threshold_m: Movement (metres) used to detect start-of-swing
            in ``"address"`` mode.
        error_arrow_stride: Draw an error vector on the 3D plot every Nth
            sample. ``<=0`` disables error arrows.
        sample_rate_hz: Output uniform timegrid rate after alignment.
        n_samples: Number of resampled points; defaults to the simulation's
            sample count when ``None``.
    """

    time_alignment: TimeAlignment = "impact"
    address_threshold_m: float = 0.005
    error_arrow_stride: int = 25
    sample_rate_hz: float = 1000.0
    n_samples: int | None = None


@dataclass(frozen=True)
class TraceCompareReport:
    """Numeric and aligned-array result of a trace comparison.

    All position errors are in **millimetres**, speeds in **mph**.

    Attributes:
        rmse_per_axis_mm: ``(3,)`` RMSE per ``[x, y, z]`` axis (mm).
        total_rmse_mm: Euclidean RMSE across the full trace (mm).
        max_error_mm: Maximum per-sample Euclidean error (mm).
        peak_speed_delta_mph: ``peak(sim) - peak(meas)`` clubhead speed (mph).
        impact_position_delta_mm: Position delta vector at the simulated
            impact frame, in mm; shape ``(3,)``.
        time_alignment_offset_s: Seconds the measured trace was shifted by.
        source_meta: Free-form mapping of provenance for downstream JSON
            serialisation.
        time: Resampled common timegrid, shape ``(N,)``.
        measured_clubhead: Aligned measured clubhead positions ``(N, 3)`` (m).
        simulated_clubhead: Simulated clubhead positions ``(N, 3)`` (m).
        measured_quat: Aligned measured quaternions ``(N, 4)``.
        simulated_quat: Simulated quaternions ``(N, 4)``.
        measured_butt: Aligned measured butt positions ``(N, 3)`` (m).
        simulated_butt: Simulated butt positions ``(N, 3)`` (m).
        measured_speed_mph: Per-sample measured clubhead speed (mph).
        simulated_speed_mph: Per-sample simulated clubhead speed (mph).
        impact_idx: Impact frame index on the common timegrid (0-based).
    """

    rmse_per_axis_mm: NDArray[np.float64]
    total_rmse_mm: float
    max_error_mm: float
    peak_speed_delta_mph: float
    impact_position_delta_mm: NDArray[np.float64]
    time_alignment_offset_s: float
    source_meta: dict[str, str]
    time: NDArray[np.float64]
    measured_clubhead: NDArray[np.float64]
    simulated_clubhead: NDArray[np.float64]
    measured_quat: NDArray[np.float64]
    simulated_quat: NDArray[np.float64]
    measured_butt: NDArray[np.float64]
    simulated_butt: NDArray[np.float64]
    measured_speed_mph: NDArray[np.float64]
    simulated_speed_mph: NDArray[np.float64]
    impact_idx: int
    options: TraceCompareOptions = field(default_factory=TraceCompareOptions)

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dict of the scalar metrics."""
        return {
            "rmse_per_axis_mm": [float(v) for v in self.rmse_per_axis_mm],
            "total_rmse_mm": float(self.total_rmse_mm),
            "max_error_mm": float(self.max_error_mm),
            "peak_speed_delta_mph": float(self.peak_speed_delta_mph),
            "impact_position_delta_mm": [
                float(v) for v in self.impact_position_delta_mm
            ],
            "time_alignment_offset_s": float(self.time_alignment_offset_s),
            "impact_idx": int(self.impact_idx),
            "source_meta": dict(self.source_meta),
            "time_alignment": self.options.time_alignment,
            "n_samples": int(self.time.shape[0]),
        }


# --- Alignment helpers -------------------------------------------------------


def _detect_address_idx(
    butt: NDArray[np.float64],
    clubhead: NDArray[np.float64],
    threshold_m: float,
) -> int:
    """Return the first index where butt or clubhead has moved > ``threshold``."""
    # ⚡ Bolt: einsum is faster than np.linalg.norm(..., axis=1) for small inner dimensions
    diff_butt = butt - butt[0:1]
    d_butt = np.sqrt(np.einsum("ij,ij->i", diff_butt, diff_butt))
    diff_club = clubhead - clubhead[0:1]
    d_club = np.sqrt(np.einsum("ij,ij->i", diff_club, diff_club))
    moved = (d_butt > threshold_m) | (d_club > threshold_m)
    where = np.where(moved)[0]
    if where.size == 0:
        return 0
    return int(where[0])


def _interp_xyz(
    sim_t: NDArray[np.float64],
    raw_t: NDArray[np.float64],
    raw_xyz: NDArray[np.float64],
) -> NDArray[np.float64]:
    out = np.empty((sim_t.shape[0], 3), dtype=np.float64)
    for k in range(3):
        out[:, k] = np.interp(sim_t, raw_t, raw_xyz[:, k])
    return out


def _slerp_series(
    sim_t: NDArray[np.float64],
    raw_t: NDArray[np.float64],
    raw_q: NDArray[np.float64],
) -> NDArray[np.float64]:
    out = np.empty((sim_t.shape[0], 4), dtype=np.float64)
    last = raw_t.shape[0] - 1
    for i, t in enumerate(sim_t):
        if t <= raw_t[0]:
            out[i] = raw_q[0]
            continue
        if t >= raw_t[last]:
            out[i] = raw_q[last]
            continue
        j = int(np.searchsorted(raw_t, t)) - 1
        j = max(0, min(j, last - 1))
        span = raw_t[j + 1] - raw_t[j]
        alpha = 0.0 if span == 0.0 else float((t - raw_t[j]) / span)
        out[i] = slerp(raw_q[j], raw_q[j + 1], alpha)
    # ⚡ Bolt: einsum is ~2x faster than np.linalg.norm(..., axis=1)
    norms = np.sqrt(np.einsum("ij,ij->i", out, out))[:, np.newaxis]
    norms[norms == 0.0] = 1.0
    return out / norms


def _clubhead_speed_mph(
    time: NDArray[np.float64], clubhead: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Per-sample clubhead speed in mph (central differences interior)."""
    n = time.shape[0]
    speeds = np.zeros(n, dtype=np.float64)
    if n < 2:
        return speeds
    for i in range(1, n - 1):
        dt = time[i + 1] - time[i - 1]
        if dt > 0:
            v = (clubhead[i + 1] - clubhead[i - 1]) / dt
            speeds[i] = float(np.linalg.norm(v))
    speeds[0] = float(
        np.linalg.norm((clubhead[1] - clubhead[0]) / max(time[1] - time[0], 1e-12))
    )
    speeds[-1] = float(
        np.linalg.norm((clubhead[-1] - clubhead[-2]) / max(time[-1] - time[-2], 1e-12))
    )
    return speeds * _MPS_TO_MPH


def _shift_for_alignment(
    target: ClubTarget, mode: TimeAlignment, address_threshold_m: float
) -> tuple[NDArray[np.float64], float]:
    """Return a shifted time vector and the shift offset in seconds."""
    t = np.asarray(target.time, dtype=np.float64).copy()
    if mode == "impact":
        # impact_idx is 1-based per CLUB_IK_SPEC.
        offset = float(t[int(target.impact_idx) - 1])
    elif mode == "address":
        idx = _detect_address_idx(target.butt, target.clubhead, address_threshold_m)
        offset = float(t[idx])
    elif mode == "none":
        offset = 0.0
    else:
        raise ValueError(f"unknown time_alignment {mode!r}")
    return t - offset, offset


def _build_common_timegrid(
    t_meas: NDArray[np.float64],
    t_sim: NDArray[np.float64],
    opts: TraceCompareOptions,
) -> NDArray[np.float64]:
    t_lo = float(max(t_meas[0], t_sim[0]))
    t_hi = float(min(t_meas[-1], t_sim[-1]))
    if t_hi <= t_lo:
        raise ValueError(
            "Aligned timegrids do not overlap; check time_alignment / inputs."
        )
    if opts.n_samples is not None and opts.n_samples >= 2:
        return np.linspace(t_lo, t_hi, int(opts.n_samples), dtype=np.float64)
    dt = 1.0 / float(opts.sample_rate_hz)
    n = max(2, int(round((t_hi - t_lo) / dt)) + 1)
    return np.linspace(t_lo, t_hi, n, dtype=np.float64)


# --- Main API ----------------------------------------------------------------


def _validate_inputs(measured: ClubTarget, simulated: ClubTarget) -> None:
    if not isinstance(measured, ClubTarget):
        raise TypeError(f"measured must be a ClubTarget, got {type(measured).__name__}")
    if not isinstance(simulated, ClubTarget):
        raise TypeError(
            f"simulated must be a ClubTarget, got {type(simulated).__name__}"
        )


def compare_clubhead_traces(
    measured: ClubTarget,
    simulated: ClubTarget,
    opts: TraceCompareOptions | None = None,
) -> TraceCompareReport:
    """Compare measured vs simulated clubhead trajectories.

    Args:
        measured: ``ClubTarget`` from a Wiffle/C3D loader.
        simulated: ``ClubTarget`` synthesised from a Simscape run.
        opts: Optional :class:`TraceCompareOptions`. Defaults applied.

    Returns:
        :class:`TraceCompareReport` with scalar metrics and aligned arrays.

    Raises:
        TypeError: If either input is not a ``ClubTarget``.
        ValueError: If the two traces have no overlapping time window
            after the selected alignment.
    """
    _validate_inputs(measured, simulated)
    o = opts or TraceCompareOptions()

    t_meas, off_meas = _shift_for_alignment(
        measured, o.time_alignment, o.address_threshold_m
    )
    t_sim, off_sim = _shift_for_alignment(
        simulated, o.time_alignment, o.address_threshold_m
    )
    common_t = _build_common_timegrid(t_meas, t_sim, o)

    meas_club = _interp_xyz(common_t, t_meas, measured.clubhead)
    sim_club = _interp_xyz(common_t, t_sim, simulated.clubhead)
    meas_butt = _interp_xyz(common_t, t_meas, measured.butt)
    sim_butt = _interp_xyz(common_t, t_sim, simulated.butt)
    meas_q = _slerp_series(common_t, t_meas, measured.club_quat)
    sim_q = _slerp_series(common_t, t_sim, simulated.club_quat)

    delta_m = sim_club - meas_club
    delta_mm = delta_m * _M_TO_MM
    rmse_axes = np.sqrt(np.mean(delta_mm**2, axis=0))
    total_rmse = float(np.sqrt(np.mean(np.sum(delta_mm**2, axis=1))))
    # ⚡ Bolt: avoiding np.linalg.norm allows max computation before sqrt, saving ~2x time
    max_err = float(np.sqrt(np.max(np.einsum("ij,ij->i", delta_mm, delta_mm))))

    speed_meas = _clubhead_speed_mph(common_t, meas_club)
    speed_sim = _clubhead_speed_mph(common_t, sim_club)
    peak_delta = float(speed_sim.max() - speed_meas.max())

    impact_idx = int(np.argmax(speed_sim))
    impact_delta = delta_mm[impact_idx]

    return TraceCompareReport(
        rmse_per_axis_mm=rmse_axes.astype(np.float64),
        total_rmse_mm=total_rmse,
        max_error_mm=max_err,
        peak_speed_delta_mph=peak_delta,
        impact_position_delta_mm=impact_delta.astype(np.float64),
        time_alignment_offset_s=float(off_meas - off_sim),
        source_meta={
            "measured_filename": measured.source.filename,
            "measured_format": measured.source.format,
            "measured_subject": measured.source.subject_id,
            "measured_trial": measured.source.trial_id,
            "simulated_filename": simulated.source.filename,
            "simulated_format": simulated.source.format,
            "simulated_subject": simulated.source.subject_id,
            "simulated_trial": simulated.source.trial_id,
        },
        time=common_t,
        measured_clubhead=meas_club,
        simulated_clubhead=sim_club,
        measured_quat=meas_q,
        simulated_quat=sim_q,
        measured_butt=meas_butt,
        simulated_butt=sim_butt,
        measured_speed_mph=speed_meas,
        simulated_speed_mph=speed_sim,
        impact_idx=impact_idx,
        options=o,
    )


# --- Plotting ----------------------------------------------------------------


def _import_pyplot():  # type: ignore[no-untyped-def]
    """Lazy import so test collection works in headless environments."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    return plt


def plot_3d_overlay(report: TraceCompareReport, ax=None):  # type: ignore[no-untyped-def]
    """Plot both clubhead paths in the world frame with error vectors.

    Returns the ``matplotlib.figure.Figure``.
    """
    plt = _import_pyplot()
    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
    m = report.measured_clubhead
    s = report.simulated_clubhead
    ax.plot(m[:, 0], m[:, 1], m[:, 2], color="tab:blue", label="measured", lw=1.5)
    ax.plot(s[:, 0], s[:, 1], s[:, 2], color="tab:red", label="simulated", lw=1.5)
    stride = report.options.error_arrow_stride
    if stride > 0:
        idx = np.arange(0, m.shape[0], stride)
        for i in idx:
            ax.plot(
                [m[i, 0], s[i, 0]],
                [m[i, 1], s[i, 1]],
                [m[i, 2], s[i, 2]],
                color="grey",
                lw=0.5,
                alpha=0.6,
            )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(
        f"Clubhead 3D trace — RMSE {report.total_rmse_mm:.1f} mm, "
        f"max {report.max_error_mm:.1f} mm"
    )
    ax.legend(loc="best")
    return fig


def plot_per_axis_timeseries(report: TraceCompareReport):  # type: ignore[no-untyped-def]
    """Three-row figure: X(t), Y(t), Z(t) for both, with shaded delta."""
    plt = _import_pyplot()
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    labels = ("X", "Y", "Z")
    t = report.time
    for k, ax in enumerate(axes):
        ax.plot(t, report.measured_clubhead[:, k], color="tab:blue", label="measured")
        ax.plot(t, report.simulated_clubhead[:, k], color="tab:red", label="simulated")
        ax.fill_between(
            t,
            report.measured_clubhead[:, k],
            report.simulated_clubhead[:, k],
            color="grey",
            alpha=0.25,
        )
        ax.set_ylabel(f"{labels[k]} (m)")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(
        "Clubhead position vs time — RMSE per axis "
        f"[{report.rmse_per_axis_mm[0]:.1f}, "
        f"{report.rmse_per_axis_mm[1]:.1f}, "
        f"{report.rmse_per_axis_mm[2]:.1f}] mm"
    )
    fig.tight_layout()
    return fig


def plot_speed_comparison(report: TraceCompareReport):  # type: ignore[no-untyped-def]
    """Clubhead speed (mph) vs time for both traces."""
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(report.time, report.measured_speed_mph, color="tab:blue", label="measured")
    ax.plot(report.time, report.simulated_speed_mph, color="tab:red", label="simulated")
    ax.axvline(report.time[report.impact_idx], color="k", lw=0.8, ls="--", alpha=0.4)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("clubhead speed (mph)")
    ax.set_title(f"Clubhead speed — peak delta {report.peak_speed_delta_mph:+.1f} mph")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _quat_to_shaft_dir(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotate the body-frame +Z axis by quaternion ``q`` to get a unit vector."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    # Apply R(q) to [0, 0, 1]:
    vx = 2 * (x * z + w * y)
    vy = 2 * (y * z - w * x)
    vz = 1 - 2 * (x * x + y * y)
    return np.array([vx, vy, vz], dtype=np.float64)


def _draw_skeleton(  # type: ignore[no-untyped-def]
    ax,
    butt: NDArray[np.float64],
    clubhead: NDArray[np.float64],
    quat: NDArray[np.float64],
    color: str,
    label: str,
) -> None:
    ax.scatter(*butt, color=color, marker="o", s=40, label=f"{label} butt")
    ax.scatter(*clubhead, color=color, marker="^", s=60, label=f"{label} head")
    ax.plot(
        [butt[0], clubhead[0]],
        [butt[1], clubhead[1]],
        [butt[2], clubhead[2]],
        color=color,
        lw=2,
    )
    shaft_len = float(np.linalg.norm(clubhead - butt))
    direction = _quat_to_shaft_dir(quat) * (shaft_len * 0.3)
    tip = clubhead + direction
    ax.plot(
        [clubhead[0], tip[0]],
        [clubhead[1], tip[1]],
        [clubhead[2], tip[2]],
        color=color,
        lw=1,
        ls=":",
    )


def plot_setup_pose_skeletons(  # type: ignore[no-untyped-def]
    measured: ClubTarget, simulated: ClubTarget
):
    """Draw setup-pose skeletons (t=0) side-by-side and overlaid.

    Returns a figure with two 3D axes: ``[side_by_side, overlay]``. The
    docstring guarantees ``len(fig.axes) == 2``.
    """
    _validate_inputs(measured, simulated)
    plt = _import_pyplot()
    fig = plt.figure(figsize=(11, 5))
    ax_side = fig.add_subplot(1, 2, 1, projection="3d")
    ax_over = fig.add_subplot(1, 2, 2, projection="3d")

    m_butt = measured.butt[0]
    m_head = measured.clubhead[0]
    m_q = measured.club_quat[0]
    s_butt = simulated.butt[0]
    s_head = simulated.clubhead[0]
    s_q = simulated.club_quat[0]

    _draw_skeleton(ax_side, m_butt, m_head, m_q, "tab:blue", "measured")
    # Offset the simulated skeleton along +X for visual clarity in side-by-side.
    offset = np.array([0.6, 0.0, 0.0])
    _draw_skeleton(
        ax_side, s_butt + offset, s_head + offset, s_q, "tab:red", "simulated"
    )
    ax_side.set_title("Setup pose — side by side (t=0)")
    ax_side.set_xlabel("X (m)")
    ax_side.set_ylabel("Y (m)")
    ax_side.set_zlabel("Z (m)")

    _draw_skeleton(ax_over, m_butt, m_head, m_q, "tab:blue", "measured")
    _draw_skeleton(ax_over, s_butt, s_head, s_q, "tab:red", "simulated")
    delta = (s_head - m_head) * _M_TO_MM
    delta_norm = float(np.linalg.norm(delta))
    geo_deg = float(
        np.degrees(quaternion_geodesic_angles(m_q[None, :], s_q[None, :])[0])
    )
    ax_over.plot(
        [m_head[0], s_head[0]],
        [m_head[1], s_head[1]],
        [m_head[2], s_head[2]],
        color="grey",
        lw=1,
    )
    ax_over.set_title(
        f"Overlay — head delta {delta_norm:.1f} mm, orient {geo_deg:.1f} deg"
    )
    ax_over.set_xlabel("X (m)")
    ax_over.set_ylabel("Y (m)")
    ax_over.set_zlabel("Z (m)")
    ax_over.legend(loc="best", fontsize=7)
    fig.tight_layout()
    return fig
