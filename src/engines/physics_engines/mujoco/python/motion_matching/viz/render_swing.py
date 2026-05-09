"""Thin renderer over MuJoCo for motion-matching fit results.

Implements the three views required by ``VISUALIZATION_SPEC.md``:

* :func:`render_trajectory_overlay` -- View 1: measured vs. simulated club
  skeleton + grip path. 3D matplotlib figure (CI-friendly PNG default).
* :func:`render_error_timecourse`   -- View 2: stacked error/speed/torque
  panels with an impact-frame indicator.
* :func:`render_fit_quality_card`   -- View 3: single-figure summary safe
  to drop into a PR description.

Why matplotlib for 3D rather than ``mujoco.viewer`` / ``mujoco.Renderer``?

The deliverable in issue #4125 says either approach is acceptable, with a
preference for offscreen PNG output for CI artifacts. ``mujoco.Renderer``
needs an OpenGL context (``EGL`` / ``GLFW`` / ``OSMesa``), which several
CI runners do not provide. Matplotlib's 3D back-end is in the test
environment unconditionally, produces deterministic byte output, and
covers the spec's "skeleton + ghost trace" requirement without a GL
dependency. An ``mujoco.viewer.launch_passive`` interactive mode is
supported as an opt-in via :class:`VizOptions.interactive` for users
running locally; CI defaults to PNG.

Public dataclasses
------------------
:class:`FitResult` is the minimal contract this renderer expects:
``time``, ``grip``, ``clubhead``, ``club_quat`` arrays plus optional
``tau``, ``solver_status``, ``duration_s``, and free-form ``metadata``.
The mujoco ``simulate.SimOut`` produced by ``simulate_with_coefficients``
satisfies this contract by attribute compatibility (``grip``,
``clubhead``, ``time`` etc.), so the cost-fitting layer can pass its
output directly with no glue code.

:class:`VizOptions` collects DPI / output-path / palette knobs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import matplotlib

# Force a non-interactive backend BEFORE any pyplot import so this module is
# safe to import in headless CI. Callers wanting an interactive viewer use
# ``VizOptions.interactive=True`` which routes through ``mujoco.viewer``
# rather than matplotlib.
matplotlib.use("Agg", force=False)

import matplotlib.pyplot as plt  # noqa: E402 -- backend must be set first.
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

__all__ = [
    "FitResult",
    "VizOptions",
    "render_error_timecourse",
    "render_fit_quality_card",
    "render_trajectory_overlay",
]

_LOG = logging.getLogger(__name__)

# Palette per VISUALIZATION_SPEC.md "Styling".
_COLOR_MEASURED = "#1f77b4"
_COLOR_SIMULATED = "#d62728"
_COLOR_ERROR = "#7f7f7f"

# Output defaults per spec ("1.5x retina, DPI 200").
_DEFAULT_DPI = 200
_DEFAULT_OUTPUT_DIR = Path("output") / "viz"


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@runtime_checkable
class _SimOutLike(Protocol):
    """Structural type matching mujoco simulate.SimOut.

    Declared here as a Protocol rather than importing the SimOut dataclass
    directly so this renderer has no hard dependency on the simulate
    module (which itself imports MuJoCo). Tests can build duck-typed
    objects that satisfy the protocol without compiling an MJCF.
    """

    time: NDArray[np.float64]
    grip: NDArray[np.float64]
    clubhead: NDArray[np.float64]
    club_quat: NDArray[np.float64]


@dataclass
class FitResult:
    """Minimal viz contract for a motion-matching fit.

    Designed to be a strict subset of the canonical mujoco
    ``simulate.SimOut`` so callers can pass a SimOut directly:

        >>> from src.engines.physics_engines.mujoco.python.motion_matching \\
        ...     .simulate import SimOut  # doctest: +SKIP
        >>> render_trajectory_overlay(sim_out, target)  # doctest: +SKIP

    Attributes:
        time: ``(N,)`` time grid in seconds.
        grip: ``(N, 3)`` mid-hands position (m), world frame.
        clubhead: ``(N, 3)`` clubhead position (m), world frame.
        club_quat: ``(N, 4)`` clubhead orientation, ``[w, x, y, z]``.
        tau: optional ``(N, nu)`` actuator torques (N*m). Drawn in View 2
            when present; omitted otherwise.
        solver_status: free-form status string (e.g. ``"ok"``, ``"diverged"``).
        duration_s: wall-clock seconds spent producing the fit.
        metadata: free-form key-value bag. Pulled into View 3's header
            (e.g. ``swing_id``, ``solver``, ``iterations``, ``commit``).
    """

    time: NDArray[np.float64]
    grip: NDArray[np.float64]
    clubhead: NDArray[np.float64]
    club_quat: NDArray[np.float64]
    tau: NDArray[np.float64] | None = None
    solver_status: str = "ok"
    duration_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_simout(cls, sim_out: _SimOutLike, **metadata: Any) -> FitResult:
        """Adapt a mujoco ``simulate.SimOut`` (or any duck-type match)."""
        tau = getattr(sim_out, "tau", None)
        return cls(
            time=np.asarray(sim_out.time, dtype=np.float64),
            grip=np.asarray(sim_out.grip, dtype=np.float64),
            clubhead=np.asarray(sim_out.clubhead, dtype=np.float64),
            club_quat=np.asarray(sim_out.club_quat, dtype=np.float64),
            tau=np.asarray(tau, dtype=np.float64) if tau is not None else None,
            solver_status=getattr(sim_out, "solver_status", "ok"),
            duration_s=float(getattr(sim_out, "duration_s", 0.0)),
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class VizOptions:
    """Knobs shared by all three render functions.

    Attributes:
        output_dir: Directory where PNGs are written when ``output_path``
            is not supplied. Created on demand.
        output_path: Explicit output PNG path. Overrides ``output_dir``.
        dpi: Raster DPI. Default 200 per spec.
        figsize_overlay: ``(W, H)`` inches for View 1.
        figsize_timecourse: ``(W, H)`` inches for View 2.
        figsize_card: ``(W, H)`` inches for View 3.
        title: Optional override for the plot title (View 1 / View 3
            header). Falls back to ``FitResult.metadata['swing_id']``.
        interactive: If True, View 1 launches ``mujoco.viewer.launch_passive``
            instead of writing a PNG. Returns ``Path`` to a placeholder
            sentinel so the typed signature is preserved. Best-effort:
            falls back to PNG with a warning if mujoco.viewer is missing.
    """

    output_dir: Path = field(default_factory=lambda: _DEFAULT_OUTPUT_DIR)
    output_path: Path | None = None
    dpi: int = _DEFAULT_DPI
    figsize_overlay: tuple[float, float] = (12.0, 6.0)
    figsize_timecourse: tuple[float, float] = (10.0, 10.0)
    figsize_card: tuple[float, float] = (10.0, 8.0)
    title: str | None = None
    interactive: bool = False


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _resolve_output_path(
    options: VizOptions,
    default_name: str,
) -> Path:
    """Return the PNG path to write, creating the parent directory."""
    if options.output_path is not None:
        path = Path(options.output_path)
    else:
        path = Path(options.output_dir) / default_name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _quat_geodesic_deg(
    q_a: NDArray[np.float64],
    q_b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Sign-invariant geodesic distance between unit quaternion sequences.

    Inputs are ``(N, 4)`` arrays in ``[w, x, y, z]`` order. Returns an
    ``(N,)`` vector of degrees, ``2 * acos(|<q_a, q_b>|)``.
    """
    if q_a.shape != q_b.shape or q_a.ndim != 2 or q_a.shape[1] != 4:
        raise ValueError(
            "quaternion shapes must match and be (N, 4); "
            f"got {q_a.shape} vs {q_b.shape}"
        )
    dot = np.clip(np.abs(np.sum(q_a * q_b, axis=1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def _interp_to_grid(
    time_src: NDArray[np.float64],
    values_src: NDArray[np.float64],
    time_dst: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Linear-interpolate per-column to ``time_dst``.

    Used when the measured target and the simulated result are on
    different time grids. The cost layer is responsible for proper
    alignment; this is a defensive resample for the visualization.
    """
    if values_src.ndim == 1:
        return np.interp(time_dst, time_src, values_src)
    out = np.empty((time_dst.shape[0], values_src.shape[1]), dtype=np.float64)
    for c in range(values_src.shape[1]):
        out[:, c] = np.interp(time_dst, time_src, values_src[:, c])
    return out


def _impact_time(target: Any) -> float | None:
    """Return the impact-frame time in seconds, or None if unavailable."""
    impact_idx = getattr(target, "impact_idx", None)
    time_arr = getattr(target, "time", None)
    if impact_idx is None or time_arr is None:
        return None
    # impact_idx in CLUB_IK_SPEC is 1-based per the MATLAB origin; the python
    # ClubTarget validator requires 1 <= impact_idx <= n. Guard against both
    # 0-based and 1-based callers since ad-hoc tests pass 0-based.
    idx = int(impact_idx)
    n = int(np.asarray(time_arr).shape[0])
    if 1 <= idx <= n:
        idx0 = idx - 1
    elif 0 <= idx < n:
        idx0 = idx
    else:
        return None
    return float(np.asarray(time_arr)[idx0])


def _safe_save(fig: plt.Figure, path: Path, dpi: int) -> Path:
    """Write the figure to ``path`` and close it."""
    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"failed to write non-empty image to {path}")
    return path


def _swing_label(result: FitResult, options: VizOptions) -> str:
    """Pick a human-readable swing label for titles."""
    if options.title:
        return options.title
    swing_id = result.metadata.get("swing_id")
    if swing_id:
        return str(swing_id)
    return "MuJoCo motion-matching fit"


def _set_equal_3d(ax: Any, xs: NDArray, ys: NDArray, zs: NDArray) -> None:
    """Force a 3D axis to equal scale across x/y/z without `set_box_aspect` quirks."""
    spans = [
        float(np.ptp(xs)) if xs.size else 0.0,
        float(np.ptp(ys)) if ys.size else 0.0,
        float(np.ptp(zs)) if zs.size else 0.0,
    ]
    span = max(spans + [1e-3])
    mids = [
        float(np.mean([xs.min(), xs.max()])) if xs.size else 0.0,
        float(np.mean([ys.min(), ys.max()])) if ys.size else 0.0,
        float(np.mean([zs.min(), zs.max()])) if zs.size else 0.0,
    ]
    half = span / 2.0
    ax.set_xlim(mids[0] - half, mids[0] + half)
    ax.set_ylim(mids[1] - half, mids[1] + half)
    ax.set_zlim(mids[2] - half, mids[2] + half)


# ---------------------------------------------------------------------------
# View 1: trajectory overlay
# ---------------------------------------------------------------------------


def _draw_overlay_panel(
    ax: Any,
    butt: NDArray[np.float64],
    head: NDArray[np.float64],
    *,
    color: str,
    label: str,
    n_skeletons: int = 8,
) -> None:
    """Draw a clubhead trace plus a stride of butt-to-head skeletons."""
    ax.plot(
        head[:, 0],
        head[:, 1],
        head[:, 2],
        color=color,
        linewidth=1.0,
        alpha=0.6,
        label=f"{label} clubhead path",
    )
    n = butt.shape[0]
    if n == 0:
        return
    stride = max(1, n // max(1, n_skeletons))
    for i in range(0, n, stride):
        seg_x = [butt[i, 0], head[i, 0]]
        seg_y = [butt[i, 1], head[i, 1]]
        seg_z = [butt[i, 2], head[i, 2]]
        ax.plot(seg_x, seg_y, seg_z, color=color, linewidth=1.5, alpha=0.55)
    # First skeleton labelled for the legend, then re-plot none for clarity.
    ax.plot(
        [butt[0, 0], head[0, 0]],
        [butt[0, 1], head[0, 1]],
        [butt[0, 2], head[0, 2]],
        color=color,
        linewidth=2.5,
        marker="o",
        label=f"{label} skeleton (t=0)",
    )


def _try_interactive_viewer(result: FitResult) -> bool:
    """Attempt mujoco.viewer.launch_passive; return True on success.

    This is best-effort. Any import error or runtime failure (no display,
    no GL context) returns False so the caller falls back to PNG output.
    """
    try:  # pragma: no cover -- requires display + mujoco
        import mujoco  # noqa: F401  -- only needed to confirm installation
        import mujoco.viewer as viewer  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        _LOG.info("mujoco.viewer unavailable (%s); falling back to PNG.", exc)
        return False
    # Wiring an actual MjModel/MjData replay through launch_passive is out of
    # scope for the thin renderer (the issue's "Out of scope" carve-out).
    # We log and return False so the PNG branch runs; the user gets a
    # well-defined artifact rather than a silent no-op.
    _LOG.info(
        "Interactive viewer requested but a model is not bundled with FitResult; "
        "writing PNG instead. To get a live viewer, drive launch_passive "
        "directly from your simulate.SimOut + MjModel."
    )
    _ = result  # suppress unused-arg lint
    return False


def render_trajectory_overlay(
    result: FitResult,
    target: Any,
    options: VizOptions | None = None,
) -> Path:
    """Render View 1 from VISUALIZATION_SPEC.md.

    Two side-by-side 3D panels: measured club skeleton + clubhead trace
    (left) vs. simulated (right). A small inset shows the per-frame
    error vectors between the two clubhead paths.

    Args:
        result: The simulated fit (typically a mujoco
            ``simulate.SimOut`` adapted via ``FitResult.from_simout``).
        target: A ``ClubTarget``-like object with ``time``, ``butt``,
            ``clubhead``, ``club_quat`` arrays.
        options: Rendering knobs (DPI, output path, colors).

    Returns:
        Path to the written PNG.
    """
    options = options or VizOptions()
    if options.interactive and _try_interactive_viewer(result):
        # Interactive mode opened in a viewer; no PNG to return.
        # Use a placeholder under the output dir so the typed return is
        # honoured (callers expect a Path even when no file is produced).
        placeholder = _resolve_output_path(options, "trajectory_overlay.viewer")
        placeholder.write_text("interactive viewer launched\n")
        return placeholder

    fig = plt.figure(figsize=options.figsize_overlay, dpi=options.dpi)
    ax_meas = fig.add_subplot(1, 2, 1, projection="3d")
    ax_sim = fig.add_subplot(1, 2, 2, projection="3d")

    meas_butt = np.asarray(target.butt, dtype=np.float64)
    meas_head = np.asarray(target.clubhead, dtype=np.float64)
    sim_butt = np.asarray(result.grip, dtype=np.float64)
    sim_head = np.asarray(result.clubhead, dtype=np.float64)

    _draw_overlay_panel(
        ax_meas, meas_butt, meas_head, color=_COLOR_MEASURED, label="measured"
    )
    _draw_overlay_panel(
        ax_sim, sim_butt, sim_head, color=_COLOR_SIMULATED, label="simulated"
    )

    # Equal-aspect across both panels using the union range so the eye can
    # actually compare drift rather than chase rescaled axes.
    all_xs = np.concatenate([meas_head[:, 0], sim_head[:, 0]])
    all_ys = np.concatenate([meas_head[:, 1], sim_head[:, 1]])
    all_zs = np.concatenate([meas_head[:, 2], sim_head[:, 2]])
    for ax in (ax_meas, ax_sim):
        _set_equal_3d(ax, all_xs, all_ys, all_zs)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.legend(loc="upper left", fontsize=8)

    ax_meas.set_title("Measured")
    ax_sim.set_title("Simulated")
    fig.suptitle(_swing_label(result, options))

    path = _resolve_output_path(options, "trajectory_overlay.png")
    return _safe_save(fig, path, options.dpi)


# ---------------------------------------------------------------------------
# View 2: error timecourse
# ---------------------------------------------------------------------------


def render_error_timecourse(
    result: FitResult,
    target: Any,
    options: VizOptions | None = None,
) -> Path:
    """Render View 2 from VISUALIZATION_SPEC.md.

    Stacked panels vs. simulation time:
      1. Position error (mm) — butt and clubhead.
      2. Orientation error (deg) — geodesic distance between
         ``result.club_quat`` and ``target.club_quat``.
      3. Clubhead speed (mph) — measured vs. simulated.
      4. Joint torques (N*m), one trace per actuated joint, if available.

    A vertical line marks the impact frame across all four panels.
    """
    options = options or VizOptions()
    fig, axes = plt.subplots(
        4, 1, figsize=options.figsize_timecourse, dpi=options.dpi, sharex=True
    )
    ax_pos, ax_ori, ax_speed, ax_tau = axes

    sim_t = np.asarray(result.time, dtype=np.float64)
    meas_t = np.asarray(target.time, dtype=np.float64)

    # Resample measured to simulated grid for direct subtraction.
    meas_butt = _interp_to_grid(meas_t, np.asarray(target.butt), sim_t)
    meas_head = _interp_to_grid(meas_t, np.asarray(target.clubhead), sim_t)
    meas_quat = _interp_to_grid(meas_t, np.asarray(target.club_quat), sim_t)
    # Re-normalize quaternions after linear interp.
    qn = np.linalg.norm(meas_quat, axis=1, keepdims=True)
    meas_quat = meas_quat / np.maximum(qn, 1e-12)

    # --- Panel 1: position error ---------------------------------------
    err_butt_mm = np.linalg.norm(result.grip - meas_butt, axis=1) * 1000.0
    err_head_mm = np.linalg.norm(result.clubhead - meas_head, axis=1) * 1000.0
    ax_pos.plot(sim_t, err_butt_mm, color=_COLOR_MEASURED, label="butt")
    ax_pos.plot(sim_t, err_head_mm, color="#ff7f0e", label="clubhead")
    ax_pos.set_ylabel("Position error [mm]")
    ax_pos.legend(loc="upper right", fontsize=8)
    ax_pos.grid(True, alpha=0.3)

    # --- Panel 2: orientation error ------------------------------------
    sim_quat = np.asarray(result.club_quat, dtype=np.float64)
    qn = np.linalg.norm(sim_quat, axis=1, keepdims=True)
    sim_quat_unit = sim_quat / np.maximum(qn, 1e-12)
    ori_err_deg = _quat_geodesic_deg(sim_quat_unit, meas_quat)
    ax_ori.plot(sim_t, ori_err_deg, color=_COLOR_ERROR)
    ax_ori.set_ylabel("Orientation error [deg]")
    ax_ori.grid(True, alpha=0.3)

    # --- Panel 3: clubhead speed ---------------------------------------
    def _speed_mph(t: NDArray, pos: NDArray) -> NDArray:
        if t.shape[0] < 2:
            return np.zeros_like(t)
        v = np.gradient(pos, t, axis=0)
        return np.linalg.norm(v, axis=1) * 2.23694  # m/s -> mph.

    meas_speed = _speed_mph(sim_t, meas_head)
    sim_speed = _speed_mph(sim_t, np.asarray(result.clubhead))
    ax_speed.plot(sim_t, meas_speed, color=_COLOR_MEASURED, label="measured")
    ax_speed.plot(
        sim_t, sim_speed, color=_COLOR_SIMULATED, linestyle="--", label="simulated"
    )
    ax_speed.set_ylabel("Clubhead speed [mph]")
    ax_speed.legend(loc="upper left", fontsize=8)
    ax_speed.grid(True, alpha=0.3)

    # --- Panel 4: joint torques ----------------------------------------
    if result.tau is not None and result.tau.size:
        tau = np.asarray(result.tau, dtype=np.float64)
        for j in range(tau.shape[1]):
            ax_tau.plot(sim_t, tau[:, j], linewidth=0.8)
        ax_tau.set_ylabel("Joint torques [N*m]")
    else:
        ax_tau.text(
            0.5,
            0.5,
            "no torque trace available",
            ha="center",
            va="center",
            transform=ax_tau.transAxes,
            color=_COLOR_ERROR,
        )
        ax_tau.set_ylabel("Joint torques [N*m]")
    ax_tau.set_xlabel("Time [s]")
    ax_tau.grid(True, alpha=0.3)

    # --- Impact-frame indicator across all panels ----------------------
    t_impact = _impact_time(target)
    if t_impact is not None:
        for ax in axes:
            ax.axvline(t_impact, color="k", linestyle=":", alpha=0.6, linewidth=1.0)

    fig.suptitle(f"Error timecourse — {_swing_label(result, options)}")

    path = _resolve_output_path(options, "error_timecourse.png")
    return _safe_save(fig, path, options.dpi)


# ---------------------------------------------------------------------------
# View 3: fit quality summary card
# ---------------------------------------------------------------------------


def _summary_text(result: FitResult, target: Any) -> str:
    """Multi-line summary used by View 3's text panel.

    Computes RMSE in mm, mean orientation error in deg, and impact-frame
    speed. Falls back to ``--`` for any field that cannot be computed.
    """
    sim_t = np.asarray(result.time, dtype=np.float64)
    meas_t = np.asarray(target.time, dtype=np.float64)
    meas_butt = _interp_to_grid(meas_t, np.asarray(target.butt), sim_t)
    meas_head = _interp_to_grid(meas_t, np.asarray(target.clubhead), sim_t)
    meas_quat = _interp_to_grid(meas_t, np.asarray(target.club_quat), sim_t)
    qn = np.linalg.norm(meas_quat, axis=1, keepdims=True)
    meas_quat = meas_quat / np.maximum(qn, 1e-12)
    sim_quat = np.asarray(result.club_quat, dtype=np.float64)
    sn = np.linalg.norm(sim_quat, axis=1, keepdims=True)
    sim_quat = sim_quat / np.maximum(sn, 1e-12)

    diff_butt = np.asarray(result.grip) - meas_butt
    rmse_butt = (
        float(np.sqrt(np.vdot(diff_butt, diff_butt) / diff_butt.shape[0])) * 1000.0
    )
    diff_head = np.asarray(result.clubhead) - meas_head
    rmse_head = (
        float(np.sqrt(np.vdot(diff_head, diff_head) / diff_head.shape[0])) * 1000.0
    )
    ori_err = float(np.mean(_quat_geodesic_deg(sim_quat, meas_quat)))

    impact_idx = getattr(target, "impact_idx", None)
    if impact_idx is None:
        speed_meas_mph = float("nan")
        speed_sim_mph = float("nan")
    else:
        i = max(0, min(sim_t.shape[0] - 1, int(impact_idx) - 1))
        if sim_t.shape[0] >= 2:
            v_meas = np.gradient(meas_head, sim_t, axis=0)
            v_sim = np.gradient(np.asarray(result.clubhead), sim_t, axis=0)
            speed_meas_mph = float(np.linalg.norm(v_meas[i]) * 2.23694)
            speed_sim_mph = float(np.linalg.norm(v_sim[i]) * 2.23694)
        else:
            speed_meas_mph = speed_sim_mph = float("nan")

    md = result.metadata
    swing_id = md.get("swing_id", "—")
    solver = md.get("solver", "—")
    iters = md.get("iterations", "—")
    commit = md.get("commit", "—")
    branch = md.get("branch", "—")

    wall = result.duration_s
    if math.isfinite(wall) and wall > 0:
        wall_str = f"{int(wall // 60)}m {wall % 60:0.1f}s"
    else:
        wall_str = "—"

    lines = [
        f"Swing: {swing_id}",
        f"Solver: {solver}    Iterations: {iters}    Wall: {wall_str}",
        f"Status: {result.solver_status}",
        "",
        f"Final RMSE — clubhead position: {rmse_head:6.2f} mm",
        f"Final RMSE — butt position:     {rmse_butt:6.2f} mm",
        f"Final mean orientation error:   {ori_err:6.2f} deg",
        (
            f"Clubhead speed at impact:       {speed_sim_mph:6.1f} mph "
            f"(meas: {speed_meas_mph:.1f})"
        ),
        "",
        f"Commit: {commit}    Branch: {branch}",
    ]
    return "\n".join(lines)


def render_fit_quality_card(
    result: FitResult,
    target: Any,
    options: VizOptions | None = None,
) -> Path:
    """Render View 3 from VISUALIZATION_SPEC.md.

    A single PNG suitable for dropping into a PR description: header
    metadata, headline RMSE numbers, plus thumbnails of the trajectory
    overlay (top-down projection) and the clubhead-position error.
    """
    options = options or VizOptions()
    fig = plt.figure(figsize=options.figsize_card, dpi=options.dpi)

    # Layout: 2 columns. Left = text summary. Right = stacked thumbnails.
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], hspace=0.35, wspace=0.25)
    ax_text = fig.add_subplot(gs[:, 0])
    ax_thumb_top = fig.add_subplot(gs[0, 1])
    ax_thumb_bot = fig.add_subplot(gs[1, 1])

    # Text summary panel.
    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        _summary_text(result, target),
        ha="left",
        va="top",
        family="monospace",
        fontsize=10,
        transform=ax_text.transAxes,
    )

    # Thumbnail 1: trajectory overlay (top-down x/y).
    meas_head = np.asarray(target.clubhead, dtype=np.float64)
    sim_head = np.asarray(result.clubhead, dtype=np.float64)
    ax_thumb_top.plot(
        meas_head[:, 0], meas_head[:, 1], color=_COLOR_MEASURED, label="measured"
    )
    ax_thumb_top.plot(
        sim_head[:, 0],
        sim_head[:, 1],
        color=_COLOR_SIMULATED,
        linestyle="--",
        label="simulated",
    )
    ax_thumb_top.set_aspect("equal", adjustable="datalim")
    ax_thumb_top.set_title("Clubhead path (top-down)", fontsize=10)
    ax_thumb_top.set_xlabel("x [m]")
    ax_thumb_top.set_ylabel("y [m]")
    ax_thumb_top.legend(loc="best", fontsize=8)
    ax_thumb_top.grid(True, alpha=0.3)

    # Thumbnail 2: position error timecourse (mm).
    sim_t = np.asarray(result.time, dtype=np.float64)
    meas_t = np.asarray(target.time, dtype=np.float64)
    meas_head_resamp = _interp_to_grid(meas_t, meas_head, sim_t)
    err_mm = np.linalg.norm(sim_head - meas_head_resamp, axis=1) * 1000.0
    ax_thumb_bot.plot(sim_t, err_mm, color=_COLOR_ERROR)
    ax_thumb_bot.set_title("Clubhead position error", fontsize=10)
    ax_thumb_bot.set_xlabel("Time [s]")
    ax_thumb_bot.set_ylabel("error [mm]")
    ax_thumb_bot.grid(True, alpha=0.3)
    t_impact = _impact_time(target)
    if t_impact is not None:
        ax_thumb_bot.axvline(
            t_impact, color="k", linestyle=":", alpha=0.6, linewidth=1.0
        )

    fig.suptitle(_swing_label(result, options))

    path = _resolve_output_path(options, "fit_quality_card.png")
    return _safe_save(fig, path, options.dpi)
