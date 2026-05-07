"""Meshcat 3D trajectory overlay (View 1 of VISUALIZATION_SPEC.md).

The overlay shows the canonical humanoid skeleton plus *both* the
measured grip path (blue, ``#1f77b4``) and the simulated grip path
(red, ``#d62728``) on the same Meshcat scene. A third grey trace draws
the per-frame error vector from simulated -> measured clubhead so you
can see *where* the fit is failing without flipping between two
viewports.

Two artefacts are written:

1. ``<out_dir>/trajectory_overlay.html`` -- a static, self-contained
   Meshcat scene (saved via ``Meshcat.StaticHtml``). Open in any
   browser; no Drake or server needed.
2. ``<out_dir>/trajectory_overlay.png`` -- a matplotlib screenshot
   fallback rendered with the ``Agg`` backend so the file is produced
   even on headless CI nodes that cannot run Meshcat. The PNG is
   *always* produced; the HTML is produced only when ``pydrake`` is
   importable.

Per CLAUDE.md, all ``pydrake`` imports are explicit
``from pydrake.X import Y`` and live inside the function so the module
imports cleanly on systems without ``pydrake`` (e.g. the standard CI
runner without the Drake extras).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.shared.python.motion_matching.club_target import ClubTarget


# Shared palette -- mirrors VISUALIZATION_SPEC.md "Styling".
COLOR_MEASURED = "#1f77b4"
COLOR_SIMULATED = "#d62728"
COLOR_ERROR = "#7f7f7f"


__all__ = [
    "DrakeFitResult",
    "OverlayArtifacts",
    "render_trajectory_overlay",
]


# ---------------------------------------------------------------------------
# Drake-side FitResult bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DrakeFitResult:
    """Drake-engine bundle consumed by every viz entry point.

    A thin, frozen wrapper around the float-pathway ``SimOut`` plus the
    minimum metadata needed to render the three views without re-running
    the simulator. Constructed by callers (e.g. the optimiser, the CLI
    ``visualize_fit`` entry point of a future issue) from the canonical
    ``simulate_with_coefficients`` output.

    Attributes:
        time:        ``(N,)`` simulation time grid in seconds. Strictly
                     increasing.
        grip:        ``(N, 3)`` simulated grip path (m). Mirrors
                     ``SimOut.grip``.
        clubhead:    ``(N, 3)`` simulated clubhead path (m).
        club_quat:   ``(N, 4)`` simulated club quaternions ``[w, x, y, z]``.
        tau:         ``(N, n_joints)`` simulated joint torques (N*m).
                     Optional; if ``None`` the joint-torque panel is
                     skipped.
        coefficients: ``(n_joints*7,)`` polynomial coefficient vector
                     reported by the optimiser.
        final_loss:  Final scalar loss from the optimiser.
        solver_status: ``"success"`` / ``"warning"`` / ``"failed"``.
        wall_clock_s: Optimiser wall-clock seconds, for the summary card.
        n_iterations: Optimiser iteration count, for the summary card.
        swing_id:    Free-form trial identifier (e.g. ``"TW_ProV1"``).
        commit_hash: Optional 7-character git commit hash.
        branch:      Optional branch name.
    """

    time: NDArray[np.float64]
    grip: NDArray[np.float64]
    clubhead: NDArray[np.float64]
    club_quat: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    final_loss: float
    tau: NDArray[np.float64] | None = None
    solver_status: str = "success"
    wall_clock_s: float = 0.0
    n_iterations: int = 0
    swing_id: str = ""
    commit_hash: str = ""
    branch: str = ""

    def __post_init__(self) -> None:
        # DbC: shape parity with ClubTarget so downstream maths just works.
        if self.time.ndim != 1:
            msg = f"time must be 1-D; got shape {self.time.shape}"
            raise ValueError(msg)
        n = self.time.shape[0]
        if n < 2:
            msg = f"time must have at least 2 samples; got {n}"
            raise ValueError(msg)
        for name, arr, cols in (
            ("grip", self.grip, 3),
            ("clubhead", self.clubhead, 3),
            ("club_quat", self.club_quat, 4),
        ):
            if arr.shape != (n, cols):
                msg = f"{name} must have shape ({n}, {cols}); got {arr.shape}"
                raise ValueError(msg)
        if self.tau is not None:
            if self.tau.ndim != 2 or self.tau.shape[0] != n:
                msg = (
                    "tau must have shape (N, n_joints) matching time length "
                    f"N={n}; got {self.tau.shape}"
                )
                raise ValueError(msg)
        if self.solver_status not in {"success", "warning", "failed"}:
            msg = (
                "solver_status must be 'success' / 'warning' / 'failed'; "
                f"got {self.solver_status!r}"
            )
            raise ValueError(msg)


# ---------------------------------------------------------------------------
# Output artefacts bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OverlayArtifacts:
    """Paths to the artefacts produced by :func:`render_trajectory_overlay`.

    ``html_path`` is ``None`` when the Drake/Meshcat path could not run
    (e.g. ``pydrake`` not importable on the host); ``png_path`` is
    always produced because matplotlib with the ``Agg`` backend works
    on every CI node.
    """

    png_path: Path
    html_path: Path | None = None
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_trajectory_overlay(
    fit: DrakeFitResult,
    target: ClubTarget,
    out_dir: Path,
    *,
    title: str | None = None,
) -> OverlayArtifacts:
    """Render the side-by-side trajectory overlay (View 1).

    The function *always* writes the PNG fallback so headless CI gets a
    stable artefact, and *additionally* writes a static Meshcat HTML
    scene when ``pydrake`` is importable on the host.

    Args:
        fit:    :class:`DrakeFitResult` from the optimiser.
        target: Canonical :class:`ClubTarget` consumed by the fit.
        out_dir: Directory to write artefacts into. Created if missing.
        title:  Optional figure title. Defaults to ``fit.swing_id`` or
                ``"Trajectory overlay"``.

    Returns:
        :class:`OverlayArtifacts` with paths to every artefact actually
        written.

    Raises:
        ValueError: if ``fit`` and ``target`` do not have the same
            number of samples (the loader/sim contract guarantees they
            do, so this is a hard fail rather than a resample).
    """
    if fit.time.shape[0] != target.time.shape[0]:
        msg = (
            "fit and target must share the canonical timegrid; got "
            f"{fit.time.shape[0]} vs {target.time.shape[0]} samples"
        )
        raise ValueError(msg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_title = title or fit.swing_id or "Trajectory overlay"

    png_path = _render_png_fallback(fit, target, out_dir, resolved_title)
    html_path = _try_render_meshcat_html(fit, target, out_dir, resolved_title)

    return OverlayArtifacts(
        png_path=png_path,
        html_path=html_path,
        meta={
            "title": resolved_title,
            "n_samples": int(fit.time.shape[0]),
            "solver_status": fit.solver_status,
        },
    )


# ---------------------------------------------------------------------------
# Matplotlib fallback (always runs)
# ---------------------------------------------------------------------------


def _render_png_fallback(
    fit: DrakeFitResult,
    target: ClubTarget,
    out_dir: Path,
    title: str,
) -> Path:
    """Render the PNG screenshot fallback.

    Two side-by-side 3D axes -- left is the measured trajectory, right
    the simulated -- with the same camera limits so visual drift is
    obvious. The grey error vectors connect simulated -> measured at a
    handful of evenly spaced frames.

    Returns the path of the PNG written.
    """
    # Local imports keep matplotlib optional at module-import time.
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.figure as _mfig
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    fig = _mfig.Figure(figsize=(12.0, 5.5), dpi=150)
    ax_left = fig.add_subplot(1, 2, 1, projection="3d")
    ax_right = fig.add_subplot(1, 2, 2, projection="3d")

    _draw_club_skeleton_panel(
        ax_left,
        butt=target.butt,
        clubhead=target.clubhead,
        color=COLOR_MEASURED,
        title="Measured",
    )
    _draw_club_skeleton_panel(
        ax_right,
        butt=fit.grip,
        clubhead=fit.clubhead,
        color=COLOR_SIMULATED,
        title="Simulated",
    )

    # Error vectors -- draw on the right panel at ~10 evenly spaced frames.
    n = fit.time.shape[0]
    indices = np.linspace(0, n - 1, num=min(10, n), dtype=int)
    for k in indices:
        ax_right.plot(
            [fit.clubhead[k, 0], target.clubhead[k, 0]],
            [fit.clubhead[k, 1], target.clubhead[k, 1]],
            [fit.clubhead[k, 2], target.clubhead[k, 2]],
            color=COLOR_ERROR,
            linewidth=0.8,
            alpha=0.6,
        )

    # Tie the camera limits across both panels so drift is legible.
    _share_axis_limits(ax_left, ax_right, fit, target)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    png_path = out_dir / "trajectory_overlay.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    return png_path


def _draw_club_skeleton_panel(
    ax: Any,
    *,
    butt: NDArray[np.float64],
    clubhead: NDArray[np.float64],
    color: str,
    title: str,
) -> None:
    """Draw a single 3D panel of the club skeleton + clubhead trace."""
    ax.plot(
        clubhead[:, 0],
        clubhead[:, 1],
        clubhead[:, 2],
        color=color,
        linewidth=1.4,
        label="clubhead path",
    )
    # Faint butt path, since it's the slow end of the skeleton.
    ax.plot(
        butt[:, 0],
        butt[:, 1],
        butt[:, 2],
        color=color,
        linewidth=0.8,
        alpha=0.4,
        label="butt path",
    )
    # Skeleton at impact-ish (mid-frame) so the figure isn't just two curves.
    mid = butt.shape[0] // 2
    ax.plot(
        [butt[mid, 0], clubhead[mid, 0]],
        [butt[mid, 1], clubhead[mid, 1]],
        [butt[mid, 2], clubhead[mid, 2]],
        color=color,
        linewidth=2.5,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(loc="upper right", fontsize=8)


def _share_axis_limits(
    ax_left: Any,
    ax_right: Any,
    fit: DrakeFitResult,
    target: ClubTarget,
) -> None:
    """Pin both 3D axes to the union bounding box of all four traces."""
    pts = np.concatenate(
        [
            np.asarray(target.butt, dtype=float),
            np.asarray(target.clubhead, dtype=float),
            np.asarray(fit.grip, dtype=float),
            np.asarray(fit.clubhead, dtype=float),
        ],
        axis=0,
    )
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    pad = 0.05 * np.maximum(hi - lo, 1.0e-3)
    for ax in (ax_left, ax_right):
        ax.set_xlim(float(lo[0] - pad[0]), float(hi[0] + pad[0]))
        ax.set_ylim(float(lo[1] - pad[1]), float(hi[1] + pad[1]))
        ax.set_zlim(float(lo[2] - pad[2]), float(hi[2] + pad[2]))


# ---------------------------------------------------------------------------
# Optional Meshcat HTML (skipped on hosts without pydrake)
# ---------------------------------------------------------------------------


def _try_render_meshcat_html(
    fit: DrakeFitResult,
    target: ClubTarget,
    out_dir: Path,
    title: str,
) -> Path | None:
    """Attempt to write a static Meshcat HTML scene.

    Returns the path on success and ``None`` on any importable-or-runtime
    failure (e.g. ``pydrake`` not installed). The PNG fallback ensures
    callers always get a usable artefact.
    """
    try:  # pragma: no cover - exercised only when pydrake is installed
        # Explicit imports per CLAUDE.md.
        from pydrake.geometry import Meshcat, Rgba, Sphere
        from pydrake.math import RigidTransform
    except Exception:
        return None

    try:  # pragma: no cover - same gating
        meshcat = Meshcat()
        # Path prefix so all this issue's geometry is removable in one go.
        prefix = "trajectory_overlay"
        meshcat.Delete(prefix)

        _publish_path_as_spheres(
            meshcat,
            f"{prefix}/measured/clubhead",
            target.clubhead,
            color=Rgba(0.122, 0.467, 0.706, 1.0),  # COLOR_MEASURED
            radius=0.012,
            sphere_cls=Sphere,
            transform_cls=RigidTransform,
        )
        _publish_path_as_spheres(
            meshcat,
            f"{prefix}/measured/butt",
            target.butt,
            color=Rgba(0.122, 0.467, 0.706, 0.5),
            radius=0.010,
            sphere_cls=Sphere,
            transform_cls=RigidTransform,
        )
        _publish_path_as_spheres(
            meshcat,
            f"{prefix}/simulated/clubhead",
            fit.clubhead,
            color=Rgba(0.839, 0.153, 0.157, 1.0),  # COLOR_SIMULATED
            radius=0.012,
            sphere_cls=Sphere,
            transform_cls=RigidTransform,
        )
        _publish_path_as_spheres(
            meshcat,
            f"{prefix}/simulated/grip",
            fit.grip,
            color=Rgba(0.839, 0.153, 0.157, 0.5),
            radius=0.010,
            sphere_cls=Sphere,
            transform_cls=RigidTransform,
        )

        html = meshcat.StaticHtml()
        html_path = out_dir / "trajectory_overlay.html"
        html_path.write_text(html, encoding="utf-8")
        # Bake the title into the HTML so the artefact is self-describing.
        html_path.write_text(
            f"<!-- {title} -->\n" + html_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return html_path
    except Exception:
        return None


def _publish_path_as_spheres(
    meshcat: Any,
    path: str,
    points: NDArray[np.float64],
    *,
    color: Any,
    radius: float,
    sphere_cls: Any,
    transform_cls: Any,
) -> None:  # pragma: no cover - only runs with pydrake
    """Publish a polyline as a chain of small spheres on the Meshcat tree.

    Meshcat's ``SetLine`` API exists in newer wheels but is awkward to
    feature-detect across versions; small spheres render identically
    everywhere and double as time markers in the static HTML.
    """
    n = points.shape[0]
    # Subsample so we don't render 1k+ spheres into the static HTML.
    step = max(1, n // 80)
    for k in range(0, n, step):
        meshcat.SetObject(f"{path}/{k}", sphere_cls(radius), color)
        p = [
            float(points[k, 0]),
            float(points[k, 1]),
            float(points[k, 2]),
        ]
        meshcat.SetTransform(f"{path}/{k}", transform_cls(p=p))
