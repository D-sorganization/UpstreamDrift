"""Animation utilities for golf swing data visualisation.

This module creates frame-by-frame matplotlib animations from recorded
simulation data, supporting:

- Trajectory playback with a time slider.
- Stick-figure skeleton animation.
- Force / torque vector evolution over time.
- Side-by-side desired-vs-actual comparison.
- Export to MP4, GIF, or image sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib import animation as mpl_animation
from matplotlib import pyplot as plt

from src.shared.python.core.contracts import precondition

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from src.shared.python.plotting.base import RecorderInterface


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class AnimationConfig:
    """Settings for swing animation rendering.

    Attributes:
        fps: Frames per second.
        interval_ms: Milliseconds between frames (derived from fps when 0).
        trail_length: Number of past positions to keep visible.
        figsize: Figure size ``(width, height)`` in inches.
        dpi: Dots per inch.
        show_vectors: Overlay force / velocity vectors on each frame.
        vector_scale: Scaling factor applied to vector magnitudes.
        skeleton_links: Pairs of body names to draw as stick-figure links.
        desired_color: RGBA for desired trajectory overlay.
        actual_color: RGBA for actual trajectory line.
    """

    fps: int = 30
    interval_ms: int = 0
    trail_length: int = 60
    figsize: tuple[float, float] = (10.0, 8.0)
    dpi: int = 100
    show_vectors: bool = True
    vector_scale: float = 0.01
    skeleton_links: list[tuple[str, str]] = field(default_factory=list)
    desired_color: str = "#00CED1"
    actual_color: str = "#FF4500"

    @property
    def effective_interval(self) -> int:
        """Return the animation interval in milliseconds, derived from fps if needed."""
        if self.interval_ms > 0:
            return self.interval_ms
        return max(1, int(1000.0 / self.fps))


# ---------------------------------------------------------------------------
# Core animator
# ---------------------------------------------------------------------------
class SwingAnimator:
    """Creates matplotlib ``FuncAnimation`` objects from simulation data.

    Typical usage::

        animator = SwingAnimator(recorder, config)
        anim = animator.create_trajectory_animation(body_names=["club_head", "r_hand"])
        anim.save("swing.mp4")
    """

    def __init__(
        self,
        recorder: RecorderInterface,
        config: AnimationConfig | None = None,
    ) -> None:
        self.recorder = recorder
        self.config = config or AnimationConfig()

    # ----- public helpers -----

    def create_trajectory_animation(
        self,
        body_names: list[str] | None = None,
        desired_positions: dict[str, np.ndarray] | None = None,
    ) -> mpl_animation.FuncAnimation:
        """Animate 3-D body trajectories with optional desired overlay.

        Args:
            body_names: Bodies to animate. Defaults to ``["club_head"]``.
            desired_positions: ``{body: (N,3)}`` desired trajectories.

        Returns:
            A ``FuncAnimation`` ready for display or ``.save()``.
        """
        body_names = body_names or ["club_head"]
        cfg = self.config

        fig = plt.figure(figsize=cfg.figsize, dpi=cfg.dpi)
        ax: Axes = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")  # type: ignore[attr-defined]
        ax.set_title("Swing Trajectory Animation")

        body_data, times = self._gather_trajectory_data(body_names)
        if len(times) == 0:
            ax.text2D(  # type: ignore[attr-defined]
                0.5, 0.5, "No trajectory data", transform=ax.transAxes, ha="center"
            )
            return mpl_animation.FuncAnimation(fig, lambda _: [], frames=1)

        self._plot_desired_trajectories(ax, desired_positions, cfg)
        lines, points = self._create_body_artists(ax, body_data, cfg)
        time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)  # type: ignore[attr-defined]
        ax.legend(loc="upper right", fontsize=8)
        self._set_axis_limits_from_data(ax, body_data)

        def _update(frame: int) -> list:
            artists: list = []
            start = max(0, frame - cfg.trail_length)
            for name, pos in body_data.items():
                seg = pos[start : frame + 1]
                if seg.ndim == 2 and seg.shape[1] >= 3:
                    lines[name].set_data(seg[:, 0], seg[:, 1])
                    lines[name].set_3d_properties(seg[:, 2])
                    points[name].set_data([pos[frame, 0]], [pos[frame, 1]])
                    points[name].set_3d_properties([pos[frame, 2]])
                artists.extend([lines[name], points[name]])
            time_text.set_text(f"t = {times[frame]:.3f} s")
            artists.append(time_text)
            return artists

        anim = mpl_animation.FuncAnimation(
            fig,
            _update,
            frames=len(times),
            interval=cfg.effective_interval,
            blit=False,
        )
        return anim

    def _gather_trajectory_data(self, body_names):
        body_data: dict[str, np.ndarray] = {}
        times = np.empty(0)
        for name in body_names:
            t, pos = self.recorder.get_time_series(f"{name}_position")
            if len(t) > 0:
                body_data[name] = np.asarray(pos)
                if len(t) > len(times):
                    times = np.asarray(t)
        return body_data, times

    def _plot_desired_trajectories(self, ax, desired_positions, cfg):
        if not desired_positions:
            return
        for name, pts in desired_positions.items():
            pts = np.asarray(pts)
            if pts.ndim == 2 and pts.shape[1] >= 3:
                ax.plot(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    color=cfg.desired_color,
                    alpha=0.4,
                    linewidth=1.5,
                    label=f"{name} desired",
                )

    def _create_body_artists(self, ax, body_data, cfg):
        lines: dict[str, Any] = {}
        points: dict[str, Any] = {}
        for name in body_data:
            (line,) = ax.plot(
                [], [], [], linewidth=1.5, color=cfg.actual_color, label=name
            )
            (pt,) = ax.plot([], [], [], "o", markersize=5, color=cfg.actual_color)
            lines[name] = line
            points[name] = pt
        return lines, points

    def _set_axis_limits_from_data(self, ax, body_data):
        all_pts = np.vstack(list(body_data.values()))
        margin = 0.1
        for setter, col in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
            lo, hi = float(all_pts[:, col].min()), float(all_pts[:, col].max())
            span = max(hi - lo, 0.1)
            setter(lo - margin * span, hi + margin * span)

    @precondition(
        lambda self, body_positions, links=None: len(body_positions) > 0,
        "Body positions dict must be non-empty",
    )
    def create_stick_figure_animation(
        self,
        body_positions: dict[str, np.ndarray],
        links: list[tuple[str, str]] | None = None,
    ) -> mpl_animation.FuncAnimation:
        """Animate a stick-figure skeleton.

        Args:
            body_positions: ``{body_name: (N,3)}`` positions over time.
            links: Pairs of body names to connect with lines.

        Returns:
            ``FuncAnimation`` for skeleton playback.
        """
        cfg = self.config
        links = links or cfg.skeleton_links
        fig = plt.figure(figsize=cfg.figsize, dpi=cfg.dpi)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Skeleton Animation")

        n_frames = max(len(p) for p in body_positions.values())

        link_lines: list[Any] = []
        for _ in links:
            (line,) = ax.plot([], [], [], "o-", linewidth=2.0, markersize=4)
            link_lines.append(line)

        # Axis limits
        all_pts = np.vstack(list(body_positions.values()))
        for setter, col in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
            lo, hi = float(all_pts[:, col].min()), float(all_pts[:, col].max())
            span = max(hi - lo, 0.1)
            setter(lo - 0.1 * span, hi + 0.1 * span)

        def _update(frame: int) -> list:
            for idx, (a, b) in enumerate(links):
                if a in body_positions and b in body_positions:
                    pa = body_positions[a][min(frame, len(body_positions[a]) - 1)]
                    pb = body_positions[b][min(frame, len(body_positions[b]) - 1)]
                    link_lines[idx].set_data([pa[0], pb[0]], [pa[1], pb[1]])
                    link_lines[idx].set_3d_properties([pa[2], pb[2]])
            return link_lines

        return mpl_animation.FuncAnimation(
            fig,
            _update,
            frames=n_frames,
            interval=cfg.effective_interval,
            blit=False,
        )

    @precondition(
        lambda self, positions, vectors, times=None, label="Force": len(positions) > 0,
        "Positions array must be non-empty",
    )
    @precondition(
        lambda self, positions, vectors, times=None, label="Force": len(positions)
        == len(vectors),
        "Positions and vectors must have the same length",
    )
    def create_vector_field_animation(
        self,
        positions: np.ndarray,
        vectors: np.ndarray,
        times: np.ndarray | None = None,
        label: str = "Force",
    ) -> mpl_animation.FuncAnimation:
        """Animate time-varying vectors (forces, velocities) at body positions.

        Args:
            positions: ``(N, 3)`` application points.
            vectors: ``(N, 3)`` vector values per frame.
            times: Optional timestamps for the annotation.
            label: Legend label for the vector arrows.

        Returns:
            ``FuncAnimation`` for vector evolution.
        """
        cfg = self.config
        fig = plt.figure(figsize=cfg.figsize, dpi=cfg.dpi)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(f"{label} Vector Animation")

        n_frames = len(positions)
        if times is None:
            times = np.arange(n_frames, dtype=float)

        quiver = ax.quiver(
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            color=cfg.actual_color,
            arrow_length_ratio=0.15,
        )
        time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)  # type: ignore[attr-defined]

        # Limits
        all_pts = np.asarray(positions)
        for setter, col in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
            lo, hi = float(all_pts[:, col].min()), float(all_pts[:, col].max())
            span = max(hi - lo, 0.1)
            setter(lo - 0.2 * span, hi + 0.2 * span)

        def _update(frame: int) -> list:
            nonlocal quiver
            quiver.remove()
            p = positions[frame]
            v = vectors[frame] * cfg.vector_scale
            quiver = ax.quiver(
                [p[0]],
                [p[1]],
                [p[2]],
                [v[0]],
                [v[1]],
                [v[2]],
                color=cfg.actual_color,
                arrow_length_ratio=0.15,
            )
            time_text.set_text(f"t = {times[frame]:.3f} s")
            return [quiver, time_text]

        return mpl_animation.FuncAnimation(
            fig,
            _update,
            frames=n_frames,
            interval=cfg.effective_interval,
            blit=False,
        )

    # ----- convenience save wrapper -----

    @staticmethod
    @precondition(
        lambda anim, path, writer="ffmpeg", fps=30, dpi=100: fps > 0,
        "Frames per second must be positive",
    )
    @precondition(
        lambda anim, path, writer="ffmpeg", fps=30, dpi=100: dpi > 0,
        "DPI must be positive",
    )
    def save_animation(
        anim: mpl_animation.FuncAnimation,
        path: str | Path,
        writer: str = "ffmpeg",
        fps: int = 30,
        dpi: int = 100,
    ) -> Path:
        """Save an animation to disk.

        Args:
            anim: The animation to save.
            path: Output file path (e.g. ``"swing.mp4"``, ``"swing.gif"``).
            writer: Matplotlib writer backend (``"ffmpeg"``, ``"pillow"``).
            fps: Frames per second in the output file.
            dpi: Output DPI.

        Returns:
            Resolved ``Path`` of the saved file.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(out), writer=writer, fps=fps, dpi=dpi)
        return out
