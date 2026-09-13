"""Synchronized 3D full humanoid skeleton renderer for physics engine forward rollouts.

Provides Design-by-Contract (DbC) kinematic frame extraction, anatomical skeleton
reconstruction (legs, pelvis, spine, torso, head, dual arms, dual wrists, club shaft,
and clubhead), and high-resolution animated GIF/MP4 export for cross-engine motion matching.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from numpy.typing import NDArray
from PIL import Image

try:
    import cv2  # type: ignore[import-untyped,import-not-found]
except ImportError:
    cv2 = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Canonical humanoid bone connections (body segment name pairs)
CANONICAL_HUMANOID_BONES: tuple[tuple[str, str], ...] = (
    # Legs
    ("left_foot", "left_shin"),
    ("left_shin", "left_thigh"),
    ("left_thigh", "pelvis"),
    ("right_foot", "right_shin"),
    ("right_shin", "right_thigh"),
    ("right_thigh", "pelvis"),
    # Spine & Torso
    ("pelvis", "lower_torso"),
    ("lower_torso", "upper_torso"),
    ("upper_torso", "head"),
    # Left Arm
    ("upper_torso", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_hand"),
    # Right Arm
    ("upper_torso", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_hand"),
    # Club
    ("right_hand", "club"),
    ("left_hand", "club"),
    ("club", "clubhead"),
)


@dataclass(frozen=True)
class HumanoidSkeletalTopology:
    """Humanoid body connectivity graph for 3D skeleton reconstruction."""

    bones: tuple[tuple[str, str], ...] = CANONICAL_HUMANOID_BONES

    def filter_bones_for_bodies(
        self, available_bodies: Sequence[str]
    ) -> list[tuple[str, str]]:
        """Filter bone pairs to those where both bodies exist in the model."""
        body_set = set(available_bodies)
        return [(b1, b2) for b1, b2 in self.bones if b1 in body_set and b2 in body_set]


@dataclass(frozen=True)
class HumanoidTrajectoryData:
    """Validated time-series trajectory of all humanoid body coordinates."""

    times: NDArray[np.float64]
    body_names: list[str]
    # Shape: (num_frames, num_bodies, 3)
    body_positions: NDArray[np.float64]
    grip_index: int
    clubhead_index: int

    def __post_init__(self) -> None:
        if self.times.ndim != 1 or len(self.times) == 0:
            raise ValueError("times must be a non-empty 1D array")
        num_times = len(self.times)
        num_bodies = len(self.body_names)
        if self.body_positions.shape != (num_times, num_bodies, 3):
            raise ValueError(
                f"body_positions shape must be ({num_times}, {num_bodies}, 3), got {self.body_positions.shape}"
            )
        if not (0 <= self.grip_index < num_bodies):
            raise IndexError("grip_index out of range")
        if not (0 <= self.clubhead_index < num_bodies):
            raise IndexError("clubhead_index out of range")

    @property
    def frame_count(self) -> int:
        return len(self.times)

    @property
    def grip_points(self) -> NDArray[np.float64]:
        return self.body_positions[:, self.grip_index, :]

    @property
    def clubhead_points(self) -> NDArray[np.float64]:
        return self.body_positions[:, self.clubhead_index, :]


def render_humanoid_frame(
    data: HumanoidTrajectoryData,
    frame_index: int,
    ax: Any,
    topology: HumanoidSkeletalTopology | None = None,
    engine_name: str = "MuJoCo",
    camera_elev: float = 20.0,
    camera_azim: float = -65.0,
) -> None:
    """Render a single 3D frame of the complete humanoid skeleton and club trajectory."""
    if topology is None:
        topology = HumanoidSkeletalTopology()

    if not (0 <= frame_index < data.frame_count):
        raise IndexError(
            f"frame_index {frame_index} out of range [0, {data.frame_count})"
        )

    ax.clear()
    t_val = float(data.times[frame_index])
    coords = {
        name: data.body_positions[frame_index, idx]
        for idx, name in enumerate(data.body_names)
    }

    # 1. Draw bones
    resolved_bones = topology.filter_bones_for_bodies(data.body_names)
    for b1, b2 in resolved_bones:
        p1 = coords[b1]
        p2 = coords[b2]
        is_club = "club" in b1 or "club" in b2
        color = "darkorange" if is_club else "royalblue"
        lw = 3.5 if is_club else 2.5
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color=color,
            linewidth=lw,
            alpha=0.9,
        )

    # 2. Draw body joints / centers
    for name, pos in coords.items():
        if name in ["world", "ball"]:
            continue
        color = "crimson" if name in ["club", "clubhead"] else "dodgerblue"
        size = 50 if name in ["head", "pelvis"] else 30
        ax.scatter([pos[0]], [pos[1]], [pos[2]], color=color, s=size, depthshade=True)

    # 3. Draw trajectory history
    grip_pts = data.grip_points
    head_pts = data.clubhead_points
    ax.plot(
        grip_pts[: frame_index + 1, 0],
        grip_pts[: frame_index + 1, 1],
        grip_pts[: frame_index + 1, 2],
        color="darkturquoise",
        linewidth=1.5,
        linestyle="--",
        label="Grip Trajectory",
    )
    ax.plot(
        head_pts[: frame_index + 1, 0],
        head_pts[: frame_index + 1, 1],
        head_pts[: frame_index + 1, 2],
        color="firebrick",
        linewidth=2.0,
        label="Clubhead Trajectory",
    )

    # 4. Perspective and limits
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(0.0, 2.0)
    ax.view_init(elev=camera_elev, azim=camera_azim)
    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Y (m)", fontsize=9)
    ax.set_zlabel("Z (m)", fontsize=9)

    cur_grip = grip_pts[frame_index]
    cur_head = head_pts[frame_index]
    hud_text = (
        f"{engine_name} Full Humanoid Forward Sim\n"
        f"Time: {t_val:0.3f}s / {data.times[-1]:0.2f}s (Frame {frame_index + 1}/{data.frame_count})\n"
        f"Grip: [{cur_grip[0]:.2f}, {cur_grip[1]:.2f}, {cur_grip[2]:.2f}] m\n"
        f"Clubhead: [{cur_head[0]:.2f}, {cur_head[1]:.2f}, {cur_head[2]:.2f}] m"
    )
    ax.text2D(
        0.03,
        0.90,
        hud_text,
        transform=ax.transAxes,
        fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.85},
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def render_humanoid_video(
    data: HumanoidTrajectoryData,
    output_path: Path | str,
    fps: int = 25,
    dpi: int = 100,
    format: str = "gif",
    engine_name: str = "MuJoCo",
) -> Path:
    """Render full animation across all frames and save to GIF or MP4."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9, 8), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    topology = HumanoidSkeletalTopology()

    frames_rgb: list[Image.Image] = []
    video_writer = None
    fmt = format.lower().strip()

    if fmt == "mp4" and cv2 is None:
        logger.warning("OpenCV not available, falling back to GIF format")
        fmt = "gif"
        out_file = out_file.with_suffix(".gif")

    try:
        for idx in range(data.frame_count):
            render_humanoid_frame(
                data, idx, ax, topology=topology, engine_name=engine_name
            )
            fig.canvas.draw()
            canvas_obj: Any = fig.canvas
            rgba = np.asarray(canvas_obj.buffer_rgba())
            rgb = rgba[:, :, :3]

            if fmt == "mp4":
                if video_writer is None:
                    h, w = rgb.shape[:2]
                    cv2_mod: Any = cv2
                    fourcc = cv2_mod.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2_mod.VideoWriter(
                        str(out_file), fourcc, fps, (w, h)
                    )
                cv2_mod = cv2
                bgr = cv2_mod.cvtColor(rgb, cv2_mod.COLOR_RGB2BGR)
                video_writer.write(bgr)
            else:
                frames_rgb.append(Image.fromarray(rgb))

        if fmt == "mp4" and video_writer is not None:
            video_writer.release()
        elif fmt == "gif" and frames_rgb:
            duration_ms = int(1000.0 / fps)
            frames_rgb[0].save(
                out_file,
                save_all=True,
                append_images=frames_rgb[1:],
                duration=duration_ms,
                loop=0,
            )

        logger.info(
            f"Rendered {engine_name} humanoid animation to {out_file} ({data.frame_count} frames)"
        )
        return out_file
    finally:
        plt.close(fig)
