"""Synchronized 3D video and animation overlay of Simscape model vs. C3D markers (#9921).

Provides Design-by-Contract (DbC) data loading, skeletal reconstruction,
residual error calculation, and animated 3D video generation (MP4/GIF).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from numpy.typing import NDArray
from PIL import Image

try:
    import cv2  # type: ignore[import-untyped,import-not-found]
except ImportError:
    cv2 = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Standard golf anatomical skeleton connections (marker name pairs)
STANDARD_BONES: tuple[tuple[str, str], ...] = (
    # Pelvis / Waist loop
    ("WaistLeft", "WaistRight"),
    ("WaistRight", "WaistRBack"),
    ("WaistRBack", "WaistLBack"),
    ("WaistLBack", "WaistLeft"),
    # Torso / Spine
    ("WaistLBack", "BackLeft"),
    ("WaistRBack", "BackRight"),
    ("BackLeft", "BackTop"),
    ("BackRight", "BackTop"),
    # Head triad
    ("BackTop", "HeadTop"),
    ("HeadTop", "HeadFront"),
    ("HeadFront", "HeadSide"),
    ("HeadSide", "HeadTop"),
    # Left shoulder and arm
    ("BackTop", "LShoulderTop"),
    ("LShoulderTop", "LShoulderBack"),
    ("LShoulderTop", "LUArmHigh"),
    ("LUArmHigh", "LElbowOut"),
    ("LElbowOut", "LWristTop"),
    # Right shoulder and arm
    ("BackTop", "RShoulderBack"),
    ("RShoulderBack", "RUArmHigh"),
    ("RUArmHigh", "RElbowOut"),
    ("RElbowOut", "RWristTop"),
    # Club / Shaft / Clubhead
    ("LWristTop", "Marker_2:2:1"),
    ("RWristTop", "Marker_2:2:1"),
    ("Marker_2:2:1", "Marker_2:2:2"),
    ("Marker_2:2:2", "Marker_2:2:3"),
    ("Marker_2:2:3", "Marker_2:2:1"),
    ("Marker_3:3:1", "Marker_3:3:2"),
    ("Marker_3:3:2", "Marker_3:3:3"),
    ("Marker_3:3:3", "Marker_3:3:1"),
    ("Marker_2:2:1", "Marker_3:3:1"),
)


@dataclass(frozen=True)
class SkeletalTopology:
    """Anatomical marker connectivity graph."""

    bones: tuple[tuple[str, str], ...] = STANDARD_BONES

    def filter_bones_for_labels(
        self, available_labels: Sequence[str]
    ) -> list[tuple[int, int]]:
        """Map bone pairs to integer index pairs present in available labels."""
        label_set = {name: idx for idx, name in enumerate(available_labels)}
        resolved: list[tuple[int, int]] = []
        for name_a, name_b in self.bones:
            if name_a in label_set and name_b in label_set:
                resolved.append((label_set[name_a], label_set[name_b]))
        return resolved


@dataclass(frozen=True)
class OverlayDataset:
    """Validated time-series dataset of target mocap and fitted model markers."""

    times: NDArray[np.float64]
    labels: list[str]
    target_points: NDArray[np.float64]
    target_valid: NDArray[np.bool_]
    model_points: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.times.ndim != 1 or len(self.times) == 0:
            raise ValueError("times must be a non-empty 1D array")
        num_times = len(self.times)
        num_markers = len(self.labels)

        if self.target_points.shape != (num_times, num_markers, 3):
            raise ValueError(
                f"target_points shape must be ({num_times}, {num_markers}, 3)"
            )
        if self.model_points.shape != (num_times, num_markers, 3):
            raise ValueError(
                f"model_points shape must be ({num_times}, {num_markers}, 3)"
            )
        if self.target_valid.shape != (num_times, num_markers):
            raise ValueError(f"target_valid shape must be ({num_times}, {num_markers})")

    @property
    def frame_count(self) -> int:
        return len(self.times)

    @property
    def marker_count(self) -> int:
        return len(self.labels)

    @property
    def duration_s(self) -> float:
        return float(self.times[-1] - self.times[0])

    def calculate_residuals_mm(
        self, frame_index: int | None = None
    ) -> NDArray[np.float64]:
        """Compute Euclidean residuals in millimetres for valid observations."""
        if frame_index is not None:
            if not 0 <= frame_index < self.frame_count:
                raise IndexError(
                    f"frame_index {frame_index} out of range [0, {self.frame_count})"
                )
            diff = (
                self.model_points[frame_index] - self.target_points[frame_index]
            ) * 1000.0
            dist = np.linalg.norm(diff, axis=-1)
            valid = self.target_valid[frame_index]
            dist[~valid] = np.nan
            return dist

        diff = (self.model_points - self.target_points) * 1000.0
        dist = np.linalg.norm(diff, axis=-1)
        dist[~self.target_valid] = np.nan
        return dist

    def rms(self, frame_index: int) -> float:
        """Instantaneous RMS error in millimetres for a specific frame."""
        residuals = self.calculate_residuals_mm(frame_index)
        valid = np.isfinite(residuals)
        if not np.any(valid):
            return 0.0
        return float(np.sqrt(np.mean(residuals[valid] ** 2)))

    def total_rms(self) -> float:
        """Overall RMS error in millimetres across all frames and valid markers."""
        residuals = self.calculate_residuals_mm()
        valid = np.isfinite(residuals)
        if not np.any(valid):
            return 0.0
        return float(np.sqrt(np.mean(residuals[valid] ** 2)))


def load_overlay_dataset(
    capture_path: Path | str,
    replay_path: Path | str,
    max_time_s: float | None = None,
) -> OverlayDataset:
    """Load, validate, and synchronize mocap target and model replay data."""
    cap_file = Path(capture_path)
    rep_file = Path(replay_path)

    if not cap_file.is_file():
        raise FileNotFoundError(f"Capture payload not found: {cap_file}")
    if not rep_file.is_file():
        raise FileNotFoundError(f"Replay payload not found: {rep_file}")

    try:
        capture_data = json.loads(cap_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON in capture file {cap_file}: {exc}") from exc

    try:
        replay_data = json.loads(rep_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON in replay file {rep_file}: {exc}") from exc

    # Determine labels
    if "labels" in replay_data:
        labels: list[str] = list(replay_data["labels"])
    elif "labels" in capture_data:
        labels = list(capture_data["labels"])
    else:
        raise ValueError("No marker labels found in either replay or capture payload")

    cap_labels: list[str] = list(capture_data.get("labels", []))
    missing = [name for name in labels if name not in cap_labels]
    if missing:
        raise ValueError(f"Capture payload missing required labels: {missing[:5]}")

    cap_indices = [cap_labels.index(name) for name in labels]

    time_arr = np.asarray(capture_data["time_s"], dtype=np.float64)
    target_pts_all = np.asarray(capture_data["points_world_m"], dtype=np.float64)
    valid_flags_all = np.asarray(capture_data["valid"], dtype=bool)

    # Replay prediction
    if "prediction_m" in replay_data:
        pred_pts_all = np.asarray(replay_data["prediction_m"], dtype=np.float64)
    elif "final_prediction_m" in replay_data:
        pred_pts_all = np.asarray(replay_data["final_prediction_m"], dtype=np.float64)
    else:
        raise ValueError(
            "Replay payload does not contain prediction_m or final_prediction_m"
        )

    num_pred_frames = pred_pts_all.shape[0]
    num_pred_markers = pred_pts_all.shape[1]

    if num_pred_markers != len(labels):
        raise ValueError(
            f"Marker count mismatch: prediction has {num_pred_markers}, expected {len(labels)}"
        )

    # Slice target to matching frames
    if num_pred_frames > len(time_arr):
        raise ValueError(
            f"Replay has {num_pred_frames} frames but capture only has {len(time_arr)}"
        )

    times = time_arr[:num_pred_frames]
    target_pts = target_pts_all[:num_pred_frames, cap_indices, :]
    valid_flags = valid_flags_all[:num_pred_frames, cap_indices]
    model_pts = pred_pts_all

    # Optional duration cutoff
    if max_time_s is not None and max_time_s > 0:
        mask = times <= (max_time_s + 1e-10)
        times = times[mask]
        target_pts = target_pts[mask]
        valid_flags = valid_flags[mask]
        model_pts = model_pts[mask]

    return OverlayDataset(
        times=times,
        labels=labels,
        target_points=target_pts,
        target_valid=valid_flags,
        model_points=model_pts,
    )


def render_overlay_frame(
    dataset: OverlayDataset,
    frame_index: int,
    ax: Any,
    topology: SkeletalTopology | None = None,
    error_threshold_mm: float = 15.0,
    camera_elev: float = 20.0,
    camera_azim: float = -65.0,
) -> dict[str, Any]:
    """Render a single 3D comparison frame on the given matplotlib 3D axis."""
    if topology is None:
        topology = SkeletalTopology()

    ax.clear()

    t_s = float(dataset.times[frame_index])
    target_coords = dataset.target_points[frame_index]
    model_coords = dataset.model_points[frame_index]
    valid_mask = dataset.target_valid[frame_index]
    residuals = dataset.calculate_residuals_mm(frame_index)
    frame_rms = dataset.rms(frame_index)

    # 1. Target mocap markers (cyan diamonds)
    if np.any(valid_mask):
        ax.scatter(
            target_coords[valid_mask, 0],
            target_coords[valid_mask, 1],
            target_coords[valid_mask, 2],
            c="deepskyblue",
            marker="D",
            s=28,
            alpha=0.85,
            label="C3D Target",
            depthshade=True,
        )

    # 2. Model fitted markers (orange circles)
    ax.scatter(
        model_coords[:, 0],
        model_coords[:, 1],
        model_coords[:, 2],
        c="darkorange",
        marker="o",
        s=36,
        alpha=0.95,
        label="Simscape Model",
        depthshade=True,
    )

    # 3. Skeletal bones
    resolved_bones = topology.filter_bones_for_labels(dataset.labels)

    # Draw model skeleton bones (orange lines)
    for idx_a, idx_b in resolved_bones:
        p_a = model_coords[idx_a]
        p_b = model_coords[idx_b]
        ax.plot(
            [p_a[0], p_b[0]],
            [p_a[1], p_b[1]],
            [p_a[2], p_b[2]],
            color="chocolate",
            linewidth=2.0,
            alpha=0.8,
        )

    # Draw target skeleton bones (faint cyan lines)
    for idx_a, idx_b in resolved_bones:
        if valid_mask[idx_a] and valid_mask[idx_b]:
            p_a = target_coords[idx_a]
            p_b = target_coords[idx_b]
            ax.plot(
                [p_a[0], p_b[0]],
                [p_a[1], p_b[1]],
                [p_a[2], p_b[2]],
                color="deepskyblue",
                linestyle="--",
                linewidth=1.2,
                alpha=0.5,
            )

    # 4. Residual error vectors (red lines connecting model to target)
    for idx in range(dataset.marker_count):
        if valid_mask[idx] and residuals[idx] > error_threshold_mm:
            m_pt = model_coords[idx]
            t_pt = target_coords[idx]
            ax.plot(
                [m_pt[0], t_pt[0]],
                [m_pt[1], t_pt[1]],
                [m_pt[2], t_pt[2]],
                color="crimson",
                linewidth=1.5,
                alpha=0.7,
            )

    # Bounding box & views
    all_pts = np.vstack([target_coords[valid_mask], model_coords])
    center = np.median(all_pts, axis=0)
    radius = 0.95  # golf swing reach sphere

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.view_init(elev=camera_elev, azim=camera_azim)
    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Y (m)", fontsize=9)
    ax.set_zlabel("Z (m)", fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    # HUD Overlay
    worst_idx = int(np.nanargmax(residuals)) if np.any(np.isfinite(residuals)) else 0
    worst_name = dataset.labels[worst_idx]
    worst_err = (
        float(residuals[worst_idx]) if np.isfinite(residuals[worst_idx]) else 0.0
    )

    hud_text = (
        f"Time: {t_s:0.3f} s (Frame {frame_index + 1}/{dataset.frame_count})\n"
        f"Instantaneous RMS: {frame_rms:5.1f} mm\n"
        f"Max Error: {worst_name} ({worst_err:5.1f} mm)"
    )
    ax.text2D(
        0.03,
        0.93,
        hud_text,
        transform=ax.transAxes,
        fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.8},
    )

    return {
        "frame_index": frame_index,
        "time_s": t_s,
        "rms_mm": frame_rms,
        "max_marker": worst_name,
        "max_error_mm": worst_err,
    }


def render_overlay_video(
    dataset: OverlayDataset,
    output_path: Path | str,
    fps: int = 30,
    dpi: int = 120,
    format: str = "mp4",
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Render full animation across all frames and save to MP4 or GIF."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9, 7), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    topology = SkeletalTopology()

    frames_rgb: list[Image.Image] = []
    video_writer = None

    fmt = format.lower().strip()
    if fmt == "mp4" and cv2 is None:
        logger.warning("OpenCV not installed; falling back to GIF format")
        fmt = "gif"
        out_file = out_file.with_suffix(".gif")

    width, height = 0, 0

    try:
        for idx in range(dataset.frame_count):
            render_overlay_frame(dataset, idx, ax, topology=topology)
            fig.canvas.draw()

            # Extract RGB buffer
            canvas_obj: Any = fig.canvas
            rgba = np.asarray(canvas_obj.buffer_rgba())
            rgb = rgba[:, :, :3]

            if fmt == "mp4":
                if video_writer is None:
                    height, width = rgb.shape[:2]
                    cv2_mod: Any = cv2
                    fourcc = cv2_mod.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2_mod.VideoWriter(
                        str(out_file), fourcc, fps, (width, height)
                    )
                cv2_mod = cv2
                bgr = cv2_mod.cvtColor(rgb, cv2_mod.COLOR_RGB2BGR)
                video_writer.write(bgr)
            else:
                img = Image.fromarray(rgb)
                frames_rgb.append(img)

            if progress_callback is not None:
                progress_callback(idx + 1, dataset.frame_count)

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
            f"Rendered overlay animation to {out_file} ({dataset.frame_count} frames)"
        )
        return out_file
    finally:
        plt.close(fig)


def main() -> None:
    """CLI entrypoint for overlay generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture", type=Path, required=True, help="Path to driver_marker_payload.json"
    )
    parser.add_argument(
        "--replay",
        type=Path,
        required=True,
        help="Path to qualified_candidate_replay.json or first_prefix_fit.json",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output file path (.mp4 or .gif)"
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--dpi", type=int, default=120, help="Resolution DPI")
    parser.add_argument(
        "--format", choices=["mp4", "gif"], default="mp4", help="Video container format"
    )
    parser.add_argument(
        "--max-time", type=float, default=None, help="Maximum duration in seconds"
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    dataset = load_overlay_dataset(args.capture, args.replay, max_time_s=args.max_time)
    logger.info(
        f"Loaded {dataset.frame_count} frames ({dataset.duration_s:0.3f}s), "
        f"{dataset.marker_count} markers. Overall RMS: {dataset.total_rms():0.2f} mm"
    )

    out = render_overlay_video(
        dataset,
        output_path=args.output,
        fps=args.fps,
        dpi=args.dpi,
        format=args.format,
        progress_callback=lambda cur, tot: (
            logger.info(f"Rendering frame {cur}/{tot}...")
            if cur % 25 == 0 or cur == tot
            else None
        ),
    )
    logger.info(f"Video complete: {out}")


if __name__ == "__main__":
    main()
