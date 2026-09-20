"""Core data structures and kinematic routines for the Tour Matching Viewer (Step 3).

Supports:
1. Pure data loading of native replay archives (.npz) and OpenSim (.mot/.sto).
2. Pure-Python forward kinematics (body_poses_from_state) verified to < 1e-9 against MuJoCo.
3. Frame generation (viewer_frame) computing world capsule segments, markers, and RMS error.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.mujoco.python.native_mjcf import (
    transform as _transform,
)
from src.shared.python.motion_matching.full_body_spec import (
    order_full_body_joints,
    upper_body_slice,
)
from src.shared.python.motion_matching.visual_skeleton import (
    WorldSegment,
    derive_visual_skeleton,
    skeleton_world_segments,
)

Array: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]

ENGINE_COLORS: dict[str, str] = {
    "mujoco": "#1f77b4",  # Blue
    "pinocchio": "#d62728",  # Red
    "drake": "#2ca02c",  # Green
    "opensim": "#9467bd",  # Purple
    "simscape": "#ff7f0e",  # Orange
    "default": "#ff7f0e",  # Orange
}


@dataclass(frozen=True)
class MultiCandidateReplay:
    """Up to 4 candidate trajectories overlaid on a shared timeline (MV-03 #10479)."""

    candidates: tuple[tuple[str, ReplayData], ...]

    def __post_init__(self) -> None:
        if len(self.candidates) > 4:
            raise ValueError("MultiCandidateReplay supports at most 4 candidates")
        if not self.candidates:
            raise ValueError("MultiCandidateReplay requires at least 1 candidate")

    @property
    def frame_count(self) -> int:
        return self.candidates[0][1].frame_count

    @property
    def time_s(self) -> NDArray[np.float64]:
        return self.candidates[0][1].time_s

    def get_per_engine_rms(self, frame_idx: int) -> dict[str, float]:
        """Compute valid marker RMS error (in meters) for each candidate at frame_idx."""
        res: dict[str, float] = {}
        for engine, replay in self.candidates:
            if frame_idx < 0 or frame_idx >= replay.frame_count:
                continue
            tm = (
                replay.target_markers_m[frame_idx]
                if replay.target_markers_m is not None
                else None
            )
            mm = (
                replay.model_markers_m[frame_idx]
                if replay.model_markers_m is not None
                else None
            )
            vm = replay.valid_mask[frame_idx] if replay.valid_mask is not None else None
            if tm is not None and mm is not None:
                diff = mm[vm] - tm[vm] if vm is not None else mm - tm
                res[engine] = float(np.sqrt(np.mean(diff**2))) if len(diff) > 0 else 0.0
            else:
                res[engine] = 0.0
        return res


@dataclass(frozen=True)
class ReplayData:
    """Loaded trajectory data ready for 3D playback and visual comparison."""

    time_s: NDArray[np.float64]
    coordinates: NDArray[np.float64]
    model_markers_m: NDArray[np.float64] | None = None
    target_markers_m: NDArray[np.float64] | None = None
    valid_mask: NDArray[np.bool_] | None = None
    coordinate_names: tuple[str, ...] | None = None

    @property
    def frame_count(self) -> int:
        return int(len(self.time_s))


@dataclass(frozen=True)
class ViewerFrame:
    """Resolved 3D rendering data for a single frame of playback."""

    segments: list[WorldSegment]
    target_markers: NDArray[np.float64] | None
    model_markers: NDArray[np.float64] | None
    valid_mask: NDArray[np.bool_] | None
    rms_error: float


def load_replay(
    path: Path | str,
    spec: Mapping[str, Any] | None = None,
) -> ReplayData:
    """Load playback trajectory from a native .npz archive or OpenSim .mot file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Replay file not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".npz":
        return _load_npz_replay(p, spec)
    if suffix in (".mot", ".sto"):
        return _load_mot_replay(p, spec)
    raise ValueError(
        f"Unsupported replay format: {suffix} (expected .npz, .mot, or .sto)"
    )


def _load_npz_replay(path: Path, spec: Mapping[str, Any] | None) -> ReplayData:
    data = np.load(path)
    if "time_s" not in data:
        raise ValueError("Missing 'time_s' in replay archive")

    time_s = np.asarray(data["time_s"], dtype=np.float64)
    if time_s.ndim != 1:
        raise ValueError("time_s must be a 1D array")
    if len(time_s) > 1 and np.any(np.diff(time_s) <= 0):
        raise ValueError("time_s timestamps must be strictly monotone")

    n_frames = len(time_s)

    if "coordinates" in data:
        coordinates = np.asarray(data["coordinates"], dtype=np.float64)
    elif "q" in data:
        coordinates = np.asarray(data["q"], dtype=np.float64)
    elif "native_state" in data:
        native_state = np.asarray(data["native_state"], dtype=np.float64)
        if spec is not None and "coordinate_order" in spec:
            n_coords = len(spec["coordinate_order"])
        else:
            n_coords = native_state.shape[1] // 2
        coordinates = native_state[:, :n_coords]
    else:
        raise ValueError("Archive must contain 'coordinates', 'q', or 'native_state'")

    if coordinates.shape[0] != n_frames:
        raise ValueError(
            f"Frame count mismatch: time_s has {n_frames}, coordinates has {coordinates.shape[0]}"
        )

    model_markers_m: NDArray[np.float64] | None = None
    for key in ("markers_m", "model_markers_m"):
        if key in data:
            model_markers_m = np.asarray(data[key], dtype=np.float64)
            if model_markers_m.shape[0] != n_frames:
                raise ValueError(
                    f"Frame count mismatch: time_s has {n_frames}, {key} has {model_markers_m.shape[0]}"
                )
            break

    target_markers_m: NDArray[np.float64] | None = None
    for key in ("target_m", "target_markers_m"):
        if key in data:
            target_markers_m = np.asarray(data[key], dtype=np.float64)
            if target_markers_m.shape[0] != n_frames:
                raise ValueError(
                    f"Frame count mismatch: time_s has {n_frames}, {key} has {target_markers_m.shape[0]}"
                )
            break

    valid_mask: NDArray[np.bool_] | None = None
    for key in ("valid", "valid_mask", "marker_validity"):
        if key in data:
            valid_mask = np.asarray(data[key], dtype=np.bool_)
            if valid_mask.shape[0] != n_frames:
                raise ValueError(
                    f"Frame count mismatch: time_s has {n_frames}, {key} has {valid_mask.shape[0]}"
                )
            break

    coord_names: tuple[str, ...] | None = None
    if "coordinate_order" in data:
        coord_names = tuple(str(x) for x in data["coordinate_order"])
    elif "manifest_json" in data:
        import json

        try:
            manifest = json.loads(str(data["manifest_json"]))
            if manifest.get("coordinate_names"):
                coord_names = tuple(manifest["coordinate_names"])
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    elif spec is not None and "coordinate_order" in spec:
        coord_names = tuple(spec["coordinate_order"])

    return ReplayData(
        time_s=time_s,
        coordinates=coordinates,
        model_markers_m=model_markers_m,
        target_markers_m=target_markers_m,
        valid_mask=valid_mask,
        coordinate_names=coord_names,
    )


def _load_mot_replay(path: Path, spec: Mapping[str, Any] | None) -> ReplayData:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    end_idx = -1
    in_degrees = False
    for i, line in enumerate(lines):
        s = line.strip()
        if "=" in s:
            k, _, v = s.partition("=")
            if k.strip().lower() == "indegrees" and v.strip().lower() == "yes":
                in_degrees = True
        if s.lower() == "endheader":
            end_idx = i
            break

    if end_idx < 0 or end_idx + 1 >= len(lines):
        raise ValueError(
            "Invalid OpenSim motion file: missing 'endheader' or column names"
        )

    col_names = lines[end_idx + 1].split()
    if not col_names or col_names[0].lower() != "time":
        raise ValueError("First column of OpenSim motion file must be 'time'")

    data_lines = [ln for ln in lines[end_idx + 2 :] if ln.strip()]
    if not data_lines:
        raise ValueError("OpenSim motion file contains no data rows")

    rows = []
    for row_str in data_lines:
        rows.append([float(val) for val in row_str.split()])
    table = np.asarray(rows, dtype=np.float64)

    time_s = table[:, 0]
    if len(time_s) > 1 and np.any(np.diff(time_s) <= 0):
        raise ValueError("time_s timestamps must be strictly monotone")

    n_frames = len(time_s)
    col_map = {name: table[:, idx] for idx, name in enumerate(col_names)}

    if spec is not None and "coordinate_order" in spec:
        coord_order = spec["coordinate_order"]
        coord_cols = []
        for name in coord_order:
            if name not in col_map:
                raise ValueError(
                    f"Required coordinate '{name}' missing from motion file"
                )
            arr = col_map[name].copy()
            # If in degrees, convert rotational coords to radians
            if (
                in_degrees
                and not any(
                    name.endswith(suffix)
                    for suffix in ("_tx", "_ty", "_tz", "_px", "_py", "_pz")
                )
                and not name.startswith("TranslationInput")
            ):
                arr = np.deg2rad(arr)
            coord_cols.append(arr)
        coordinates = np.column_stack(coord_cols)
        coord_names: tuple[str, ...] | None = tuple(coord_order)
    else:
        coordinates = table[:, 1:]
        coord_names = tuple(col_names[1:])

    return ReplayData(
        time_s=time_s,
        coordinates=coordinates,
        model_markers_m=None,
        target_markers_m=None,
        valid_mask=None,
        coordinate_names=coord_names,
    )


def body_poses_from_state(
    spec: Mapping[str, Any],
    q: Sequence[float] | Mapping[str, float] | NDArray[np.float64],
    coordinate_names: Sequence[str] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute 4x4 rigid body poses from coordinate state using pure-Python FK.

    Matches MuJoCo forward kinematics (xpos and xmat) to < 1e-9 across all joints.
    """
    ordered_joints = order_full_body_joints(spec)

    if isinstance(q, Mapping):
        coord_map = dict(q)
    elif coordinate_names is not None:
        coord_map = {
            name: float(val) for name, val in zip(coordinate_names, q, strict=True)
        }
    else:
        # Default joint ordering matching the depth-first kinematic tree
        children_by_parent: dict[str, list[Mapping[str, Any]]] = {}
        for joint in ordered_joints:
            children_by_parent.setdefault(joint["parent"], []).append(joint)

        default_names: list[str] = []

        def traverse(body: str) -> None:
            for j in children_by_parent.get(body, []):
                for prim in j["primitives"]:
                    default_names.append(prim["coordinate"])
                traverse(j["child"])

        traverse("world")
        coord_map = {
            name: float(val) for name, val in zip(default_names, q, strict=True)
        }

    offsets: dict[str, NDArray[np.float64]] = {"world": np.eye(4)}
    poses: dict[str, NDArray[np.float64]] = {"world": np.eye(4)}

    for joint in ordered_joints:
        parent = joint["parent"]
        child = joint["child"]
        child_to_follower = _transform(joint["child_to_follower"])
        child_offset: NDArray[np.float64] = np.asarray(
            np.linalg.inv(child_to_follower), dtype=np.float64
        )
        offsets[child] = child_offset

        parent_offset = offsets[parent]
        m_rel = parent_offset @ _transform(joint["parent_to_base"])
        b_pos = m_rel[:3, 3]
        r_b = m_rel[:3, :3]

        p_p = poses[parent][:3, 3]
        r_p = poses[parent][:3, :3]

        curr_pos = p_p + r_p @ b_pos
        curr_r = r_p @ r_b

        for prim in joint["primitives"]:
            kind = prim["primitive"]
            val = coord_map[prim["coordinate"]]
            axis = np.eye(3)["xyz".index(kind[1])]
            if kind[0] == "P":
                curr_pos = curr_pos + curr_r @ (axis * val)
            elif kind[0] == "R":
                curr_r = curr_r @ Rotation.from_rotvec(axis * val).as_matrix()

        t_child = np.eye(4)
        t_child[:3, :3] = curr_r
        t_child[:3, 3] = curr_pos
        poses[child] = t_child

    return poses


def viewer_frame(
    spec: Mapping[str, Any],
    replay: ReplayData,
    frame_idx: int,
) -> ViewerFrame:
    """Compute visual skeleton segments, markers, and RMS error for a single playback frame."""
    if frame_idx < 0 or frame_idx >= replay.frame_count:
        raise IndexError(
            f"frame_idx {frame_idx} out of range [0, {replay.frame_count})"
        )

    skeleton = derive_visual_skeleton(spec)
    q_frame = replay.coordinates[frame_idx]
    coord_names = replay.coordinate_names or tuple(spec["coordinate_order"])
    poses = body_poses_from_state(spec, q_frame, coordinate_names=coord_names)
    segments = skeleton_world_segments(skeleton, poses)

    target_markers = (
        replay.target_markers_m[frame_idx]
        if replay.target_markers_m is not None
        else None
    )
    model_markers = (
        replay.model_markers_m[frame_idx]
        if replay.model_markers_m is not None
        else None
    )
    valid_mask = replay.valid_mask[frame_idx] if replay.valid_mask is not None else None

    rms_error = 0.0
    if target_markers is not None and model_markers is not None:
        if valid_mask is not None:
            diff = model_markers[valid_mask] - target_markers[valid_mask]
        else:
            diff = model_markers - target_markers
        if len(diff) > 0:
            rms_error = float(np.sqrt(np.mean(diff**2)))

    return ViewerFrame(
        segments=segments,
        target_markers=target_markers,
        model_markers=model_markers,
        valid_mask=valid_mask,
        rms_error=rms_error,
    )


def _render_single_gif_frame(
    ax: Any,
    spec: Mapping[str, Any],
    replays: tuple[tuple[str, ReplayData], ...],
    frame_idx: int,
) -> None:
    """Render 3D visual segments and markers for one frame onto axes."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    ax.clear()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(0.0, 2.0)
    ax.view_init(elev=20, azim=45)

    for engine, rep in replays:
        if frame_idx >= rep.frame_count:
            continue
        vframe = viewer_frame(spec, rep, frame_idx)
        lines = [[seg.start_m, seg.end_m] for seg in vframe.segments]
        eng_col = ENGINE_COLORS.get(engine, ENGINE_COLORS["default"])
        if lines:
            ax.add_collection3d(
                Line3DCollection(lines, colors=eng_col, linewidths=2.0, alpha=0.8)
            )
        if vframe.model_markers is not None:
            ax.scatter(
                vframe.model_markers[:, 0],
                vframe.model_markers[:, 1],
                vframe.model_markers[:, 2],
                color=eng_col,
                s=15,
            )


def export_animation_gif(
    replay: ReplayData | MultiCandidateReplay,
    spec: Mapping[str, Any],
    output_path: Path | str,
    *,
    fps: int = 30,
    dpi: int = 80,
    max_frames: int | None = None,
) -> Path:
    """Export 3D trajectory playback as an animated GIF."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from PIL import Image

    matplotlib.use("Agg")
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fig = Figure(figsize=(5, 5), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")

    total_frames = replay.frame_count
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    replays = (
        replay.candidates
        if isinstance(replay, MultiCandidateReplay)
        else (("default", replay),)
    )

    frames: list[Image.Image] = []
    for i in range(total_frames):
        _render_single_gif_frame(ax, spec, replays, i)
        canvas.draw()
        buf = canvas.buffer_rgba()
        img = Image.frombuffer(
            "RGBA", canvas.get_width_height(), buf, "raw", "RGBA", 0, 1
        )
        frames.append(img.convert("RGB"))

    if frames:
        duration_ms = max(10, int(1000 / max(1, fps)))
        frames[0].save(
            out_p,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
    return out_p
