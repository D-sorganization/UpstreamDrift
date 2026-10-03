"""Core data structures and kinematic routines for the Tour Matching Viewer (Step 3).

Supports:
1. Pure data loading of native replay archives (.npz) and OpenSim (.mot/.sto).
2. Pure-Python forward kinematics (body_poses_from_state) verified to < 1e-9 against MuJoCo.
3. Frame generation (viewer_frame) computing world capsule segments, markers, and RMS error.
"""

from __future__ import annotations

import math
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
    "myosuite": "#8c564b",  # Brown
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
                # Canonical receipt RMS: per-marker 3D distances pooled over the
                # frame's valid markers (matches compute_shared_metrics).
                if len(diff) > 0:
                    rms_sq = np.sum(diff**2, axis=-1)
                    res[engine] = float(np.sqrt(np.mean(rms_sq)))
                else:
                    res[engine] = 0.0
            else:
                res[engine] = 0.0
        return res


def _make_readonly(arr: np.ndarray | None) -> None:
    """Set numpy array flags to read-only while satisfying Law of Demeter."""
    if arr is not None and isinstance(arr, np.ndarray):
        flags = arr.flags
        flags.writeable = False


@dataclass(frozen=True)
class ReplayData:
    """Loaded trajectory data ready for 3D playback and visual comparison."""

    time_s: NDArray[np.float64]
    coordinates: NDArray[np.float64]
    model_markers_m: NDArray[np.float64] | None = None
    target_markers_m: NDArray[np.float64] | None = None
    valid_mask: NDArray[np.bool_] | None = None
    coordinate_names: tuple[str, ...] | None = None
    drive_mode: str = "kinematic_prescribed"
    marker_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _make_readonly(self.time_s)
        _make_readonly(self.coordinates)
        _make_readonly(self.model_markers_m)
        _make_readonly(self.target_markers_m)
        _make_readonly(self.valid_mask)

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


@dataclass(frozen=True)
class MarkerResidual:
    """Per-marker error within a single frame."""

    marker_name: str
    marker_idx: int
    error_m: float


@dataclass(frozen=True)
class FrameResidual:
    """Error metrics and worst marker for a single time frame."""

    frame_idx: int
    time_s: float
    phase: str
    rms_error_m: float
    worst_marker: MarkerResidual | None
    valid_markers: int = 0


@dataclass(frozen=True)
class ResidualSummary:
    """Comprehensive residual timeline across an entire candidate swing."""

    worst_frame_idx: int
    worst_time_s: float
    worst_phase: str
    worst_marker_name: str
    worst_marker_idx: int
    max_marker_error_m: float
    worst_frame_rms_m: float
    frame_residuals: tuple[FrameResidual, ...]
    marker_names: tuple[str, ...]
    residual_vectors: tuple[NDArray[np.float64], ...]

    @property
    def mean_rms_m(self) -> float:
        """Mean RMS error across frames with observed markers, in meters."""
        observed = [fr for fr in self.frame_residuals if fr.valid_markers > 0]
        if not observed:
            return 0.0
        return float(np.mean([fr.rms_error_m for fr in observed]))

    def worst_frame_for_marker(self, marker: str | int) -> int:
        """Return the discrete frame index with highest error for a given marker."""
        if isinstance(marker, int):
            idx = marker
        else:
            norm_marker = marker.strip().lower()
            idx = -1
            for i, name in enumerate(self.marker_names):
                if name.strip().lower() == norm_marker:
                    idx = i
                    break
            if idx == -1:
                raise KeyError(f"Unknown marker name: {marker}")

        best_frame = 0
        best_val = -1.0
        for f_idx, vec in enumerate(self.residual_vectors):
            if idx < len(vec):
                x = vec[idx]
                err = float(
                    math.sqrt(x.dot(x))
                )  # ⚡ Bolt: math.sqrt(ndarray.dot) is faster than np.linalg.norm for small 1D arrays
                if err > best_val:
                    best_val = err
                    best_frame = f_idx
        return best_frame

    def worst_frame_for_phase(self, phase: str) -> int:
        """Return the discrete frame index with highest RMS within a specified swing phase."""
        norm_phase = phase.strip().lower()
        matching_frames = [
            fr for fr in self.frame_residuals if norm_phase in fr.phase.lower()
        ]
        if not matching_frames:
            return self.worst_frame_idx
        worst_fr = max(matching_frames, key=lambda fr: fr.rms_error_m)
        return worst_fr.frame_idx


@dataclass(frozen=True)
class ClubOnlyCompareView:
    """Observed club frames vs inferred body candidates for CO-09 UI (#10613)."""

    trial_id: str
    trial_clock_hz: float
    native_time_s: NDArray[np.float64]
    observed_mid_hands_m: NDArray[np.float64]
    observed_face_m: NDArray[np.float64]
    predicted_mid_hands_m: NDArray[np.float64] | None
    predicted_face_m: NDArray[np.float64] | None
    predicted_unavailable_reason: str | None
    body_candidate_ids: tuple[str, ...]
    error_time_tradeoffs: tuple[Mapping[str, float], ...]
    infeasible_models: tuple[Mapping[str, str], ...]
    legend: tuple[Mapping[str, str], ...]
    body_motion_disclaimer: str
    display_status: str
    native_g1_pass: bool

    def __post_init__(self) -> None:
        n = int(self.native_time_s.shape[0])
        if n < 2:
            raise ValueError("native_time_s must have at least 2 samples")
        for name, arr in (
            ("observed_mid_hands_m", self.observed_mid_hands_m),
            ("observed_face_m", self.observed_face_m),
        ):
            if arr.shape != (n, 3) or not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be finite shape ({n}, 3)")
        if self.predicted_mid_hands_m is None or self.predicted_face_m is None:
            if not self.predicted_unavailable_reason:
                raise ValueError("predicted_unavailable_reason required")
        if not self.native_g1_pass and self.display_status == "native_verified":
            raise ValueError("unqualified compare view cannot appear native_verified")
        if "not measured" not in self.body_motion_disclaimer.lower():
            raise ValueError("body_motion_disclaimer must deny measured body motion")


def club_only_compare_from_ui_result(
    result: Any,
    *,
    predicted_mid_hands_m: NDArray[np.float64] | None = None,
    predicted_face_m: NDArray[np.float64] | None = None,
) -> ClubOnlyCompareView:
    """Build a viewer compare payload from a club-only UI match result."""
    from src.shared.python.motion_matching.club_only.ui_integration import (
        ClubOnlyUiResult,
        build_club_only_result_view,
        build_observed_inferred_legend,
    )

    if not isinstance(result, ClubOnlyUiResult):
        raise TypeError("result must be ClubOnlyUiResult")
    view = build_club_only_result_view(result)
    legend = tuple(entry.as_dict() for entry in build_observed_inferred_legend(view))
    obs = result.observation
    predicted_reason = None
    pred_mid = predicted_mid_hands_m
    pred_face = predicted_face_m
    if pred_mid is None or pred_face is None:
        predicted_reason = (
            "predicted club frame series unavailable: fast-match scored "
            "observation residuals without emitting a continuous predicted "
            "club trajectory package"
        )
        pred_mid = None
        pred_face = None
    return ClubOnlyCompareView(
        trial_id=view.trial_id,
        trial_clock_hz=float(view.trial_clock_hz),
        native_time_s=np.asarray(obs.native_time_s, dtype=np.float64).copy(),
        observed_mid_hands_m=np.asarray(obs.mid_hands_xyz, dtype=np.float64).copy(),
        observed_face_m=np.asarray(obs.face_xyz, dtype=np.float64).copy(),
        predicted_mid_hands_m=(
            None if pred_mid is None else np.asarray(pred_mid, dtype=np.float64).copy()
        ),
        predicted_face_m=(
            None
            if pred_face is None
            else np.asarray(pred_face, dtype=np.float64).copy()
        ),
        predicted_unavailable_reason=predicted_reason,
        body_candidate_ids=tuple(view.candidate_ids),
        error_time_tradeoffs=tuple(dict(x) for x in view.error_time_tradeoffs),
        infeasible_models=tuple(dict(x) for x in view.infeasible_models),
        legend=legend,
        body_motion_disclaimer=view.body_motion_disclaimer,
        display_status=view.display_status.value,
        native_g1_pass=bool(view.native_g1_pass),
    )


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
        # Canonical receipt RMS (tour_metrics.compute_shared_metrics): pool the
        # per-marker 3D distances over the frame's valid markers, never the
        # flattened XYZ components.
        if len(diff) > 0:
            rms_sq = np.sum(diff**2, axis=-1)
            rms_error = float(np.sqrt(np.mean(rms_sq)))

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

    frames: list[Any] = []
    for i in range(total_frames):
        _render_single_gif_frame(ax, spec, replays, i)
        frames.append(_capture_canvas_frame(canvas))

    return _save_gif_frames(frames, out_p, fps)


def _resolve_marker_names(
    tm: NDArray[np.float64] | None,
    mm: NDArray[np.float64] | None,
    marker_names: Sequence[str] | None,
) -> tuple[str, ...]:
    if marker_names is not None:
        return tuple(marker_names)
    n_m = tm.shape[1] if tm is not None else (mm.shape[1] if mm is not None else 0)
    return tuple(f"Marker_{i}" for i in range(n_m))


def _resolve_phase_boundaries(
    n_frames: int, event_indices: Mapping[str, int] | None
) -> tuple[int, int, int]:
    if event_indices:
        return (
            event_indices.get("Top", int(0.35 * n_frames)),
            event_indices.get("Impact", int(0.60 * n_frames)),
            event_indices.get("Finish", max(0, n_frames - 1)),
        )
    return int(0.35 * n_frames), int(0.60 * n_frames), max(0, n_frames - 1)


def _get_swing_phase(k: int, top_idx: int, impact_idx: int, n_frames: int) -> str:
    if k < top_idx:
        return "Address / Backswing"
    if k < impact_idx:
        return "Top / Early Downswing"
    if k <= impact_idx + max(1, int(0.15 * n_frames)):
        return "Impact Zone"
    return "Finish"


def _evaluate_single_frame_residual(
    k: int,
    t_k: float,
    phase_k: str,
    tm: NDArray[np.float64] | None,
    mm: NDArray[np.float64] | None,
    vm: NDArray[np.bool_] | None,
    names: tuple[str, ...],
) -> tuple[FrameResidual, NDArray[np.float64], MarkerResidual | None, float]:
    if tm is not None and mm is not None and k < len(tm) and k < len(mm):
        diff = mm[k] - tm[k]
        mask_k = (
            vm[k] if vm is not None and k < len(vm) else np.ones(len(diff), dtype=bool)
        )
        valid_diff = diff[mask_k]
        # Canonical receipt RMS (tour_metrics.compute_shared_metrics): pool the
        # per-marker 3D distances over the frame's valid markers, never the
        # flattened XYZ components. An empty valid set is unobserved: claim no
        # worst marker and keep the frame out of global-worst/mean statistics.
        if valid_diff.size == 0:
            frame_res = FrameResidual(k, t_k, phase_k, 0.0, None, 0)
            return frame_res, diff, None, 0.0
        per_marker_sq = np.sum(valid_diff**2, axis=-1)
        rms_k = float(np.sqrt(np.mean(per_marker_sq))) if per_marker_sq.size else 0.0
        sq_diff = np.einsum("ij,ij->i", diff, diff)
        sq_diff_masked = np.where(mask_k, sq_diff, -1.0)
        worst_m_idx = int(np.argmax(sq_diff_masked))
        worst_m_err = float(
            np.sqrt(sq_diff[worst_m_idx])
        )  # ⚡ Bolt: np.sqrt(np.einsum) avoids temporary allocations and is faster than np.linalg.norm(..., axis=1)
        worst_m_name = (
            names[worst_m_idx] if worst_m_idx < len(names) else f"Marker_{worst_m_idx}"
        )
        marker_res = MarkerResidual(worst_m_name, worst_m_idx, worst_m_err)
        frame_res = FrameResidual(k, t_k, phase_k, rms_k, marker_res, int(mask_k.sum()))
        return frame_res, diff, marker_res, rms_k

    diff = np.zeros((len(names), 3), dtype=np.float64)
    frame_res = FrameResidual(k, t_k, phase_k, 0.0, None, 0)
    return frame_res, diff, None, 0.0


def compute_residual_summary(
    replay: ReplayData,
    marker_names: Sequence[str] | None = None,
    event_indices: Mapping[str, int] | None = None,
) -> ResidualSummary:
    """Compute per-frame, per-marker, and per-phase residual errors across the swing."""
    n_frames = replay.frame_count
    tm = replay.target_markers_m
    mm = replay.model_markers_m
    vm = replay.valid_mask
    names = _resolve_marker_names(
        tm, mm, marker_names or getattr(replay, "marker_names", None)
    )
    top_idx, impact_idx, _ = _resolve_phase_boundaries(n_frames, event_indices)

    frame_residuals_list: list[FrameResidual] = []
    residual_vectors_list: list[NDArray[np.float64]] = []
    global_max_err = -1.0
    global_worst_frame = 0
    global_worst_marker_idx = 0
    global_worst_marker_name = names[0] if names else "None"
    global_worst_frame_rms = 0.0

    for k in range(n_frames):
        t_k = float(replay.time_s[k])
        phase_k = _get_swing_phase(k, top_idx, impact_idx, n_frames)
        f_res, diff, m_res, rms_k = _evaluate_single_frame_residual(
            k, t_k, phase_k, tm, mm, vm, names
        )
        frame_residuals_list.append(f_res)
        residual_vectors_list.append(diff)
        if m_res is not None and m_res.error_m > global_max_err:
            global_max_err = m_res.error_m
            global_worst_frame = k
            global_worst_marker_idx = m_res.marker_idx
            global_worst_marker_name = m_res.marker_name
            global_worst_frame_rms = rms_k

    worst_phase = _get_swing_phase(global_worst_frame, top_idx, impact_idx, n_frames)
    worst_time = (
        float(replay.time_s[global_worst_frame])
        if global_worst_frame < len(replay.time_s)
        else 0.0
    )

    return ResidualSummary(
        worst_frame_idx=global_worst_frame,
        worst_time_s=worst_time,
        worst_phase=worst_phase,
        worst_marker_name=global_worst_marker_name,
        worst_marker_idx=global_worst_marker_idx,
        max_marker_error_m=max(0.0, global_max_err),
        worst_frame_rms_m=global_worst_frame_rms,
        frame_residuals=tuple(frame_residuals_list),
        marker_names=names,
        residual_vectors=tuple(residual_vectors_list),
    )


def _render_still_geometry(ax: Any, vframe: ViewerFrame, engine_name: str) -> None:
    """Render skeleton lines, target dots, model dots, and residual vectors into 3D axes."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    eng_color = ENGINE_COLORS.get(engine_name.lower(), ENGINE_COLORS["default"])
    lines = [[seg.start_m, seg.end_m] for seg in vframe.segments]
    if lines:
        ax.add_collection3d(
            Line3DCollection(lines, colors=eng_color, linewidths=2.5, alpha=0.85)
        )

    valid = (
        vframe.valid_mask
        if vframe.valid_mask is not None
        else (
            np.ones(len(vframe.target_markers), dtype=bool)
            if vframe.target_markers is not None
            else np.array([], dtype=bool)
        )
    )
    if vframe.target_markers is not None:
        tm = vframe.target_markers[valid]
        if len(tm) > 0:
            ax.scatter(
                tm[:, 0],
                tm[:, 1],
                tm[:, 2],
                color="black",
                s=24,
                label="Observed Dots",
                alpha=0.9,
            )

    if vframe.model_markers is not None:
        mm = vframe.model_markers
        ax.scatter(
            mm[:, 0],
            mm[:, 1],
            mm[:, 2],
            color=eng_color,
            s=20,
            label=f"Model ({engine_name.capitalize()})",
            alpha=0.9,
        )
        if vframe.target_markers is not None:
            res_lines = [
                [m_pos, t_pos]
                for m_pos, t_pos, is_v in zip(
                    mm, vframe.target_markers, valid, strict=False
                )
                if is_v
            ]
            if res_lines:
                ax.add_collection3d(
                    Line3DCollection(
                        res_lines,
                        colors="#d9534f",
                        linewidths=1.2,
                        linestyles="--",
                        alpha=0.75,
                    )
                )


def _setup_3d_axes(ax: Any) -> None:
    """Setup standard 3D viewport limits and elevation/azimuth angles."""
    ax.clear()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(0.0, 2.0)
    ax.view_init(elev=20, azim=45)


def _capture_canvas_frame(canvas: Any) -> Any:
    """Draw canvas and convert rgba buffer to an RGB PIL Image."""
    from PIL import Image

    canvas.draw()
    buf = canvas.buffer_rgba()
    img = Image.frombuffer("RGBA", canvas.get_width_height(), buf, "raw", "RGBA", 0, 1)
    return img.convert("RGB")


def _save_gif_frames(frames: Sequence[Any], out_p: Path, fps: int) -> Path:
    """Write sequential RGB PIL images to an animated GIF."""
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


@dataclass(frozen=True)
class BoardExportMetadata:
    """Metadata displayed on board-ready stills and animations."""

    candidate_hash: str = "unknown"
    engine_name: str = "default"
    drive_mode: str = "torque_driven"
    verdict: str = "UNVERIFIED"
    evidence_link: str | None = None


def _resolve_export_metadata(
    metadata: BoardExportMetadata | None,
    kwargs: Mapping[str, Any],
) -> BoardExportMetadata:
    """Resolve explicit metadata or synthesize from legacy keyword arguments."""
    return metadata or BoardExportMetadata(
        candidate_hash=str(kwargs.get("candidate_hash", "unknown")),
        engine_name=str(kwargs.get("engine_name", "default")),
        drive_mode=str(kwargs.get("drive_mode", "torque_driven")),
        verdict=str(kwargs.get("verdict", "UNVERIFIED")),
        evidence_link=kwargs.get("evidence_link"),
    )


def export_board_ready_still(
    replay: ReplayData,
    spec: Mapping[str, Any],
    frame_idx: int,
    output_path: Path | str,
    *,
    metadata: BoardExportMetadata | None = None,
    **kwargs: Any,
) -> Path:
    """Export a board-ready still image with model vs observed dots and honest residual captions."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    meta = _resolve_export_metadata(metadata, kwargs)

    matplotlib.use("Agg")
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fig = Figure(figsize=(8, 7), dpi=150)
    canvas = FigureCanvasAgg(fig)
    ax: Any = fig.add_subplot(111, projection="3d")
    _setup_3d_axes(ax)

    vframe = viewer_frame(spec, replay, frame_idx)
    _render_still_geometry(ax, vframe, meta.engine_name)

    t_val = float(replay.time_s[frame_idx]) if frame_idx < len(replay.time_s) else 0.0
    rms_mm = vframe.rms_error * 1000.0
    caption = (
        f"Candidate: {meta.candidate_hash[:12]} | Engine: {meta.engine_name.capitalize()} | Drive: {meta.drive_mode.replace('_', ' ').title()}\n"
        f"Frame {frame_idx + 1}/{replay.frame_count} ({t_val:.3f} s) | RMS Error: {rms_mm:.2f} mm | Verdict: {meta.verdict}"
    )
    if meta.evidence_link:
        caption += f"\nEvidence Link: {meta.evidence_link}"
    fig.suptitle(caption, fontsize=10, fontweight="bold", y=0.96)
    ax.legend(loc="upper right", fontsize=8)

    canvas.draw()
    fig.savefig(out_p, bbox_inches="tight")
    return out_p


def export_board_ready_video(
    replay: ReplayData,
    spec: Mapping[str, Any],
    output_path: Path | str,
    *,
    metadata: BoardExportMetadata | None = None,
    fps: int = 20,
    dpi: int = 100,
    max_frames: int | None = None,
    **kwargs: Any,
) -> Path:
    """Export animated board-ready video/GIF with observed dots, residual vectors, and metadata."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    meta = _resolve_export_metadata(metadata, kwargs)

    matplotlib.use("Agg")
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fig = Figure(figsize=(7, 6), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax: Any = fig.add_subplot(111, projection="3d")

    total_frames = replay.frame_count
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    frames: list[Any] = []
    for k in range(total_frames):
        _setup_3d_axes(ax)

        vframe = viewer_frame(spec, replay, k)
        _render_still_geometry(ax, vframe, meta.engine_name)

        t_val = float(replay.time_s[k]) if k < len(replay.time_s) else 0.0
        rms_mm = vframe.rms_error * 1000.0
        caption = (
            f"Candidate: {meta.candidate_hash[:12]} | Engine: {meta.engine_name.capitalize()} | Drive: {meta.drive_mode.replace('_', ' ').title()}\n"
            f"Frame {k + 1}/{replay.frame_count} ({t_val:.3f} s) | RMS: {rms_mm:.2f} mm | Verdict: {meta.verdict}"
        )
        if meta.evidence_link:
            caption += f" | Link: {meta.evidence_link}"
        fig.suptitle(caption, fontsize=9, fontweight="bold", y=0.97)

        frames.append(_capture_canvas_frame(canvas))

    return _save_gif_frames(frames, out_p, fps)
