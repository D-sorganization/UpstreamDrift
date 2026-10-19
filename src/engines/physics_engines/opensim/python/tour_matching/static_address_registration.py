"""Static address-hold trunk marker re-registration for OpenSim (#13434).

Re-places the 14 trunk markers on pelvis and torso using a quiet segment at the
start of the capture (held address) under a declared static anatomical pose:
1. Lumbar extension, bending, and rotation are locked to 0 (neutral spine).
2. Pelvis orientation is set directly from the waist-marker cluster anatomical
   axes (x anterior, y superior, z right) and solved numerically in OpenSim.
3. Remaining coordinates (pelvis translations, legs, arms) are solved by IK
   using ONLY non-trunk markers.
4. The 14 trunk markers are re-placed as the mean observed position in each
   marker's parent body frame over the quiet segment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.opensim.python.tour_matching import read_trc
from src.engines.physics_engines.opensim.python.tour_matching.marker_set import (
    MarkerPlacement,
    attach_marker_set,
    parse_model,
    write_model,
)
from src.shared.python.contracts import ensure, require
from src.shared.python.motion_matching.tour_capture_contract import TourCapture

LOGGER = logging.getLogger("static_address_registration")

TRUNK_MARKERS: tuple[str, ...] = (
    "WaistLeft",
    "WaistRight",
    "WaistLBack",
    "WaistRBack",
    "BackTop",
    "BackLeft",
    "BackRight",
    "HeadTop",
    "HeadFront",
    "HeadSide",
    "LShoulderTop",
    "LShoulderBack",
    "RShoulderTop",
    "RShoulderBack",
)

SYMMETRY_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("pelvis_front", "WaistLeft", "WaistRight"),
    ("pelvis_back", "WaistLBack", "WaistRBack"),
    ("back", "BackLeft", "BackRight"),
    ("shoulders", "LShoulderBack", "RShoulderBack"),
)


@dataclass(frozen=True)
class StaticRegistrationResult:
    """Summary of static address trunk marker re-registration."""

    quiet_frames: tuple[int, ...]
    quiet_time_range_s: tuple[float, float]
    quiet_duration_s: float
    speed_threshold_mps: float
    pelvis_rotation_coords: dict[str, float]
    pelvis_fit_residual_rad: float
    non_trunk_marker_rms_m: float
    old_offsets: dict[str, tuple[str, tuple[float, float, float]]]
    new_offsets: dict[str, tuple[str, tuple[float, float, float]]]
    re_registered_markers: tuple[str, ...]
    unregistered_markers: tuple[str, ...]
    displacements_mm: dict[str, float]
    symmetry_before: dict[str, dict[str, float]]
    symmetry_after: dict[str, dict[str, float]]


# ---------------------------------------------------------------------------
# Law of Demeter (LoD) single-hop OpenSim helpers
# ---------------------------------------------------------------------------


def _get_body_transform(
    model: Any, state: Any, body_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Single-hop query for body transform (rotation matrix 3x3, origin 3) in ground."""
    body = model.getBodySet().get(body_name)
    xform = body.getTransformInGround(state)
    rot = xform.R().asMat33()
    r = np.array([[rot.get(i, j) for j in range(3)] for i in range(3)], dtype=float)
    p = np.array(xform.p().to_numpy(), dtype=float)
    return r, p


def _get_marker_parent_body(marker: Any) -> str:
    """Single-hop extraction of marker parent body name."""
    frame_name = str(marker.getParentFrameName())
    return frame_name.split("/")[-1]


def _extract_model_marker_offsets(
    model: Any,
) -> dict[str, tuple[str, tuple[float, float, float]]]:
    """Extract all marker placements (body, (x, y, z)) from an OpenSim Model."""
    markers = model.getMarkerSet()
    out = {}
    for i in range(markers.getSize()):
        m = markers.get(i)
        name = str(m.getName())
        body = _get_marker_parent_body(m)
        loc = tuple(float(v) for v in m.get_location().to_numpy())
        out[name] = (body, (loc[0], loc[1], loc[2]))
    return out


# ---------------------------------------------------------------------------
# Quiet Segment & Waist Cluster
# ---------------------------------------------------------------------------


def quiet_segment_frames(
    capture: TourCapture,
    trunk_markers: Sequence[str] = TRUNK_MARKERS,
    threshold_mps: float = 0.05,
    min_duration_s: float = 0.05,
) -> np.ndarray:
    """Find initial quiet frames where median trunk marker speed < threshold_mps.

    Requires at least min_duration_s of quiet segment from the first frame.
    """
    require(capture.frames > 1, "Capture must have at least 2 frames")
    require(threshold_mps > 0.0, "Speed threshold must be positive")
    require(min_duration_s > 0.0, "Minimum duration must be positive")

    indices = [capture.index(m) for m in trunk_markers if m in capture.labels]
    require(len(indices) > 0, "At least one trunk marker must exist in capture")

    dt = np.diff(capture.time_s)
    pts = capture.points_m[:, indices, :]
    diffs = np.diff(pts, axis=0)
    speeds = np.linalg.norm(diffs / dt[:, None, None], axis=-1)

    valid_mask = capture.valid[:-1, indices] & capture.valid[1:, indices]
    masked_speeds = np.where(valid_mask, speeds, np.nan)
    median_speeds = np.nanmedian(masked_speeds, axis=1)

    exceed = np.where(median_speeds >= threshold_mps)[0]
    last_quiet_idx = int(exceed[0]) if len(exceed) > 0 else len(median_speeds)

    quiet_count = last_quiet_idx + 1
    duration = float(capture.time_s[quiet_count - 1] - capture.time_s[0])
    require(
        duration >= min_duration_s,
        f"Quiet segment duration {duration:.4f} s < minimum {min_duration_s:.4f} s",
    )

    frames = np.arange(quiet_count, dtype=int)
    ensure(len(frames) >= 2, "Quiet segment must span at least 2 frames")
    return frames


def waist_cluster_frame(waist_points: Mapping[str, np.ndarray]) -> np.ndarray:
    """Compute OpenSim pelvis body orientation from waist cluster anatomical axes.

    Anterior: a = normalize(mean(WaistLeft, WaistRight) - mean(WaistLBack, WaistRBack))
    Left: l = normalize(mean(WaistLeft, WaistLBack) - mean(WaistRight, WaistRBack))
    OpenSim pelvis convention: x anterior (a), z right (-l), y superior (z cross x).
    """
    wl = np.asarray(waist_points["WaistLeft"], dtype=float)
    wr = np.asarray(waist_points["WaistRight"], dtype=float)
    wlb = np.asarray(waist_points["WaistLBack"], dtype=float)
    wrb = np.asarray(waist_points["WaistRBack"], dtype=float)

    front = 0.5 * (wl + wr)
    back = 0.5 * (wlb + wrb)
    a = front - back
    norm_a = float(np.linalg.norm(a))
    require(norm_a > 1e-6, "Anterior waist vector norm must be positive")
    a = a / norm_a

    l_mid = 0.5 * (wl + wlb)
    r_mid = 0.5 * (wr + wrb)
    l_raw = l_mid - r_mid
    l_orth = l_raw - float(np.dot(l_raw, a)) * a
    norm_l = float(np.linalg.norm(l_orth))
    require(norm_l > 1e-6, "Lateral waist vector norm must be positive")
    l_orth = l_orth / norm_l

    x = a
    z = -l_orth
    y = np.cross(z, x)
    norm_y = float(np.linalg.norm(y))
    require(norm_y > 1e-6, "Superior waist vector norm must be positive")
    y = y / norm_y

    R = np.column_stack([x, y, z])
    det = float(np.linalg.det(R))
    ensure(abs(det - 1.0) < 1e-6, "Waist cluster frame must be right-handed (det = +1)")
    ensure(
        bool(np.allclose(R.T @ R, np.eye(3), atol=1e-6)),
        "Waist cluster frame must be orthonormal",
    )
    return R


# ---------------------------------------------------------------------------
# Coordinate Solvers & Helpers
# ---------------------------------------------------------------------------


def solve_pelvis_rotations(
    model: Any, target_R: np.ndarray
) -> tuple[dict[str, float], float]:
    """Solve pelvis_tilt, pelvis_list, pelvis_rotation to match target_R in ground."""
    require(target_R.shape == (3, 3), "target_R must be 3x3")
    state = model.initSystem()
    coords = model.getCoordinateSet()
    c_tilt = coords.get("pelvis_tilt")
    c_list = coords.get("pelvis_list")
    c_rot = coords.get("pelvis_rotation")

    def residual(angles: np.ndarray) -> np.ndarray:
        c_tilt.setValue(state, float(angles[0]), False)
        c_list.setValue(state, float(angles[1]), False)
        c_rot.setValue(state, float(angles[2]), False)
        model.realizePosition(state)
        r_current, _ = _get_body_transform(model, state, "pelvis")
        return Rotation.from_matrix(r_current.T @ target_R).as_rotvec()

    seed = np.zeros(3)
    sol = least_squares(residual, seed, xtol=1e-15, ftol=1e-15, gtol=1e-15)
    res_norm = float(np.linalg.norm(sol.fun))
    require(res_norm < 1e-6, f"Pelvis rotation residual {res_norm:.2e} >= 1e-6 rad")

    out_coords = {
        "pelvis_tilt": float(sol.x[0]),
        "pelvis_list": float(sol.x[1]),
        "pelvis_rotation": float(sol.x[2]),
    }
    return out_coords, res_norm


def _solve_quiet_ik(
    model_path: Path,
    trc_path: Path,
    start_time: float,
    end_time: float,
    non_trunk_markers: Sequence[str],
    coord_names: Sequence[str],
    work_dir: Path,
) -> np.ndarray:
    """Run OpenSim InverseKinematicsTool for non-trunk markers on locked static model."""
    import opensim  # type: ignore[import-not-found]

    tool = opensim.InverseKinematicsTool()
    model = opensim.Model(str(model_path))
    model.initSystem()
    tool.setModel(model)
    tool.setMarkerDataFileName(str(trc_path))
    tool.setStartTime(start_time)
    tool.setEndTime(end_time)
    tool.setResultsDir(str(work_dir))
    out_mot = work_dir / "ik_quiet.mot"
    tool.setOutputMotionFileName(str(out_mot))

    # MarkersReference gives markers without a task its default weight (1.0),
    # so every other model marker needs an explicit task with apply=False.
    tasks = tool.getIKTaskSet()
    keep = set(non_trunk_markers)
    marker_set = model.getMarkerSet()
    for i in range(marker_set.getSize()):
        marker_name = str(marker_set.get(i).getName())
        task = opensim.IKMarkerTask()
        task.setName(marker_name)
        task.setWeight(1.0)
        task.setApply(marker_name in keep)
        tasks.cloneAndAppend(task)
    tool.run()

    table = opensim.TimeSeriesTable(str(out_mot))
    in_degrees = table.getTableMetaDataAsString("inDegrees") == "yes"
    q = np.column_stack(
        [np.array(table.getDependentColumn(n).to_numpy()) for n in coord_names]
    )
    if in_degrees:
        # Convert rotational coordinates to radians
        coords = model.getCoordinateSet()
        rot_mask = [
            coords.get(n).getMotionType() == opensim.Coordinate.Rotational
            for n in coord_names
        ]
        q[:, rot_mask] = np.radians(q[:, rot_mask])
    return q


def _compute_symmetry(
    offsets: Mapping[str, tuple[str, tuple[float, float, float]]],
) -> dict[str, dict[str, float]]:
    """Compute |z_L + z_R| and |x_L - x_R| asymmetry (mm) for bilateral pairs."""
    return {
        pair: {
            "asymmetry_z_mm": float(
                abs(offsets[left][1][2] + offsets[right][1][2]) * 1e3
            ),
            "asymmetry_x_mm": float(
                abs(offsets[left][1][0] - offsets[right][1][0]) * 1e3
            ),
        }
        for pair, left, right in SYMMETRY_PAIRS
        if left in offsets and right in offsets
    }


def _compute_quiet_poses_and_rms(
    locked_osim: Path,
    q_quiet: np.ndarray,
    coord_names: Sequence[str],
    non_trunk: Sequence[str],
    old_offsets: Mapping[str, tuple[str, tuple[float, float, float]]],
    raw_cap: TourCapture,
    quiet_frames: np.ndarray,
) -> tuple[list[dict[str, tuple[np.ndarray, np.ndarray]]], float]:
    """Evaluate body poses per quiet frame and return (poses, non_trunk_rms)."""
    import opensim  # type: ignore[import-not-found]

    eval_model = opensim.Model(str(locked_osim))
    eval_state = eval_model.initSystem()
    eval_coords = eval_model.getCoordinateSet()
    eval_bodies = sorted({old_offsets[m][0] for m in non_trunk} | {"pelvis", "torso"})
    quiet_poses = []
    for row in q_quiet:
        for i, name in enumerate(coord_names):
            c = eval_coords.get(name)
            if not c.getDefaultLocked():
                c.setValue(eval_state, float(row[i]), False)
        eval_model.realizePosition(eval_state)
        poses = {b: _get_body_transform(eval_model, eval_state, b) for b in eval_bodies}
        quiet_poses.append(poses)

    rms_sq_list = []
    for f_idx, q_idx in enumerate(quiet_frames):
        for m_name in non_trunk:
            col = raw_cap.index(m_name)
            if raw_cap.valid[q_idx, col]:
                body, offset = old_offsets[m_name]
                r_b, p_b = quiet_poses[f_idx][body]
                diff = r_b @ np.array(offset) + p_b - raw_cap.points_m[q_idx, col]
                rms_sq_list.append(np.sum(diff**2))
    non_trunk_rms = float(np.sqrt(np.mean(rms_sq_list))) if rms_sq_list else 0.0
    return quiet_poses, non_trunk_rms


def _recompute_trunk_offsets(
    old_offsets: Mapping[str, tuple[str, tuple[float, float, float]]],
    raw_cap: TourCapture,
    quiet_frames: np.ndarray,
    quiet_poses: list[dict[str, tuple[np.ndarray, np.ndarray]]],
) -> tuple[
    dict[str, tuple[str, tuple[float, float, float]]],
    list[str],
    list[str],
    dict[str, float],
]:
    """Recompute 14 trunk marker offsets from observations in parent bodies."""
    frame_to_idx = {int(q_f): i for i, q_f in enumerate(quiet_frames)}
    new_offsets = dict(old_offsets)
    re_registered = []
    unregistered = []
    displacements = {}

    for name in TRUNK_MARKERS:
        if name not in old_offsets:
            continue
        body, old_loc = old_offsets[name]
        if name not in raw_cap.labels:
            unregistered.append(name)
            displacements[name] = 0.0
            continue
        col = raw_cap.index(name)
        valid_frames = [f for f in quiet_frames if raw_cap.valid[f, col]]
        if not valid_frames:
            unregistered.append(name)
            displacements[name] = 0.0
            LOGGER.info("Marker %s has no quiet observations; keeping old offset", name)
            continue

        body_offsets = []
        for f in valid_frames:
            r_b, p_b = quiet_poses[frame_to_idx[f]][body]
            obs = raw_cap.points_m[f, col]
            body_offsets.append(r_b.T @ (obs - p_b))

        mean_loc = tuple(float(v) for v in np.mean(body_offsets, axis=0))
        new_offsets[name] = (body, mean_loc)
        re_registered.append(name)
        disp_mm = float(np.linalg.norm(np.array(mean_loc) - np.array(old_loc)) * 1e3)
        displacements[name] = disp_mm

    return new_offsets, re_registered, unregistered, displacements


# ---------------------------------------------------------------------------
# Public Registration Pipeline
# ---------------------------------------------------------------------------


def register_static_trunk_markers(
    model_path: Path | str,
    trc_path: Path | str,
    speed_threshold_mps: float = 0.05,
    min_duration_s: float = 0.05,
) -> StaticRegistrationResult:
    """Re-register the 14 trunk markers using the static address hold."""
    import opensim  # type: ignore[import-not-found]

    model_p = Path(model_path).resolve()
    trc_p = Path(trc_path).resolve()
    require(model_p.is_file(), f"Model file not found: {model_p}")
    require(trc_p.is_file(), f"TRC file not found: {trc_p}")

    model_orig = opensim.Model(str(model_p))
    old_offsets = _extract_model_marker_offsets(model_orig)
    raw_cap = read_trc(trc_p)

    quiet_frames = quiet_segment_frames(
        raw_cap,
        trunk_markers=TRUNK_MARKERS,
        threshold_mps=speed_threshold_mps,
        min_duration_s=min_duration_s,
    )
    t_start = float(raw_cap.time_s[quiet_frames[0]])
    t_end = float(raw_cap.time_s[quiet_frames[-1]])
    duration = t_end - t_start

    waist_means = {}
    for name in ("WaistLeft", "WaistRight", "WaistLBack", "WaistRBack"):
        idx = raw_cap.index(name)
        val = raw_cap.valid[quiet_frames, idx]
        require(bool(np.any(val)), f"Waist marker {name} lacks quiet observations")
        waist_means[name] = np.mean(raw_cap.points_m[quiet_frames[val], idx], axis=0)

    target_R_pelvis = waist_cluster_frame(waist_means)
    pelvis_rot_coords, pelvis_res = solve_pelvis_rotations(model_orig, target_R_pelvis)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        static_model = opensim.Model(str(model_p))
        c_set = static_model.getCoordinateSet()

        # Set & lock pelvis rotations and lumbar coordinates to 0
        for c_name, val in pelvis_rot_coords.items():
            coord = c_set.get(c_name)
            coord.setDefaultValue(val)
            coord.setDefaultLocked(True)
        for c_name in ("lumbar_extension", "lumbar_bending", "lumbar_rotation"):
            coord = c_set.get(c_name)
            coord.setDefaultValue(0.0)
            coord.setDefaultLocked(True)

        locked_osim = tmp_path / "locked_static.osim"
        static_model.printToXML(str(locked_osim))

        # Solve non-trunk coordinates
        coord_names = [str(c_set.get(i).getName()) for i in range(c_set.getSize())]
        non_trunk = [
            m for m in old_offsets if m not in TRUNK_MARKERS and m in raw_cap.labels
        ]
        q_quiet = _solve_quiet_ik(
            locked_osim,
            trc_p,
            t_start,
            t_end,
            non_trunk,
            coord_names,
            tmp_path,
        )
        require(
            q_quiet.shape[0] == len(quiet_frames),
            f"IK returned {q_quiet.shape[0]} rows for {len(quiet_frames)} quiet frames",
        )
        quiet_poses, non_trunk_rms = _compute_quiet_poses_and_rms(
            locked_osim,
            q_quiet,
            coord_names,
            non_trunk,
            old_offsets,
            raw_cap,
            quiet_frames,
        )

    new_offsets, re_registered, unregistered, displacements = _recompute_trunk_offsets(
        old_offsets, raw_cap, quiet_frames, quiet_poses
    )
    sym_before = _compute_symmetry(old_offsets)
    sym_after = _compute_symmetry(new_offsets)

    return StaticRegistrationResult(
        quiet_frames=tuple(int(f) for f in quiet_frames),
        quiet_time_range_s=(t_start, t_end),
        quiet_duration_s=duration,
        speed_threshold_mps=speed_threshold_mps,
        pelvis_rotation_coords=pelvis_rot_coords,
        pelvis_fit_residual_rad=pelvis_res,
        non_trunk_marker_rms_m=non_trunk_rms,
        old_offsets=old_offsets,
        new_offsets=new_offsets,
        re_registered_markers=tuple(re_registered),
        unregistered_markers=tuple(unregistered),
        displacements_mm=displacements,
        symmetry_before=sym_before,
        symmetry_after=sym_after,
    )


def write_registered_model(
    model_path: Path | str,
    new_offsets: Mapping[str, tuple[str, tuple[float, float, float]]],
    output_path: Path | str,
) -> Path:
    """Write an updated OpenSim .osim model with new_offsets."""
    model_p = Path(model_path).resolve()
    output_p = Path(output_path).resolve()
    tree = parse_model(model_p)
    placements = {
        name: MarkerPlacement(body, offset)
        for name, (body, offset) in new_offsets.items()
    }
    attach_marker_set(tree, placements)
    return write_model(tree, output_p)
