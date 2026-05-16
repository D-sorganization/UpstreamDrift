"""FSP engine integration — Phase 2 of the Functional Swing Plane epic.

Bridges simulation frame sequences to the Rust SVD primitives (Phase 1,
``upstream_physics.fsp``) and provides the full analysis pipeline:
  1. Detect the Mid-Downswing -> Mid-Follow-through analysis window.
  2. Extract clubhead and hand trajectories from the window.
  3. Fit a best-fit plane (FSP) and compute golf-specific metrics.

Graceful degradation: if the Rust extension is not built, a pure-NumPy
SVD fallback is used automatically so tests always pass without a Maturin
build.

Design by Contract:
  - ``detect_md_mf_window`` raises ``ValueError`` if fewer than 5 frames
    are supplied.
  - ``extract_clubhead_trajectory`` raises ``ValueError`` if
    ``end_idx <= start_idx``.
  - All public functions use type hints and avoid ``print()``.

Coordinate convention (inherited from Phase 1):
  Z-up -- ground plane is z = 0, vertical is +Z.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rust extension -- try to import; fall back to pure NumPy
# ---------------------------------------------------------------------------

try:
    from upstream_physics import fsp as _rust_fsp  # type: ignore[import]

    _RUST_AVAILABLE = True
    logger.debug("upstream_physics.fsp Rust extension loaded")
except ImportError:
    _rust_fsp = None  # type: ignore[assignment]
    _RUST_AVAILABLE = False
    logger.debug("upstream_physics.fsp not available -- using NumPy SVD fallback")


# ---------------------------------------------------------------------------
# Plane fallback type (mirrors Rust Plane attributes)
# ---------------------------------------------------------------------------


class _NumpyPlane:
    """Pure-Python plane with the same interface as the Rust ``Plane``."""

    __slots__ = ("normal", "centroid")

    def __init__(self, normal: np.ndarray, centroid: np.ndarray) -> None:
        self.normal: np.ndarray = normal
        self.centroid: np.ndarray = centroid

    def __repr__(self) -> str:
        return f"_NumpyPlane(normal={self.normal}, centroid={self.centroid})"


# ---------------------------------------------------------------------------
# Internal plane-fitting helpers
# ---------------------------------------------------------------------------

# Joint name candidates for the clubhead position, searched in order.
_CLUBHEAD_KEYS: tuple[str, ...] = ("clubhead", "club_head", "grip_end")

# Joint name candidates for the lead-hand position.
_HAND_KEYS: tuple[str, ...] = ("lead_hand", "hand", "wrist", "grip")


def _fit_plane(pts: np.ndarray) -> Any:
    """Fit a best-fit plane to *pts* (N, 3) using Rust SVD or NumPy fallback.

    Args:
        pts: Point cloud with shape (N, 3) and N >= 3.

    Returns:
        A plane object with ``.normal`` and ``.centroid`` attributes.

    Raises:
        ValueError: If fewer than 3 rows or if points are degenerate.
    """
    if len(pts) < 3:
        raise ValueError(f"At least 3 points required to fit a plane; got {len(pts)}")

    if _RUST_AVAILABLE and _rust_fsp is not None:
        points_list = pts.tolist()
        return _rust_fsp.calculate_fsp(points_list)

    # Pure-NumPy SVD fallback
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    norm = float(np.linalg.norm(normal))
    if norm > 0.0:
        normal = normal / norm
    return _NumpyPlane(normal=normal, centroid=centroid)


def _plane_normal_array(plane: Any) -> np.ndarray:
    """Extract the normal as a float64 numpy array from any plane type."""
    n = plane.normal
    if isinstance(n, (list, tuple)):
        return np.asarray(n, dtype=np.float64)
    return np.asarray(n, dtype=np.float64)


def _plane_centroid_array(plane: Any) -> np.ndarray:
    """Extract the centroid as a float64 numpy array from any plane type."""
    c = plane.centroid
    if isinstance(c, (list, tuple)):
        return np.asarray(c, dtype=np.float64)
    return np.asarray(c, dtype=np.float64)


def _slope_deg(plane: Any) -> float:
    """Angle between the plane normal and vertical (+Z), in [0, 90] degrees."""
    if _RUST_AVAILABLE and _rust_fsp is not None:
        return float(_rust_fsp.fsp_slope_deg(plane))
    normal = _plane_normal_array(plane)
    nz = float(np.clip(abs(normal[2]), -1.0, 1.0))
    return float(np.degrees(np.arccos(nz)))


def _direction_deg(plane: Any, target_line: np.ndarray) -> float:
    """Azimuth of the FSP relative to *target_line*, in (-180, 180] degrees."""
    if _RUST_AVAILABLE and _rust_fsp is not None:
        tl = (
            target_line.tolist()
            if hasattr(target_line, "tolist")
            else list(target_line)
        )
        return float(_rust_fsp.fsp_direction_deg(plane, tl))
    # NumPy fallback: project target_line onto the plane and measure azimuth.
    normal = _plane_normal_array(plane)
    dot = float(np.dot(target_line, normal))
    proj = target_line - dot * normal
    proj_mag = float(np.linalg.norm(proj))
    if proj_mag < 1e-12:
        return 0.0
    proj_unit = proj / proj_mag
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    ref_dot = float(np.dot(ref, normal))
    ref_proj = ref - ref_dot * normal
    ref_mag = float(np.linalg.norm(ref_proj))
    if ref_mag < 1e-12:
        return float(np.degrees(np.arctan2(float(proj[1]), float(proj[0]))))
    ref_unit = ref_proj / ref_mag
    cos_theta = float(np.clip(np.dot(ref_unit, proj_unit), -1.0, 1.0))
    cross = np.cross(ref_unit, proj_unit)
    sign = float(np.sign(np.dot(cross, normal)))
    return float(sign * np.degrees(np.arccos(cos_theta)))


def _point_to_plane_distance(point: np.ndarray, plane: Any) -> float:
    """Signed perpendicular distance from *point* to *plane*."""
    if _RUST_AVAILABLE and _rust_fsp is not None:
        pt = point.tolist() if hasattr(point, "tolist") else list(point)
        return float(_rust_fsp.point_to_fsp_distance(pt, plane))
    normal = _plane_normal_array(plane)
    centroid = _plane_centroid_array(plane)
    return float(np.dot(point - centroid, normal))


# ---------------------------------------------------------------------------
# Public data class
# ---------------------------------------------------------------------------


@dataclass
class FspResult:
    """Results of the Functional Swing Plane analysis.

    Attributes:
        plane: Best-fit plane object (Rust ``Plane`` or ``_NumpyPlane``).
            Has ``.normal`` and ``.centroid`` attributes.
        slope_deg: Angle between the plane normal and vertical, in [0, 90].
        direction_deg: Azimuth of the FSP relative to the target line,
            in (-180, 180].
        clubhead_deviations: Signed distances of each clubhead position
            from the FSP, shape (N,).
        hand_deviations: Signed distances of each lead-hand position
            from the FSP, shape (N,).
    """

    plane: Any
    slope_deg: float
    direction_deg: float
    clubhead_deviations: np.ndarray
    hand_deviations: np.ndarray


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_md_mf_window(sim_frames: list[Any]) -> tuple[int, int]:
    """Return (start_idx, end_idx) for Mid-Downswing to Mid-Follow-through.

    Two strategies are tried in order:

    1. Phase labels -- if any frame contains a ``"phase"`` key, the index
       of the first ``"mid_downswing"`` frame is used as *start* and the
       first ``"mid_follow_through"`` frame as *end*.
    2. Velocity heuristic -- find the frame with the maximum clubhead
       speed (used as the impact proxy), then use
       ``round(impact * 0.70)`` as *start* and
       ``min(impact + round((n - impact - 1) * 0.60), n - 1)`` as *end*.

    Design by Contract:
        Precondition: ``len(sim_frames) >= 5`` -- raises ``ValueError`` otherwise.

    Args:
        sim_frames: Ordered list of simulation frame dictionaries.

    Returns:
        ``(start_idx, end_idx)`` -- both are valid indices into *sim_frames*
        with ``start_idx < end_idx``.

    Raises:
        ValueError: If fewer than 5 frames are provided.
    """
    n = len(sim_frames)
    if n < 5:
        raise ValueError(f"detect_md_mf_window requires at least 5 frames; got {n}")

    # Strategy 1: phase labels
    md_idx, mf_idx = _find_phase_indices(sim_frames)
    if md_idx is not None and mf_idx is not None and md_idx < mf_idx:
        logger.debug(
            "detect_md_mf_window: using phase labels (%d -> %d)", md_idx, mf_idx
        )
        return int(md_idx), int(mf_idx)

    # Strategy 2: velocity heuristic
    impact_idx = _find_impact_by_velocity(sim_frames)
    start_idx = max(0, round(impact_idx * 0.70))
    end_idx = min(impact_idx + max(1, round((n - impact_idx - 1) * 0.60)), n - 1)

    # Ensure start < end
    if start_idx >= end_idx:
        start_idx = max(0, end_idx - 1)

    logger.debug(
        "detect_md_mf_window: velocity heuristic, impact=%d, window=%d->%d",
        impact_idx,
        start_idx,
        end_idx,
    )
    return int(start_idx), int(end_idx)


def extract_clubhead_trajectory(
    sim_frames: list[Any], start_idx: int, end_idx: int
) -> np.ndarray:
    """Extract clubhead 3-D positions from ``frames[start_idx:end_idx+1]``.

    Key resolution order for clubhead position:
    ``"clubhead"`` -> ``"club_head"`` -> ``"grip_end"`` ->
    joint with maximum velocity (dynamic fallback).

    Design by Contract:
        Precondition: ``end_idx > start_idx`` -- raises ``ValueError`` otherwise.

    Args:
        sim_frames: Full ordered list of simulation frame dictionaries.
        start_idx: Index of the first frame to include (inclusive).
        end_idx: Index of the last frame to include (inclusive).

    Returns:
        Float64 NumPy array of shape ``(N, 3)`` where
        ``N = end_idx - start_idx + 1``.

    Raises:
        ValueError: If ``end_idx <= start_idx``.
    """
    if end_idx <= start_idx:
        raise ValueError(
            f"end_idx must be greater than start_idx; "
            f"got start_idx={start_idx}, end_idx={end_idx}"
        )

    window = sim_frames[start_idx : end_idx + 1]
    positions: list[np.ndarray] = []

    for frame in window:
        pos = _extract_position_from_frame(frame, _CLUBHEAD_KEYS)
        positions.append(pos)

    return np.asarray(positions, dtype=np.float64)


def compute_swing_fsp(sim_frames: list[Any]) -> FspResult:
    """Full FSP pipeline: detect window -> extract trajectory -> fit plane.

    Gracefully falls back to pure-NumPy SVD if the ``upstream_physics``
    Rust extension is not built.

    Args:
        sim_frames: Ordered list of simulation frame dictionaries.

    Returns:
        :class:`FspResult` with plane, slope, direction, and per-point
        deviations for both clubhead and lead-hand trajectories.

    Raises:
        ValueError: Propagated from :func:`detect_md_mf_window` (< 5 frames)
            or if the trajectory is degenerate (< 3 non-collinear points).
    """
    start_idx, end_idx = detect_md_mf_window(sim_frames)
    clubhead_traj = extract_clubhead_trajectory(sim_frames, start_idx, end_idx)

    plane = _fit_plane(clubhead_traj)
    slope = _slope_deg(plane)

    # Target line heuristic: direction from first to last clubhead point
    target_line = _estimate_target_line(clubhead_traj)
    direction = _direction_deg(plane, target_line)

    # Signed deviations for clubhead
    clubhead_devs = np.array(
        [_point_to_plane_distance(pt, plane) for pt in clubhead_traj],
        dtype=np.float64,
    )

    # Lead-hand deviations over the same window
    hand_traj = _extract_hand_trajectory(sim_frames, start_idx, end_idx)
    hand_devs = np.array(
        [_point_to_plane_distance(pt, plane) for pt in hand_traj],
        dtype=np.float64,
    )

    logger.debug(
        "compute_swing_fsp: slope=%.2f direction=%.2f n_frames=%d rust=%s",
        slope,
        direction,
        len(clubhead_traj),
        _RUST_AVAILABLE,
    )

    return FspResult(
        plane=plane,
        slope_deg=float(slope),
        direction_deg=float(direction),
        clubhead_deviations=clubhead_devs,
        hand_deviations=hand_devs,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_phase_indices(
    sim_frames: list[Any],
) -> tuple[int | None, int | None]:
    """Return (mid_downswing_idx, mid_follow_through_idx) from phase labels.

    Returns (None, None) if no frame has a "phase" key or if either label
    is missing.
    """
    md_idx: int | None = None
    mf_idx: int | None = None

    for i, frame in enumerate(sim_frames):
        if not isinstance(frame, dict):
            continue
        phase = frame.get("phase")
        if phase == "mid_downswing" and md_idx is None:
            md_idx = i
        elif phase == "mid_follow_through" and mf_idx is None:
            mf_idx = i

    return md_idx, mf_idx


def _find_impact_by_velocity(sim_frames: list[Any]) -> int:
    """Return the index of the frame with maximum clubhead speed.

    Falls back to the midpoint if no velocity data is found.
    """
    best_idx = len(sim_frames) // 2
    best_speed = -1.0

    for i, frame in enumerate(sim_frames):
        if not isinstance(frame, dict):
            continue
        vel = frame.get("clubhead_velocity")
        if vel is None:
            continue
        speed = float(np.linalg.norm(np.asarray(vel, dtype=np.float64)))
        if speed > best_speed:
            best_speed = speed
            best_idx = i

    return best_idx


def _extract_position_from_frame(
    frame: Any, key_candidates: tuple[str, ...]
) -> np.ndarray:
    """Extract a 3-D position from *frame* by trying *key_candidates* in order.

    If none of the candidate keys are found, falls back to the first 3-D
    array value found, or zeros as a last resort.

    Args:
        frame: A simulation frame dictionary.
        key_candidates: Ordered sequence of key names to try.

    Returns:
        Float64 array of shape (3,).
    """
    if isinstance(frame, dict):
        for key in key_candidates:
            val = frame.get(key)
            if val is not None:
                return np.asarray(val, dtype=np.float64)

        # Dynamic fallback: first value that looks like a 3-D position
        for val in frame.values():
            try:
                arr = np.asarray(val, dtype=np.float64)
                if arr.shape == (3,):
                    return arr
            except (ValueError, TypeError):
                continue

    # Last resort: zeros
    logger.warning(
        "Could not find position key in frame; returning zeros. Keys tried: %s",
        key_candidates,
    )
    return np.zeros(3, dtype=np.float64)


def _extract_hand_trajectory(
    sim_frames: list[Any], start_idx: int, end_idx: int
) -> np.ndarray:
    """Extract lead-hand 3-D positions from the given window.

    Args:
        sim_frames: Full frame list.
        start_idx: First frame index (inclusive).
        end_idx: Last frame index (inclusive).

    Returns:
        Float64 array of shape (N, 3).
    """
    window = sim_frames[start_idx : end_idx + 1]
    positions = [_extract_position_from_frame(frame, _HAND_KEYS) for frame in window]
    return np.asarray(positions, dtype=np.float64)


def _estimate_target_line(trajectory: np.ndarray) -> np.ndarray:
    """Estimate the target-line direction from a clubhead trajectory.

    Uses the vector from the first to the last point as a proxy for the
    intended shot direction.  Falls back to ``[0, 1, 0]`` (pure-Y) if the
    trajectory has zero length.

    Args:
        trajectory: Float64 array of shape (N, 3), N >= 2.

    Returns:
        Unit vector of shape (3,).
    """
    if len(trajectory) < 2:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    direction = trajectory[-1] - trajectory[0]
    mag = float(np.linalg.norm(direction))
    if mag < 1e-12:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return direction / mag
