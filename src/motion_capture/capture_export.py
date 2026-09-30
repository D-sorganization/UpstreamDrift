"""Pure, data-free export and transformation functions for motion capture (Issue #11163).

This module provides reusable, reproducible, data-free functions for motion
capture processing and C3D export pipelines (such as ``capture-O`` and future
captures).

Design by Contract (DbC) is enforced on all functions using
:mod:`src.shared.python.contracts`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from src.shared.python.contracts import ensure, require

__all__ = [
    "detect_impact_frame",
    "fill_short_gaps",
    "relabel_markers",
    "subject_parameters",
    "to_capture_frame",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. fill_short_gaps
# ---------------------------------------------------------------------------


def _fill_column_linear(col: np.ndarray, max_gap: int) -> int:
    """Linear fill of internal NaN runs <= max_gap in a 1D column (in-place).

    Returns the count of samples filled.
    """
    n = col.shape[0]
    filled_count = 0
    i = 0
    while i < n:
        if not np.isnan(col[i]):
            i += 1
            continue
        start = i
        while i < n and np.isnan(col[i]):
            i += 1
        end = i
        gap_len = end - start
        # Gap must be bounded by valid samples on both sides:
        # start > 0 (not leading) and end < n (not trailing).
        if start > 0 and end < n and gap_len <= max_gap:
            y0 = col[start - 1]
            y1 = col[end]
            # Linearly interpolate between y0 and y1
            col[start:end] = np.linspace(y0, y1, gap_len + 2)[1:-1]
            filled_count += gap_len
    return filled_count


def fill_short_gaps(x: np.ndarray, max_gap: int) -> tuple[np.ndarray, int]:
    """Linear fill of NaN runs of length <= ``max_gap``.

    Only gaps that are bounded by valid samples on both sides are filled.
    Leading and trailing gaps, and gaps strictly longer than ``max_gap``,
    remain NaN. Operates per column for 1D ``(N,)`` and 2D ``(N, 3)`` arrays.
    The input array is not mutated.

    Preconditions:
        - ``x`` is a ``numpy.ndarray`` with 1D ``(N,)`` or 2D ``(N, 3)`` shape.
        - ``max_gap`` is an integer >= 1.

    Postconditions:
        - Returned array shape and dtype match input.
        - Existing valid samples in ``x`` are preserved.
        - Returned count of filled samples is >= 0.

    Args:
        x: Input array, shape ``(N,)`` or ``(N, 3)``.
        max_gap: Maximum consecutive NaN frames to linearly interpolate.

    Returns:
        A tuple of ``(filled_array, samples_filled)``.

    Raises:
        PreconditionError / TypeError: If ``x`` is not an ndarray or ``max_gap`` is invalid.
        PreconditionError / ValueError: If array shape is neither 1D nor ``(N, 3)``.
    """
    require(isinstance(x, np.ndarray), "x must be a numpy.ndarray", type(x).__name__)
    require(x.ndim in (1, 2), "x must be 1D (N,) or 2D (N, 3)", x.shape)
    if x.ndim == 2:
        require(x.shape[1] == 3, "2D array must have shape (N, 3)", x.shape)
    require(
        isinstance(max_gap, (int, np.integer)) and max_gap >= 1,
        "max_gap must be an integer >= 1",
        max_gap,
    )

    out = x.astype(np.float64, copy=True)
    if out.shape[0] == 0:
        return out, 0

    total_filled = 0
    if out.ndim == 1:
        total_filled = _fill_column_linear(out, int(max_gap))
    else:
        for c in range(out.shape[1]):
            total_filled += _fill_column_linear(out[:, c], int(max_gap))

    ensure(out.shape == x.shape, "Output shape must match input shape")
    ensure(total_filled >= 0, "Samples filled count must be non-negative")
    valid_mask = ~np.isnan(x)
    ensure(
        bool(np.all(out[valid_mask] == x[valid_mask])),
        "Pre-existing valid samples must be preserved",
    )
    return out, total_filled


# ---------------------------------------------------------------------------
# 2. relabel_markers
# ---------------------------------------------------------------------------


def _validate_relabel_inputs(
    points: np.ndarray, labels: Sequence[str], target_labels: Sequence[str]
) -> None:
    """Validate the argument types and shapes for :func:`relabel_markers`."""
    require(
        isinstance(points, np.ndarray),
        "points must be a numpy.ndarray",
        type(points).__name__,
    )
    require(points.ndim == 3, "points must be a 3D array", points.shape)
    require(
        isinstance(labels, (list, tuple)),
        "labels must be a sequence of strings",
    )
    require(
        isinstance(target_labels, (list, tuple)),
        "target_labels must be a sequence of strings",
    )
    require(
        points.shape[1] == len(labels),
        "points.shape[1] must match len(labels)",
        (points.shape[1], len(labels)),
    )


def _resolve_relabel_layout(points: np.ndarray, layout: str | None) -> bool:
    """Return whether ``points`` uses the coords-first ``(3, M, N)`` layout."""
    if layout is not None:
        canonical_layout = layout.strip().lower()
        require(
            canonical_layout
            in ("(3, m, n)", "coords_first", "(n, m, 3)", "frames_first"),
            "Invalid layout specified",
            layout,
        )
        return canonical_layout in ("(3, m, n)", "coords_first")
    return points.shape[0] == 3 and points.shape[2] != 3


def _require_targets_resolvable(
    target_labels: Sequence[str],
    labels: Sequence[str],
    label_map: dict[str, int],
    opt_set: set[str],
) -> None:
    """Raise if any non-optional target label is missing from the source labels."""
    for tgt in target_labels:
        if tgt not in label_map and tgt not in opt_set:
            require(
                False,
                f"Missing required target label: {tgt!r}. Source labels: {list(labels)!r}",
                tgt,
            )


def _build_relabeled_points(
    points: np.ndarray,
    target_labels: Sequence[str],
    label_map: dict[str, int],
    *,
    is_coords_first: bool,
) -> np.ndarray:
    """Assemble the reordered, NaN-padded output array for ``target_labels``."""
    m_target = len(target_labels)
    if is_coords_first:
        n_frames = points.shape[2]
        out = np.full((3, m_target, n_frames), np.nan, dtype=np.float64)
        for j, tgt in enumerate(target_labels):
            if tgt in label_map:
                out[:, j, :] = points[:, label_map[tgt], :]
    else:
        n_frames = points.shape[0]
        out = np.full((n_frames, m_target, 3), np.nan, dtype=np.float64)
        for j, tgt in enumerate(target_labels):
            if tgt in label_map:
                out[:, j, :] = points[:, label_map[tgt], :]
    return out


def relabel_markers(
    points: np.ndarray,
    labels: Sequence[str],
    target_labels: Sequence[str],
    optional_labels: Sequence[str] | set[str] | None = None,
    *,
    layout: str | None = None,
) -> np.ndarray:
    """Reorder points array to ``target_labels``, padding missing optional labels with NaN.

    Supports two 3D layouts:
    - ``(3, M, N)`` (coords-first, e.g. ezc3d): axis 0 is spatial coordinate (X, Y, Z),
      axis 1 is marker (len M), axis 2 is frame (len N).
    - ``(N, M, 3)`` (frames-first): axis 0 is frame (len N), axis 1 is marker (len M),
      axis 2 is spatial coordinate (X, Y, Z).

    Preconditions:
        - ``points`` is a 3D numpy array.
        - ``points.shape[1] == len(labels)``.
        - Every label in ``target_labels`` must be in ``labels`` or ``optional_labels``.

    Postconditions:
        - Output array shape is ``(3, len(target_labels), N)`` or ``(N, len(target_labels), 3)``
          matching the input layout.
        - For target labels in ``labels``, data matches the source marker.
        - For target labels missing from ``labels`` (in ``optional_labels``), slice is NaN.

    Args:
        points: Source marker positions array, shape ``(3, M, N)`` or ``(N, M, 3)``.
        labels: Source marker labels corresponding to axis 1.
        target_labels: Desired output marker labels in order.
        optional_labels: Labels allowed to be missing from ``labels``.
        layout: Explicit layout string, either ``"(3, M, N)"`` / ``"coords_first"``
            or ``"(N, M, 3)"`` / ``"frames_first"``. If None, inferred from shape.

    Returns:
        Reordered and padded points array.

    Raises:
        PreconditionError / TypeError: If types are invalid.
        PreconditionError / ValueError: If shape, dimension, or label counts mismatch,
            or if a required target label is missing.
    """
    _validate_relabel_inputs(points, labels, target_labels)

    is_coords_first = _resolve_relabel_layout(points, layout)

    opt_set = set(optional_labels) if optional_labels is not None else set()
    label_map = {lbl: i for i, lbl in enumerate(labels)}

    _require_targets_resolvable(target_labels, labels, label_map, opt_set)

    out = _build_relabeled_points(
        points, target_labels, label_map, is_coords_first=is_coords_first
    )

    ensure(out.shape[1] == len(target_labels), "Output marker count mismatch")
    return out


# ---------------------------------------------------------------------------
# 3. to_capture_frame
# ---------------------------------------------------------------------------


_AXIS_DIRECTIONS: dict[str, np.ndarray] = {
    "forward": np.array([1.0, 0.0, 0.0]),
    "front": np.array([1.0, 0.0, 0.0]),
    "north": np.array([1.0, 0.0, 0.0]),
    "backward": np.array([-1.0, 0.0, 0.0]),
    "back": np.array([-1.0, 0.0, 0.0]),
    "south": np.array([-1.0, 0.0, 0.0]),
    "left": np.array([0.0, 1.0, 0.0]),
    "west": np.array([0.0, 1.0, 0.0]),
    "right": np.array([0.0, -1.0, 0.0]),
    "east": np.array([0.0, -1.0, 0.0]),
    "up": np.array([0.0, 0.0, 1.0]),
    "down": np.array([0.0, 0.0, -1.0]),
}


def _parse_axis_convention(conv_str: str) -> np.ndarray:
    """Parse a convention string e.g. 'x-forward,y-left,z-up' into a 3x3 basis matrix.

    Columns of the returned matrix represent the physical basis vectors for
    the local (x, y, z) axes.
    """
    require(
        isinstance(conv_str, str) and len(conv_str.strip()) > 0,
        "Convention string must be non-empty",
        conv_str,
    )
    tokens = re.split(r"[,;]+", conv_str.strip())
    axis_map: dict[str, np.ndarray] = {}
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        m = re.match(r"^([xyzXYZ])[-:=_]([a-zA-Z]+)$", tok)
        require(
            m is not None,
            f"Invalid axis convention token: {tok!r} in {conv_str!r}",
            tok,
        )
        assert m is not None  # for mypy
        axis_name = m.group(1).lower()
        direction_name = m.group(2).lower()
        require(
            direction_name in _AXIS_DIRECTIONS,
            f"Unknown direction {direction_name!r} in {conv_str!r}",
            direction_name,
        )
        require(
            axis_name not in axis_map,
            f"Duplicate axis {axis_name!r} in {conv_str!r}",
            axis_name,
        )
        axis_map[axis_name] = _AXIS_DIRECTIONS[direction_name]

    require(
        set(axis_map.keys()) == {"x", "y", "z"},
        f"Convention must define all three axes x, y, z; got {list(axis_map.keys())!r}",
        conv_str,
    )
    m_mat = np.column_stack([axis_map["x"], axis_map["y"], axis_map["z"]])
    det_m = float(np.linalg.det(m_mat))
    require(
        abs(abs(det_m) - 1.0) < 1e-6,
        f"Axis directions must form an orthogonal triad; got det={det_m}",
        conv_str,
    )
    return m_mat


def to_capture_frame(
    points: np.ndarray,
    *,
    from_axes: str,
    to_axes: str,
    axis: int | None = None,
) -> np.ndarray:
    """Convert points between named axis conventions via a signed permutation matrix.

    Converts coordinates between named spatial conventions such as
    ``"x-forward,y-left,z-up"`` and ``"x-forward,y-up,z-right"``.
    Preserves handedness with an exact ``det(R) == +1`` postcondition.

    Preconditions:
        - ``points`` is a numpy ndarray containing a dimension of size 3.
        - ``from_axes`` and ``to_axes`` specify valid orthonormal axis triads.

    Postconditions:
        - Transformation rotation matrix satisfies ``det(R) == +1``.
        - Output shape matches input shape.

    Args:
        points: Coordinate array, e.g. ``(3,)``, ``(N, 3)``, ``(N, M, 3)``, or ``(3, M, N)``.
        from_axes: Source axis convention description (e.g. ``"x-forward,y-up,z-right"``).
        to_axes: Target axis convention description (e.g. ``"x-forward,y-left,z-up"``).
        axis: Dimension along which 3D coordinates lie. Defaults to -1 (last axis)
            unless points is 3D with shape[0] == 3 and shape[2] != 3 (coords-first).

    Returns:
        Transformed array with coordinates in target convention.

    Raises:
        PreconditionError / TypeError: If inputs are invalid.
        PreconditionError / ValueError: If axes are non-orthogonal or syntax is invalid.
        PostconditionError: If conversion would invert handedness (``det(R) != +1``).
    """
    require(
        isinstance(points, np.ndarray),
        "points must be a numpy.ndarray",
        type(points).__name__,
    )
    require(
        3 in points.shape,
        "points must contain at least one dimension of length 3",
        points.shape,
    )

    m_from = _parse_axis_convention(from_axes)
    m_to = _parse_axis_convention(to_axes)

    # R maps coordinates in frame_from to frame_to:
    # p_phys = M_from @ v_from = M_to @ v_to => v_to = (M_to^T @ M_from) @ v_from
    r_mat = m_to.T @ m_from

    det_r = float(np.linalg.det(r_mat))
    ensure(
        abs(det_r - 1.0) < 1e-6,
        f"Axis conversion must preserve handedness with det(R) = +1; got det(R) = {det_r:.4f}",
        det_r,
    )

    # Determine coordinate axis
    if axis is None:
        if points.ndim == 3 and points.shape[0] == 3 and points.shape[-1] != 3:
            target_axis = 0
        else:
            target_axis = -1
    else:
        target_axis = axis

    if target_axis in (-1, points.ndim - 1):
        require(points.shape[-1] == 3, "Last dimension must have size 3", points.shape)
        # v_to^T = v_from^T @ R^T
        out = points.astype(np.float64) @ r_mat.T
    elif target_axis == 0:
        require(points.shape[0] == 3, "First dimension must have size 3", points.shape)
        out = np.tensordot(r_mat, points.astype(np.float64), axes=([1], [0]))
    else:
        require(False, f"Unsupported transformation axis: {target_axis}", target_axis)

    ensure(out.shape == points.shape, "Output shape must match input shape")
    return out


# ---------------------------------------------------------------------------
# 4. detect_impact_frame
# ---------------------------------------------------------------------------


def _validate_impact_frame_inputs(club_head: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Validate :func:`detect_impact_frame` inputs and return the valid-frame mask."""
    require(
        isinstance(club_head, np.ndarray),
        "club_head must be a numpy.ndarray",
        type(club_head).__name__,
    )
    require(isinstance(t, np.ndarray), "t must be a numpy.ndarray", type(t).__name__)
    require(
        club_head.ndim == 2 and club_head.shape[1] == 3,
        "club_head must have shape (N, 3)",
        club_head.shape,
    )
    require(t.ndim == 1, "t must be 1D (N,)", t.shape)
    require(
        len(club_head) == len(t),
        "club_head and t must have the same length",
        (len(club_head), len(t)),
    )
    require(len(t) >= 2, "At least 2 frames are required", len(t))
    require(
        bool(np.all(np.diff(t) > 0)),
        "Time array t must be strictly increasing",
    )

    valid_mask = ~np.isnan(club_head).any(axis=1)
    require(
        bool(np.sum(valid_mask) >= 2),
        "At least 2 valid frames in club_head are required",
        int(np.sum(valid_mask)),
    )
    return valid_mask


def _compute_frame_speeds(
    club_head: np.ndarray, t: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    """Return per-frame club-head speeds via central/one-sided differences.

    Frames adjacent to a NaN gap never bridge across it; frames with no
    usable neighbour remain NaN.
    """
    n_frames = len(t)
    speeds = np.full(n_frames, np.nan, dtype=np.float64)
    for i in range(n_frames):
        if not valid_mask[i]:
            continue
        has_prev = (i > 0) and valid_mask[i - 1]
        has_next = (i < n_frames - 1) and valid_mask[i + 1]

        if has_prev and has_next:
            v = (club_head[i + 1] - club_head[i - 1]) / (t[i + 1] - t[i - 1])
        elif has_prev:
            v = (club_head[i] - club_head[i - 1]) / (t[i] - t[i - 1])
        elif has_next:
            v = (club_head[i + 1] - club_head[i]) / (t[i + 1] - t[i])
        else:
            continue
        speeds[i] = float(np.linalg.norm(v))
    return speeds


def _select_impact_frame_index(
    speeds: np.ndarray,
    n_frames: int,
    downswing_window: tuple[int, int] | None,
) -> int:
    """Return the index of maximum speed within an optional downswing window."""
    search_speeds = np.copy(speeds)
    if downswing_window is not None:
        start_w, end_w = downswing_window
        require(
            0 <= start_w < end_w <= n_frames,
            "Invalid downswing_window range",
            downswing_window,
        )
        search_speeds[:start_w] = np.nan
        search_speeds[end_w:] = np.nan

    require(
        bool(np.any(~np.isnan(search_speeds))),
        "No valid speed values could be evaluated in search range",
    )
    return int(np.nanargmax(search_speeds))


def _log_impact_frame_result(
    impact_idx: int,
    n_frames: int,
    valid_mask: np.ndarray,
    speeds: np.ndarray,
    t: np.ndarray,
) -> None:
    """Log a dropout warning or the detected impact frame, as appropriate."""
    pre_gap = (impact_idx > 0) and not valid_mask[impact_idx - 1]
    post_gap = (impact_idx < n_frames - 1) and not valid_mask[impact_idx + 1]
    if pre_gap or post_gap:
        logger.warning(
            "Dropout or NaN gap detected adjacent to impact frame %d (pre_gap=%s, post_gap=%s); "
            "returned nearest valid frame.",
            impact_idx,
            pre_gap,
            post_gap,
        )
    else:
        logger.info(
            "Detected impact at frame %d (t=%.4f s, speed=%.2f m/s).",
            impact_idx,
            t[impact_idx],
            speeds[impact_idx],
        )


def detect_impact_frame(
    club_head: np.ndarray,
    t: np.ndarray,
    *,
    downswing_window: tuple[int, int] | None = None,
) -> int:
    """Find the frame of maximum club-head speed within the downswing.

    Robust to NaN gaps: ignores gap frames, never picks an interpolated spike,
    and if a dropout occurs at the impact frame, reports it and returns the
    nearest valid frame.

    Preconditions:
        - ``club_head`` is a 2D ``numpy.ndarray`` with shape ``(N, 3)`` and N >= 2.
        - ``t`` is a 1D ``numpy.ndarray`` with shape ``(N,)`` strictly increasing.
        - At least 2 valid (non-NaN) frames in ``club_head``.

    Postconditions:
        - Returned index is within ``[0, N - 1]``.
        - Returned frame is valid (not NaN in ``club_head``).

    Args:
        club_head: Club-head positions in metres, shape ``(N, 3)``.
        t: Strictly increasing timestamps in seconds, shape ``(N,)``.
        downswing_window: Optional ``(start_frame, end_frame)`` frame range
            limiting the search to the downswing. Defaults to the entire swing.

    Returns:
        0-indexed integer frame of maximum club-head speed.

    Raises:
        PreconditionError / TypeError: If inputs are not ndarrays.
        PreconditionError / ValueError: If shapes mismatch, t is not monotonic,
            or insufficient valid frames exist.
    """
    valid_mask = _validate_impact_frame_inputs(club_head, t)
    n_frames = len(t)

    speeds = _compute_frame_speeds(club_head, t, valid_mask)
    impact_idx = _select_impact_frame_index(speeds, n_frames, downswing_window)
    _log_impact_frame_result(impact_idx, n_frames, valid_mask, speeds, t)

    ensure(0 <= impact_idx < n_frames, "impact_frame index out of bounds")
    ensure(
        valid_mask[impact_idx],
        "impact_frame must be a valid frame, not a gap frame",
    )
    ensure(not np.isnan(speeds[impact_idx]), "impact_frame must have valid speed")
    return impact_idx


# ---------------------------------------------------------------------------
# 5. subject_parameters
# ---------------------------------------------------------------------------


def subject_parameters(
    height_m: float,
    mass_kg: float,
    subject_id: str = "capture-O",
) -> dict[str, Any]:
    """Validate plausible human body parameters and return C3D SUBJECT group parameters.

    Preconditions:
        - ``height_m`` is in plausible range ``[1.2, 2.3]`` metres.
        - ``mass_kg`` is in plausible range ``[35.0, 200.0]`` kg.
        - Neutral subject id is a non-empty string.

    Postconditions:
        - Returns a dictionary containing ``HEIGHT_M``, ``MASS_KG``, and ``ID``.

    Args:
        height_m: Golfer height in metres. Must be in 1.2–2.3 m.
        mass_kg: Golfer body mass in kilograms. Must be in 35–200 kg.
        subject_id: Neutral capture identifier (default: ``"capture-O"``).

    Returns:
        Dictionary with C3D ``SUBJECT`` group parameters.

    Raises:
        PreconditionError / TypeError: If types are invalid.
        PreconditionError / ValueError: If height or mass are outside plausible ranges.
    """
    require(
        isinstance(height_m, (int, float)) and not isinstance(height_m, bool),
        "height_m must be a number",
        type(height_m).__name__,
    )
    require(
        isinstance(mass_kg, (int, float)) and not isinstance(mass_kg, bool),
        "mass_kg must be a number",
        type(mass_kg).__name__,
    )
    sid = subject_id
    require(
        isinstance(sid, str) and len(sid.strip()) > 0,
        "Subject ID must be a non-empty string",
        sid,
    )
    require(
        1.2 <= float(height_m) <= 2.3,
        "height_m must be within plausible human range [1.2, 2.3] m",
        height_m,
    )
    require(
        35.0 <= float(mass_kg) <= 200.0,
        "mass_kg must be within plausible human range [35.0, 200.0] kg",
        mass_kg,
    )

    params: dict[str, Any] = {
        "HEIGHT_M": float(height_m),
        "MASS_KG": float(mass_kg),
        "ID": str(sid),
    }
    ensure(
        "HEIGHT_M" in params and "MASS_KG" in params and "ID" in params,
        "Result dictionary must contain HEIGHT_M, MASS_KG, and ID",
    )
    return params
