"""
Gap-filling strategies for motion capture data.

Part of issue #4564. Handles marker occlusion and missing keypoints
using interpolation and reconstruction strategies.

The PCA reconstruction inner loop (SVD on the visible-row submatrix plus
per-row least-squares), and the LINEAR/CUBIC marker interpolation (issue
#8927), are routed through the Rust ``upstream-mocap-preproc`` kernel when
available; the pure-Python implementation in ``_gap_fill_pure_python.py`` is
preserved as a fallback for environments that do not ship the wheel.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from ..contracts import KeypointFrame, KeypointSequence, MarkerFrame, MarkerTrajectory

try:  # pragma: no cover - import guard
    import upstream_mocap_preproc as _rust_kernel  # type: ignore[import-not-found]

    _RUST_AVAILABLE = True
except ImportError:  # pragma: no cover
    _rust_kernel = None  # type: ignore[assignment]
    _RUST_AVAILABLE = False


class GapFillStrategy(str, Enum):
    """Gap-filling strategies."""

    LINEAR = "linear"  # Linear interpolation
    CUBIC = "cubic"  # Cubic spline interpolation
    PCA = "pca"  # PCA reconstruction for marker occlusion
    NEAREST = "nearest"  # Nearest neighbor


def gap_fill(
    data: KeypointSequence | MarkerTrajectory,
    strategy: GapFillStrategy = GapFillStrategy.LINEAR,
    max_gap: int = 10,
) -> KeypointSequence | MarkerTrajectory:
    """
    Fill gaps in motion capture data.

    Args:
        data: Input keypoint sequence or marker trajectory
        strategy: Gap-filling strategy to use
        max_gap: Maximum gap size to fill (frames)

    Returns:
        Data with gaps filled

    Raises:
        ValueError: If data type is unsupported
    """
    if isinstance(data, KeypointSequence):
        return _gap_fill_keypoints(data, strategy, max_gap)
    if isinstance(data, MarkerTrajectory):
        return _gap_fill_markers(data, strategy, max_gap)
    raise ValueError(f"Unsupported data type: {type(data)}")


def _gap_fill_keypoints(
    seq: KeypointSequence,
    strategy: GapFillStrategy,
    max_gap: int,
) -> KeypointSequence:
    """Fill gaps in keypoint sequence."""
    if len(seq.frames) < 2:
        return seq

    # Find gaps (frames with low confidence keypoints)
    gap_indices = _find_gaps_keypoints(seq.frames)

    if not gap_indices:
        return seq

    # Fill gaps
    filled_frames = _fill_gaps_keypoints(seq.frames, gap_indices, strategy, max_gap)

    return KeypointSequence(
        id=seq.id,
        frames=filled_frames,
        calibration=seq.calibration,
        metadata={**seq.metadata, "gap_filled": True, "strategy": strategy.value},
    )


def _gap_fill_markers(
    traj: MarkerTrajectory,
    strategy: GapFillStrategy,
    max_gap: int,
) -> MarkerTrajectory:
    """Fill gaps in marker trajectory."""
    if len(traj.frames) < 2:
        return traj

    # Find gaps (occluded markers)
    gap_indices = _find_gaps_markers(traj.frames)

    if not gap_indices:
        return traj

    # PCA operates on the full trajectory (low-rank reconstruction needs
    # the full visible-row submatrix as basis), with linear-interpolation
    # fallback for any frames it cannot reconstruct.
    if strategy == GapFillStrategy.PCA:
        filled_frames = _pca_reconstruct_markers(traj.frames, gap_indices, max_gap)
    elif strategy in (GapFillStrategy.LINEAR, GapFillStrategy.CUBIC):
        # LINEAR is the pipeline default (pipeline.py GapFillStep), so this
        # is the hottest path — dispatch to Rust when available (issue #8927).
        filled_frames = _interp_reconstruct_markers(
            traj.frames, gap_indices, strategy, max_gap
        )
    else:
        filled_frames = _fill_gaps_markers(traj.frames, gap_indices, strategy, max_gap)

    return MarkerTrajectory(
        id=traj.id,
        frames=filled_frames,
        calibration=traj.calibration,
        subject_id=traj.subject_id,
        metadata={**traj.metadata, "gap_filled": True, "strategy": strategy.value},
    )


def _find_gaps_keypoints(frames: list[KeypointFrame]) -> list[tuple[int, int]]:
    """Find gap indices in keypoint frames."""
    gaps = []
    gap_start = None

    for i, frame in enumerate(frames):
        # Check if any keypoint has low confidence
        has_low_conf = any(kp.confidence < 0.5 for kp in frame.keypoints)

        if has_low_conf and gap_start is None:
            gap_start = i
        elif not has_low_conf and gap_start is not None:
            gaps.append((gap_start, i - 1))
            gap_start = None

    if gap_start is not None:
        gaps.append((gap_start, len(frames) - 1))

    return gaps


def _find_gaps_markers(frames: list[MarkerFrame]) -> list[tuple[int, int]]:
    """Find gap indices (contiguous occluded-frame runs) in marker frames.

    Vectorized (issue #8927): builds an ``(n_frames, n_markers)`` occlusion
    matrix and reduces it with ``ndarray.any(axis=1)``, then finds run
    boundaries via a numpy diff instead of the previous per-frame Python
    ``any()`` call plus a hand-rolled ``gap_start``/``gap_end`` state
    machine.
    """
    n_frames = len(frames)
    if n_frames == 0:
        return []

    marker_names = list(frames[0].markers.keys())
    n_markers = max(len(marker_names), 1)
    reference_keys = frames[0].markers.keys()
    occ_matrix = np.zeros((n_frames, n_markers), dtype=bool)

    for i, frame in enumerate(frames):
        markers = frame.markers
        if markers.keys() == reference_keys:
            # Fast path: uniform marker schema, fully vectorizable reduction.
            for j, name in enumerate(marker_names):
                occ_matrix[i, j] = markers[name].occluded
        else:
            # Rare non-uniform schema: fall back to a per-frame scan, but
            # still fold the result into the same matrix/any() reduction.
            occ_matrix[i, 0] = any(m.occluded for m in markers.values())

    has_occluded = np.asarray(occ_matrix.any(axis=1))
    return _runs_from_bool_mask(has_occluded)


def _runs_from_bool_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive ``(start, end)`` index pairs for contiguous True-runs."""
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _fill_gaps_keypoints(
    frames: list[KeypointFrame],
    gaps: list[tuple[int, int]],
    strategy: GapFillStrategy,
    max_gap: int,
) -> list[KeypointFrame]:
    """Fill gaps in keypoint frames."""
    filled = list(frames)

    for start, end in gaps:
        gap_size = end - start + 1
        if gap_size > max_gap:
            continue  # Skip gaps that are too large

        if strategy == GapFillStrategy.LINEAR:
            filled = _linear_interp_keypoints(filled, start, end)
        elif strategy == GapFillStrategy.CUBIC:
            filled = _cubic_interp_keypoints(filled, start, end)
        elif strategy == GapFillStrategy.NEAREST:
            filled = _nearest_interp_keypoints(filled, start, end)
        elif strategy == GapFillStrategy.PCA:
            # Keypoint PCA falls back to linear (PCA is implemented for
            # marker trajectories where the rigid-body subspace is dense).
            filled = _linear_interp_keypoints(filled, start, end)

    return filled


def _fill_gaps_markers(
    frames: list[MarkerFrame],
    gaps: list[tuple[int, int]],
    strategy: GapFillStrategy,
    max_gap: int,
) -> list[MarkerFrame]:
    """Fill gaps in marker frames."""
    filled = list(frames)

    for start, end in gaps:
        gap_size = end - start + 1
        if gap_size > max_gap:
            continue  # Skip gaps that are too large

        if strategy == GapFillStrategy.LINEAR:
            filled = _linear_interp_markers(filled, start, end)
        elif strategy == GapFillStrategy.CUBIC:
            filled = _cubic_interp_markers(filled, start, end)
        elif strategy == GapFillStrategy.NEAREST:
            filled = _nearest_interp_markers(filled, start, end)

    return filled


def _linear_interp_keypoints(
    frames: list[KeypointFrame],
    start: int,
    end: int,
) -> list[KeypointFrame]:
    """Linear interpolation for keypoints."""
    if start == 0 or end >= len(frames):
        return frames

    before = frames[start - 1]
    after = frames[end + 1] if end + 1 < len(frames) else None

    for i in range(start, min(end + 1, len(frames))):
        frame = frames[i]
        new_keypoints = []

        for j, kp in enumerate(frame.keypoints):
            if kp.confidence < 0.5:
                # Interpolate from before/after
                if after and j < len(before.keypoints) and j < len(after.keypoints):
                    t = (i - start + 1) / (end - start + 2)
                    kp_before = before.keypoints[j]
                    kp_after = after.keypoints[j]

                    z_val: float | None
                    if kp_before.z is not None and kp_after.z is not None:
                        z_val = kp_before.z + t * (kp_after.z - kp_before.z)
                    else:
                        z_val = None
                    new_kp = Keypoint(
                        x=kp_before.x + t * (kp_after.x - kp_before.x),
                        y=kp_before.y + t * (kp_after.y - kp_before.y),
                        z=z_val,
                        confidence=0.5,  # Mark as interpolated
                        name=kp.name,
                    )
                    new_keypoints.append(new_kp)
                else:
                    new_keypoints.append(kp)
            else:
                new_keypoints.append(kp)

        frames[i] = KeypointFrame(
            timestamp=frame.timestamp,
            keypoints=new_keypoints,
            schema_name=frame.schema_name,
            frame_index=frame.frame_index,
        )

    return frames


def _linear_interp_markers(
    frames: list[MarkerFrame],
    start: int,
    end: int,
) -> list[MarkerFrame]:
    """Linear interpolation for markers."""
    if start == 0 or end >= len(frames):
        return frames

    before = frames[start - 1]
    after = frames[end + 1] if end + 1 < len(frames) else None

    for i in range(start, min(end + 1, len(frames))):
        frame = frames[i]
        new_markers = dict(frame.markers)

        for name, marker in frame.markers.items():
            if (
                marker.occluded
                and after
                and name in before.markers
                and name in after.markers
            ):
                t = (i - start + 1) / (end - start + 2)
                m_before = before.markers[name]
                m_after = after.markers[name]

                new_markers[name] = Marker(
                    name=name,
                    x=m_before.x + t * (m_after.x - m_before.x),
                    y=m_before.y + t * (m_after.y - m_before.y),
                    z=m_before.z + t * (m_after.z - m_before.z),
                    residual=None,
                    occluded=False,
                )

        frames[i] = MarkerFrame(
            timestamp=frame.timestamp,
            markers=new_markers,
            frame_index=frame.frame_index,
        )

    return frames


def _cubic_interp_keypoints(
    frames: list[KeypointFrame],
    start: int,
    end: int,
) -> list[KeypointFrame]:
    """Cubic spline interpolation for keypoints (placeholder)."""
    # For now, fall back to linear
    return _linear_interp_keypoints(frames, start, end)


def _cubic_interp_markers(
    frames: list[MarkerFrame],
    start: int,
    end: int,
) -> list[MarkerFrame]:
    """Cubic spline interpolation for markers (placeholder)."""
    # For now, fall back to linear
    return _linear_interp_markers(frames, start, end)


def _nearest_interp_keypoints(
    frames: list[KeypointFrame],
    start: int,
    end: int,
) -> list[KeypointFrame]:
    """Nearest neighbor interpolation for keypoints."""
    if start == 0:
        return frames

    before = frames[start - 1]

    for i in range(start, min(end + 1, len(frames))):
        frame = frames[i]
        new_keypoints = []

        for j, kp in enumerate(frame.keypoints):
            if kp.confidence < 0.5 and j < len(before.keypoints):
                kp_before = before.keypoints[j]
                new_kp = Keypoint(
                    x=kp_before.x,
                    y=kp_before.y,
                    z=kp_before.z,
                    confidence=0.5,
                    name=kp.name,
                )
                new_keypoints.append(new_kp)
            else:
                new_keypoints.append(kp)

        frames[i] = KeypointFrame(
            timestamp=frame.timestamp,
            keypoints=new_keypoints,
            schema_name=frame.schema_name,
            frame_index=frame.frame_index,
        )

    return frames


def _nearest_interp_markers(
    frames: list[MarkerFrame],
    start: int,
    end: int,
) -> list[MarkerFrame]:
    """Nearest neighbor interpolation for markers."""
    if start == 0:
        return frames

    before = frames[start - 1]

    for i in range(start, min(end + 1, len(frames))):
        frame = frames[i]
        new_markers = dict(frame.markers)

        for name, marker in frame.markers.items():
            if marker.occluded and name in before.markers:
                m_before = before.markers[name]
                new_markers[name] = Marker(
                    name=name,
                    x=m_before.x,
                    y=m_before.y,
                    z=m_before.z,
                    residual=None,
                    occluded=False,
                )

        frames[i] = MarkerFrame(
            timestamp=frame.timestamp,
            markers=new_markers,
            frame_index=frame.frame_index,
        )

    return frames


# Import Marker and Keypoint for type hints / construction
from ..contracts import Keypoint, Marker


def _interp_reconstruct_markers(
    frames: list[MarkerFrame],
    gap_indices: list[tuple[int, int]],
    strategy: GapFillStrategy,
    max_gap: int,
) -> list[MarkerFrame]:
    """Fill LINEAR/CUBIC marker gaps.

    Dispatches to the Rust ``upstream_mocap_preproc.linear_gap_fill`` /
    ``cubic_gap_fill`` kernels when available (mirrors the
    ``_pca_reconstruct_markers`` dispatch pattern below), otherwise falls
    back to the pure-Python per-gap interpolation in ``_fill_gaps_markers``.
    """
    if _RUST_AVAILABLE and frames:
        return _interp_reconstruct_markers_rust(frames, gap_indices, strategy, max_gap)
    return _fill_gaps_markers(list(frames), gap_indices, strategy, max_gap)


def _interp_reconstruct_markers_rust(
    frames: list[MarkerFrame],
    gap_indices: list[tuple[int, int]],
    strategy: GapFillStrategy,
    max_gap: int,
) -> list[MarkerFrame]:
    """Rust-backed linear/cubic interpolation.

    Stacks the trajectory into an ``(n_frames, n_markers, 3)`` array plus an
    ``(n_frames, n_markers)`` occlusion mask via ``_stack_marker_matrix``,
    calls the matching Rust kernel, then unstacks the result back into
    ``MarkerFrame`` objects via ``_unstack_marker_matrix``.

    The Rust kernel enforces ``max_gap`` per marker column independently
    (each point's own contiguous occluded run). The pre-Rust Python
    ``_fill_gaps_markers``/``_find_gaps_markers`` path is more conservative:
    it unions occlusion across *all* markers into shared gap windows first
    (``gap_indices``) and skips a window entirely if that combined window
    exceeds ``max_gap``, even for a marker whose own run within it is short.
    To keep the Rust dispatch behaviorally identical to that existing
    fallback (issue #8927 is a performance change, not a semantics change),
    frames belonging to an oversized combined window are excluded from the
    unstack step via ``_large_gap_frames``, the same helper
    ``_pca_reconstruct_markers`` already uses for this purpose.
    """
    marker_names = list(frames[0].markers.keys())
    n_frames = len(frames)
    n_markers = len(marker_names)

    matrix, occ_mask = _stack_marker_matrix(frames, marker_names)
    data = matrix.reshape(n_frames, n_markers, 3)
    mask = occ_mask.reshape(n_frames, n_markers, 3).any(axis=2)

    kernel = (
        _rust_kernel.linear_gap_fill  # type: ignore[union-attr]
        if strategy == GapFillStrategy.LINEAR
        else _rust_kernel.cubic_gap_fill  # type: ignore[union-attr]
    )
    filled, filled_mask = kernel(data, mask, int(max_gap))
    filled_matrix = np.asarray(filled).reshape(n_frames, n_markers * 3)
    filled_mask = np.asarray(filled_mask).copy()

    skip_frames = _large_gap_frames(gap_indices, max_gap)
    if skip_frames:
        skip_idx = np.fromiter(skip_frames, dtype=np.intp, count=len(skip_frames))
        filled_mask[skip_idx, :] = True  # treat as "kernel could not fill"

    return _unstack_marker_matrix(frames, marker_names, filled_matrix, filled_mask)


def _unstack_marker_matrix(
    frames: list[MarkerFrame],
    marker_names: list[str],
    filled_matrix: np.ndarray,
    filled_mask: np.ndarray,
) -> list[MarkerFrame]:
    """Inverse of ``_stack_marker_matrix``: rebuild frames from a filled matrix.

    ``filled_mask`` is the kernel's post-fill occlusion mask (``(n_frames,
    n_markers)``): still ``True`` where the kernel could not fill an entry
    (e.g. the gap exceeded ``max_gap``, or there was no anchor on one side).
    Only markers that were occluded going in *and* got filled
    (``not filled_mask[i, j]``) are replaced; everything else keeps its
    original value/occlusion state.
    """
    filled_frames = list(frames)
    for i, frame in enumerate(frames):
        new_markers: dict[str, Marker] | None = None
        for j, name in enumerate(marker_names):
            marker = frame.markers.get(name)
            if marker is None or not marker.occluded or filled_mask[i, j]:
                continue
            if new_markers is None:
                new_markers = dict(frame.markers)
            base = 3 * j
            new_markers[name] = Marker(
                name=name,
                x=float(filled_matrix[i, base]),
                y=float(filled_matrix[i, base + 1]),
                z=float(filled_matrix[i, base + 2]),
                residual=None,
                occluded=False,
            )
        if new_markers is not None:
            filled_frames[i] = MarkerFrame(
                timestamp=frame.timestamp,
                markers=new_markers,
                frame_index=frame.frame_index,
            )
    return filled_frames


def _pca_reconstruct_markers(
    frames: list[MarkerFrame],
    gap_indices: list[tuple[int, int]],
    max_gap: int,
    rank: int | None = None,
) -> list[MarkerFrame]:
    """Reconstruct occluded markers via low-rank SVD projection.

    Dispatches to the Rust ``upstream_mocap_preproc.pca_gap_fill`` kernel
    when available, otherwise falls back to the pure-Python implementation.
    """
    if _RUST_AVAILABLE and frames:
        return _pca_reconstruct_markers_rust(frames, gap_indices, max_gap, rank)
    return _pca_reconstruct_markers_python(frames, gap_indices, max_gap, rank)


def _pca_reconstruct_markers_rust(
    frames: list[MarkerFrame],
    gap_indices: list[tuple[int, int]],
    max_gap: int,
    rank: int | None,
) -> list[MarkerFrame]:
    """Rust-backed PCA reconstruction. Stacks frames, calls kernel, rebuilds frames."""
    n_frames = len(frames)
    marker_names = list(frames[0].markers.keys())
    n_markers = len(marker_names)

    data = np.zeros((n_frames, n_markers, 3), dtype=np.float64)
    mask = np.zeros((n_frames, n_markers), dtype=bool)
    for i, frame in enumerate(frames):
        for j, name in enumerate(marker_names):
            m = frame.markers.get(name)
            if m is None:
                mask[i, j] = True
                continue
            data[i, j, 0] = m.x
            data[i, j, 1] = m.y
            data[i, j, 2] = m.z
            if m.occluded:
                mask[i, j] = True

    filled, filled_mask, pca_success = _rust_kernel.pca_gap_fill(  # type: ignore[union-attr]
        data, mask, int(max_gap), rank
    )
    filled = np.asarray(filled)
    filled_mask = np.asarray(filled_mask)
    pca_success = np.asarray(pca_success)

    # Reassemble frames with the filled entries.
    filled_frames = list(frames)
    for i, frame in enumerate(frames):
        if not pca_success[i]:
            continue
        new_markers = dict(frame.markers)
        for j, name in enumerate(marker_names):
            if mask[i, j] and not filled_mask[i, j]:
                new_markers[name] = Marker(
                    name=name,
                    x=float(filled[i, j, 0]),
                    y=float(filled[i, j, 1]),
                    z=float(filled[i, j, 2]),
                    residual=None,
                    occluded=False,
                )
        filled_frames[i] = MarkerFrame(
            timestamp=frame.timestamp,
            markers=new_markers,
            frame_index=frame.frame_index,
        )

    # Linear fallback for frames PCA could not handle.
    remaining_gaps = _find_gaps_markers(filled_frames)
    if remaining_gaps:
        filled_frames = _fill_gaps_markers(
            filled_frames, remaining_gaps, GapFillStrategy.LINEAR, max_gap
        )
    return filled_frames


def _pca_reconstruct_markers_python(
    frames: list[MarkerFrame],
    gap_indices: list[tuple[int, int]],
    max_gap: int,
    rank: int | None = None,
) -> list[MarkerFrame]:
    """Pure-Python PCA reconstruction (preserves original numpy-only path).

    Algorithm:
    1. Stack the trajectory into matrix M of shape (n_frames, n_markers * 3).
    2. Identify rows with zero occlusions ("visible rows") to form the basis.
    3. Compute SVD of the visible-rows submatrix and truncate to rank k
       (default: min(6, rank(M_visible))).
    4. For each occluded frame, solve for the basis coefficients using only
       the visible coordinates, then back-fill the occluded entries from the
       projection.
    5. Frames that cannot be reconstructed (e.g. all markers occluded, or
       gap exceeds max_gap, or rank-deficient basis) fall back to linear
       interpolation per the existing strategy.

    Postcondition: every frame whose gap is bounded by max_gap and whose
    visible markers can be projected onto the basis has its previously
    occluded markers filled with finite values and ``occluded=False``.
    """
    n_frames = len(frames)
    if n_frames < 2:
        return list(frames)

    marker_names = list(frames[0].markers.keys())
    matrix, occ_mask = _stack_marker_matrix(frames, marker_names)
    basis = _build_marker_pca_basis(matrix, occ_mask, rank)
    if basis is None:
        return _fill_gaps_markers(
            list(frames), gap_indices, GapFillStrategy.LINEAR, max_gap
        )

    mean, basis_vectors, basis_rank = basis
    skip_frames = _large_gap_frames(gap_indices, max_gap)
    filled_frames, pca_failed_frames = _apply_marker_pca_projection(
        frames=frames,
        marker_names=marker_names,
        matrix=matrix,
        occ_mask=occ_mask,
        mean=mean,
        basis_vectors=basis_vectors,
        basis_rank=basis_rank,
        skip_frames=skip_frames,
    )

    if pca_failed_frames:
        remaining_gaps = _find_gaps_markers(filled_frames)
        if remaining_gaps:
            filled_frames = _fill_gaps_markers(
                filled_frames, remaining_gaps, GapFillStrategy.LINEAR, max_gap
            )

    return filled_frames


def _stack_marker_matrix(
    frames: list[MarkerFrame],
    marker_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return marker coordinate matrix and matching occlusion mask."""
    n_dims = len(marker_names) * 3
    matrix = np.zeros((len(frames), n_dims), dtype=float)
    occ_mask = np.zeros((len(frames), n_dims), dtype=bool)
    for i, frame in enumerate(frames):
        for j, name in enumerate(marker_names):
            marker = frame.markers.get(name)
            base = 3 * j
            if marker is None:
                occ_mask[i, base : base + 3] = True
                continue
            matrix[i, base] = marker.x
            matrix[i, base + 1] = marker.y
            matrix[i, base + 2] = marker.z
            if marker.occluded:
                occ_mask[i, base : base + 3] = True
    return matrix, occ_mask


def _build_marker_pca_basis(
    matrix: np.ndarray,
    occ_mask: np.ndarray,
    rank: int | None,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Build a PCA basis from fully visible marker rows."""
    fully_visible_rows = ~occ_mask.any(axis=1)
    n_visible = int(fully_visible_rows.sum())
    if n_visible < 2:
        return None

    visible_matrix = matrix[fully_visible_rows]
    mean = visible_matrix.mean(axis=0)
    centered = visible_matrix - mean
    try:
        _, singular_values, basis_rows = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    effective_rank = _effective_marker_pca_rank(singular_values)
    if effective_rank == 0:
        return None
    basis_rank = min(6, effective_rank) if rank is None else min(rank, effective_rank)
    return mean, basis_rows[:basis_rank].T, basis_rank


def _effective_marker_pca_rank(singular_values: np.ndarray) -> int:
    """Return rank after filtering near-zero PCA noise components."""
    if singular_values.size == 0 or singular_values.max() == 0.0:
        return 0
    relative_threshold = 0.01 * float(singular_values.max())
    return int((relative_threshold < singular_values).sum())


def _large_gap_frames(gap_indices: list[tuple[int, int]], max_gap: int) -> set[int]:
    """Return frame indices whose gaps exceed the PCA fill limit."""
    frames_to_skip: set[int] = set()
    for start, end in gap_indices:
        gap_size = end - start + 1
        if gap_size > max_gap:
            frames_to_skip.update(range(start, end + 1))
    return frames_to_skip


def _apply_marker_pca_projection(
    *,
    frames: list[MarkerFrame],
    marker_names: list[str],
    matrix: np.ndarray,
    occ_mask: np.ndarray,
    mean: np.ndarray,
    basis_vectors: np.ndarray,
    basis_rank: int,
    skip_frames: set[int],
) -> tuple[list[MarkerFrame], set[int]]:
    """Project occluded marker rows onto the PCA basis."""
    filled_frames = list(frames)
    pca_failed_frames: set[int] = set()

    for i in range(len(frames)):
        if not occ_mask[i].any():
            continue  # nothing occluded
        if i in skip_frames:
            pca_failed_frames.add(i)
            continue

        visible_idx = np.where(~occ_mask[i])[0]
        if visible_idx.size < basis_rank:
            # Under-determined: not enough visible coords to fit k coeffs
            pca_failed_frames.add(i)
            continue

        # Solve V_k[visible] @ c = (M[i, visible] - mean[visible])
        system_matrix = basis_vectors[visible_idx]
        target = matrix[i, visible_idx] - mean[visible_idx]
        try:
            coeffs, *_ = np.linalg.lstsq(system_matrix, target, rcond=None)
        except np.linalg.LinAlgError:
            pca_failed_frames.add(i)
            continue

        reconstructed = mean + basis_vectors @ coeffs
        if not np.all(np.isfinite(reconstructed)):
            pca_failed_frames.add(i)
            continue

        filled_frames[i] = _replace_occluded_markers(
            filled_frames[i], marker_names, occ_mask[i], reconstructed
        )

    return filled_frames, pca_failed_frames


def _replace_occluded_markers(
    frame: MarkerFrame,
    marker_names: list[str],
    row_mask: np.ndarray,
    reconstructed: np.ndarray,
) -> MarkerFrame:
    """Return a frame with reconstructed coordinates for occluded markers."""
    new_markers = dict(frame.markers)
    for j, name in enumerate(marker_names):
        base = 3 * j
        if row_mask[base]:
            new_markers[name] = Marker(
                name=name,
                x=float(reconstructed[base]),
                y=float(reconstructed[base + 1]),
                z=float(reconstructed[base + 2]),
                residual=None,
                occluded=False,
            )
    return MarkerFrame(
        timestamp=frame.timestamp,
        markers=new_markers,
        frame_index=frame.frame_index,
    )
