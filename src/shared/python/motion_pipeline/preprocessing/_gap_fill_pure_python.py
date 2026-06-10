"""
Gap-filling strategies for motion capture data.

Part of issue #4564. Handles marker occlusion and missing keypoints
using interpolation and reconstruction strategies.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from ..contracts import (
    Keypoint,
    KeypointFrame,
    KeypointSequence,
    Marker,
    MarkerFrame,
    MarkerTrajectory,
)


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
    """Find gap indices in marker frames."""
    gaps = []
    gap_start = None

    for i, frame in enumerate(frames):
        # Check if any marker is occluded
        has_occluded = any(m.occluded for m in frame.markers.values())

        if has_occluded and gap_start is None:
            gap_start = i
        elif not has_occluded and gap_start is not None:
            gaps.append((gap_start, i - 1))
            gap_start = None

    if gap_start is not None:
        gaps.append((gap_start, len(frames) - 1))

    return gaps


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
                # Interpolate from before/after. Both neighbour frames must
                # actually have keypoint j — consecutive frames can carry
                # different keypoint counts, and indexing ``before`` without
                # checking its length raised IndexError (the guard checked
                # only ``after``).
                if after and j < len(after.keypoints) and j < len(before.keypoints):
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


def _marker_matrix(
    frames: list[MarkerFrame], marker_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    n_frames = len(frames)
    n_dims = len(marker_names) * 3
    matrix = np.zeros((n_frames, n_dims), dtype=float)
    occ_mask = np.zeros((n_frames, n_dims), dtype=bool)

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


def _linear_marker_fallback(
    frames: list[MarkerFrame],
    gap_indices: list[tuple[int, int]],
    max_gap: int,
) -> list[MarkerFrame]:
    return _fill_gaps_markers(
        list(frames), gap_indices, GapFillStrategy.LINEAR, max_gap
    )


def _pca_marker_basis(
    matrix: np.ndarray,
    occ_mask: np.ndarray,
    rank: int | None,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    fully_visible_rows = ~occ_mask.any(axis=1)
    if int(fully_visible_rows.sum()) < 2:
        return None

    visible_matrix = matrix[fully_visible_rows]
    mean = visible_matrix.mean(axis=0)
    centered = visible_matrix - mean

    try:
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    eps = max(singular_values.shape[0], 1) * np.finfo(float).eps
    eps *= singular_values.max() if singular_values.size else 1.0
    effective_rank = int((eps < singular_values).sum())
    if effective_rank == 0:
        return None

    basis_rank = min(6, effective_rank) if rank is None else min(rank, effective_rank)
    return mean, vt[:basis_rank].T, basis_rank


def _frames_exceeding_gap_limit(
    gap_indices: list[tuple[int, int]],
    max_gap: int,
) -> set[int]:
    skip_frames: set[int] = set()
    for start, end in gap_indices:
        if end - start + 1 > max_gap:
            skip_frames.update(range(start, end + 1))
    return skip_frames


def _project_marker_frame(
    frame: MarkerFrame,
    frame_index: int,
    marker_names: list[str],
    matrix: np.ndarray,
    occ_mask: np.ndarray,
    mean: np.ndarray,
    basis: np.ndarray,
    basis_rank: int,
) -> MarkerFrame | None:
    visible_idx = np.where(~occ_mask[frame_index])[0]
    if visible_idx.size < basis_rank:
        return None

    try:
        coeffs, *_ = np.linalg.lstsq(
            basis[visible_idx],
            matrix[frame_index, visible_idx] - mean[visible_idx],
            rcond=None,
        )
    except np.linalg.LinAlgError:
        return None

    reconstructed = mean + basis @ coeffs
    if not np.all(np.isfinite(reconstructed)):
        return None

    new_markers = dict(frame.markers)
    for j, name in enumerate(marker_names):
        base = 3 * j
        if occ_mask[frame_index, base]:
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


def _pca_reconstruct_markers(
    frames: list[MarkerFrame],
    gap_indices: list[tuple[int, int]],
    max_gap: int,
    rank: int | None = None,
) -> list[MarkerFrame]:
    """Reconstruct occluded markers via low-rank SVD projection.

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
    if len(frames) < 2:
        return list(frames)

    marker_names = list(frames[0].markers.keys())
    matrix, occ_mask = _marker_matrix(frames, marker_names)
    basis_parts = _pca_marker_basis(matrix, occ_mask, rank)
    if basis_parts is None:
        return _linear_marker_fallback(frames, gap_indices, max_gap)
    mean, basis, basis_rank = basis_parts
    skip_frames = _frames_exceeding_gap_limit(gap_indices, max_gap)

    filled_frames = list(frames)
    pca_failed_frames: set[int] = set()

    for i, frame in enumerate(frames):
        if not occ_mask[i].any():
            continue
        if i in skip_frames:
            pca_failed_frames.add(i)
            continue

        reconstructed_frame = _project_marker_frame(
            frame, i, marker_names, matrix, occ_mask, mean, basis, basis_rank
        )
        if reconstructed_frame is None:
            pca_failed_frames.add(i)
            continue

        filled_frames[i] = reconstructed_frame

    if pca_failed_frames:
        remaining_gaps = _find_gaps_markers(filled_frames)
        if remaining_gaps:
            filled_frames = _fill_gaps_markers(
                filled_frames, remaining_gaps, GapFillStrategy.LINEAR, max_gap
            )

    return filled_frames
