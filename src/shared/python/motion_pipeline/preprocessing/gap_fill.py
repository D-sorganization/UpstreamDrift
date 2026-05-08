"""
Gap-filling strategies for motion capture data.

Part of issue #4564. Handles marker occlusion and missing keypoints
using interpolation and reconstruction strategies.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np

from ..contracts import KeypointFrame, KeypointSequence, MarkerFrame, MarkerTrajectory


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
    elif isinstance(data, MarkerTrajectory):
        return _gap_fill_markers(data, strategy, max_gap)
    else:
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
        metadata={**seq.metadata, "gap_filled": True, "strategy": strategy.value}
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
    
    # Fill gaps
    filled_frames = _fill_gaps_markers(traj.frames, gap_indices, strategy, max_gap)
    
    return MarkerTrajectory(
        id=traj.id,
        frames=filled_frames,
        calibration=traj.calibration,
        subject_id=traj.subject_id,
        metadata={**traj.metadata, "gap_filled": True, "strategy": strategy.value}
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
                if after and j < len(after.keypoints):
                    t = (i - start + 1) / (end - start + 2)
                    kp_before = before.keypoints[j]
                    kp_after = after.keypoints[j]
                    
                    new_kp = Keypoint(
                        x=kp_before.x + t * (kp_after.x - kp_before.x),
                        y=kp_before.y + t * (kp_after.y - kp_before.y),
                        z=kp_before.z + t * (kp_after.z - kp_before.z) if kp_before.z is not None else None,
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
            if marker.occluded and after and name in before.markers and name in after.markers:
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


# Import Marker for type hints
from ..contracts import Marker