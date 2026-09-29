"""Unit tests for motion_pipeline.preprocessing.resample."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.shared.python.motion_pipeline.contracts import (
    Keypoint,
    KeypointFrame,
    KeypointSequence,
    Marker,
    MarkerFrame,
    MarkerTrajectory,
)
from src.shared.python.motion_pipeline.preprocessing.resample import resample

from ._local_fixtures import make_keypoint_sequence, make_marker_trajectory

_FINITE_COORD = st.floats(
    min_value=-100.0,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False,
)


def _affine_keypoint_sequence(
    *,
    num_frames: int,
    num_keypoints: int,
    fps: float,
    base: float,
    slope: float,
    confidence: float,
) -> KeypointSequence:
    frames: list[KeypointFrame] = []
    for frame_index in range(num_frames):
        timestamp = frame_index / fps
        keypoints = [
            Keypoint(
                x=base + keypoint_index + slope * timestamp,
                y=base - keypoint_index + (slope * 0.5) * timestamp,
                z=base + (keypoint_index * 0.25) - (slope * 0.25) * timestamp,
                confidence=confidence,
                name=f"kp_{keypoint_index}",
            )
            for keypoint_index in range(num_keypoints)
        ]
        frames.append(
            KeypointFrame(
                timestamp=timestamp,
                keypoints=keypoints,
                schema_name="custom",
                frame_index=frame_index,
            )
        )
    return KeypointSequence(id="affine_keypoints", frames=frames)


def _affine_marker_trajectory(
    *,
    num_frames: int,
    num_markers: int,
    fps: float,
    base: float,
    slope: float,
) -> MarkerTrajectory:
    frames: list[MarkerFrame] = []
    marker_names = [f"M{marker_index}" for marker_index in range(num_markers)]
    for frame_index in range(num_frames):
        timestamp = frame_index / fps
        markers = {
            name: Marker(
                name=name,
                x=base + marker_index + slope * timestamp,
                y=base - marker_index + (slope * 0.25) * timestamp,
                z=base + (marker_index * 0.5) - (slope * 0.5) * timestamp,
                residual=float(marker_index),
                occluded=False,
            )
            for marker_index, name in enumerate(marker_names)
        }
        frames.append(
            MarkerFrame(
                timestamp=timestamp,
                markers=markers,
                frame_index=frame_index,
            )
        )
    return MarkerTrajectory(id="affine_markers", frames=frames, subject_id="subject")


def test_resample_unsupported_type_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        resample("not-a-sequence", target_fps=120.0)  # type: ignore[arg-type]


def test_resample_keypoints_short_sequence_unchanged() -> None:
    seq = make_keypoint_sequence(num_frames=1, num_kp=2, fps=30.0)
    out = resample(seq, target_fps=120.0)
    assert out.num_frames == 1


def test_resample_markers_short_trajectory_unchanged() -> None:
    traj = make_marker_trajectory(num_frames=1, fps=30.0)
    out = resample(traj, target_fps=120.0)
    assert out.num_frames == 1


def test_resample_keypoints_upsamples_frame_count() -> None:
    seq = make_keypoint_sequence(num_frames=30, num_kp=1, fps=30.0)
    duration = seq.frames[-1].timestamp - seq.frames[0].timestamp
    out = resample(seq, target_fps=120.0, source_fps=30.0)
    expected = int(duration * 120.0) + 1
    assert isinstance(out, KeypointSequence)
    assert out.num_frames == expected
    assert out.metadata.get("resampled") is True
    assert out.metadata.get("source_fps") == 30.0
    assert out.metadata.get("target_fps") == 120.0


def test_resample_keypoints_preserves_endpoints() -> None:
    seq = make_keypoint_sequence(num_frames=30, num_kp=1, fps=30.0)
    out = resample(seq, target_fps=100.0)
    assert out.frames[0].timestamp == pytest.approx(seq.frames[0].timestamp)
    assert out.frames[-1].timestamp == pytest.approx(seq.frames[-1].timestamp)


def test_resample_markers_increases_frame_count() -> None:
    traj = make_marker_trajectory(num_frames=30, fps=60.0)
    out = resample(traj, target_fps=1000.0)
    assert isinstance(out, MarkerTrajectory)
    # 1000Hz should produce far more frames than 30
    assert out.num_frames > 30
    assert out.metadata.get("target_fps") == 1000.0


def test_resample_round_trip_preserves_signal_shape() -> None:
    """60Hz -> 120Hz -> 60Hz round-trip should approximate the original."""
    seq = make_keypoint_sequence(num_frames=30, num_kp=1, fps=60.0)
    up = resample(seq, target_fps=120.0, source_fps=60.0)
    down = resample(up, target_fps=60.0, source_fps=120.0)

    raw = np.array([f.keypoints[0].x for f in seq.frames])
    rt = np.array([f.keypoints[0].x for f in down.frames])
    # Allow small numerical drift; both should have same length to within 1
    n = min(len(raw), len(rt))
    np.testing.assert_allclose(raw[:n], rt[:n], atol=1e-6)


def test_resample_keypoints_timestamp_grid_uniform() -> None:
    seq = make_keypoint_sequence(num_frames=30, num_kp=1, fps=30.0)
    out = resample(seq, target_fps=100.0)
    timestamps = np.array([f.timestamp for f in out.frames])
    diffs = np.diff(timestamps)
    np.testing.assert_allclose(diffs, diffs[0], atol=1e-6)


def test_resample_marker_trajectory_id_preserved() -> None:
    traj = make_marker_trajectory(num_frames=10, fps=30.0)
    out = resample(traj, target_fps=60.0)
    assert out.id == traj.id


@given(
    num_frames=st.integers(min_value=2, max_value=8),
    num_keypoints=st.integers(min_value=1, max_value=5),
    fps=st.integers(min_value=15, max_value=120),
    target_multiplier=st.integers(min_value=2, max_value=4),
    base=_FINITE_COORD,
    slope=_FINITE_COORD,
    confidence=st.floats(
        min_value=0.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
@settings(max_examples=40, deadline=None)
def test_resample_keypoints_preserves_affine_paths_and_labels(
    num_frames: int,
    num_keypoints: int,
    fps: int,
    target_multiplier: int,
    base: float,
    slope: float,
    confidence: float,
) -> None:
    source_fps = float(fps)
    seq = _affine_keypoint_sequence(
        num_frames=num_frames,
        num_keypoints=num_keypoints,
        fps=source_fps,
        base=base,
        slope=slope,
        confidence=confidence,
    )

    out = resample(
        seq,
        target_fps=source_fps * target_multiplier,
        source_fps=source_fps,
    )

    assert isinstance(out, KeypointSequence)
    assert out.num_keypoints == num_keypoints
    assert [kp.name for kp in out.frames[0].keypoints] == [
        f"kp_{keypoint_index}" for keypoint_index in range(num_keypoints)
    ]
    for frame in out.frames:
        for keypoint_index, keypoint in enumerate(frame.keypoints):
            timestamp = frame.timestamp
            assert keypoint.confidence == pytest.approx(confidence)
            assert keypoint.x == pytest.approx(
                base + keypoint_index + slope * timestamp
            )
            assert keypoint.y == pytest.approx(
                base - keypoint_index + (slope * 0.5) * timestamp
            )
            assert keypoint.z == pytest.approx(
                base + (keypoint_index * 0.25) - (slope * 0.25) * timestamp
            )


@given(
    num_frames=st.integers(min_value=2, max_value=8),
    num_markers=st.integers(min_value=1, max_value=5),
    fps=st.integers(min_value=15, max_value=120),
    target_multiplier=st.integers(min_value=2, max_value=4),
    base=_FINITE_COORD,
    slope=_FINITE_COORD,
)
@settings(max_examples=40, deadline=None)
def test_resample_markers_preserves_affine_paths_and_marker_surface(
    num_frames: int,
    num_markers: int,
    fps: int,
    target_multiplier: int,
    base: float,
    slope: float,
) -> None:
    source_fps = float(fps)
    traj = _affine_marker_trajectory(
        num_frames=num_frames,
        num_markers=num_markers,
        fps=source_fps,
        base=base,
        slope=slope,
    )

    out = resample(
        traj,
        target_fps=source_fps * target_multiplier,
        source_fps=source_fps,
    )

    assert isinstance(out, MarkerTrajectory)
    assert out.subject_id == traj.subject_id
    assert out.marker_names == [
        f"M{marker_index}" for marker_index in range(num_markers)
    ]
    for frame in out.frames:
        assert frame.marker_names == out.marker_names
        for marker_index, marker_name in enumerate(out.marker_names):
            marker = frame.markers[marker_name]
            timestamp = frame.timestamp
            assert marker.name == marker_name
            assert marker.residual == pytest.approx(float(marker_index))
            assert marker.occluded is False
            assert marker.x == pytest.approx(base + marker_index + slope * timestamp)
            assert marker.y == pytest.approx(
                base - marker_index + (slope * 0.25) * timestamp
            )
            assert marker.z == pytest.approx(
                base + (marker_index * 0.5) - (slope * 0.5) * timestamp
            )


def test_pure_python_resample_parity() -> None:
    from src.shared.python.motion_pipeline.preprocessing._resample_pure_python import (
        resample as pure_resample,
    )

    # Test standard parity
    seq = make_keypoint_sequence(num_frames=30, num_kp=1, fps=30.0)
    out = pure_resample(seq, target_fps=120.0, source_fps=30.0)
    assert out.num_frames > 30
    assert out.metadata.get("resampled") is True

    traj = make_marker_trajectory(num_frames=30, fps=60.0)
    out_traj = pure_resample(traj, target_fps=120.0, source_fps=60.0)
    assert out_traj.num_frames > 30

    # Test error condition
    with pytest.raises(ValueError, match="Unsupported"):
        pure_resample("invalid", target_fps=60.0)  # type: ignore[arg-type]

    # Test short inputs (len < 2)
    short_seq = make_keypoint_sequence(num_frames=1, num_kp=1, fps=30.0)
    assert pure_resample(short_seq, target_fps=120.0) is short_seq

    short_traj = make_marker_trajectory(num_frames=1, fps=60.0)
    assert pure_resample(short_traj, target_fps=120.0) is short_traj


def test_resample_estimate_fps_edge_cases() -> None:
    from src.shared.python.motion_pipeline.preprocessing.resample import _estimate_fps
    from src.shared.python.motion_pipeline.preprocessing._resample_pure_python import (
        _estimate_fps as pure_estimate_fps,
    )
    from src.shared.python.motion_pipeline.contracts import KeypointFrame

    # dt <= 0
    frames_bad = [
        KeypointFrame.model_construct(
            timestamp=1.0, keypoints=[], schema_name="custom", frame_index=0
        ),
        KeypointFrame.model_construct(
            timestamp=0.5, keypoints=[], schema_name="custom", frame_index=1
        ),
    ]
    assert _estimate_fps(frames_bad) == 30.0
    assert pure_estimate_fps(frames_bad) == 30.0

    # len < 2
    frames_short = [
        KeypointFrame.model_construct(
            timestamp=1.0, keypoints=[], schema_name="custom", frame_index=0
        ),
    ]
    assert _estimate_fps(frames_short) == 30.0
    assert pure_estimate_fps(frames_short) == 30.0
