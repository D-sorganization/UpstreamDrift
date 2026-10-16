"""Tests for the shared marker frame source behind every engine's video (#12878)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.export import export_video
from src.shared.python.motion_matching.video_frame_source import (
    MarkerFrameSource,
    VideoSourceError,
    load_marker_frame_source,
    video_source_blocker,
)

pytestmark = [pytest.mark.unit]

FRAMES, MARKERS = 12, 5


def _markers(offset: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(7)
    base = rng.normal(size=(1, MARKERS, 3)) * 0.2
    drift = np.linspace(0, 0.3, FRAMES)[:, None, None]
    return base + drift + offset


@pytest.fixture
def replay_record(tmp_path: Path) -> Path:
    """The per-engine forward-replay layout (markers_m / target_m / valid)."""
    path = tmp_path / "returned-replay.npz"
    np.savez(
        path,
        time_s=np.linspace(0.0, 1.1, FRAMES),
        markers_m=_markers(0.01),
        target_m=_markers(),
        valid=np.ones((FRAMES, MARKERS), dtype=bool),
        labels=np.array([f"m{i}" for i in range(MARKERS)]),
    )
    return path


def test_replay_record_layout_is_normalised(replay_record: Path) -> None:
    source = load_marker_frame_source(replay_record)

    assert source.frame_count == FRAMES
    assert source.model_markers_m.shape == (FRAMES, MARKERS, 3)
    assert source.marker_validity is not None
    assert source.marker_validity.dtype == bool


def test_package_layout_is_normalised(tmp_path: Path) -> None:
    path = tmp_path / "candidate.npz"
    np.savez(
        path,
        time_s=np.linspace(0.0, 1.1, FRAMES),
        model_markers_m=_markers(0.01),
        target_markers_m=_markers(),
        marker_validity=np.ones((FRAMES, MARKERS), dtype=bool),
    )

    assert load_marker_frame_source(path).frame_count == FRAMES


def test_missing_markers_error_names_what_the_archive_contains(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dynamics_record.npz"
    np.savez(path, time_s=np.linspace(0, 1, 5), q=np.zeros((5, 3)), tau=np.zeros(5))

    with pytest.raises(VideoSourceError, match=r"dynamics_record\.npz.*q, tau"):
        load_marker_frame_source(path)


def test_missing_file_is_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="nope.npz"):
        load_marker_frame_source(tmp_path / "nope.npz")


def test_single_frame_is_rejected() -> None:
    with pytest.raises(VideoSourceError, match="at least 2 frames"):
        MarkerFrameSource(
            time_s=np.zeros(1),
            model_markers_m=np.zeros((1, 2, 3)),
            target_markers_m=np.zeros((1, 2, 3)),
        )


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(VideoSourceError, match="target_markers_m"):
        MarkerFrameSource(
            time_s=np.arange(4.0),
            model_markers_m=np.zeros((4, 2, 3)),
            target_markers_m=np.zeros((3, 2, 3)),
        )


def test_blocker_is_none_when_playable_and_text_otherwise(
    replay_record: Path, tmp_path: Path
) -> None:
    assert video_source_blocker(replay_record) is None
    assert "no results archive" in (video_source_blocker(None) or "")
    assert "not found" in (video_source_blocker(tmp_path / "x.npz") or "")


def test_export_video_plays_a_replay_record(
    replay_record: Path, tmp_path: Path
) -> None:
    out = export_video(
        replay_record, "mujoco", tmp_path / "replay.mp4", size_px=(160, 160)
    )

    assert out.is_file() and out.stat().st_size > 0
