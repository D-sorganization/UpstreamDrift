"""Unit tests for Simscape C3D video overlay visualization (#9921).

Tests adhere strictly to Design by Contract (DbC), Law of Demeter (LoD),
and Test-Driven Development (TDD).
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from src.motion_capture.simscape_c3d_video_overlay import (
    OverlayDataset,
    SkeletalTopology,
    load_overlay_dataset,
    render_overlay_frame,
    render_overlay_video,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def sample_overlay_data(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal synthetic capture payload and replay prediction for testing."""
    times = [0.0, 0.05, 0.10]
    labels = ["WaistLeft", "WaistRight", "BackTop", "HeadTop", "LWristTop"]
    num_times = len(times)
    num_markers = len(labels)

    # Synthetic target points in meters (times, markers, 3)
    target_pts = np.zeros((num_times, num_markers, 3), dtype=float)
    for t_idx in range(num_times):
        for m_idx in range(num_markers):
            target_pts[t_idx, m_idx] = [0.1 * m_idx, 0.05 * t_idx, 1.0 + 0.1 * m_idx]

    # One marker invalid at t=2
    valid_flags = np.ones((num_times, num_markers), dtype=bool)
    valid_flags[2, 3] = False  # HeadTop invalid at t=2

    capture_payload = {
        "time_s": times,
        "labels": labels,
        "points_world_m": target_pts.tolist(),
        "valid": valid_flags.tolist(),
        "source_sha256": "dummy_sha256",
    }
    capture_file = tmp_path / "test_driver_marker_payload.json"
    capture_file.write_text(json.dumps(capture_payload), encoding="utf-8")

    # Model prediction with slight offset (e.g. 5mm error)
    model_pts = target_pts.copy()
    model_pts[:, :, 0] += 0.005  # +5mm in X

    replay_payload = {
        "required_release": "R2025b",
        "duration_s": 0.10,
        "prediction_m": model_pts.tolist(),
    }
    replay_file = tmp_path / "test_qualified_candidate_replay.json"
    replay_file.write_text(json.dumps(replay_payload), encoding="utf-8")

    return capture_file, replay_file


def test_overlay_dataset_contracts(sample_overlay_data: tuple[Path, Path]) -> None:
    capture_file, replay_file = sample_overlay_data
    dataset = load_overlay_dataset(capture_file, replay_file)

    assert dataset.frame_count == 3
    assert dataset.marker_count == 5
    assert dataset.duration_s == 0.10
    assert dataset.labels == [
        "WaistLeft",
        "WaistRight",
        "BackTop",
        "HeadTop",
        "LWristTop",
    ]

    # Check RMS calculation
    rms_0 = dataset.rms(0)
    assert 4.9 <= rms_0 <= 5.1  # exactly ~5.0 mm offset

    # Overall RMS across all valid frames
    total_rms = dataset.total_rms()
    assert 4.9 <= total_rms <= 5.1


def test_load_dataset_rejects_missing_or_invalid_files(tmp_path: Path) -> None:
    non_existent = tmp_path / "non_existent.json"
    with pytest.raises(FileNotFoundError):
        load_overlay_dataset(non_existent, tmp_path / "foo.json")

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_overlay_dataset(bad_json, bad_json)


def test_load_dataset_rejects_mismatched_marker_counts(
    tmp_path: Path, sample_overlay_data: tuple[Path, Path]
) -> None:
    capture_file, _ = sample_overlay_data
    bad_replay = tmp_path / "bad_replay.json"
    bad_replay.write_text(
        json.dumps({"prediction_m": [[[0, 0, 0]]]}),  # only 1 marker
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Marker count mismatch"):
        load_overlay_dataset(capture_file, bad_replay)


def test_skeletal_topology_resolves_bones() -> None:
    topology = SkeletalTopology()
    assert len(topology.bones) > 0
    # Check that bones are pairs of strings
    for bone in topology.bones:
        assert isinstance(bone, tuple)
        assert len(bone) == 2
        assert isinstance(bone[0], str)
        assert isinstance(bone[1], str)


def test_render_overlay_frame_produces_valid_artists(
    sample_overlay_data: tuple[Path, Path],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    capture_file, replay_file = sample_overlay_data
    dataset = load_overlay_dataset(capture_file, replay_file)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    frame_info = render_overlay_frame(dataset, frame_index=0, ax=ax)
    plt.close(fig)

    assert frame_info["frame_index"] == 0
    assert frame_info["time_s"] == 0.0
    assert "rms_mm" in frame_info
    assert frame_info["rms_mm"] > 0


def test_render_overlay_video_gif(
    sample_overlay_data: tuple[Path, Path], tmp_path: Path
) -> None:
    capture_file, replay_file = sample_overlay_data
    dataset = load_overlay_dataset(capture_file, replay_file)

    output_gif = tmp_path / "test_animation.gif"
    rendered_path = render_overlay_video(
        dataset, output_path=output_gif, fps=10, dpi=60, format="gif"
    )

    assert rendered_path.exists()
    assert rendered_path.stat().st_size > 0
    assert rendered_path.suffix == ".gif"


def test_render_overlay_video_mp4(
    sample_overlay_data: tuple[Path, Path], tmp_path: Path
) -> None:
    capture_file, replay_file = sample_overlay_data
    dataset = load_overlay_dataset(capture_file, replay_file)

    output_mp4 = tmp_path / "test_animation.mp4"
    rendered_path = render_overlay_video(
        dataset, output_path=output_mp4, fps=10, dpi=60, format="mp4"
    )

    assert rendered_path.exists()
    assert rendered_path.stat().st_size > 0
    assert rendered_path.suffix == ".mp4"
