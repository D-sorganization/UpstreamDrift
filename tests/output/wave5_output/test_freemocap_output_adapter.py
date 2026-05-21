"""Tests for FreeMoCap output adapter (parser + exporters)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.motion_capture.freemocap_ingest.output_adapter import (
    FreeMoCapOutputAdapter,
    LandmarkFrame,
    LandmarkPoint,
    LandmarkSession,
)


def _write_landmarks_csv(path: Path, n_frames: int = 3) -> None:
    """Write a minimal landmark CSV with nose + left_shoulder + conf."""
    header = [
        "frame_number",
        "timestamp",
        "nose_x",
        "nose_y",
        "nose_z",
        "nose_conf",
        "left_shoulder_x",
        "left_shoulder_y",
        "left_shoulder_z",
        "left_shoulder_conf",
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(n_frames):
            w.writerow([i, i * 0.033, 0.1, 0.2, 0.3, 0.95, 0.4, 0.5, 0.6, 0.8])


def test_landmark_point_defaults():
    p = LandmarkPoint(name="nose", x=1.0, y=2.0, z=3.0)
    assert p.confidence == 1.0
    assert p.visible is True


def test_landmark_frame_get_point():
    f = LandmarkFrame(
        frame_number=0,
        timestamp=0.0,
        points=[
            LandmarkPoint("a", 1, 2, 3),
            LandmarkPoint("b", 4, 5, 6),
        ],
    )
    assert f.get_point("a").x == 1
    assert f.get_point("missing") is None


def test_landmark_frame_to_array_empty():
    f = LandmarkFrame(frame_number=0, timestamp=0.0, points=[])
    arr = f.to_array()
    assert arr.shape == (0, 4)


def test_landmark_frame_to_array_with_points():
    f = LandmarkFrame(
        frame_number=0,
        timestamp=0.0,
        points=[LandmarkPoint("a", 1, 2, 3, confidence=0.9)],
    )
    arr = f.to_array()
    assert arr.shape == (1, 4)
    assert arr[0, 0] == 1
    assert arr[0, 3] == 0.9


def test_landmark_session_to_array_empty():
    s = LandmarkSession(session_id="x")
    arr = s.to_array()
    assert arr.shape == (0, 0, 4)


def test_landmark_session_to_array_stack():
    pts = [LandmarkPoint("a", 1, 2, 3), LandmarkPoint("b", 4, 5, 6)]
    s = LandmarkSession(
        session_id="x",
        frames=[
            LandmarkFrame(0, 0.0, pts),
            LandmarkFrame(1, 0.1, pts),
        ],
    )
    arr = s.to_array()
    assert arr.shape == (2, 2, 4)


def test_parse_requires_existing_dir(tmp_path: Path):
    adapter = FreeMoCapOutputAdapter(tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="Output directory not found"):
        adapter.parse()


def test_parse_requires_landmark_csv(tmp_path: Path):
    adapter = FreeMoCapOutputAdapter(tmp_path)
    with pytest.raises(FileNotFoundError, match="No landmark CSV"):
        adapter.parse()


def test_parse_reads_landmark_csv(tmp_path: Path):
    csv_path = tmp_path / "freemocap_3d_landmarks_main.csv"
    _write_landmarks_csv(csv_path, n_frames=4)

    adapter = FreeMoCapOutputAdapter(tmp_path)
    session = adapter.parse()
    assert len(session.frames) == 4
    assert session.frames[0].frame_number == 0
    # Two landmarks defined: nose, left_shoulder
    names = {p.name for p in session.frames[0].points}
    assert "nose" in names
    assert "left_shoulder" in names

    # Confidence properly parsed (bug fix: was AttributeError on list.get)
    nose = session.frames[0].get_point("nose")
    assert nose is not None
    assert nose.confidence == pytest.approx(0.95)
    assert nose.visible is True


def test_parse_low_confidence_not_visible(tmp_path: Path):
    csv_path = tmp_path / "freemocap_3d_landmarks_main.csv"
    header = [
        "frame_number",
        "timestamp",
        "nose_x",
        "nose_y",
        "nose_z",
        "nose_conf",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow([0, 0.0, 0.1, 0.2, 0.3, 0.2])  # low conf

    adapter = FreeMoCapOutputAdapter(tmp_path)
    session = adapter.parse()
    nose = session.frames[0].get_point("nose")
    assert nose.confidence == pytest.approx(0.2)
    assert nose.visible is False


def test_parse_with_calibration_and_metadata(tmp_path: Path):
    csv_path = tmp_path / "freemocap_3d_landmarks_main.csv"
    _write_landmarks_csv(csv_path, n_frames=2)

    (tmp_path / "camera_calibration.json").write_text(json.dumps({"fx": 500}))
    (tmp_path / "recording_metadata.json").write_text(json.dumps({"fps": 30}))

    adapter = FreeMoCapOutputAdapter(tmp_path)
    session = adapter.parse(session_id="custom")
    assert session.session_id == "custom"
    assert session.calibration == {"fx": 500}
    assert session.metadata == {"fps": 30}


def test_parse_skips_blank_rows(tmp_path: Path):
    csv_path = tmp_path / "freemocap_3d_landmarks_main.csv"
    with open(csv_path, "w", newline="") as f:
        f.write("frame_number,timestamp,nose_x,nose_y,nose_z,nose_conf\n")
        f.write("0,0.0,0.1,0.2,0.3,0.9\n")
        f.write("\n")
        f.write("1,0.1,0.4,0.5,0.6,0.8\n")

    adapter = FreeMoCapOutputAdapter(tmp_path)
    session = adapter.parse()
    assert len(session.frames) == 2


def test_get_session_returns_last_parsed(tmp_path: Path):
    csv_path = tmp_path / "freemocap_3d_landmarks_main.csv"
    _write_landmarks_csv(csv_path)
    adapter = FreeMoCapOutputAdapter(tmp_path)
    assert adapter.get_session() is None
    adapter.parse()
    assert adapter.get_session() is not None


def test_export_to_numpy_without_parse_raises(tmp_path: Path):
    adapter = FreeMoCapOutputAdapter(tmp_path)
    with pytest.raises(ValueError, match="No session loaded"):
        adapter.export_to_numpy(tmp_path / "out.npy")


def test_export_to_csv_without_parse_raises(tmp_path: Path):
    adapter = FreeMoCapOutputAdapter(tmp_path)
    with pytest.raises(ValueError, match="No session loaded"):
        adapter.export_to_csv(tmp_path / "out.csv")


def test_export_to_numpy_roundtrip(tmp_path: Path):
    csv_path = tmp_path / "freemocap_3d_landmarks_main.csv"
    _write_landmarks_csv(csv_path, n_frames=3)

    adapter = FreeMoCapOutputAdapter(tmp_path)
    adapter.parse()
    out = tmp_path / "data.npy"
    arr = adapter.export_to_numpy(out)
    assert out.exists()
    loaded = np.load(out)
    np.testing.assert_array_equal(arr, loaded)
    assert loaded.shape[0] == 3  # frames
    assert loaded.shape[2] == 4  # x, y, z, conf


def test_export_to_csv_roundtrip(tmp_path: Path):
    csv_path = tmp_path / "freemocap_3d_landmarks_main.csv"
    _write_landmarks_csv(csv_path, n_frames=2)

    adapter = FreeMoCapOutputAdapter(tmp_path)
    adapter.parse()
    out = tmp_path / "export.csv"
    adapter.export_to_csv(out)
    assert out.exists()

    with open(out) as f:
        reader = csv.reader(f)
        rows = list(reader)
    header = rows[0]
    assert header[:2] == ["frame_number", "timestamp"]
    assert len(rows) == 3  # header + 2 frames
