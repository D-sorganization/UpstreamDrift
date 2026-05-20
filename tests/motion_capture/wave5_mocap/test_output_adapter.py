"""Fast tests for motion_capture.freemocap_ingest.output_adapter."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from motion_capture.freemocap_ingest.output_adapter import (
    MEDIAPIPE_LANDMARKS,
    FreeMoCapOutputAdapter,
    LandmarkFrame,
    LandmarkPoint,
    LandmarkSession,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Dataclass behaviour
# ---------------------------------------------------------------------------


def test_landmark_point_defaults() -> None:
    p = LandmarkPoint(name="nose", x=1.0, y=2.0, z=3.0)
    assert p.confidence == 1.0
    assert p.visible is True


def test_landmark_frame_get_point_found_and_missing() -> None:
    f = LandmarkFrame(
        frame_number=1,
        timestamp=0.5,
        points=[LandmarkPoint("nose", 0.0, 0.0, 0.0)],
    )
    assert f.get_point("nose").name == "nose"
    assert f.get_point("missing") is None


def test_landmark_frame_to_array_with_points() -> None:
    f = LandmarkFrame(
        frame_number=0,
        timestamp=0.0,
        points=[
            LandmarkPoint("a", 1.0, 2.0, 3.0, confidence=0.9),
            LandmarkPoint("b", 4.0, 5.0, 6.0, confidence=0.8),
        ],
    )
    arr = f.to_array()
    assert arr.shape == (2, 4)
    np.testing.assert_allclose(arr[0], [1.0, 2.0, 3.0, 0.9])
    np.testing.assert_allclose(arr[1], [4.0, 5.0, 6.0, 0.8])


def test_landmark_frame_to_array_empty() -> None:
    f = LandmarkFrame(frame_number=0, timestamp=0.0)
    arr = f.to_array()
    assert arr.shape == (0, 4)


def test_landmark_session_to_array_empty() -> None:
    s = LandmarkSession(session_id="x")
    arr = s.to_array()
    assert arr.shape == (0, 0, 4)


def test_landmark_session_to_array_stacks_frames() -> None:
    points = [LandmarkPoint("a", 1.0, 2.0, 3.0)]
    s = LandmarkSession(
        session_id="x",
        frames=[
            LandmarkFrame(0, 0.0, points=points),
            LandmarkFrame(1, 0.1, points=points),
        ],
    )
    arr = s.to_array()
    assert arr.shape == (2, 1, 4)


def test_mediapipe_landmarks_list_nonempty() -> None:
    assert len(MEDIAPIPE_LANDMARKS) > 0
    assert "nose" in MEDIAPIPE_LANDMARKS


# ---------------------------------------------------------------------------
# Fixtures helpers
# ---------------------------------------------------------------------------


def _write_landmarks_csv(
    path: Path,
    *,
    landmarks: tuple[str, ...] = ("nose", "left_hip"),
    rows: int = 3,
    include_conf: bool = True,
    bad_row: bool = False,
) -> None:
    header = ["frame_num", "timestamp"]
    for name in landmarks:
        header.extend([f"{name}_x", f"{name}_y", f"{name}_z"])
        if include_conf:
            header.append(f"{name}_conf")
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i in range(rows):
            row: list[object] = [i, i * 0.1]
            for j, _ in enumerate(landmarks):
                row.extend([j + 0.1, j + 0.2, j + 0.3])
                if include_conf:
                    row.append(0.9)
            writer.writerow(row)
        if bad_row:
            # row that should be skipped (non-numeric coordinate)
            row = [rows, rows * 0.1]
            for _ in landmarks:
                row.extend(["NaNbad", 0.0, 0.0])
                if include_conf:
                    row.append(0.0)
            writer.writerow(row)
        # Empty row should be tolerated (skipped)
        writer.writerow([])


# ---------------------------------------------------------------------------
# Adapter behaviour
# ---------------------------------------------------------------------------


def test_adapter_init_resolves_path(tmp_path: Path) -> None:
    a = FreeMoCapOutputAdapter(tmp_path)
    assert a.output_dir == tmp_path.expanduser().resolve()
    assert a.get_session() is None


def test_parse_missing_output_dir(tmp_path: Path) -> None:
    a = FreeMoCapOutputAdapter(tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="Output directory"):
        a.parse()


def test_parse_no_landmark_files(tmp_path: Path) -> None:
    a = FreeMoCapOutputAdapter(tmp_path)
    with pytest.raises(FileNotFoundError, match="No landmark CSV"):
        a.parse()


def test_parse_happy_path_with_conf(tmp_path: Path) -> None:
    csv_path = tmp_path / "freemocap_3d_landmarks_main.csv"
    _write_landmarks_csv(csv_path, rows=4)

    # Calibration + metadata files
    (tmp_path / "camera_calibration.json").write_text(
        json.dumps({"cameras": ["c0", "c1"]})
    )
    (tmp_path / "recording_metadata.json").write_text(json.dumps({"fps": 30}))

    adapter = FreeMoCapOutputAdapter(tmp_path)
    session = adapter.parse(session_id="custom-id")

    assert session.session_id == "custom-id"
    assert len(session.frames) == 4
    assert session.calibration == {"cameras": ["c0", "c1"]}
    assert session.metadata == {"fps": 30}

    first = session.frames[0]
    assert first.frame_number == 0
    assert first.timestamp == pytest.approx(0.0)
    names = {p.name for p in first.points}
    assert names == {"nose", "left_hip"}
    # confidence preserved (0.9 → visible)
    for p in first.points:
        assert p.confidence == pytest.approx(0.9)
        assert p.visible is True

    assert adapter.get_session() is session


def test_parse_default_session_id_from_parent(tmp_path: Path) -> None:
    out = tmp_path / "session_42" / "freemocap_output"
    out.mkdir(parents=True)
    _write_landmarks_csv(out / "freemocap_3d_landmarks_a.csv")
    adapter = FreeMoCapOutputAdapter(out)
    session = adapter.parse()
    assert session.session_id == "session_42"


def test_parse_csv_without_conf_defaults_to_one(tmp_path: Path) -> None:
    csv_path = tmp_path / "freemocap_3d_landmarks_x.csv"
    _write_landmarks_csv(csv_path, include_conf=False, rows=2)
    adapter = FreeMoCapOutputAdapter(tmp_path)
    session = adapter.parse()
    assert all(p.confidence == 1.0 for f in session.frames for p in f.points)
    assert all(p.visible for f in session.frames for p in f.points)


def test_parse_skips_bad_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "freemocap_3d_landmarks_b.csv"
    _write_landmarks_csv(csv_path, rows=2, bad_row=True)
    adapter = FreeMoCapOutputAdapter(tmp_path)
    session = adapter.parse()
    # 2 good rows; bad row's points were skipped so frame exists but with 0 points
    # Empty trailing row dropped entirely
    assert len(session.frames) == 3
    assert session.frames[-1].points == []


def test_export_to_numpy_requires_session(tmp_path: Path) -> None:
    a = FreeMoCapOutputAdapter(tmp_path)
    with pytest.raises(ValueError, match="No session"):
        a.export_to_numpy(tmp_path / "out.npy")


def test_export_to_csv_requires_session(tmp_path: Path) -> None:
    a = FreeMoCapOutputAdapter(tmp_path)
    with pytest.raises(ValueError, match="No session"):
        a.export_to_csv(tmp_path / "out.csv")


def test_export_to_numpy_roundtrip(tmp_path: Path) -> None:
    csv_path = tmp_path / "freemocap_3d_landmarks_a.csv"
    _write_landmarks_csv(csv_path, rows=3)
    adapter = FreeMoCapOutputAdapter(tmp_path)
    adapter.parse()
    out = tmp_path / "out.npy"
    arr = adapter.export_to_numpy(out)
    assert out.exists()
    reloaded = np.load(out)
    np.testing.assert_array_equal(arr, reloaded)
    assert arr.shape[0] == 3  # frames


def test_export_to_csv_roundtrip(tmp_path: Path) -> None:
    csv_path = tmp_path / "freemocap_3d_landmarks_a.csv"
    _write_landmarks_csv(csv_path, rows=2)
    adapter = FreeMoCapOutputAdapter(tmp_path)
    adapter.parse()

    out = tmp_path / "out.csv"
    adapter.export_to_csv(out)
    assert out.exists()

    with open(out, newline="") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    assert header[0] == "frame_number"
    assert header[1] == "timestamp"
    # 4 columns per landmark
    assert (len(header) - 2) % 4 == 0
    assert len(rows) == 3  # header + 2 data rows


def test_load_calibration_missing(tmp_path: Path) -> None:
    a = FreeMoCapOutputAdapter(tmp_path)
    assert a._load_calibration() is None


def test_load_metadata_missing(tmp_path: Path) -> None:
    a = FreeMoCapOutputAdapter(tmp_path)
    assert a._load_metadata() == {}


def test_find_landmarks_csv(tmp_path: Path) -> None:
    (tmp_path / "freemocap_3d_landmarks_one.csv").write_text("x")
    (tmp_path / "freemocap_3d_landmarks_two.csv").write_text("x")
    (tmp_path / "unrelated.csv").write_text("x")
    a = FreeMoCapOutputAdapter(tmp_path)
    found = a._find_landmarks_csv()
    assert len(found) == 2
