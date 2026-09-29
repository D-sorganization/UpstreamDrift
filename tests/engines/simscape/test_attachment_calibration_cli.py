"""Regression contracts for the native-pose attachment calibration experiment."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pytestmark = [pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[3]
RUNNER = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/reproduction/calibrate_pose_attachments.py"
)


def fixture_paths(tmp_path: Path) -> list[str]:
    frames = tmp_path / "frames"
    frames.mkdir()
    seed = {
        "source_sha256": "capture",
        "geometry_in": [14.5, 12],
        "labels": ["moving", "origin"],
        "body_names": ["B", "B"],
        "offsets_m": [[0, 0, 0], [0, 0, 0]],
        "initial_state_verified": True,
    }
    points = []
    for i, angle in enumerate([0, 0.3, 0.7, 1.2]):
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        points.append([(rotation @ [0.1, 0.2, 0.3]).tolist(), [0, 0, 0]])
        record = {
            "frame_index": i + 1,
            "time_s": i / 10,
            "source_sha256": "capture",
            "geometry_in": [14.5, 12],
            "body_names": ["B"],
            "origins_m": [[0, 0, 0]],
            "rotations_world_from_body": [rotation.tolist()],
            "saved_marker_max_difference_m": 0,
        }
        (frames / f"frame-{i + 1:05}.json").write_text(json.dumps(record))
    points[1][0] = [999, 999, 999]
    capture = {
        "source_sha256": "capture",
        "labels": seed["labels"],
        "time_s": [0, 0.1, 0.2, 0.3],
        "points_world_m": points,
        "valid": [[True, True], [False, True], [True, True], [True, True]],
    }
    for name, value in [("seed", seed), ("capture", capture)]:
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    return [
        "--repo",
        str(ROOT),
        "--frames",
        str(frames),
        "--seed",
        str(tmp_path / "seed.json"),
        "--capture",
        str(tmp_path / "capture.json"),
        "--output",
        str(tmp_path / "result"),
    ]


def test_recovers_fixed_offsets_and_excludes_invalid_observation(
    tmp_path: Path,
) -> None:
    args = fixture_paths(tmp_path)
    process = subprocess.run(
        [sys.executable, str(RUNNER), *args], capture_output=True, text=True
    )
    assert process.returncode == 0, process.stderr
    candidate = json.loads((tmp_path / "result/candidate_seed.json").read_text())
    expected = np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]])
    assert np.allclose(candidate["offsets_m"], expected, atol=1e-12)
    assert candidate["initial_state_verified"] is False
    report = json.loads((tmp_path / "result/calibration_report.json").read_text())
    assert report["valid_observation_count"] == 7
    assert report["fixed_pose_after_rms_m"] < 1e-12
    assert report["fixed_pose_before_rms_m"] > 0.1


def test_rejects_capture_identity_mismatch(tmp_path: Path) -> None:
    args = fixture_paths(tmp_path)
    path = tmp_path / "capture.json"
    capture = json.loads(path.read_text())
    capture["source_sha256"] = "different"
    path.write_text(json.dumps(capture))
    process = subprocess.run(
        [sys.executable, str(RUNNER), *args], capture_output=True, text=True
    )
    assert process.returncode != 0
    assert "capture identity" in process.stderr
    assert not (tmp_path / "result/candidate_seed.json").exists()
