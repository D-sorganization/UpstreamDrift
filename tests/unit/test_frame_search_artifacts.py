"""Unit tests for frame_search_artifacts and the manifest run-dir hook."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

ARTIFACTS_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "MachineLearning"
    / "frame_search_artifacts.py"
)
PREPARE_PATH = ARTIFACTS_PATH.parent / "prepare_frame_by_frame_search.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def artifacts():
    return _load("frame_search_artifacts", ARTIFACTS_PATH)


@pytest.fixture
def prepare():
    return _load("prepare_frame_by_frame_search", PREPARE_PATH)


def test_progress_csv_round_trip(tmp_path: Path, artifacts) -> None:
    progress = tmp_path / "progress.csv"
    artifacts.write_progress_header(progress)
    rows = [
        artifacts.ProgressRow(
            frame_idx=i,
            selected_candidate=i % 3,
            score=0.1 * i,
            wall_clock_s=1.5,
            timestamp=f"2026-05-05T00:00:0{i}",
        )
        for i in range(1, 4)
    ]
    for row in rows:
        artifacts.append_progress_row(progress, row)

    read_back = artifacts.read_progress(progress)
    assert read_back == rows


def test_checkpoint_manifest_match(tmp_path: Path, prepare, artifacts) -> None:
    # Build a minimal target CSV and column manifest.
    desired = tmp_path / "desired.csv"
    desired.write_text(
        "time,ClubLogs_CHGlobalPosition_1\n0.0,0.0\n0.01,0.1\n0.02,0.2\n",
        encoding="utf-8",
    )
    column_manifest = tmp_path / "column_manifest.json"
    column_manifest.write_text(
        json.dumps(
            {
                "input_columns": {
                    "applied_controls": ["LSLogs_ActuatorTorqueX"],
                }
            }
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "manifest.json"
    run_dir = tmp_path / "run"

    manifest = prepare.build_search_manifest(
        desired_target_csv=desired,
        column_manifest=column_manifest,
        output_json=output_json,
        torque_output_csv=tmp_path / "torques.csv",
        polynomial_output_mat=tmp_path / "polys.mat",
        run_dir=run_dir,
        checkpoint_interval_frames=5,
    )

    assert manifest["outputs"]["run_dir"] == str(run_dir)
    assert manifest["checkpoint"]["interval_frames"] == 5
    assert run_dir.is_dir()
    copy = run_dir / "manifest.json"
    assert copy.is_file()
    # The manifest copy in the run directory hashes to the same value as
    # the original on-disk manifest, so MATLAB and Python agree on the SHA.
    assert artifacts.manifest_sha256(copy) == artifacts.manifest_sha256(output_json)


def test_artifacts_reader_handles_partial_run(tmp_path: Path, artifacts) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    progress = run_dir / "progress.csv"
    artifacts.write_progress_header(progress)
    artifacts.append_progress_row(
        progress,
        artifacts.ProgressRow(1, 0, 0.5, 1.2, "2026-05-05T00:00:01"),
    )
    artifacts.append_progress_row(
        progress,
        artifacts.ProgressRow(2, 1, 0.4, 1.3, "2026-05-05T00:00:02"),
    )
    # Simulate a partially-flushed third row (truncated mid-line).
    with progress.open("a", encoding="utf-8") as handle:
        handle.write("3,2,not_a_number")

    status = artifacts.load_run_status(run_dir)
    assert status.last_frame_idx == 2
    assert status.has_checkpoint is False
    assert status.manifest_sha256 is None
    assert pytest.approx(status.total_wall_clock_s) == 2.5
    assert len(status.rows) == 2


def test_detect_stale_flags_old_progress(tmp_path: Path, artifacts) -> None:
    progress = tmp_path / "progress.csv"
    artifacts.write_progress_header(progress)
    artifacts.append_progress_row(
        progress,
        artifacts.ProgressRow(1, 0, 0.0, 1.0, "2026-05-05T00:00:01"),
    )
    mtime = progress.stat().st_mtime
    # Pretend "now" is well past the stale threshold.
    assert artifacts.detect_stale(
        progress,
        expected_frame_seconds=1.0,
        multiplier=2.0,
        now_seconds=mtime + 60.0,
    )
    assert not artifacts.detect_stale(
        progress,
        expected_frame_seconds=1.0,
        multiplier=2.0,
        now_seconds=mtime + 0.5,
    )


def test_detect_stale_rejects_nonpositive(tmp_path: Path, artifacts) -> None:
    progress = tmp_path / "progress.csv"
    progress.write_text("frame_idx\n", encoding="utf-8")
    with pytest.raises(ValueError):
        artifacts.detect_stale(progress, expected_frame_seconds=0.0)


def test_load_summary_missing_returns_empty(tmp_path: Path, artifacts) -> None:
    assert artifacts.load_summary(tmp_path) == {}


def test_load_summary_reads_json(tmp_path: Path, artifacts) -> None:
    (tmp_path / "summary.json").write_text(
        json.dumps({"total_frames": 42}), encoding="utf-8"
    )
    assert artifacts.load_summary(tmp_path) == {"total_frames": 42}


def test_load_run_status_with_stale_check(
    tmp_path: Path, artifacts, monkeypatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    progress = run_dir / "progress.csv"
    artifacts.write_progress_header(progress)
    artifacts.append_progress_row(
        progress,
        artifacts.ProgressRow(1, 0, 0.0, 1.0, "2026-05-05T00:00:01"),
    )
    # Backdate mtime to force stale detection.
    old = time.time() - 600.0
    import os

    os.utime(progress, (old, old))
    status = artifacts.load_run_status(
        run_dir, expected_frame_seconds=1.0, stale_lock_multiplier=2.0
    )
    assert status.is_stale is True
