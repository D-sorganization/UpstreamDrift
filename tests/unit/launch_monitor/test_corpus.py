"""Tests for the private-corpus loader over synthetic Parquet fixtures."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from src.shared.python.launch_monitor.corpus import (
    corpus_dataset_path,
    load_private_corpus,
)


pytestmark = pytest.mark.unit


def _synthetic_checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "checkout"
    dataset = checkout / "data" / "authority" / "database" / "shot_corpus_parquet"
    rows = pd.DataFrame(
        {
            "monitor": ["TrackMan", "FlightScope Mevo+"],
            "file": ["a.csv", "b.csv"],
            "row_index": [0, 0],
            "club": ["Driver", "7 Iron"],
            "club_speed_mph": [100.0, 80.0],
            "ball_speed_mph": [150.0, 110.0],
            "smash_factor": [1.5, 1.375],
            "launch_angle_deg": [12.0, 18.0],
            "launch_direction_deg": [1.0, -0.5],
            "spin_rate_rpm": [2700.0, 6500.0],
            "back_spin_rpm": [2600.0, 6400.0],
            "side_spin_rpm": [300.0, -200.0],
            "spin_axis_deg": [4.0, -2.0],
            "attack_angle_deg": [-1.2, -4.0],
            "club_path_deg": [0.5, 1.5],
            "face_angle_deg": [0.2, 0.8],
            "carry_yd": [250.0, 165.0],
            "total_yd": [270.0, 172.0],
            "apex_native": [95.0, 28.0],
            "descent_angle_deg": [38.0, 45.0],
            "native_json": ["{}", "{}"],
        }
    )
    for source_id, group in (
        ("synthetic_trackman", rows.iloc[:1]),
        ("synthetic_mevo", rows.iloc[1:]),
    ):
        partition = dataset / f"source_id={source_id}"
        partition.mkdir(parents=True)
        group.to_parquet(partition / "part-0.parquet", index=False)
    return checkout


def test_load_private_corpus_converts_to_canonical_units(tmp_path: Path) -> None:
    frame = load_private_corpus(root=_synthetic_checkout(tmp_path))

    assert len(frame) == 2
    row = frame.set_index("session_id").loc["synthetic_trackman"]
    assert row["ball_speed"] == pytest.approx(150.0 * 0.44704)
    assert row["launch_angle"] == pytest.approx(math.radians(12.0))
    assert row["spin_rate"] == pytest.approx(2700.0 * math.pi / 30.0)
    assert row["carry_distance"] == pytest.approx(250.0 * 0.9144)
    assert row["monitor_vendor"] == "TrackMan"
    assert row["observation_kind"] == "shot"
    assert "apex_native" not in frame.columns
    assert frame["shot_id"].nunique() == 2


def test_source_and_metric_selection(tmp_path: Path) -> None:
    checkout = _synthetic_checkout(tmp_path)
    frame = load_private_corpus(
        root=checkout,
        sources=["synthetic_mevo"],
        metrics=["ball_speed", "carry_distance"],
    )
    assert set(frame["session_id"].astype(str)) == {"synthetic_mevo"}
    assert "ball_speed" in frame.columns
    assert "spin_rate" not in frame.columns
    with pytest.raises(ValueError, match="Unknown corpus sources"):
        load_private_corpus(root=checkout, sources=["nope"])
    with pytest.raises(ValueError, match="Unknown corpus metrics"):
        load_private_corpus(root=checkout, metrics=["warp_speed"])


def test_missing_root_and_missing_dataset_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LAUNCH_MONITOR_DATA_ROOT", raising=False)
    with pytest.raises(FileNotFoundError, match="LAUNCH_MONITOR_DATA_ROOT"):
        corpus_dataset_path()
    with pytest.raises(FileNotFoundError, match="shot corpus dataset not found"):
        load_private_corpus(root=tmp_path / "empty")


def test_lateral_flight_and_capture_columns_reach_canonical_schema(
    tmp_path: Path,
) -> None:
    """The #18/#19 corpus columns convert into the canonical schema."""
    checkout = tmp_path / "checkout"
    dataset = checkout / "data" / "authority" / "database" / "shot_corpus_parquet"
    partition = dataset / "source_id=synthetic_new"
    partition.mkdir(parents=True)
    rows = pd.DataFrame(
        {
            "monitor": ["TrackMan"],
            "file": ["a.csv"],
            "row_index": [0],
            "club": ["Driver"],
            "club_speed_mph": [100.0],
            "ball_speed_mph": [150.0],
            "smash_factor": [1.5],
            "launch_angle_deg": [12.0],
            "launch_direction_deg": [1.0],
            "spin_rate_rpm": [2700.0],
            "back_spin_rpm": [2600.0],
            "side_spin_rpm": [300.0],
            "spin_axis_deg": [4.0],
            "attack_angle_deg": [-1.2],
            "club_path_deg": [0.5],
            "face_angle_deg": [0.2],
            "carry_yd": [250.0],
            "total_yd": [270.0],
            "apex_native": [95.0],
            "descent_angle_deg": [38.0],
            "lateral_carry_yd": [-12.5],
            "flight_time_s": [6.2],
            "captured_at": ["2023-08-07T00:00:00"],
            "native_json": ["{}"],
        }
    )
    rows.to_parquet(partition / "part-0.parquet", index=False)

    frame = load_private_corpus(root=checkout)

    row = frame.iloc[0]
    assert row["lateral_carry"] == pytest.approx(-12.5 * 0.9144)  # yards -> m
    assert row["flight_time"] == pytest.approx(6.2)
    assert row["captured_at"] == "2023-08-07T00:00:00"


def test_corpus_predating_the_new_columns_still_loads(tmp_path: Path) -> None:
    """An older pinned corpus lacks the columns; the loader must not fail."""
    frame = load_private_corpus(root=_synthetic_checkout(tmp_path))

    assert len(frame) == 2
    assert "lateral_carry" not in frame.columns
    assert "captured_at" not in frame.columns
