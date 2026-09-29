from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tools.launch_monitor_model import (
    ColumnMapping,
    ImportOptions,
    LaunchMonitorProject,
    detect_profile,
    import_session,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

FIXTURES = Path(__file__).parents[2] / "fixtures" / "launch_monitor"


@pytest.mark.parametrize(
    ("filename", "profile"),
    [
        ("trackman.csv", "trackman"),
        ("foresight.csv", "foresight"),
        ("flightscope.csv", "flightscope"),
        ("garmin.csv", "garmin"),
        ("skytrak.csv", "skytrak"),
        ("uneekor.csv", "uneekor"),
    ],
)
def test_detects_vendor_profiles(filename: str, profile: str) -> None:
    headers = pd.read_csv(FIXTURES / filename, nrows=0).columns.tolist()
    result = detect_profile(headers)
    assert result.profile_id == profile
    assert result.confidence >= 0.5


def test_trackman_import_converts_units_and_preserves_provenance() -> None:
    session = import_session(FIXTURES / "trackman.csv")
    shots = session.shots
    assert session.manifest.profile_id == "trackman"
    assert session.manifest.file_sha256
    assert session.manifest.row_count == 2
    assert shots.loc[0, "club_speed"] == pytest.approx(88 * 0.44704)
    assert shots.loc[0, "carry_distance"] == pytest.approx(170 * 0.9144)
    assert shots.loc[0, "attack_angle"] == pytest.approx(np.deg2rad(-3.5))
    assert shots.loc[0, "spin_rate"] == pytest.approx(6100 * 2 * np.pi / 60)
    assert shots.loc[0, "source_row"] == 2
    assert "source::Club Speed (mph)" in shots.columns
    assert shots.loc[0, "source::Club Speed (mph)"] == 88
    assert session.manifest.metric_sources["club_speed"] == "Club Speed (mph)"
    assert set(shots["status::club_speed"]) == {"reported"}


def test_generic_json_import_uses_explicit_mapping(tmp_path: Path) -> None:
    source = tmp_path / "shots.json"
    source.write_text(
        json.dumps([{"speed": 100.0, "launch": 12.0, "note": "fit"}]),
        encoding="utf-8",
    )
    options = ImportOptions(
        profile_id="generic",
        mappings=(
            ColumnMapping("speed", "ball_speed", "mph"),
            ColumnMapping("launch", "launch_angle", "deg"),
        ),
        session_name="Mapped JSON",
    )
    session = import_session(source, options)
    assert session.shots.loc[0, "ball_speed"] == pytest.approx(44.704)
    assert session.shots.loc[0, "launch_angle"] == pytest.approx(np.deg2rad(12))
    assert session.shots.loc[0, "source::note"] == "fit"


def test_gspro_open_connect_nested_json_is_flattened(tmp_path: Path) -> None:
    source = tmp_path / "gspro.json"
    source.write_text(
        json.dumps(
            {
                "DeviceID": "Test Device",
                "Units": "Yards",
                "BallData": {
                    "Speed": 150.0,
                    "HLA": 1.0,
                    "VLA": 12.0,
                    "TotalSpin": 2500.0,
                    "BackSpin": 2450.0,
                },
                "ClubData": {"Speed": 101.0, "AngleOfAttack": -1.5},
            }
        ),
        encoding="utf-8",
    )
    session = import_session(source)
    assert session.manifest.profile_id == "gspro"
    assert session.shots.loc[0, "ball_speed"] == pytest.approx(150 * 0.44704)
    assert session.shots.loc[0, "club_speed"] == pytest.approx(101 * 0.44704)
    assert "source::BallData.Speed" in session.shots


@pytest.mark.parametrize(
    ("headers", "profile"),
    [
        (
            [
                "Club Speed (mph)",
                "Ball Speed (mph)",
                "Face to Path (deg)",
                "Carry Distance (yd)",
            ],
            "full_swing",
        ),
        (
            [
                "Smash Factor",
                "Launch Direction (deg)",
                "Shot Type",
                "Carry Distance (yd)",
            ],
            "rapsodo",
        ),
    ],
)
def test_detects_additional_common_profiles(headers: list[str], profile: str) -> None:
    assert detect_profile(headers).profile_id == profile


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".xlsx"])
def test_generic_tabular_formats_use_same_mapping(tmp_path: Path, suffix: str) -> None:
    pytest.importorskip("openpyxl")
    source = tmp_path / f"shots{suffix}"
    frame = pd.DataFrame({"speed": [90.0, 91.0], "distance": [150.0, 152.0]})
    if suffix == ".xlsx":
        frame.to_excel(source, index=False)
    else:
        frame.to_csv(source, index=False, sep="\t" if suffix == ".tsv" else ",")
    session = import_session(
        source,
        ImportOptions(
            profile_id="generic",
            mappings=(
                ColumnMapping("speed", "club_speed", "mph"),
                ColumnMapping("distance", "carry_distance", "yd"),
            ),
        ),
    )
    assert len(session.shots) == 2
    assert session.shots.loc[0, "club_speed"] == pytest.approx(90 * 0.44704)


def test_project_aggregates_sessions_and_round_trips(tmp_path: Path) -> None:
    project = LaunchMonitorProject("Player Study")
    project.add_session(import_session(FIXTURES / "trackman.csv"))
    project.add_session(import_session(FIXTURES / "garmin.csv"))
    project.record_actions(({"action": "filter", "column": "club"},))
    combined = project.combined_shots()
    assert len(combined) == 4
    assert set(combined["monitor_vendor"]) == {"TrackMan", "Garmin"}

    destination = tmp_path / "study.lmproject"
    project.save(destination)
    restored = LaunchMonitorProject.load(destination)
    assert restored.name == "Player Study"
    assert len(restored.sessions) == 2
    assert restored.audit_log == [{"action": "filter", "column": "club"}]
    pd.testing.assert_frame_equal(
        restored.combined_shots().reset_index(drop=True),
        combined.reset_index(drop=True),
        check_dtype=False,
    )
