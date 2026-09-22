"""CO-00 workbook identity, event-label normalization and loader regressions."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.motion_matching.club_only.workbook_identity import (
    CLUB_DATA_SHA256,
    EXPECTED_SAMPLE_COUNTS,
    IDENTITY_SCHEMA,
    UNIT_AUTHORITY,
    WIFFLE_PROV1_SHA256,
    build_club_workbook_identity,
    count_numeric_samples,
    read_sheet_event_samples,
    verify_workbook_hash,
)
from src.shared.python.motion_matching.loaders.event_labels import (
    axis_component,
    normalize_event_label,
    parse_event_marker_cells,
)
from src.shared.python.motion_matching.loaders.excel import read_excel_event_markers
from src.engines.physics_engines.pinocchio.python.motion_training.club_trajectory_parser import (
    ClubTrajectoryParser,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
CLUB_DATA = REPO_ROOT / "data" / "Club_Data.xlsx"
WIFFLE = (
    REPO_ROOT / "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/"
    "golf_gui/Motion Capture Plotter/Wiffle_ProV1_club_3D_data.xlsx"
)


def test_normalize_event_label_accepts_bare_and_equals() -> None:
    assert normalize_event_label("A=") == "A"
    assert normalize_event_label("A") == "A"
    assert normalize_event_label(" T= ") == "T"
    assert normalize_event_label("CHS") == "CHS"
    assert normalize_event_label("CHS=") == "CHS"
    assert normalize_event_label("Wiffle ball") is None
    assert normalize_event_label(None) is None


def test_parse_event_marker_cells_a_equals_regression() -> None:
    parsed = parse_event_marker_cells(
        ["Wiffle ball", None, "A=", 240, "T=", 412, "I=", 519, "F=", 832, "CHS", 114.5]
    )
    assert parsed == {"A": 240.0, "T": 412.0, "I": 519.0, "F": 832.0, "CHS": 114.5}


def test_axis_component_preserves_real_zero() -> None:
    assert axis_component(0.0, default=1.0) == 0.0
    assert axis_component(None, default=1.0) == 1.0
    assert axis_component(float("nan"), default=0.0) == 0.0


def test_workbook_hashes_match_frozen_manifests() -> None:
    assert verify_workbook_hash(CLUB_DATA, CLUB_DATA_SHA256) == CLUB_DATA_SHA256
    assert verify_workbook_hash(WIFFLE, WIFFLE_PROV1_SHA256) == WIFFLE_PROV1_SHA256


def test_workbook_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    fake = tmp_path / "fake.xlsx"
    fake.write_bytes(b"not-a-workbook")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_workbook_hash(fake, CLUB_DATA_SHA256)


def test_trailing_blanks_excluded_from_sample_counts() -> None:
    for sheet, expected in EXPECTED_SAMPLE_COUNTS.items():
        if sheet == "Filtering Experiments":
            continue
        count = count_numeric_samples(CLUB_DATA, sheet)
        assert count == expected
        assert count < 885


def test_read_excel_event_markers_tw_wiffle_a_equals() -> None:
    markers = read_excel_event_markers(CLUB_DATA, "TW_wiffle")
    assert markers.A_sample == 240.0
    assert markers.T_sample == 412.0
    assert markers.I_sample == 519.0
    assert markers.F_sample == 832.0
    assert markers.CHS_mph == 114.5


def test_read_excel_event_markers_bare_labels() -> None:
    markers = read_excel_event_markers(CLUB_DATA, "TW_ProV1")
    assert markers.A_sample == 240.0
    assert markers.I_sample == 525.0


def test_shared_and_identity_events_agree_on_all_trials() -> None:
    for sheet in ("TW_wiffle", "TW_ProV1", "GW_wiffle", "GW_ProV11"):
        shared = read_sheet_event_samples(CLUB_DATA, sheet)
        excel = read_excel_event_markers(CLUB_DATA, sheet)
        assert shared["A"] == excel.A_sample
        assert shared["T"] == excel.T_sample
        assert shared["I"] == excel.I_sample
        assert shared["F"] == excel.F_sample
        assert shared["CHS"] == excel.CHS_mph


def test_pinocchio_parser_accepts_bare_and_equals_events() -> None:
    parser = ClubTrajectoryParser(CLUB_DATA)
    equals_events = parser._parse_events_list(
        ["Wiffle", None, "A=", 240, "T=", 412, "I=", 519, "F=", 832, "CHS", 114.5]
    )
    bare_events = parser._parse_events_list(
        ["ProV1", None, "A", 240, "T", 418, "I", 525, "F", 725, "CHS", 114.5]
    )
    assert equals_events.address == 240
    assert equals_events.impact == 519
    assert bare_events.address == 240
    assert bare_events.impact == 525


def test_build_club_workbook_identity_four_trial_table() -> None:
    identity = build_club_workbook_identity(REPO_ROOT)
    assert identity.schema == IDENTITY_SCHEMA
    assert len(identity.manifests) == 2
    assert all(m.verified for m in identity.manifests)
    assert len(identity.trials) == 4

    by_id = {trial.trial_id: trial for trial in identity.trials}
    assert by_id["TW_ProV1"].alias_sheets == ("Filtering Experiments",)
    assert by_id["TW_ProV1"].lineage_id == by_id["TW_ProV1"].lineage_id
    assert by_id["GW_wiffle"].ball_label is None
    assert by_id["GW_wiffle"].ball_label_status == "conflict_sheet_name_vs_a1"
    assert by_id["TW_wiffle"].event_samples["A"] == 240.0
    assert by_id["TW_wiffle"].time_range_s[0] == pytest.approx(-2.158333333333333)
    assert identity.units.to_meters_scale == UNIT_AUTHORITY.to_meters_scale
    assert identity.units.declared_units == "inches"
    assert identity.units.reviewed_units == "centimetres"
    assert identity.events.sample_rate_hz == 240.0
    assert identity.orientation.status_when_derived == "derived_not_measured"


def test_identity_write_json_round_trip(tmp_path: Path) -> None:
    identity = build_club_workbook_identity(REPO_ROOT)
    out = tmp_path / "club_workbook_identity.json"
    identity.write_json(out)
    payload = out.read_text(encoding="utf-8")
    assert IDENTITY_SCHEMA in payload
    assert CLUB_DATA_SHA256 in payload
