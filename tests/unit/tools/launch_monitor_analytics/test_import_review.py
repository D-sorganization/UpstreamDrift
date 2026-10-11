"""Tests for the Qt-free import-review helpers (issue #11987).

Extracted from ``src/tools/launch_monitor_analytics/widgets.py``'s
``ImportMappingDialog`` so the header-reading, auto-mapping and
``ImportOptions``-building logic can be exercised without PyQt6, and reused by
the web API's ``/v2/import/preview`` and ``/v2/import`` routes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tools.launch_monitor_analytics.import_review import (
    MEASUREMENT_STATUSES,
    RETAIN_ONLY,
    auto_mappings,
    build_import_options,
    mapping_targets,
    read_headers,
)
from src.tools.launch_monitor_model import IDENTITY_COLUMNS, METRICS

pytestmark = pytest.mark.unit

_FIXTURES = Path(__file__).parents[3] / "fixtures" / "launch_monitor"


def test_read_headers_returns_csv_columns() -> None:
    headers = read_headers(_FIXTURES / "trackman.csv")
    assert headers[:5] == [
        "Shot",
        "Date",
        "Time",
        "Club",
        "Club Speed (mph)",
    ]


def test_read_headers_rejects_unsupported_extension(tmp_path: Path) -> None:
    bogus = tmp_path / "shots.xyz"
    bogus.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        read_headers(bogus)


def test_mapping_targets_starts_with_retain_only_then_identity_then_metrics() -> None:
    targets = mapping_targets()
    assert targets[0] == RETAIN_ONLY
    assert targets[1 : 1 + len(IDENTITY_COLUMNS)] == list(IDENTITY_COLUMNS)
    assert "date" in targets
    assert "time" in targets
    assert targets[-len(METRICS) :] == list(METRICS)


def test_auto_mappings_maps_trackman_headers_to_canonical_targets() -> None:
    headers = read_headers(_FIXTURES / "trackman.csv")
    mapping = auto_mappings("trackman", headers)
    assert mapping["Club Speed (mph)"] == "club_speed"
    assert mapping["Carry (yd)"] == "carry_distance"


def test_auto_mappings_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="Unknown import profile"):
        auto_mappings("not-a-real-profile", ["Club Speed (mph)"])


def test_build_import_options_skips_retain_only_rows() -> None:
    options = build_import_options(
        profile_id="trackman",
        rows=[
            ("Club Speed (mph)", "club_speed", "", 1.0, "reported"),
            ("Notes", RETAIN_ONLY, "", 1.0, "reported"),
        ],
        session_name="",
        default_session_name="fallback",
        player="",
        monitor_model="",
        software_version="",
    )
    assert len(options.mappings) == 1
    assert options.mappings[0].source_column == "Club Speed (mph)"


def test_build_import_options_applies_multiplier_and_status() -> None:
    options = build_import_options(
        profile_id="trackman",
        rows=[("Club Speed (mph)", "club_speed", "mph", -1.0, "measured")],
        session_name="My Session",
        default_session_name="fallback",
        player="",
        monitor_model="",
        software_version="",
    )
    mapping = options.mappings[0]
    assert mapping.multiplier == -1.0
    assert mapping.measurement_status == "measured"
    assert mapping.source_unit == "mph"
    assert options.session_name == "My Session"


def test_build_import_options_blank_unit_becomes_none() -> None:
    options = build_import_options(
        profile_id="trackman",
        rows=[("Club Speed (mph)", "club_speed", "   ", 1.0, "reported")],
        session_name="",
        default_session_name="fallback",
        player="",
        monitor_model="",
        software_version="",
    )
    assert options.mappings[0].source_unit is None


def test_build_import_options_blank_session_name_falls_back_to_default() -> None:
    options = build_import_options(
        profile_id="trackman",
        rows=[],
        session_name="   ",
        default_session_name="fallback-name",
        player="",
        monitor_model="",
        software_version="",
    )
    assert options.session_name == "fallback-name"


def test_build_import_options_blank_optional_strings_become_none() -> None:
    options = build_import_options(
        profile_id="trackman",
        rows=[],
        session_name="session",
        default_session_name="fallback",
        player="  ",
        monitor_model="  ",
        software_version="  ",
    )
    assert options.player is None
    assert options.monitor_model is None
    assert options.software_version is None


def test_build_import_options_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="Unknown import profile"):
        build_import_options(
            profile_id="not-a-real-profile",
            rows=[],
            session_name="",
            default_session_name="fallback",
            player="",
            monitor_model="",
            software_version="",
        )


def test_build_import_options_rejects_unknown_target() -> None:
    with pytest.raises(ValueError, match="Unknown target column"):
        build_import_options(
            profile_id="trackman",
            rows=[("Club Speed (mph)", "not_a_real_target", "", 1.0, "reported")],
            session_name="",
            default_session_name="fallback",
            player="",
            monitor_model="",
            software_version="",
        )


def test_build_import_options_rejects_invalid_multiplier() -> None:
    with pytest.raises(ValueError, match="multiplier must be 1.0 or -1.0"):
        build_import_options(
            profile_id="trackman",
            rows=[("Club Speed (mph)", "club_speed", "", 2.0, "reported")],
            session_name="",
            default_session_name="fallback",
            player="",
            monitor_model="",
            software_version="",
        )


def test_build_import_options_rejects_invalid_measurement_status() -> None:
    with pytest.raises(ValueError, match="measurement_status must be one of"):
        build_import_options(
            profile_id="trackman",
            rows=[("Club Speed (mph)", "club_speed", "", 1.0, "bogus-status")],
            session_name="",
            default_session_name="fallback",
            player="",
            monitor_model="",
            software_version="",
        )


def test_measurement_statuses_matches_documented_five_values() -> None:
    assert MEASUREMENT_STATUSES == (
        "reported",
        "measured",
        "estimated",
        "derived",
        "unknown",
    )
