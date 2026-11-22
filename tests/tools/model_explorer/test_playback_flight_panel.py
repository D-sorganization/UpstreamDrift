"""Shot card of the live Results Playback: pure data, no Qt (#15150 G5b)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from src.tools.model_explorer import playback_flight_panel as shot
from src.tools.native_viewer_export.display_units import DisplayUnits

pytestmark = pytest.mark.unit

CFG = shot.load_card_settings()
FULL = shot.ShotSummary(
    "at impact", True, 114.7, 1.2, 4.7, 169.6, 10.1, 1815.7, -3.4, 222.5, 252.3,
    18.8, 26.9,
)  # fmt: skip


def _values(summary: shot.ShotSummary, *units: str) -> dict[str, str]:
    rows = shot.shot_rows(summary, DisplayUnits(*units), CFG)
    return {row.label: row.value for row in rows}


def test_rows_show_every_number_in_yards_and_mph_in_card_order() -> None:
    assert _values(FULL, "yd", "mph") == {
        "Status": "at impact",
        "Club Speed": "115 mph",
        "Face to Path": "+1.2 deg",
        "Attack Angle": "+4.7 deg",
        "Ball Speed": "170 mph",
        "Launch Angle": "10.1 deg",
        "Spin Rate": "1816 rpm",
        "Spin Axis": "3.4 deg L",
        "Carry": "243 yd",
        "Total": "276 yd",
        "Apex": "21 yd",
        "Side": "29 yd R",
    }
    sections = [r.section for r in shot.shot_rows(FULL, DisplayUnits(), CFG)]
    assert sections == sorted(sections, key=list(CFG.sections).index)


def test_rows_follow_the_metric_units() -> None:
    values = _values(FULL, "m", "m/s")
    assert values["Club Speed"] == "51 m/s" and values["Ball Speed"] == "76 m/s"
    assert values["Carry"] == "222 m" and values["Side"] == "27 m R"


def test_no_contact_and_missing_flight_replace_the_ball_rows() -> None:
    missed = _values(replace(FULL, contact=False), "yd", "mph")
    assert missed["Ball Flight"] == "no contact: no flight"
    assert "Carry" not in missed and missed["Club Speed"] == "115 mph"
    reason = "unavailable: upstream-physics Rust kernel not found"
    gone = _values(replace(FULL, flight_reason=reason), "yd", "mph")
    assert gone["Ball Flight"] == f"{CFG.flight_unavailable_text}: {reason}"
    assert "Ball Speed" not in gone and gone["Attack Angle"] == "+4.7 deg"
    assert _values(replace(FULL, total_m=None), "yd", "mph")["Total"] == "--"


def test_summary_reads_the_exporter_panel_and_flight() -> None:
    rows = (
        NS(key="clubhead_speed", value=114.7),
        NS(key="attack_angle_deg", value=None),
    )
    panel = NS(status="at impact", card=NS(rows=rows), verdict=NS(contact=True))
    flight = NS(
        path_m=np.zeros((2, 3)), status="model prediction", ball_speed_mph=170.0,
        launch_deg=10.0, spin_rpm=1800.0, spin_axis_deg=2.0, carry_m=220.0,
        apex_m=19.0, lateral_m=27.0, aim_lateral_m=None, rollout=NS(total_m=250.0),
    )  # fmt: skip
    summary = shot.summary_from_shot(panel, flight)
    assert summary.club_speed_mph == 114.7 and summary.attack_angle_deg is None
    assert (summary.total_m, summary.side_m, summary.flight_reason) == (250, 27, None)
    flight.path_m, flight.status = None, "no impact detected"
    assert shot.summary_from_shot(panel, flight).flight_reason == "no impact detected"
    panel.verdict = NS(contact=False)
    assert shot.summary_from_shot(panel, flight).contact is False
    assert shot.summary_from_shot(None, None).flight_reason


def test_contract_and_settings(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="units"):
        shot.shot_rows(FULL, "yd", CFG)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ShotSummary"):
        shot.shot_rows(None, DisplayUnits(), CFG)  # type: ignore[arg-type]
    bad = tmp_path / "card.json"
    bad.write_text('{"title": "Shot"}', encoding="utf-8")
    with pytest.raises(ValueError, match="card settings"):
        shot.load_card_settings(bad)
    assert shot.club_label(b'{"club": {"name": "iron7"}}', CFG) == "7-Iron"
    assert shot.club_label(b"{}", CFG) == CFG.default_club


def test_a_non_bundle_run_says_the_card_needs_a_bundle(tmp_path: Path) -> None:
    record = tmp_path / "dynamics_record.npz"
    np.savez(record, q=np.zeros((3, 2)), time_s=np.arange(3.0))
    summary = shot.summary_for_path(record)
    assert summary.club_speed_mph is None and summary.status == CFG.needs_bundle_text
