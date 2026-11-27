"""Real impact-card output retains its units at the playback boundary (#15524)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.impact_parameters import ClubheadSeries
from src.shared.python.impact_parameters.extract import MPS_TO_MPH
from src.tools.model_explorer.shot_card_rows import (
    load_card_settings,
    shot_rows,
    summary_from_shot,
)
from src.tools.native_viewer_export import impact_factors
from src.tools.native_viewer_export.ball import BallPlacement
from src.tools.native_viewer_export.display_units import DisplayUnits

pytestmark = [pytest.mark.integration, pytest.mark.headless_safe]
SPEED_MPS = 40.0
CONTACT_S = 0.1 + 0.4 / 360.0
RADIUS_M = 0.021335


def _panel(speed_unit: str, miss_m: float = 0.0) -> impact_factors.FactorsPanel:
    # An analytic face crosses the ball between samples; no capture or engine.
    times = np.arange(72) / 360.0
    centre = np.column_stack(
        [
            SPEED_MPS * (times - CONTACT_S),
            times * 0,
            times * 0,
        ]
    )
    series = ClubheadSeries(
        times_s=times,
        face_center_m=centre,
        velocity_mps=np.tile([SPEED_MPS, 0.0, 0.0], (len(times), 1)),
        face_normal=np.tile([1.0, 0.0, 0.0], (len(times), 1)),
        toe_axis=np.tile([0.0, -1.0, 0.0], (len(times), 1)),
        grip_axis=np.tile([0.0, 0.0, 1.0], (len(times), 1)),
    )
    ball = BallPlacement((RADIUS_M, miss_m, 0.0), RADIUS_M, "synthetic")
    return impact_factors.panel_from_series(
        series,
        CONTACT_S,
        ball,
        impact_factors.load_factor_settings(),
        speed_unit=speed_unit,
    )


@pytest.mark.parametrize("input_unit", ["mph", "m/s"])
@pytest.mark.parametrize(
    ("output_unit", "expected"), [("mph", "89 mph"), ("m/s", "40 m/s")]
)
def test_real_impact_speed_survives_playback_unit_selection(
    input_unit: str,
    output_unit: str,
    expected: str,
) -> None:
    panel = _panel(input_unit)
    summary = summary_from_shot(panel, None)
    assert summary.contact
    assert summary.club_speed_mph == pytest.approx(SPEED_MPS * MPS_TO_MPH)
    rows = shot_rows(summary, DisplayUnits("m", output_unit), load_card_settings())
    values = {row.label: row.value for row in rows}
    assert values["Club Speed"] == expected
    assert summary.attack_angle_deg == pytest.approx(0.0, abs=1e-10)
    assert summary.face_to_path_deg == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("input_unit", ["mph", "m/s"])
def test_missed_ball_keeps_club_speed_without_launch_rows(input_unit: str) -> None:
    summary = summary_from_shot(_panel(input_unit, miss_m=0.2), None)
    assert not summary.contact
    rows = shot_rows(summary, DisplayUnits(), load_card_settings())
    values = {row.label: row.value for row in rows}
    assert values["Club Speed"] == "89 mph"
    assert values["Ball Flight"] == "no contact: no flight"
    assert "Ball Speed" not in values


def test_unavailable_impact_speed_stays_blank() -> None:
    panel = impact_factors.no_impact_panel(impact_factors.load_factor_settings())
    summary = summary_from_shot(panel, None)
    assert summary.club_speed_mph is None
    rows = shot_rows(summary, DisplayUnits(), load_card_settings())
    assert {row.label: row.value for row in rows}["Club Speed"] == "--"
