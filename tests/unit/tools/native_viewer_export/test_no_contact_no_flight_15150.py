"""One no-contact decision gates the panel and the flight (#15150, C22)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.impact_parameters import ClubheadSeries
from src.tools.native_viewer_export import (
    contact_verdict,
    flight_tracer as tracer,
    impact_factors as factors,
)
from src.tools.native_viewer_export.ball import BallPlacement

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

RATE_HZ, RADIUS_M, SPEED_MPS, CONTACT_S = 360.0, 0.021335, 40.0, 0.1
DEFAULTS = tracer.load_tracer_settings()
PANEL = factors.load_factor_settings()


def _series() -> ClubheadSeries:
    t = np.arange(0.0, 0.2, 1.0 / RATE_HZ)
    centre = np.column_stack([SPEED_MPS * (t - CONTACT_S), 0 * t, 0 * t])
    return ClubheadSeries(
        times_s=t,
        face_center_m=centre,
        velocity_mps=np.gradient(centre, t, axis=0),
        face_normal=np.tile([1.0, 0.0, 0.0], (len(t), 1)),
        toe_axis=np.tile([0.0, -1.0, 0.0], (len(t), 1)),
        grip_axis=np.tile([0.0, 0.0, 1.0], (len(t), 1)),
    )


def _ball(offset_m: float) -> BallPlacement:
    return BallPlacement((RADIUS_M, offset_m, 0.0), RADIUS_M, "test")


def test_verdict_reads_the_card_location_row() -> None:
    assert contact_verdict.verdict_for(_series(), CONTACT_S, _ball(0.0)).contact
    miss = contact_verdict.verdict_for(_series(), CONTACT_S, _ball(0.2))
    assert not miss.contact
    assert contact_verdict.verdict_for(_series(), CONTACT_S, None).contact


def test_panel_and_flight_share_the_decision() -> None:
    series, ball = _series(), _ball(0.2)
    panel = factors.panel_from_series(series, CONTACT_S, ball, PANEL)
    assert "no contact" in panel.status and not panel.verdict.contact
    play = tracer.playback_from_series(series, CONTACT_S, ball, DEFAULTS)
    assert play.path_m is None
    assert play.status == tracer.NO_CONTACT_STATUS
    assert tracer.NO_CONTACT_STATUS == "no contact: no flight"
    assert tracer.manifest_record(SimpleNamespace(flight=play))["unavailable"] == (
        tracer.NO_CONTACT_STATUS
    )


def test_a_struck_ball_still_flies() -> None:
    play = tracer.playback_from_series(_series(), CONTACT_S, _ball(0.0), DEFAULTS)
    assert play.path_m is not None


def test_flight_for_export_reads_a_given_verdict(monkeypatch) -> None:
    def boom(_swing):
        raise AssertionError("the decision must not be recomputed")

    monkeypatch.setattr(tracer, "club_series", boom)
    swing = SimpleNamespace(
        ball=_ball(0.0), has_impact_provenance=True, impact_time_s=CONTACT_S
    )
    settings = SimpleNamespace(ball_flight=DEFAULTS, impact_time_s=None)
    miss = contact_verdict.ContactVerdict(contact=False)
    play = tracer.flight_for_export(swing, settings, miss)
    assert play.path_m is None and play.status == tracer.NO_CONTACT_STATUS


def test_strike_fallback_names_the_address_ball(monkeypatch) -> None:
    from src.tools.native_viewer_export import ball as ball_mod

    class Boom:
        def __init__(self, *_a):
            raise ValueError("face never passes the ball")

    monkeypatch.setattr(ball_mod, "ClubPoseModel", Boom)
    nominal = BallPlacement((0.0, 0.0, RADIUS_M), RADIUS_M, ball_mod.SOURCE_ADDRESS)
    swing = SimpleNamespace(bundle=SimpleNamespace(spec_bytes=b"{}"))
    kept, record = ball_mod.strike_ball(swing, nominal, CONTACT_S)
    assert kept is nominal or kept.position_m == nominal.position_m
    assert record["fallback"] == ball_mod.FALLBACK_ADDRESS
    assert "face never passes the ball" in record["unavailable"]
