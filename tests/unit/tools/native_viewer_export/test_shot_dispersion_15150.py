"""Shot dispersion core (#15150), no MuJoCo needed."""

from __future__ import annotations

import numpy as np
import pytest

from src.tools.native_viewer_export import shot_dispersion as sd
from src.tools.native_viewer_export.shot_dispersion import (
    DispersionSettings,
    SampleOutcome,
    ShotSetup,
)

pytestmark = pytest.mark.unit


def _cfg(**over):
    base = {"n_shots": 20, "seed": 0, "club_roll_sd_deg": 0.5}
    return DispersionSettings(**{"ball_position_sd_m": 0.005, **base, **over})


def _linear(setup: ShotSetup) -> SampleOutcome:
    metrics = dict.fromkeys(sd.METRICS, 1.0)
    metrics.update(
        side_m=6.6 * setup.club_roll_deg, carry_m=250 - 20 * setup.ball_shift_m
    )
    trace = np.zeros((3, 3))
    return SampleOutcome(setup, None, metrics, trace)


def test_seed_determinism_count_and_zero_sd():
    assert sd.sample_setups(_cfg()) == sd.sample_setups(_cfg())
    assert sd.sample_setups(_cfg()) != sd.sample_setups(_cfg(seed=1))
    assert len(sd.sample_setups(_cfg())) == 20
    flat = _cfg(club_roll_sd_deg=0.0, ball_position_sd_m=0.0, base_club_roll_deg=-2.0)
    assert set(sd.sample_setups(flat)) == {ShotSetup(-2.0, 0.0)}


def test_invalid_settings_raise():
    for bad in ({"n_shots": 0}, {"club_roll_sd_deg": -1.0}, {"base_club_roll_deg": 99}):
        with pytest.raises(ValueError):
            _cfg(**bad)


def test_joint_fit_recovers_slopes_and_keeps_failed_shot():
    setups = sd.sample_setups(_cfg())

    def run(setup: ShotSetup) -> SampleOutcome:
        return (
            SampleOutcome(setup, "no flight") if setup is setups[0] else _linear(setup)
        )

    result = sd.run_dispersion(None, None, setups, evaluate=run)
    slopes = result.sensitivity()
    assert slopes["side_m"]["club_roll_deg"] == pytest.approx(6.6)
    assert slopes["side_m"]["ball_shift_m"] == pytest.approx(0.0, abs=1e-6)
    assert slopes["carry_m"]["ball_shift_m"] == pytest.approx(-20.0)
    shots = result.to_dict()["shots"]
    assert len(shots) == 20 and shots[0]["reason"] == "no flight"
