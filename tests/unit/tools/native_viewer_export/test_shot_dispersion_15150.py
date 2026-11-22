"""Shot dispersion from small random variations (#15150), no MuJoCo needed."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.tools.native_viewer_export import shot_dispersion as sd
from src.tools.native_viewer_export.shot_dispersion import (
    DispersionSettings,
    SampleOutcome,
    ShotSetup,
)

pytestmark = pytest.mark.unit

SIDE_PER_DEG = 6.6
CARRY_PER_M = -20.0


def _settings(**over):
    base = {
        "n_shots": 10,
        "seed": 0,
        "club_roll_sd_deg": 0.5,
        "ball_position_sd_m": 0.005,
    }
    return DispersionSettings(**{**base, **over})


def _trace(carry: float, side: float) -> np.ndarray:
    t = np.linspace(0.0, 1.0, 5)
    return np.column_stack([carry * t, side * t, 30.0 * t * (1.0 - t)])


def _linear(setup: ShotSetup) -> SampleOutcome:
    side = SIDE_PER_DEG * setup.club_roll_deg
    carry = 250.0 + CARRY_PER_M * setup.ball_shift_m
    metrics = dict.fromkeys(sd.METRICS, 1.0)
    metrics.update(carry_m=carry, side_m=side)
    return SampleOutcome(setup, None, metrics, _trace(carry, side))


def test_same_seed_identical_and_different_seed_differs():
    a, b = sd.sample_setups(_settings()), sd.sample_setups(_settings())
    assert a == b
    assert a != sd.sample_setups(_settings(seed=1))


def test_sample_count_and_spread():
    setups = sd.sample_setups(_settings(n_shots=400, club_roll_sd_deg=0.5))
    assert len(setups) == 400
    assert np.std([s.club_roll_deg for s in setups]) == pytest.approx(0.5, rel=0.15)


def test_zero_sd_gives_base():
    cfg = _settings(
        club_roll_sd_deg=0.0,
        ball_position_sd_m=0.0,
        base_club_roll_deg=-2.0,
        base_ball_shift_m=0.01,
    )
    assert set(sd.sample_setups(cfg)) == {ShotSetup(-2.0, 0.01)}


def test_values_clipped_to_limits():
    setups = sd.sample_setups(_settings(club_roll_sd_deg=1000.0, n_shots=50))
    assert max(abs(s.club_roll_deg) for s in setups) <= 20.0


@pytest.mark.parametrize(
    "bad",
    [
        {"n_shots": 0},
        {"club_roll_sd_deg": -1.0},
        {"ball_position_sd_m": float("nan")},
        {"base_club_roll_deg": 99.0},
        {"ellipse_sigma": 0.0},
    ],
)
def test_invalid_settings_raise(bad):
    with pytest.raises(ValueError):
        _settings(**bad)


def test_packaged_settings_load():
    cfg = sd.load_dispersion_settings()
    assert (cfg.n_shots, cfg.seed) == (10, 0)
    raw = json.loads(sd.SETTINGS_PATH.read_text())
    assert "not a measurement" in raw["sources"]["club_roll_sd_deg"]


def test_summary_and_sensitivity_recover_linear_model():
    setups = sd.sample_setups(_settings(n_shots=50))
    result = sd.run_dispersion(None, None, setups, evaluate=_linear)
    sides = [SIDE_PER_DEG * s.club_roll_deg for s in setups]
    summary = result.summary()
    assert summary["side_m"]["mean"] == pytest.approx(np.mean(sides))
    assert summary["side_m"]["std"] == pytest.approx(np.std(sides))
    slopes = result.sensitivity()
    assert slopes["side_m"]["club_roll_deg"] == pytest.approx(SIDE_PER_DEG)
    assert slopes["carry_m"]["ball_shift_m"] == pytest.approx(CARRY_PER_M)
    assert slopes["side_m"]["ball_shift_m"] == pytest.approx(0.0, abs=1e-6)
    assert json.dumps(result.to_dict())


def test_unavailable_shot_is_kept_with_reason():
    def evaluate(setup):
        if setup.club_roll_deg > 0:
            return SampleOutcome(setup, reason="not shown, test")
        return _linear(setup)

    setups = sd.sample_setups(_settings())
    result = sd.run_dispersion(None, None, setups, evaluate=evaluate)
    assert len(result.outcomes) == len(setups)
    failed = [s for s in result.to_dict()["shots"] if not s["available"]]
    assert failed and all(s["reason"] == "not shown, test" for s in failed)


def test_slope_degenerate_and_empty():
    assert sd.slopes({"a": [1.0, 1.0]}, [2.0, 3.0]) == {"a": None}
    with pytest.raises(ValueError):
        sd.slopes({"a": [1.0]}, [1.0, 2.0])
    with pytest.raises(ValueError):
        sd.run_dispersion(None, None, [], evaluate=_linear)


def test_ellipse_axes_match_covariance():
    rng = np.random.default_rng(3)
    pts = rng.standard_normal((4000, 2)) * [4.0, 1.0]
    centre, width, height, angle = sd.landing_ellipse(pts, 1.0)
    assert width == pytest.approx(8.0, rel=0.1)
    assert height == pytest.approx(2.0, rel=0.1)
    assert abs(angle) < 5.0 or abs(abs(angle) - 180.0) < 5.0
    assert sd.landing_ellipse(pts[:2], 1.0) is None


def test_plot_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    from src.tools.native_viewer_export.shot_dispersion_plot import draw_dispersion

    cfg = sd.load_dispersion_settings()
    result = sd.run_dispersion(None, None, sd.sample_setups(cfg), evaluate=_linear)
    out = draw_dispersion(result, cfg, tmp_path / "d.png")
    assert out.stat().st_size > 1000


def test_plot_refuses_when_no_flight(tmp_path):
    pytest.importorskip("matplotlib")
    from src.tools.native_viewer_export.shot_dispersion_plot import draw_dispersion

    cfg = sd.load_dispersion_settings()
    result = sd.run_dispersion(
        None, None, [ShotSetup(0.0, 0.0)], evaluate=lambda s: SampleOutcome(s, "x")
    )
    with pytest.raises(ValueError):
        draw_dispersion(result, cfg, tmp_path / "d.png")
