"""Monte Carlo make-probability physics tests (epic #14141, putting v4 step 9b).

Invariants only, never a snapshot of this model's own output. Kept
small (``n_trials``, distances, flat greens) so every test stays well
under the 20 s budget; bounded-iteration ``solve_aim`` dominates cost.
"""

from __future__ import annotations

import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.putting_dynamics import load_reference_fixtures
from src.shared.python.sg_optimizer.shot_model.putting_physics import (
    StrokeNoise,
    simulated_make_probability,
    variance_attribution,
)

pytestmark = pytest.mark.unit

#: Not calibrated against any measured player or tour data (see the
#: xfail test at the bottom of this file).
_NOISE = StrokeNoise(face_sd_deg=3.0, speed_cv=0.08, read_sd_deg=2.0)
_ZERO_NOISE = StrokeNoise(face_sd_deg=0.0, speed_cv=0.0, read_sd_deg=0.0)


def test_zero_noise_holes_with_probability_one_on_flat_green() -> None:
    p = simulated_make_probability(6.0, _ZERO_NOISE, slope_pct=0.0, n_trials=4, seed=0)
    assert p == 1.0


def test_make_probability_non_increasing_with_distance_common_random_numbers() -> None:
    p_near = simulated_make_probability(4.0, _NOISE, slope_pct=0.0, n_trials=8, seed=7)
    p_far = simulated_make_probability(12.0, _NOISE, slope_pct=0.0, n_trials=8, seed=7)
    assert p_far <= p_near


def test_fixed_seed_is_deterministic() -> None:
    first = simulated_make_probability(6.0, _NOISE, slope_pct=0.0, n_trials=4, seed=3)
    second = simulated_make_probability(6.0, _NOISE, slope_pct=0.0, n_trials=4, seed=3)
    assert first == second


def test_larger_face_sd_gives_lower_or_equal_make_probability() -> None:
    small_face = StrokeNoise(face_sd_deg=1.0, speed_cv=0.08, read_sd_deg=2.0)
    big_face = StrokeNoise(face_sd_deg=6.0, speed_cv=0.08, read_sd_deg=2.0)
    p_small = simulated_make_probability(
        8.0, small_face, slope_pct=0.0, n_trials=8, seed=1
    )
    p_big = simulated_make_probability(8.0, big_face, slope_pct=0.0, n_trials=8, seed=1)
    assert p_big <= p_small


@pytest.mark.parametrize("noise", [_NOISE, _ZERO_NOISE])
def test_variance_attribution_shares_in_unit_interval_and_sum_to_one(
    noise: StrokeNoise,
) -> None:
    # Zero noise exercises the "no channel shows a loss" all-0.0 fallback
    # (dividing by a zero total would otherwise raise).
    shares = variance_attribution(6.0, noise, slope_pct=0.0, n_trials=4, seed=2)
    assert set(shares) == {"face", "speed", "read"}
    assert all(0.0 <= share <= 1.0 for share in shares.values())
    total = sum(shares.values())
    assert total == pytest.approx(1.0) or total == pytest.approx(0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"distance_ft": 0.0, "noise": _NOISE},
        {"distance_ft": 6.0, "noise": _NOISE, "n_trials": 0},
        {"distance_ft": 6.0, "noise": _NOISE, "stimp_ft": -1.0},
    ],
)
def test_invalid_inputs_raise_value_error(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        simulated_make_probability(**kwargs)


@pytest.mark.parametrize(
    "bad_noise_kwargs",
    [
        {"face_sd_deg": -1.0, "speed_cv": 0.0, "read_sd_deg": 0.0},
        {"face_sd_deg": 0.0, "speed_cv": 0.0, "read_sd_deg": float("nan")},
    ],
)
def test_stroke_noise_rejects_invalid_channels(bad_noise_kwargs: dict) -> None:
    with pytest.raises((ValueError, ContractViolationError)):
        StrokeNoise(**bad_noise_kwargs)


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Fixture 'broadie2012_shotlink_50pct_putt_distance_ft' "
        "(src/config/putting_reference_fixtures.json) records the tour "
        "one-putt-50% distance; no calibrated tour StrokeNoise profile "
        "exists yet, so this guess cannot be expected to match it."
    ),
)
def test_make_probability_at_broadie_distance_is_uncalibrated() -> None:
    fixtures = {f.fixture_id: f for f in load_reference_fixtures()}
    fixture = fixtures["broadie2012_shotlink_50pct_putt_distance_ft"]
    distance_ft = float(fixture.expected.value)
    guess_noise = StrokeNoise(face_sd_deg=1.5, speed_cv=0.05, read_sd_deg=1.0)
    p = simulated_make_probability(
        distance_ft, guess_noise, slope_pct=0.0, n_trials=8, seed=0
    )
    assert p == pytest.approx(0.5, abs=0.02)
