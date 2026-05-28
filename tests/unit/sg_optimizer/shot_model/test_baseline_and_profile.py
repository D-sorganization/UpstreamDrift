"""Baseline YAML loading + PlayerProfile round-trip."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.sg_optimizer.shot_model.baseline import load_baseline
from src.shared.python.sg_optimizer.shot_model.player_profile import (
    ClubSkill,
    PlayerProfile,
    PuttingSkill,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
PGA_TOUR_YAML = REPO_ROOT / "data" / "sg_optimizer" / "baselines" / "pga_tour.yaml"


def test_load_pga_tour_baseline():
    bag = load_baseline(PGA_TOUR_YAML)
    assert "driver" in bag.clubs
    assert "7_iron" in bag.clubs
    driver = bag.get("driver")
    assert driver.carry_mean > 250
    assert driver.sigma_lat > driver.sigma_long  # tour drivers spray lateral


def test_unknown_club_raises():
    bag = load_baseline(PGA_TOUR_YAML)
    with pytest.raises(ContractViolationError):
        bag.get("nonexistent")


def test_effective_distribution_scales_sigma():
    bag = load_baseline(PGA_TOUR_YAML)
    profile = PlayerProfile(
        name="test",
        baseline=str(PGA_TOUR_YAML),
        clubs={"7_iron": ClubSkill(skill_mult_long=1.5, skill_mult_lat=2.0)},
    )
    base = bag.get("7_iron").distribution()
    eff = profile.effective_distribution("7_iron", bag)
    assert eff.sigma_long == pytest.approx(base.sigma_long * 1.5)
    assert eff.sigma_lat == pytest.approx(base.sigma_lat * 2.0)


def test_profile_yaml_round_trip(tmp_path):
    p = PlayerProfile(
        name="dieter",
        baseline=str(PGA_TOUR_YAML),
        clubs={"driver": ClubSkill(skill_mult_lat=1.2, distance_offset=-15.0)},
        putting=PuttingSkill(
            make_pct_multipliers={3.0: 0.95, 8.0: 1.0, 25.0: 1.05},
            three_putt_avoidance=0.9,
        ),
        last_updated=datetime(2026, 5, 26, tzinfo=timezone.utc),
        notes="round-trip",
    )
    out = tmp_path / "profile.yaml"
    p.to_yaml(out)
    q = PlayerProfile.from_yaml(out)
    assert q.name == "dieter"
    assert q.clubs["driver"].skill_mult_lat == pytest.approx(1.2)
    assert q.clubs["driver"].distance_offset == pytest.approx(-15.0)
    assert q.putting.make_pct_multipliers[8.0] == pytest.approx(1.0)
    assert q.putting.three_putt_avoidance == pytest.approx(0.9)
    assert q.notes == "round-trip"


def test_club_skill_rejects_zero_mult():
    with pytest.raises(ContractViolationError):
        ClubSkill(skill_mult_long=0.0)
