"""CourseConditions invariants and preset behaviour."""

from __future__ import annotations

import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.sg_optimizer.course.conditions import (
    CourseConditions,
    GreenModel,
    RoughModel,
    TreeModel,
)


def test_rough_distance_decreases_with_severity():
    light = RoughModel.preset("light").distance_multiplier()
    heavy = RoughModel.preset("heavy").distance_multiplier()
    us_open = RoughModel.preset("us_open").distance_multiplier()
    assert light > heavy > us_open
    assert us_open < 0.85  # us_open severity 0.95 → ~ -18%


def test_rough_dispersion_increases_with_severity():
    assert (
        RoughModel.preset("us_open").dispersion_multiplier()
        > RoughModel.preset("heavy").dispersion_multiplier()
        > RoughModel.preset("light").dispersion_multiplier()
    )


def test_flyer_probability_peaks_at_medium_rough():
    light = RoughModel(severity=0.20).flyer_probability()
    med = RoughModel(severity=0.50).flyer_probability()
    us_open = RoughModel(severity=0.95).flyer_probability()
    assert med > light
    assert med > us_open


def test_trees_jail_forces_punch_out():
    assert TreeModel.preset("jail").is_forced_punch_out()
    assert not TreeModel.preset("decorative").is_forced_punch_out()


def test_greens_stimp_modifier_decreases_make_pct():
    slow = GreenModel.preset("slow")
    fast = GreenModel.preset("fast")
    assert slow.make_pct_modifier(20.0) > fast.make_pct_modifier(20.0)


def test_invalid_stimp_rejected():
    with pytest.raises(ContractViolationError):
        GreenModel(stimp=20.0)


def test_yaml_round_trip(tmp_path):
    conditions = CourseConditions.tournament()
    path = tmp_path / "conditions.yaml"
    conditions.to_yaml(path)
    loaded = CourseConditions.from_yaml(path)
    assert loaded.rough.severity == pytest.approx(conditions.rough.severity)
    assert loaded.greens.stimp == pytest.approx(conditions.greens.stimp)
    assert loaded.trees.penalization == pytest.approx(conditions.trees.penalization)


def test_yaml_preset_form(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "rough: {preset: heavy}\n"
        "trees: {preset: dense}\n"
        "greens: {preset: masters}\n"
        "pin_position_difficulty: 0.8\n"
    )
    c = CourseConditions.from_yaml(path)
    assert c.rough.severity == pytest.approx(RoughModel.preset("heavy").severity)
    assert c.greens.stimp == pytest.approx(GreenModel.preset("masters").stimp)
