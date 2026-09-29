"""Putting model — make-% monotonic in distance, leave widens with stimp."""

from __future__ import annotations

import pytest

from src.shared.python.sg_optimizer.course.conditions import GreenModel
from src.shared.python.sg_optimizer.shot_model.player_profile import PuttingSkill
from src.shared.python.sg_optimizer.shot_model.putting import (
    leave_distance_distribution,
    make_probability,
)


def test_make_probability_monotonic_decreasing():
    profile = PuttingSkill()
    greens = GreenModel.preset("medium")
    p3 = make_probability(3.0, profile, greens)
    p10 = make_probability(10.0, profile, greens)
    p30 = make_probability(30.0, profile, greens)
    assert p3 > p10 > p30
    assert p3 > 0.85
    assert p30 < 0.20


def test_short_putt_better_than_tour_with_skill():
    profile = PuttingSkill(make_pct_multipliers={3.0: 1.05, 25.0: 1.0})
    greens = GreenModel.preset("medium")
    p = make_probability(3.0, profile, greens)
    p_base = make_probability(3.0, PuttingSkill(), greens)
    assert p > p_base


def test_faster_greens_reduce_long_putt_make_more_than_short_relatively():
    """Relative make-% hit grows with distance (proportional, not absolute)."""
    profile = PuttingSkill()
    slow = GreenModel.preset("slow")
    fast = GreenModel.preset("fast")
    short_ratio = make_probability(4.0, profile, fast) / make_probability(
        4.0, profile, slow
    )
    long_ratio = make_probability(20.0, profile, fast) / make_probability(
        20.0, profile, slow
    )
    assert long_ratio < short_ratio < 1.01  # fast/slow ratio falls with distance


def test_leave_distribution_widens_with_stimp():
    profile = PuttingSkill()
    slow = leave_distance_distribution(30.0, profile, GreenModel.preset("slow"))
    fast = leave_distance_distribution(30.0, profile, GreenModel.preset("fast"))
    assert fast.sigma_log > slow.sigma_log


def test_expected_three_putt_probability_increases_with_stimp():
    """Faster greens broaden the leave distribution → more tail mass above a
    typical 3-putt threshold (4 ft come-backer)."""
    profile = PuttingSkill()
    slow = leave_distance_distribution(30.0, profile, GreenModel.preset("slow"))
    fast = leave_distance_distribution(30.0, profile, GreenModel.preset("fast"))
    assert fast.expected_three_putt_probability(
        4.0
    ) > slow.expected_three_putt_probability(4.0)


def test_invalid_distance_rejected():
    from src.shared.python.contracts import ContractViolationError

    with pytest.raises(ContractViolationError):
        make_probability(0.0, PuttingSkill(), GreenModel.preset("medium"))
