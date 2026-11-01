"""Hypothesis properties every putting provider must satisfy (#14177)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.shared.python.sg_optimizer.course.conditions import GreenModel
from src.shared.python.sg_optimizer.shot_model.player_profile import PuttingSkill
from src.shared.python.sg_optimizer.shot_model.putting_provider import (
    CurvePuttingProvider,
    LogisticPuttingProvider,
    PuttingProvider,
    SimulatedPuttingCurve,
)

pytestmark = pytest.mark.unit

GRID = (2.0, 4.0, 8.0, 15.0, 30.0, 60.0)
distance = st.floats(min_value=0.5, max_value=90.0, allow_nan=False)
stimp = st.floats(min_value=8.0, max_value=14.0, allow_nan=False)
# On greens slower than stimp 10 the green model raises long putts more than
# short ones, so a curve with a flat stretch can rise slightly with distance.
fast_stimp = st.floats(min_value=10.0, max_value=14.0, allow_nan=False)


@st.composite
def curves(draw: st.DrawFn) -> SimulatedPuttingCurve:
    """Simulated curves with make rate non-increasing in distance."""
    rates = sorted(
        draw(st.lists(st.floats(0.0, 1.0), min_size=len(GRID), max_size=len(GRID))),
        reverse=True,
    )
    positive = st.floats(0.05, 20.0)
    scales = draw(st.lists(positive, min_size=len(GRID), max_size=len(GRID)))
    sigmas = draw(st.lists(st.floats(0.0, 2.0), min_size=len(GRID), max_size=len(GRID)))
    return SimulatedPuttingCurve(GRID, tuple(rates), tuple(scales), tuple(sigmas))


providers = st.one_of(
    st.just(LogisticPuttingProvider()), curves().map(CurvePuttingProvider)
)


@given(provider=providers, near=distance, far=distance, speed=fast_stimp)
@settings(max_examples=60, deadline=None)
def test_make_probability_is_bounded_and_non_increasing(
    provider: PuttingProvider, near: float, far: float, speed: float
) -> None:
    near, far = sorted((near, far))
    greens = GreenModel(stimp=speed)
    p_near = provider.make_probability(near, PuttingSkill(), greens)
    p_far = provider.make_probability(far, PuttingSkill(), greens)
    assert 0.0 <= p_far <= 1.0 and 0.0 <= p_near <= 1.0
    assert p_far <= p_near + 1e-12


@given(provider=providers, d=distance, slow=stimp, fast=stimp)
@settings(max_examples=60, deadline=None)
def test_leave_never_tightens_on_faster_greens(
    provider: PuttingProvider, d: float, slow: float, fast: float
) -> None:
    slow, fast = sorted((slow, fast))
    a = provider.leave_distribution(d, PuttingSkill(), GreenModel(stimp=slow))
    b = provider.leave_distribution(d, PuttingSkill(), GreenModel(stimp=fast))
    assert a.mean_ft > 0.0 and b.sigma_log >= a.sigma_log - 1e-12
