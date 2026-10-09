"""Tests for the recorder's ground-reaction breakdown collection (GCV-5, #11711).

``GenericPhysicsRecorder.record_step`` samples the engine's
``get_ground_reaction_breakdown()`` (when present) and
``get_ground_reaction_series()`` stacks them into a GCV-1
:class:`GroundReactionSeries`, the same way ``ground_reaction_service.py``
reads an engine's own ``get_ground_reaction_series``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.biomechanics.ground_reaction import (
    ContactSet,
    analyze_ground_reaction,
)
from src.shared.python.dashboard.recorder import GenericPhysicsRecorder

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _breakdown(fz: float):
    left = ContactSet(np.array([[0.0, 0.0, fz]]), np.array([[0.0, 0.15, 0.0]]))
    return analyze_ground_reaction(
        {"left": left, "right": ContactSet.empty()}, (0.0, 0.0, 0.95)
    )


class _FakeEngine:
    """Minimal duck-typed engine: batched state plus a GRF breakdown source."""

    def __init__(self, breakdowns):
        self._breakdowns = list(breakdowns)
        self._calls = 0
        self.t = 0.0

    def get_full_state(self):
        self.t += 0.01
        return {"q": np.zeros(1), "v": np.zeros(1), "t": self.t, "M": None}

    def get_ground_reaction_breakdown(self):
        breakdown = self._breakdowns[self._calls]
        self._calls += 1
        return breakdown


class _FakeEngineNoGrf:
    """An engine that does not implement get_ground_reaction_breakdown."""

    def __init__(self) -> None:
        self.t = 0.0

    def get_full_state(self):
        self.t += 0.01
        return {"q": np.zeros(1), "v": np.zeros(1), "t": self.t, "M": None}


def _record(engine, steps: int) -> GenericPhysicsRecorder:
    recorder = GenericPhysicsRecorder(engine)
    recorder.start()
    for _ in range(steps):
        recorder.record_step()
    recorder.stop()
    return recorder


def test_series_has_the_recorded_length_and_times() -> None:
    engine = _FakeEngine([_breakdown(784.0), _breakdown(700.0), _breakdown(650.0)])
    recorder = _record(engine, 3)

    series = recorder.get_ground_reaction_series()

    assert series is not None
    assert series.times_s.shape == (3,)
    np.testing.assert_allclose(series.times_s, [0.01, 0.02, 0.03])
    np.testing.assert_allclose(series.force_n["left"][0], (0.0, 0.0, 784.0))


def test_engine_without_the_method_gives_no_series() -> None:
    recorder = _record(_FakeEngineNoGrf(), 2)

    assert recorder.get_ground_reaction_series() is None


def test_a_none_breakdown_at_any_sample_makes_the_series_unavailable() -> None:
    engine = _FakeEngine([_breakdown(784.0), None, _breakdown(650.0)])
    recorder = _record(engine, 2)

    assert recorder.get_ground_reaction_series() is None


def test_reset_clears_previously_recorded_breakdowns() -> None:
    engine = _FakeEngine([_breakdown(784.0)])
    recorder = _record(engine, 1)
    assert recorder.get_ground_reaction_series() is not None

    recorder.reset()

    assert recorder.get_ground_reaction_series() is None


def test_no_samples_recorded_gives_no_series() -> None:
    recorder = GenericPhysicsRecorder(_FakeEngine([]))
    assert recorder.get_ground_reaction_series() is None
