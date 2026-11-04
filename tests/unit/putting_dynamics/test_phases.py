"""Ball phase marks for playback jumps (#14473, epic #14452)."""

from __future__ import annotations

import pytest

from src.shared.python.putting_dynamics import (
    Mode,
    PhaseMark,
    PutterState,
    TrajectorySample,
    phase_marks,
    simulate_strike,
)
from src.shared.python.putting_dynamics.green_spec import GreenSpec

pytestmark = pytest.mark.unit


def _sample(t: float, mode: Mode) -> TrajectorySample:
    return TrajectorySample(t, t, 0.0, 0.0, 1.0, 0.0, mode)


def test_a_mark_opens_every_run_of_a_mode() -> None:
    modes = [Mode.AIRBORNE, Mode.SLIDE, Mode.SLIDE, Mode.AIRBORNE, Mode.ROLL]
    samples = [_sample(0.1 * i, m) for i, m in enumerate(modes)]
    assert phase_marks(samples) == (
        PhaseMark(Mode.AIRBORNE, 0.0, 0),
        PhaseMark(Mode.SLIDE, pytest.approx(0.1), 1),
        PhaseMark(Mode.AIRBORNE, pytest.approx(0.3), 3),
        PhaseMark(Mode.ROLL, pytest.approx(0.4), 4),
    )


def test_a_struck_putt_skids_then_rolls_to_rest() -> None:
    # Golden parity: the marks are read off the core's own trajectory.
    surface = GreenSpec(grade_percent=1.0).surface()
    result = simulate_strike(PutterState.typical(1.8), surface)
    marks = phase_marks(result.samples)
    assert marks[0].index == 0 and marks[0].t_s == result.samples[0].t_s
    for mark in marks:
        assert result.samples[mark.index].mode is mark.mode
        assert result.samples[mark.index].t_s == mark.t_s
        if mark.index:
            assert result.samples[mark.index - 1].mode is not mark.mode
    kinds = [m.mode for m in marks]
    assert Mode.SLIDE in kinds and Mode.ROLL in kinds
    assert kinds.index(Mode.SLIDE) < kinds.index(Mode.ROLL)
    assert kinds[-1] is result.samples[-1].mode


def test_marks_need_time_ordered_samples() -> None:
    with pytest.raises(ValueError, match="sample"):
        phase_marks([])
    with pytest.raises(ValueError, match="time"):
        phase_marks([_sample(0.2, Mode.ROLL), _sample(0.1, Mode.ROLL)])
