"""Phase-correct bar paths of the multi-phase lifts (#11745).

:mod:`src.shared.python.lifting.phase_bar_paths` routes a bar convention to
the vendored Tools bar model of its lift: the snatch (hanging through the
pulls, overhead from the catch on) and the full clean and jerk (hanging
through the pulls, front rack from the catch to the jerk drive, overhead from
the overhead catch on).  Every other convention is the optimizer's
configuration-only ``bar_position``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.lifting import bundles
from src.shared.python.lifting.phase_bar_paths import (
    PHASE_BAR_CONVENTIONS,
    phase_bar_path,
)

pytestmark = pytest.mark.unit

#: Bar height per optimizer convention: hanging, on the shoulders, overhead.
_HEIGHT = {"deadlift": -1.0, "clean_and_jerk": 0.0, "jerk": 1.0, "snatch": 1.0}


class _Dyn:
    """Bar at the angle sum, at the convention's fixed height."""

    def bar_position(self, q: np.ndarray, convention: str) -> np.ndarray:
        return np.array([float(np.sum(q)), _HEIGHT[convention]])


def _tools(module: str, attribute: str):
    tools = pytest.importorskip(f"movement_optimizer.exercises.{module}")
    if not hasattr(tools, attribute):
        pytest.skip(f"vendored Tools predates {attribute}")
    return tools


def test_phase_conventions_are_the_multi_phase_lifts() -> None:
    assert set(PHASE_BAR_CONVENTIONS) == {"snatch", "clean_and_jerk"}


def test_other_conventions_use_the_configuration_bar() -> None:
    angles = np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.5]])
    path = phase_bar_path(_Dyn(), angles, np.array([0.0, 1.0]), "deadlift")
    np.testing.assert_allclose(path, [[0.6, -1.0], [0.5, -1.0]])


def test_bundles_delegates_to_the_phase_bar_path() -> None:
    angles = np.zeros((3, 3))
    time_s = np.array([0.0, 0.5, 1.0])
    np.testing.assert_array_equal(
        bundles._bar_path_m(_Dyn(), angles, time_s, "deadlift"),
        phase_bar_path(_Dyn(), angles, time_s, "deadlift"),
    )


def test_clean_and_jerk_bar_hangs_racks_then_goes_overhead() -> None:
    cj = _tools("clean_and_jerk", "clean_and_jerk_bar_path")
    times = {**cj.CLEAN_AND_JERK_PHASES, **cj.CLEAN_AND_JERK_EVENTS}
    time_s = np.linspace(2.0, 6.0, 401)  # starts away from zero on purpose
    path = phase_bar_path(_Dyn(), np.zeros((time_s.size, 3)), time_s, "clean_and_jerk")
    frac = (time_s - time_s[0]) / (time_s[-1] - time_s[0])
    pull = frac <= times["turnover"]
    rack = (frac >= times["rack_catch"]) & (frac <= times["drive_extension"])
    overhead = frac >= times["overhead_catch"]
    assert np.all(path[pull, 1] == -1.0)
    assert np.all(path[rack, 1] == 0.0)
    assert np.all(path[overhead, 1] == 1.0)


def test_snatch_bar_still_hangs_then_goes_overhead() -> None:
    snatch = _tools("snatch", "snatch_bar_path")
    time_s = np.linspace(1.0, 3.0, 21)
    path = phase_bar_path(_Dyn(), np.zeros((time_s.size, 3)), time_s, "snatch")
    frac = (time_s - time_s[0]) / (time_s[-1] - time_s[0])
    assert np.all(path[frac <= snatch.SNATCH_PHASES["turnover"], 1] == -1.0)
    assert np.all(path[frac >= snatch.SNATCH_PHASES["catch"], 1] == 1.0)


def test_phase_bar_path_rejects_a_degenerate_time_span() -> None:
    with pytest.raises(ValueError, match="time_s"):
        phase_bar_path(_Dyn(), np.zeros((2, 3)), np.array([1.0, 1.0]), "snatch")
    with pytest.raises(ValueError, match="time_s"):
        phase_bar_path(_Dyn(), np.zeros((2, 3)), np.array([0.0]), "snatch")
