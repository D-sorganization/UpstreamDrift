"""The workbench model announces every shot before it runs (issue #9545).

A job runner needs two things from the model and nothing more: a count of
shots done, for progress, and a point between shots where a cancellation
can stop the run. The ``on_shot`` hook is both; whatever it raises ends
the evaluation, so a cancelled sweep does not run its remaining shots.
"""

from __future__ import annotations

import pytest

from src.tools.bunker_shot_gui.design import (
    SandCondition,
    SolverSetup,
    SwingSetup,
    WedgeDesign,
)
from src.tools.bunker_shot_gui.model import WorkbenchModel

pytestmark = [pytest.mark.integration]

DESIGN = WedgeDesign(name="A", grind_preset="sm9_58_m")
SAND = SandCondition()
SWING = SwingSetup(clubhead_speed_mps=25.0, attack_angle_deg=-6.0)


class _Stop(Exception):
    """Raised by the hook to end the run."""


def test_the_hook_is_called_once_for_the_nominal_shot() -> None:
    calls: list[int] = []
    model = WorkbenchModel(on_shot=lambda: calls.append(1))
    model.evaluate(DESIGN, SAND, SWING, include_playability=False)
    assert len(calls) == 1


def test_the_hook_is_called_for_every_sweep_point() -> None:
    calls: list[int] = []
    settings = SolverSetup(playability_points=2)
    model = WorkbenchModel(settings, on_shot=lambda: calls.append(1))
    model.evaluate(DESIGN, SAND, SWING)
    assert len(calls) == 1 + settings.playability_points**2


def test_a_raising_hook_stops_the_run_before_the_next_shot() -> None:
    calls: list[int] = []

    def hook() -> None:
        calls.append(1)
        if len(calls) == 2:
            raise _Stop

    model = WorkbenchModel(SolverSetup(playability_points=2), on_shot=hook)
    with pytest.raises(_Stop):
        model.evaluate(DESIGN, SAND, SWING)
    assert len(calls) == 2


def test_no_hook_is_the_default() -> None:
    WorkbenchModel().evaluate(DESIGN, SAND, SWING, include_playability=False)
