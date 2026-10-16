"""Regression for UD#12912: OpenSim Run used removed 3.x Manager API."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.engines.physics_engines.opensim.python.opensim_golf.core import (
    GolfSwingModel,
)

MODEL = (
    Path(__file__).resolve().parents[3]
    / "src/engines/physics_engines/opensim/models/golf_humanoid.osim"
)


class _Manager4x:
    """Fake OpenSim 4.x Manager: has NO setInitialTime/setFinalTime."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def initialize(self, state):
        self.calls.append(("initialize", state.t))

    def integrate(self, t_final):
        self.calls.append(("integrate", t_final))
        return _State(t_final)


class _State:
    def __init__(self, t: float) -> None:
        self.t = t

    def setTime(self, t: float) -> None:  # noqa: N802 - OpenSim API name
        self.t = t


@pytest.mark.unit
def test_advance_state_uses_4x_api_only() -> None:
    model = GolfSwingModel.__new__(GolfSwingModel)
    model._manager = _Manager4x()
    model._state = _State(0.0)
    model._advance_state(0.5, 0.75)
    assert model._manager.calls == [("initialize", 0.5), ("integrate", 0.75)]
    assert model._state.t == 0.75


@pytest.mark.unit
def test_advance_state_rejects_non_increasing_interval() -> None:
    model = GolfSwingModel.__new__(GolfSwingModel)
    model._manager = _Manager4x()
    model._state = _State(0.0)
    with pytest.raises(ValueError):
        model._advance_state(1.0, 1.0)


@pytest.mark.unit
def test_run_simulation_real_opensim_golf_humanoid() -> None:
    pytest.importorskip("opensim")
    model = GolfSwingModel(str(MODEL))
    model.duration = 0.03
    model.dt = 0.01
    result = model.run_simulation()
    assert len(result.time) == 3
