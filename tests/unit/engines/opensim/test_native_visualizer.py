from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

CallLog = list[tuple[str, object | None]]


class FakeY:
    def __init__(self, calls: CallLog) -> None:
        self._calls = calls

    def set(self, idx: int, value: float) -> None:
        self._calls.append((f"y.set.{idx}", value))


class FakeState:
    def __init__(self, calls: CallLog) -> None:
        self._calls = calls
        self.y = FakeY(calls)

    def setTime(self, value: float) -> None:
        self._calls.append(("state.setTime", value))

    def getY(self) -> FakeY:
        self._calls.append(("state.getY", None))
        return self.y


class FakeSimbodyVisualizer:
    def __init__(self, calls: CallLog) -> None:
        self._calls = calls

    def setShowSimTime(self, value: bool) -> None:
        self._calls.append(("simbody.setShowSimTime", value))


class FakeVisualizerHandle:
    def __init__(self, calls: CallLog) -> None:
        self._calls = calls

    def updSimbodyVisualizer(self) -> FakeSimbodyVisualizer:
        self._calls.append(("visualizer.updSimbodyVisualizer", None))
        return FakeSimbodyVisualizer(self._calls)


class FakeDisplayVisualizer:
    def __init__(self, calls: CallLog) -> None:
        self._calls = calls

    def show(self, state: FakeState) -> None:
        self._calls.append(("display.show", state))


class FakeModel:
    def __init__(self, calls: CallLog) -> None:
        self._calls = calls
        self.state = FakeState(calls)

    def setUseVisualizer(self, value: bool) -> None:
        self._calls.append(("model.setUseVisualizer", value))

    def initSystem(self) -> FakeState:
        self._calls.append(("model.initSystem", None))
        return self.state

    def updVisualizer(self) -> FakeVisualizerHandle:
        self._calls.append(("model.updVisualizer", None))
        return FakeVisualizerHandle(self._calls)

    def realizePosition(self, state: FakeState) -> None:
        self._calls.append(("model.realizePosition", state))

    def getVisualizer(self) -> FakeDisplayVisualizer:
        self._calls.append(("model.getVisualizer", None))
        return FakeDisplayVisualizer(self._calls)


def test_native_visualizer_module_does_not_import_opensim_on_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "opensim", raising=False)

    module = importlib.import_module(
        "src.engines.physics_engines.opensim.python.motion_matching.viz.native"
    )

    assert "opensim" not in sys.modules
    assert module.render_with_opensim_visualizer is not None


def test_native_visualizer_raises_typed_unavailable_error_without_opensim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(
        "src.engines.physics_engines.opensim.python.motion_matching.viz.native"
    )

    def fail_import(name: str):
        if name == "opensim":
            raise ImportError("missing test opensim")
        return importlib.import_module(name)

    monkeypatch.setattr(module.importlib, "import_module", fail_import)

    sim = SimpleNamespace(time=np.array([0.0]), states=np.zeros((1, 1)))
    with pytest.raises(module.OpenSimVisualizerUnavailableError, match="opensim"):
        module.render_with_opensim_visualizer(model=object(), sim_out=sim)


def test_native_visualizer_drives_fake_model_without_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(
        "src.engines.physics_engines.opensim.python.motion_matching.viz.native"
    )
    monkeypatch.setitem(sys.modules, "opensim", SimpleNamespace(Visualizer=object))

    calls: CallLog = []
    sim = SimpleNamespace(
        time=np.array([0.0, 0.1]),
        states=np.array([[1.0, 2.0], [3.0, 4.0]]),
    )

    module.render_with_opensim_visualizer(
        model=FakeModel(calls), sim_out=sim, realtime_factor=float("inf")
    )

    assert ("model.setUseVisualizer", True) in calls
    assert ("simbody.setShowSimTime", True) in calls
    assert calls.count(("model.getVisualizer", None)) == 2
    assert ("y.set.0", 1.0) in calls
    assert ("y.set.1", 4.0) in calls
