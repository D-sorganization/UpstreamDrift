"""Regression tests for Drake service model-loading type contracts."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from src.shared.python.pose_interchange.services.drake import DrakeKinematicsService


class _FakePlant:
    def Finalize(self) -> None:
        return None

    def GetMyContextFromRoot(self, context: object) -> object:
        return context

    def num_positions(self) -> int:
        return 0


class _FakeDiagram:
    def CreateDefaultContext(self) -> object:
        return object()


class _FakeBuilder:
    def Build(self) -> _FakeDiagram:
        return _FakeDiagram()


class _FakeSimulator:
    def __init__(self, diagram: object, context: object) -> None:
        self.diagram = diagram
        self.context = context

    def Initialize(self) -> None:
        return None


def test_drake_load_passes_pathlike_to_parser(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class _FakeParser:
        def __init__(self, plant: object) -> None:
            captured["plant"] = plant

        def AddModels(self, model_path: Path) -> None:
            captured["model_path"] = model_path

    parsing = types.ModuleType("pydrake.multibody.parsing")
    parsing.Parser = _FakeParser
    plant = types.ModuleType("pydrake.multibody.plant")
    plant.AddMultibodyPlantSceneGraph = lambda builder, time_step: (
        _FakePlant(),
        object(),
    )
    framework = types.ModuleType("pydrake.systems.framework")
    framework.DiagramBuilder = _FakeBuilder
    analysis = types.ModuleType("pydrake.systems.analysis")
    analysis.Simulator = _FakeSimulator

    modules = {
        "pydrake": types.ModuleType("pydrake"),
        "pydrake.multibody": types.ModuleType("pydrake.multibody"),
        "pydrake.multibody.parsing": parsing,
        "pydrake.multibody.plant": plant,
        "pydrake.systems": types.ModuleType("pydrake.systems"),
        "pydrake.systems.framework": framework,
        "pydrake.systems.analysis": analysis,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    model_path = tmp_path / "model.urdf"
    DrakeKinematicsService().load(model_path)

    assert captured["model_path"] == model_path
    assert isinstance(captured["model_path"], Path)
