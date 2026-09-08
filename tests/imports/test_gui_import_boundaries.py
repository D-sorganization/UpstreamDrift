"""Import-boundary regressions for headless GUI collection.

Issue #3910: GUI modules may be collected in environments where optional
plotting/OpenGL dependencies are absent or unsafe to import. Collection must
stay deterministic; runtime widgets can still degrade when instantiated.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import patch

import pytest

pytest.importorskip("PyQt6")


@contextmanager
def _blocked_imports(*blocked_roots: str) -> Iterator[None]:
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict | None = None,
        locals: dict | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if any(name == root or name.startswith(root + ".") for root in blocked_roots):
            raise ImportError(f"blocked optional import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import
    try:
        yield
    finally:
        builtins.__import__ = real_import


@contextmanager
def _dropped_modules(*module_names: str) -> Iterator[None]:
    """Drop modules from ``sys.modules`` for the block, then restore.

    Issue #9387: the drops are load-bearing for import-boundary testing,
    but they must not leak: entries that existed before the block are
    re-bound to their prior objects in a ``finally``, and entries that
    did not exist are removed again so a replacement import performed
    inside the block never survives it.
    """
    saved = [(name, sys.modules.pop(name, None)) for name in module_names]
    try:
        yield
    finally:
        for name, module in saved:
            if module is not None:
                sys.modules[name] = module
            else:
                sys.modules.pop(name, None)


def _fake_perturbation_panel_module() -> ModuleType:
    module = ModuleType("src.shared.python.pendulum_simulator.gui.perturbation_panel")

    class _PerturbationPanel:
        pass

    module.PerturbationPanel = _PerturbationPanel
    return module


def _fake_simulation_panel_module() -> ModuleType:
    module = ModuleType("src.shared.python.pendulum_simulator.gui.simulation_panel")

    class _SimulationPanel:
        pass

    module.SimulationPanel = _SimulationPanel
    return module


def test_torque_history_widget_imports_without_pyqtgraph() -> None:
    with (
        _dropped_modules(
            "pyqtgraph",
            "src.shared.python.pendulum_simulator.gui.torque_history_widget",
        ),
        _blocked_imports("pyqtgraph"),
    ):
        module = importlib.import_module(
            "src.shared.python.pendulum_simulator.gui.torque_history_widget"
        )

    assert module.TorqueHistoryWidget.__name__ == "TorqueHistoryWidget"


def test_panel_builders_imports_without_pyqtgraph() -> None:
    with _dropped_modules(
        "pyqtgraph",
        "src.shared.python.pendulum_simulator.gui.perturbation_panel",
        "src.shared.python.pendulum_simulator.gui.simulation_panel",
        "src.shared.python.pendulum_simulator.gui.torque_history_widget",
        "src.shared.python.pendulum_simulator.gui.panel_builders",
    ):
        fake_modules = {
            "src.shared.python.pendulum_simulator.gui.perturbation_panel": (
                _fake_perturbation_panel_module()
            ),
            "src.shared.python.pendulum_simulator.gui.simulation_panel": (
                _fake_simulation_panel_module()
            ),
        }
        with patch.dict(sys.modules, fake_modules), _blocked_imports("pyqtgraph"):
            module = importlib.import_module(
                "src.shared.python.pendulum_simulator.gui.panel_builders"
            )

    assert module.build_double_panel.__name__ == "build_double_panel"


def test_visualization_widget_imports_without_qt_openglwidgets() -> None:
    with (
        _dropped_modules(
            "PyQt6.QtOpenGLWidgets",
            "src.tools.model_explorer.visualization_widget",
        ),
        _blocked_imports("PyQt6.QtOpenGLWidgets"),
    ):
        module = importlib.import_module(
            "src.tools.model_explorer.visualization_widget"
        )

    assert module.VisualizationWidget.__name__ == "VisualizationWidget"
