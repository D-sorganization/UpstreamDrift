"""Tests for ``src.launchers.exercise_dashboard``.

The class itself is a QMainWindow, so a real ``QApplication`` is needed,
but the engine-specific child dashboards are mocked to avoid Drake /
MuJoCo / Pinocchio imports.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _make_qwidget_factory(monkeypatch):
    """Return a factory that produces a fake QWidget per call."""
    from PyQt6.QtWidgets import QLabel

    def factory(*a, **kw):
        return QLabel("stub")

    return factory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_child_dashboards(monkeypatch):
    """Replace each engine-specific dashboard with a real QLabel widget.

    Returns a dict of MagicMock factories keyed by engine class name so
    callers can assert which ones were invoked.
    """
    from PyQt6.QtWidgets import QLabel

    drake = MagicMock(side_effect=lambda **kw: QLabel("drake"))
    mujoco = MagicMock(side_effect=lambda **kw: QLabel("mujoco"))
    pino = MagicMock(side_effect=lambda **kw: QLabel("pino"))

    import src.launchers.drake_dashboard as drake_mod
    import src.launchers.mujoco_dashboard as mujoco_mod
    import src.launchers.pinocchio_dashboard as pino_mod

    monkeypatch.setattr(drake_mod, "DrakeDashboard", drake)
    monkeypatch.setattr(mujoco_mod, "MuJoCoDashboard", mujoco)
    monkeypatch.setattr(pino_mod, "PinocchioDashboard", pino)
    return {
        "DrakeDashboard": drake,
        "MuJoCoDashboard": mujoco,
        "PinocchioDashboard": pino,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExerciseDashboard:
    def test_title_uses_exercise_name(
        self, qapp, patched_child_dashboards, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "src.launchers.exercise_dashboard.discover_exercise",
            lambda x: ["MuJoCo_Models"],
        )
        from src.launchers.exercise_dashboard import ExerciseDashboard

        win = ExerciseDashboard("gait")
        try:
            assert "Gait" in win.windowTitle()
            # JaxSim is always appended as an analysis backend (issue #6658).
            assert win.engines == ["MuJoCo_Models", "JaxSim_Models"]
        finally:
            win.close()

    def test_empty_discovery_falls_back(
        self, qapp, patched_child_dashboards, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "src.launchers.exercise_dashboard.discover_exercise", lambda x: []
        )
        from src.launchers.exercise_dashboard import ExerciseDashboard

        win = ExerciseDashboard("squat")
        try:
            assert set(win.engines) == {
                "MuJoCo_Models",
                "Drake_Models",
                "Pinocchio_Models",
                "JaxSim_Models",
            }
        finally:
            win.close()

    def test_first_engine_loaded_at_construction(
        self, qapp, patched_child_dashboards, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "src.launchers.exercise_dashboard.discover_exercise",
            lambda x: ["MuJoCo_Models"],
        )
        from src.launchers.exercise_dashboard import ExerciseDashboard

        win = ExerciseDashboard("gait")
        try:
            patched_child_dashboards["MuJoCoDashboard"].assert_called_once_with(
                exercise_filter="gait"
            )
        finally:
            win.close()

    def test_swap_to_drake(self, qapp, patched_child_dashboards, monkeypatch) -> None:
        monkeypatch.setattr(
            "src.launchers.exercise_dashboard.discover_exercise",
            lambda x: ["MuJoCo_Models", "Drake_Models"],
        )
        from src.launchers.exercise_dashboard import ExerciseDashboard

        win = ExerciseDashboard("gait")
        try:
            win._on_engine_changed("Drake_Models")
            patched_child_dashboards["DrakeDashboard"].assert_called_once_with(
                exercise_filter="gait"
            )
        finally:
            win.close()

    def test_swap_to_pinocchio(
        self, qapp, patched_child_dashboards, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "src.launchers.exercise_dashboard.discover_exercise",
            lambda x: ["Pinocchio_Models"],
        )
        from src.launchers.exercise_dashboard import ExerciseDashboard

        win = ExerciseDashboard("run")
        try:
            patched_child_dashboards["PinocchioDashboard"].assert_called_once_with(
                exercise_filter="run"
            )
        finally:
            win.close()

    def test_jaxsim_always_offered_as_engine(
        self, qapp, patched_child_dashboards, monkeypatch
    ) -> None:
        # JaxSim has no sibling model repo, so it is appended to the selector
        # regardless of what discover_exercise returns (issue #6658).
        monkeypatch.setattr(
            "src.launchers.exercise_dashboard.discover_exercise",
            lambda x: ["MuJoCo_Models"],
        )
        from src.launchers.exercise_dashboard import ExerciseDashboard

        win = ExerciseDashboard("gait")
        try:
            assert "JaxSim_Models" in win.engines
        finally:
            win.close()

    def test_swap_to_jaxsim_uses_capability_dashboard(self, qapp, monkeypatch) -> None:
        monkeypatch.setattr(
            "src.launchers.exercise_dashboard.discover_exercise",
            lambda x: ["MuJoCo_Models"],
        )
        from src.launchers.exercise_dashboard import ExerciseDashboard
        from src.launchers.jaxsim_dashboard import JaxSimDashboard

        win = ExerciseDashboard("gait")
        try:
            win._on_engine_changed("JaxSim_Models")
            assert isinstance(win._current_widget, JaxSimDashboard)
            # Capability-driven: partial contact-forces feature is greyed out.
            assert not win._current_widget.feature_controls[
                "Contact forces"
            ].isEnabled()
        finally:
            win.close()

    def test_opensim_shows_placeholder(self, qapp, monkeypatch) -> None:
        monkeypatch.setattr(
            "src.launchers.exercise_dashboard.discover_exercise",
            lambda x: ["OpenSim_Models"],
        )
        from PyQt6.QtWidgets import QLabel

        from src.launchers.exercise_dashboard import ExerciseDashboard

        win = ExerciseDashboard("gait")
        try:
            assert isinstance(win._current_widget, QLabel)
            assert "OpenSim" in win._current_widget.text()
        finally:
            win.close()

    def test_unknown_engine_shows_placeholder(self, qapp, monkeypatch) -> None:
        monkeypatch.setattr(
            "src.launchers.exercise_dashboard.discover_exercise",
            lambda x: ["Bogus_Engine"],
        )
        from PyQt6.QtWidgets import QLabel

        from src.launchers.exercise_dashboard import ExerciseDashboard

        win = ExerciseDashboard("gait")
        try:
            assert isinstance(win._current_widget, QLabel)
            assert "Unknown engine" in win._current_widget.text()
        finally:
            win.close()

    def test_engine_constructor_error_shows_label(self, qapp, monkeypatch) -> None:
        import src.launchers.drake_dashboard as drake_mod

        monkeypatch.setattr(
            "src.launchers.exercise_dashboard.discover_exercise",
            lambda x: ["Drake_Models"],
        )
        monkeypatch.setattr(
            drake_mod,
            "DrakeDashboard",
            MagicMock(side_effect=RuntimeError("kaboom")),
        )
        from PyQt6.QtWidgets import QLabel

        from src.launchers.exercise_dashboard import ExerciseDashboard

        win = ExerciseDashboard("gait")
        try:
            assert isinstance(win._current_widget, QLabel)
            assert "Error loading Drake_Models" in win._current_widget.text()
            assert "kaboom" in win._current_widget.text()
        finally:
            win.close()


class TestGetDockableUi:
    def test_uses_env_variable(
        self, qapp, patched_child_dashboards, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "src.launchers.exercise_dashboard.discover_exercise",
            lambda x: ["MuJoCo_Models"],
        )
        monkeypatch.setenv("BIOMECH_EXERCISE", "squat")
        from src.launchers.exercise_dashboard import get_dockable_ui

        win = get_dockable_ui()
        try:
            assert "Squat" in win.windowTitle()
        finally:
            win.close()

    def test_defaults_to_gait(
        self, qapp, patched_child_dashboards, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "src.launchers.exercise_dashboard.discover_exercise",
            lambda x: ["MuJoCo_Models"],
        )
        monkeypatch.delenv("BIOMECH_EXERCISE", raising=False)
        from src.launchers.exercise_dashboard import get_dockable_ui

        win = get_dockable_ui()
        try:
            assert "Gait" in win.windowTitle()
        finally:
            win.close()
