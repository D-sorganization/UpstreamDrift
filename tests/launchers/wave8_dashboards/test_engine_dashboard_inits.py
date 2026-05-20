"""Tests for the deferred-engine dashboards (drake, mujoco, pinocchio).

Each dashboard subclass defers the heavy physics-engine import to
``__init__``.  These tests stub the engine modules via ``patch.dict
(sys.modules, ...)`` and verify model-discovery logic, title formatting,
and graceful error handling.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine_module(modpath: str, attr: str) -> tuple[ModuleType, MagicMock]:
    """Build a fake engine module with a callable engine class.

    Returns (module, engine_instance_mock).  Each call to the engine
    class returns the same instance for assertion purposes.
    """
    module = ModuleType(modpath)
    instance = MagicMock(name=f"{attr}_instance")
    engine_cls = MagicMock(name=attr, return_value=instance)
    engine_cls.__name__ = attr
    setattr(module, attr, engine_cls)
    return module, instance


@pytest.fixture
def stub_unified_window():
    """Patch UnifiedDashboardWindow.__init__ to a no-op widget-free init.

    This avoids constructing real Qt widgets when the dashboard subclass
    calls ``super().__init__(engine, title=...)``.
    """
    captured: dict[str, object] = {}

    def fake_init(self, engine, *, title: str | None = None, **kw) -> None:
        captured["engine"] = engine
        captured["title"] = title

    with patch(
        "src.shared.python.dashboard.window.UnifiedDashboardWindow.__init__",
        new=fake_init,
    ):
        yield captured


# ---------------------------------------------------------------------------
# Drake
# ---------------------------------------------------------------------------


DRAKE_MOD = "src.engines.physics_engines.drake.python.drake_physics_engine"


class TestDrakeDashboard:
    def _patch_engine(self):
        mod, inst = _make_engine_module(DRAKE_MOD, "DrakePhysicsEngine")
        return patch.dict(sys.modules, {DRAKE_MOD: mod}), inst

    def test_default_title(self, stub_unified_window) -> None:
        patcher, _ = self._patch_engine()
        with patcher:
            from src.launchers.drake_dashboard import DrakeDashboard

            DrakeDashboard()
        assert stub_unified_window["title"] == "Drake Golf Analysis Dashboard"

    def test_exercise_filter_appends_title(self, stub_unified_window) -> None:
        patcher, _ = self._patch_engine()
        with (
            patcher,
            patch(
                "src.shared.python.config.model_source_providers.drake_models_source",
                side_effect=RuntimeError("no models"),
            ),
        ):
            from src.launchers.drake_dashboard import DrakeDashboard

            DrakeDashboard(exercise_filter="gait")
        assert "Gait" in stub_unified_window["title"]

    def test_model_path_calls_load_from_path(self, stub_unified_window) -> None:
        patcher, instance = self._patch_engine()
        with patcher:
            from src.launchers.drake_dashboard import DrakeDashboard

            DrakeDashboard(model_path="/some/model.urdf")
        instance.load_from_path.assert_called_once_with("/some/model.urdf")

    def test_load_from_path_exception_swallowed(self, stub_unified_window) -> None:
        patcher, instance = self._patch_engine()
        instance.load_from_path.side_effect = RuntimeError("bad model")
        with patcher:
            from src.launchers.drake_dashboard import DrakeDashboard

            DrakeDashboard(model_path="/oops.urdf")  # should not raise

    def test_exercise_filter_discovers_model(
        self, stub_unified_window, tmp_path: Path
    ) -> None:
        exercise = "gait"
        ex_dir = tmp_path / "exercises" / exercise
        ex_dir.mkdir(parents=True)
        urdf = ex_dir / "model.urdf"
        urdf.write_text("<robot/>")

        patcher, instance = self._patch_engine()
        with (
            patcher,
            patch(
                "src.shared.python.config.model_source_providers.drake_models_source",
                return_value=tmp_path,
            ),
        ):
            from src.launchers.drake_dashboard import DrakeDashboard

            DrakeDashboard(exercise_filter=exercise)

        instance.load_from_path.assert_called_once()
        called_path = instance.load_from_path.call_args[0][0]
        assert called_path.endswith("model.urdf")

    def test_exercise_filter_no_models_dir(self, stub_unified_window) -> None:
        patcher, instance = self._patch_engine()
        with (
            patcher,
            patch(
                "src.shared.python.config.model_source_providers.drake_models_source",
                return_value=Path("/definitely/does/not/exist"),
            ),
        ):
            from src.launchers.drake_dashboard import DrakeDashboard

            DrakeDashboard(exercise_filter="gait")
        instance.load_from_path.assert_not_called()


def test_drake_main_with_model_arg(stub_unified_window) -> None:
    mod, _ = _make_engine_module(DRAKE_MOD, "DrakePhysicsEngine")
    fake_app = MagicMock()
    fake_app.exec.return_value = 0
    with (
        patch.dict(sys.modules, {DRAKE_MOD: mod}),
        patch("src.launchers.drake_dashboard.get_qapp", return_value=fake_app),
        patch("src.shared.python.logging_pkg.logging_config.configure_gui_logging"),
        patch("sys.argv", ["drake_dashboard", "--model", "/x/y.urdf"]),
        patch("src.shared.python.dashboard.window.UnifiedDashboardWindow.show"),
    ):
        from src.launchers.drake_dashboard import main

        with pytest.raises(SystemExit):
            main()
    fake_app.exec.assert_called_once()


# ---------------------------------------------------------------------------
# MuJoCo
# ---------------------------------------------------------------------------


MUJOCO_MOD = (
    "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine"
)


class TestMuJoCoDashboard:
    def _patch_engine(self):
        mod, inst = _make_engine_module(MUJOCO_MOD, "MuJoCoPhysicsEngine")
        return patch.dict(sys.modules, {MUJOCO_MOD: mod}), inst

    def test_default_title(self, stub_unified_window) -> None:
        patcher, _ = self._patch_engine()
        with patcher:
            from src.launchers.mujoco_dashboard import MuJoCoDashboard

            MuJoCoDashboard()
        assert "MuJoCo" in stub_unified_window["title"]
        assert "Unified" in stub_unified_window["title"]

    def test_exercise_filter_title(self, stub_unified_window) -> None:
        patcher, _ = self._patch_engine()
        with (
            patcher,
            patch(
                "src.shared.python.config.model_source_providers.mujoco_models_source",
                side_effect=RuntimeError("nope"),
            ),
        ):
            from src.launchers.mujoco_dashboard import MuJoCoDashboard

            MuJoCoDashboard(exercise_filter="squat")
        assert "Squat" in stub_unified_window["title"]

    def test_exercise_filter_loads_xml(
        self, stub_unified_window, tmp_path: Path
    ) -> None:
        ex_dir = tmp_path / "exercises" / "gait"
        ex_dir.mkdir(parents=True)
        (ex_dir / "scene.xml").write_text("<mujoco/>")
        patcher, instance = self._patch_engine()
        with (
            patcher,
            patch(
                "src.shared.python.config.model_source_providers.mujoco_models_source",
                return_value=tmp_path,
            ),
        ):
            from src.launchers.mujoco_dashboard import MuJoCoDashboard

            MuJoCoDashboard(exercise_filter="gait")
        instance.load_from_path.assert_called_once()

    def test_main_invokes_launch_dashboard(self) -> None:
        mod, _ = _make_engine_module(MUJOCO_MOD, "MuJoCoPhysicsEngine")
        with (
            patch.dict(sys.modules, {MUJOCO_MOD: mod}),
            patch("src.launchers.mujoco_dashboard.launch_dashboard") as ld,
        ):
            from src.launchers.mujoco_dashboard import main

            main()
        ld.assert_called_once()
        _, kw = ld.call_args
        assert kw["title"] == "MuJoCo Golf Analysis Dashboard (Unified)"


# ---------------------------------------------------------------------------
# Pinocchio
# ---------------------------------------------------------------------------


PINO_MOD = "src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine"


class TestPinocchioDashboard:
    def _patch_engine(self):
        mod, inst = _make_engine_module(PINO_MOD, "PinocchioPhysicsEngine")
        return patch.dict(sys.modules, {PINO_MOD: mod}), inst

    def test_default_title(self, stub_unified_window) -> None:
        patcher, _ = self._patch_engine()
        with patcher:
            from src.launchers.pinocchio_dashboard import PinocchioDashboard

            PinocchioDashboard()
        assert stub_unified_window["title"] == "Pinocchio Golf Analysis Dashboard"

    def test_exercise_filter_title(self, stub_unified_window) -> None:
        patcher, _ = self._patch_engine()
        with (
            patcher,
            patch(
                "src.shared.python.config.model_source_providers.pinocchio_models_source",
                side_effect=RuntimeError("nope"),
            ),
        ):
            from src.launchers.pinocchio_dashboard import PinocchioDashboard

            PinocchioDashboard(exercise_filter="run")
        assert "Run" in stub_unified_window["title"]

    def test_exercise_filter_loads_urdf(
        self, stub_unified_window, tmp_path: Path
    ) -> None:
        ex_dir = tmp_path / "exercises" / "gait"
        ex_dir.mkdir(parents=True)
        (ex_dir / "leg.urdf").write_text("<robot/>")
        patcher, instance = self._patch_engine()
        with (
            patcher,
            patch(
                "src.shared.python.config.model_source_providers.pinocchio_models_source",
                return_value=tmp_path,
            ),
        ):
            from src.launchers.pinocchio_dashboard import PinocchioDashboard

            PinocchioDashboard(exercise_filter="gait")
        instance.load_from_path.assert_called_once()

    def test_main_invokes_launch_dashboard(self) -> None:
        mod, _ = _make_engine_module(PINO_MOD, "PinocchioPhysicsEngine")
        with (
            patch.dict(sys.modules, {PINO_MOD: mod}),
            patch("src.launchers.pinocchio_dashboard.launch_dashboard") as ld,
        ):
            from src.launchers.pinocchio_dashboard import main

            main()
        ld.assert_called_once()
        _, kw = ld.call_args
        assert kw["title"] == "Pinocchio Golf Analysis Dashboard"
