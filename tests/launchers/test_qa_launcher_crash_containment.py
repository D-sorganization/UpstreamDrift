"""Regression tests for the functional-QA launcher crash class (epic #8062).

The shared rule these tests defend:

    A missing optional dependency (MuJoCo DLL, MATLAB, a GUI extra) must never
    take down the launcher. The user must get a specific, actionable message.

Issues covered: #8065, #8066, #8068, #8069, #8070, #8084, #8086, #8087.
"""

from __future__ import annotations

import builtins
import importlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.launchers.launcher_crash_policy import classify_crash
from src.launchers.launcher_failure_messages import describe_launch_failure
from src.launchers.launcher_package_mains import resolve_package_main_module

REPO_ROOT = Path(__file__).resolve().parents[2]

WIN_DLL_INIT_ERROR = (
    "[WinError 1114] A dynamic link library (DLL) initialization routine failed"
)


# ---------------------------------------------------------------------------
# #8066 / #8070 / #8072 / #8084 - the crash dialog must not kill the launcher
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCrashPolicy:
    def test_recoverable_failure_keeps_launcher_alive(self) -> None:
        action = classify_crash(ModuleNotFoundError, launcher_is_alive=True)
        assert action.quit_application is False
        assert action.show_dialog is True

    @pytest.mark.parametrize(
        "exc_type", [OSError, RuntimeError, ImportError, AttributeError, ValueError]
    )
    def test_every_ordinary_exception_is_contained(
        self, exc_type: type[BaseException]
    ) -> None:
        assert (
            classify_crash(exc_type, launcher_is_alive=True).quit_application is False
        )

    def test_startup_failure_still_exits(self) -> None:
        action = classify_crash(OSError, launcher_is_alive=False)
        assert action.quit_application is True
        assert action.show_dialog is True

    def test_system_exit_is_honoured_silently(self) -> None:
        action = classify_crash(SystemExit, launcher_is_alive=True)
        assert action.quit_application is True
        assert action.show_dialog is False

    def test_non_exception_type_is_rejected(self) -> None:
        with pytest.raises(TypeError):
            classify_crash("OSError", launcher_is_alive=True)  # type: ignore[arg-type]


@pytest.mark.unit
class TestFailureMessages:
    def test_missing_module_is_named_with_install_command(self) -> None:
        exc = ModuleNotFoundError("No module named 'sidekick.lab.bio._c3d_marker_set'")
        message = describe_launch_failure(exc, "Pose Studio")

        assert "Pose Studio" in message
        assert "sidekick.lab.bio._c3d_marker_set" in message
        assert "pip install upstream-drift[gui-tools]" in message
        assert "still running" in message
        assert "Traceback" not in message

    def test_dll_initialization_failure_gets_specific_advice(self) -> None:
        message = describe_launch_failure(
            OSError(WIN_DLL_INIT_ERROR), "Gait Model", package_hint="mujoco"
        )

        assert "1114" in message
        assert "mujoco" in message
        assert "Visual C++" in message

    def test_missing_matlab_executable_is_explained(self) -> None:
        exc = FileNotFoundError(
            "[WinError 2] The system cannot find the file specified"
        )
        message = describe_launch_failure(exc, "Simscape 2D")

        assert "MATLAB" in message
        assert "still running" in message

    def test_user_facing_exception_message_passes_through(self) -> None:
        class _Actionable(RuntimeError):
            is_user_facing_message = True

        message = describe_launch_failure(_Actionable("Install Drake first."), "Tile")
        assert message == "Install Drake first."

    def test_empty_tile_name_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            describe_launch_failure(OSError("boom"), "  ")


# ---------------------------------------------------------------------------
# #8084 - a broken MuJoCo wheel must not break the launcher's import chain
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_launcher_startup_imports_survive_broken_mujoco(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bunkershot3d (reached from the launcher's startup imports) must import
    even when ``import mujoco`` raises OSError rather than ImportError."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mujoco" or name.startswith("mujoco."):
            raise OSError(WIN_DLL_INIT_ERROR)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    for module in [m for m in list(importlib.sys.modules) if m.startswith("mujoco")]:
        monkeypatch.delitem(importlib.sys.modules, module, raising=False)

    module = importlib.import_module("src.bunkershot3d.backends.mpm.driver")
    assert module.mujoco is None


@pytest.mark.unit
def test_bunkershot3d_package_does_not_eagerly_import_backends() -> None:
    """The package __init__ must stay lazy so importing any BunkerShot3D symbol
    does not drag native backends (and MuJoCo) into the launcher process."""
    source = (REPO_ROOT / "src" / "bunkershot3d" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert "from .backends import" not in source
    assert "__getattr__" in source

    package = importlib.import_module("src.bunkershot3d")
    assert package.WrenchTrace.__name__ == "WrenchTrace"


# ---------------------------------------------------------------------------
# #8065 / #8069 / #8086 - package mains and entry points
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPackageMainResolution:
    def test_pendulum_main_keeps_the_full_src_path(self) -> None:
        script = (
            REPO_ROOT
            / "src"
            / "shared"
            / "python"
            / "pendulum_simulator"
            / "__main__.py"
        )
        assert script.exists()
        assert (
            resolve_package_main_module(script, REPO_ROOT)
            == "src.shared.python.pendulum_simulator"
        )

    def test_tools_package_main_keeps_the_src_prefix(self) -> None:
        script = REPO_ROOT / "src" / "tools" / "pose_studio" / "__main__.py"
        if not script.exists():
            pytest.skip("pose_studio package main not present")
        assert resolve_package_main_module(script, REPO_ROOT) == "src.tools.pose_studio"

    def test_plain_script_is_not_treated_as_a_package_main(self) -> None:
        script = REPO_ROOT / "src" / "launchers" / "shot_tracer.py"
        assert resolve_package_main_module(script, REPO_ROOT) is None

    def test_main_without_package_init_is_rejected(self, tmp_path: Path) -> None:
        loose = tmp_path / "pkg"
        loose.mkdir()
        (loose / "__main__.py").write_text("", encoding="utf-8")
        assert resolve_package_main_module(loose / "__main__.py", tmp_path) is None

    def test_non_path_argument_is_rejected(self) -> None:
        with pytest.raises(TypeError):
            resolve_package_main_module("x/__main__.py", REPO_ROOT)  # type: ignore[arg-type]


@pytest.mark.unit
def test_special_app_handler_runs_package_mains_as_modules() -> None:
    """Running a ``__main__.py`` as a bare script makes its relative imports
    fail and the child exits instantly (#8065)."""
    from src.launchers.launcher_model_handlers import SpecialAppHandler

    model = MagicMock()
    model.id = "pendulum_simulator"
    model.name = "Pendulum Simulator"
    model.path = "src/shared/python/pendulum_simulator/__main__.py"
    model.type = "special_app"
    model.source_root = None
    model.provider = None

    process_manager = MagicMock()
    process_manager.launch_module.return_value = object()

    assert SpecialAppHandler().launch(model, REPO_ROOT, process_manager) is True

    process_manager.launch_script.assert_not_called()
    process_manager.launch_module.assert_called_once()
    assert (
        process_manager.launch_module.call_args.kwargs["module_name"]
        == "src.shared.python.pendulum_simulator"
    )


@pytest.mark.unit
def test_shot_tracer_script_has_an_entry_point() -> None:
    """The launcher spawns this file directly; without a __main__ guard the
    child did nothing and exited 0 (#8069)."""
    source = (REPO_ROOT / "src" / "launchers" / "shot_tracer.py").read_text(
        encoding="utf-8"
    )
    assert 'if __name__ == "__main__":' in source
    assert source.rstrip().endswith("main()")


@pytest.mark.unit
def test_pendulum_gui_perturbation_import_resolves() -> None:
    """The panel imported ``..perturbation.config`` (a package that does not
    exist) instead of ``...perturbation.config`` (#8065)."""
    import importlib.util

    assert importlib.util.find_spec("src.shared.python.perturbation.config") is not None
    source = (
        REPO_ROOT / "src/shared/python/pendulum_simulator/gui/perturbation_panel.py"
    ).read_text(encoding="utf-8")
    assert "from ...perturbation.config import" in source


# ---------------------------------------------------------------------------
# #8087 - every visible provider tile has an executable, contained handler
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProviderAssetHandler:
    @pytest.mark.parametrize(
        "model_type", ["mjcf", "urdf", "osim", "sdformat-1.8", "SDFormat-1.8"]
    )
    def test_registry_resolves_every_provider_asset_type(self, model_type: str) -> None:
        from src.launchers.launcher_model_handlers import ModelHandlerRegistry

        assert ModelHandlerRegistry().get_handler(model_type) is not None

    def test_every_declared_engine_viewer_exists(self) -> None:
        """A moved viewer must surface here, not as a dead tile at runtime."""
        from src.launchers.launcher_provider_asset_handler import _ENGINE_VIEWERS

        missing = [
            rel
            for rel in set(_ENGINE_VIEWERS.values())
            if not (REPO_ROOT / rel).exists()
        ]
        assert not missing, f"Declared engine viewers do not exist: {missing}"

    def test_missing_engine_runtime_yields_actionable_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.launchers import launcher_provider_asset_handler as mod

        monkeypatch.setattr(mod, "is_engine_runtime_available", lambda _engine: False)

        model = MagicMock()
        model.id = "drake_models-squat"
        model.name = "Drake Squat"
        model.type = "sdformat-1.8"
        model.engine_type = "drake"

        with pytest.raises(mod.EngineRuntimeUnavailableError) as excinfo:
            mod.ProviderModelAssetHandler().launch(model, REPO_ROOT, MagicMock())

        message = str(excinfo.value)
        assert "Drake" in message
        assert "pip install drake" in message
        assert "still running" in message
        assert excinfo.value.is_user_facing_message is True

    def test_specific_handlers_still_win_over_the_asset_handler(self) -> None:
        from src.launchers.launcher_model_handlers import (
            ModelHandlerRegistry,
            SpecialAppHandler,
        )

        handler = ModelHandlerRegistry().get_handler("special_app")
        assert isinstance(handler, SpecialAppHandler)
