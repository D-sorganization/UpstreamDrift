"""Comprehensive regression tests for launcher issues fixed in session 2026-03-23.

These tests ensure that every launch-related bug we've fixed can never
recur without being caught by CI. Each test documents the original failure
and validates the fix.
"""

from __future__ import annotations

import enum
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# PR #2086: ThemePreset and apply_golf_suite_style must be importable.
# ---------------------------------------------------------------------------


class TestThemeBackwardCompat:
    """PR #2086: ThemePreset and apply_golf_suite_style must be importable."""

    def test_theme_preset_importable(self) -> None:
        from src.shared.python.theme import ThemePreset

        assert issubclass(ThemePreset, enum.Enum)

    def test_theme_preset_has_dark_light(self) -> None:
        from src.shared.python.theme import ThemePreset

        assert ThemePreset.DARK.value == "Dark"
        assert ThemePreset.LIGHT.value == "Light"
        assert ThemePreset.HIGH_CONTRAST.value == "High Contrast"

    def test_apply_golf_suite_style_importable(self) -> None:
        from src.shared.python.theme import apply_golf_suite_style

        assert callable(apply_golf_suite_style)

    def test_apply_golf_suite_style_no_error(self) -> None:
        from src.shared.python.theme import apply_golf_suite_style

        apply_golf_suite_style()  # Must not raise


# ---------------------------------------------------------------------------
# PR #2086: launcher_theme.py must use current ThemeManager API.
# ---------------------------------------------------------------------------


class TestLauncherThemeApi:
    """PR #2086: launcher_theme.py must use current ThemeManager API."""

    @staticmethod
    def _get_launcher_theme_source() -> str:
        mod = importlib.import_module("src.launchers.launcher_theme")
        source_file = mod.__file__
        assert source_file is not None
        return Path(source_file).read_text(encoding="utf-8")

    def test_no_stale_set_theme(self) -> None:
        source = self._get_launcher_theme_source()
        assert "manager.set_theme(" not in source

    def test_no_stale_load_saved_theme(self) -> None:
        source = self._get_launcher_theme_source()
        assert "load_saved_theme" not in source

    def test_no_stale_get_available_fleet_themes(self) -> None:
        source = self._get_launcher_theme_source()
        assert "get_available_fleet_themes" not in source

    def test_no_stale_set_fleet_theme(self) -> None:
        source = self._get_launcher_theme_source()
        assert "set_fleet_theme" not in source


# ---------------------------------------------------------------------------
# PR #2086: Engine API must serialize numpy arrays to JSON.
# ---------------------------------------------------------------------------


class TestEngineNumpySerialization:
    """PR #2086: Engine API must serialize numpy arrays to JSON."""

    def test_sanitize_numpy_array(self) -> None:
        from src.api.routes.engines import _sanitize_for_json

        result = _sanitize_for_json(np.array([1.0, 2.0]))
        assert result == [1.0, 2.0]
        assert isinstance(result, list)

    def test_sanitize_numpy_scalar(self) -> None:
        from src.api.routes.engines import _sanitize_for_json

        assert isinstance(_sanitize_for_json(np.int64(42)), int)
        assert isinstance(_sanitize_for_json(np.float64(3.14)), float)

    def test_sanitize_nested(self) -> None:
        from src.api.routes.engines import _sanitize_for_json

        data = {"state": np.array([1.0]), "n": np.int32(5)}
        result = _sanitize_for_json(data)
        assert isinstance(result["state"], list)
        assert isinstance(result["n"], int)

    def test_sanitize_none(self) -> None:
        from src.api.routes.engines import _sanitize_for_json

        assert _sanitize_for_json(None) is None

    def test_sanitize_plain_types_pass_through(self) -> None:
        from src.api.routes.engines import _sanitize_for_json

        assert _sanitize_for_json("hello") == "hello"
        assert _sanitize_for_json(42) == 42
        assert _sanitize_for_json(3.14) == 3.14


# ---------------------------------------------------------------------------
# PR #2087: signal_toolkit must import contracts via absolute path.
# ---------------------------------------------------------------------------


class TestSignalToolkitImport:
    """PR #2087: signal_toolkit must import contracts via absolute path."""

    def test_contracts_require_importable(self) -> None:
        from src.shared.python.contracts import require

        assert callable(require)

    def test_signal_core_importable(self) -> None:
        from src.shared.python.signal_toolkit.core import Signal

        assert Signal is not None

    def test_absolute_contracts_import_in_signal_core(self) -> None:
        """signal_toolkit/core.py must try absolute import path for contracts."""
        source_path = (
            REPO_ROOT
            / "vendor"
            / "ud-tools"
            / "src"
            / "shared"
            / "python"
            / "signal_toolkit"
            / "core.py"
        )
        assert source_path.exists(), f"Expected {source_path} to exist"
        source = source_path.read_text(encoding="utf-8")
        assert (
            "src.shared.python.contracts" in source
        ), "signal_toolkit/core.py must try absolute import path for contracts"


# ---------------------------------------------------------------------------
# PR #2088: Engine manager must find all engines correctly.
# ---------------------------------------------------------------------------


class TestEngineDiscoveryPaths:
    """PR #2088: Engine manager must find all engines including MyoSuite."""

    def test_prefers_src_engines(self) -> None:
        """EngineManager should use src/engines when the repo root is passed."""
        from src.shared.python.engine_core.engine_manager import EngineManager

        em = EngineManager(REPO_ROOT)
        assert "src" in str(
            em.engines_root
        ), f"Expected engines_root under src/, got {em.engines_root}"

    def test_myosuite_not_myosim(self) -> None:
        """MYOSIM engine path should point to myosuite, not myosim."""
        from src.shared.python.engine_core.engine_manager import EngineManager
        from src.shared.python.engine_core.engine_registry import EngineType

        em = EngineManager(REPO_ROOT)
        assert (
            em.engine_paths[EngineType.MYOSIM].name == "myosuite"
        ), f"Expected 'myosuite', got '{em.engine_paths[EngineType.MYOSIM].name}'"


# ---------------------------------------------------------------------------
# PR #2088: Window must not exceed screen size.
# ---------------------------------------------------------------------------


class TestWindowSizing:
    """PR #2088: Window must not hardcode 1400x900 size."""

    def test_initial_resize_not_hardcoded_1400(self) -> None:
        """upstream_drift_launcher.py must not hardcode resize(1400, 900)."""
        source_path = REPO_ROOT / "src" / "launchers" / "upstream_drift_launcher.py"
        assert source_path.exists(), f"Expected {source_path} to exist"
        source = source_path.read_text(encoding="utf-8")
        assert (
            "self.resize(1400, 900)" not in source
        ), "Window size must be screen-aware, not hardcoded 1400x900"


# ---------------------------------------------------------------------------
# PR #2087: MuJoCo launcher _get_launch_env must include shared_python.
# ---------------------------------------------------------------------------


class TestSubprocessPythonpath:
    """PR #2087/#2089: Subprocesses must have complete PYTHONPATH."""

    def test_mujoco_launcher_has_get_launch_env(self) -> None:
        from src.launchers.mujoco_unified_launcher import MujocoUnifiedLauncher

        assert hasattr(MujocoUnifiedLauncher, "_get_launch_env")

    def test_mujoco_launch_env_has_shared_python(self) -> None:
        from src.launchers.mujoco_unified_launcher import MujocoUnifiedLauncher

        env = MujocoUnifiedLauncher._get_launch_env()
        # shared_python must be on PYTHONPATH
        pythonpath = env.get("PYTHONPATH", "")
        assert (
            "shared" in pythonpath and "python" in pythonpath
        ), f"Expected shared/python in PYTHONPATH, got: {pythonpath}"

    def test_process_manager_env_has_shared_python(self) -> None:
        """ProcessManager.get_subprocess_env must include src/shared/python."""
        from src.launchers.launcher_process_manager import ProcessManager

        pm = ProcessManager(REPO_ROOT)
        env = pm.get_subprocess_env()
        pythonpath = env.get("PYTHONPATH", "")
        shared_python = str(REPO_ROOT / "src" / "shared" / "python")
        assert (
            shared_python in pythonpath
        ), f"Expected {shared_python} in PYTHONPATH, got: {pythonpath}"

    def test_process_manager_env_has_repo_root(self) -> None:
        """ProcessManager.get_subprocess_env must include repo root."""
        from src.launchers.launcher_process_manager import ProcessManager

        pm = ProcessManager(REPO_ROOT)
        env = pm.get_subprocess_env()
        pythonpath = env.get("PYTHONPATH", "")
        assert (
            str(REPO_ROOT) in pythonpath
        ), f"Expected {REPO_ROOT} in PYTHONPATH, got: {pythonpath}"


# ---------------------------------------------------------------------------
# PR #2089: Drake must use ModuleHandler, not ScriptHandler.
# ---------------------------------------------------------------------------


class TestDrakeModuleHandler:
    """PR #2089: Drake must use ModuleHandler, not ScriptHandler."""

    def test_drake_handler_exists(self) -> None:
        from src.launchers.launcher_model_handlers import ModelHandlerRegistry

        registry = ModelHandlerRegistry()
        handler = registry.get_handler("drake")
        assert handler is not None, "Drake must have a registered handler"

    def test_drake_is_module_handler(self) -> None:
        from src.launchers.launcher_model_handlers import (
            ModelHandlerRegistry,
            ModuleHandler,
        )

        registry = ModelHandlerRegistry()
        handler = registry.get_handler("drake")
        assert isinstance(handler, ModuleHandler), (
            f"Drake must use ModuleHandler (not ScriptHandler) due to relative imports, "
            f"got {type(handler).__name__}"
        )

    def test_drake_golf_handler_exists(self) -> None:
        from src.launchers.launcher_model_handlers import ModelHandlerRegistry

        registry = ModelHandlerRegistry()
        handler = registry.get_handler("drake_golf")
        assert handler is not None, "drake_golf must have a registered handler"


# ---------------------------------------------------------------------------
# Launch diagnostics: _execute_local_launch logs failure details.
# ---------------------------------------------------------------------------


class TestLaunchDiagnostics:
    """Launch failure diagnostics provide actionable error information."""

    def test_execute_local_launch_logs_failure(self) -> None:
        """When handler.launch() returns False, detailed diagnostics are logged."""
        from src.launchers.launcher_simulation import LauncherSimulationMixin

        mixin = LauncherSimulationMixin.__new__(LauncherSimulationMixin)

        mock_model = MagicMock()
        mock_model.name = "TestEngine"
        mock_model.type = "test_type"
        mock_model.path = "some/path"

        mock_handler = MagicMock()
        mock_handler.launch.return_value = False

        mock_registry = MagicMock()
        mock_registry.get_handler.return_value = mock_handler

        mixin.model_handler_registry = mock_registry
        mixin.process_manager = MagicMock()
        mixin.show_toast = MagicMock()
        mixin.lbl_status = MagicMock()

        with (
            patch("src.launchers.launcher_simulation.REPOS_ROOT", REPO_ROOT),
            patch("src.launchers.launcher_simulation.logger") as mock_logger,
        ):
            mixin._execute_local_launch(mock_model)

        # Must have logged the failure with useful context
        mock_logger.error.assert_called()
        call_args = str(mock_logger.error.call_args)
        assert (
            "TestEngine" in call_args or "test_type" in call_args
        ), f"Error log must mention engine name or type, got: {call_args}"

    def test_execute_local_launch_success_path(self) -> None:
        """When handler.launch() returns True, success toast is shown."""
        from src.launchers.launcher_simulation import LauncherSimulationMixin

        mixin = LauncherSimulationMixin.__new__(LauncherSimulationMixin)

        mock_model = MagicMock()
        mock_model.name = "TestEngine"
        mock_model.type = "test_type"
        mock_model.path = "some/path"

        mock_handler = MagicMock()
        mock_handler.launch.return_value = True

        mock_registry = MagicMock()
        mock_registry.get_handler.return_value = mock_handler

        mixin.model_handler_registry = mock_registry
        mixin.process_manager = MagicMock()
        mixin.show_toast = MagicMock()
        mixin.lbl_status = MagicMock()

        with patch("src.launchers.launcher_simulation.REPOS_ROOT", REPO_ROOT):
            mixin._execute_local_launch(mock_model)

        mixin.show_toast.assert_called_once()
        assert "Launched" in mixin.show_toast.call_args[0][0]


# ---------------------------------------------------------------------------
# ProcessManager immediate-death detection.
# ---------------------------------------------------------------------------


class TestProcessImmediateDeathDetection:
    """ProcessManager must detect and log immediate process death."""

    def test_launch_script_logs_immediate_exit(self, tmp_path: Path) -> None:
        """If a launched script exits immediately, stderr is captured and logged."""
        from src.launchers.launcher_process_manager import ProcessManager

        # Create a script that exits with error immediately.
        # Patch validate_script_path because this test exercises process-lifecycle
        # behaviour, not input-sanitization (which is tested separately).
        bad_script = tmp_path / "bad_script.py"
        bad_script.write_text("import sys; sys.exit(1)\n")

        pm = ProcessManager(REPO_ROOT)
        with (
            patch("src.launchers.launcher_process_manager.validate_script_path"),
            patch(
                "src.launchers.launcher_process_manager.secure_popen",
                side_effect=lambda cmd, cwd=None, suite_root=None, **kw: __import__(
                    "subprocess"
                ).Popen(cmd, cwd=str(cwd) if cwd else None, **kw),
            ),
            patch.object(pm, "_emit_output") as mock_emit,
        ):
            process = pm.launch_script("test_bad", bad_script, tmp_path)
            assert process is not None, "launch_script should return a process"

            # Wait for the process to finish
            process.wait(timeout=5)

            # The _stream_output thread should eventually emit the exit code
            # Give the thread a moment to finish reading
            import time

            time.sleep(1.0)

            # Check that exit code was logged
            calls = [str(c) for c in mock_emit.call_args_list]
            exit_logged = any("exited with code" in c for c in calls)
            assert (
                exit_logged
            ), f"Expected 'exited with code' in output, got calls: {calls}"

    def test_launch_module_logs_immediate_exit(self, tmp_path: Path) -> None:
        """If a launched module exits immediately, the exit is detected."""
        from src.launchers.launcher_process_manager import ProcessManager

        # Create a minimal module that fails.
        # Patch secure_popen's cwd check by using the real subprocess.Popen
        # with a relaxed cwd so that process-lifecycle behaviour can be tested
        # independently of security validation (tested separately).
        mod_dir = tmp_path / "failing_mod"
        mod_dir.mkdir()
        (mod_dir / "__main__.py").write_text(
            "import sys; print('failing', file=sys.stderr); sys.exit(2)\n"
        )

        pm = ProcessManager(REPO_ROOT)
        with (
            patch(
                "src.launchers.launcher_process_manager.secure_popen",
                side_effect=lambda cmd, cwd=None, suite_root=None, **kw: __import__(
                    "subprocess"
                ).Popen(cmd, cwd=str(cwd) if cwd else None, **kw),
            ),
            patch.object(pm, "_emit_output") as mock_emit,
        ):
            process = pm.launch_module("test_bad_mod", "failing_mod", tmp_path)
            assert process is not None

            process.wait(timeout=5)

            import time

            time.sleep(1.0)

            calls = [str(c) for c in mock_emit.call_args_list]
            exit_logged = any("exited with code" in c for c in calls)
            assert exit_logged, f"Expected exit code in output, got: {calls}"
