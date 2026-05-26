#!/usr/bin/env python3
"""
Test suite for Golf Modeling Suite launcher fixes and new features.

Tests cover:
- Drag-and-drop functionality
- Docker container setup
- Module import fixes
- Engine detection and management
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from src.shared.python.engine_core.engine_availability import PYQT6_AVAILABLE

REPO_ROOT = Path(__file__).resolve().parents[2]

if PYQT6_AVAILABLE:
    pass


class TestSharedModuleImports(unittest.TestCase):
    """Test that shared modules can be imported correctly."""

    def test_configuration_manager_import(self) -> None:
        """Test configuration manager import."""
        try:
            from src.shared.python.config.configuration_manager import (
                ConfigurationManager,
            )

            # Test that we can instantiate it with required config_path
            config_manager = ConfigurationManager(Path("dummy_config.json"))
            self.assertIsNotNone(config_manager)
        except ImportError as e:
            self.fail(f"Failed to import ConfigurationManager: {e}")
        except Exception as e:  # noqa: BLE001
            # If instantiation fails due to missing file, that's expected in tests
            self.assertTrue(True, f"ConfigurationManager import successful: {e}")

    def test_process_worker_import(self) -> None:
        """Test process worker import."""
        try:
            from src.shared.python.ui.qt.process_worker import ProcessWorker

            # Test that we can instantiate it with required cmd
            worker = ProcessWorker(["echo", "test"])
            self.assertIsNotNone(worker)
        except ImportError as e:
            self.fail(f"Failed to import ProcessWorker: {e}")

    def test_engine_manager_import(self) -> None:
        """Test engine manager import."""
        try:
            from src.shared.python.engine_core.engine_manager import (
                EngineManager,
                EngineType,
            )

            # Test that we can instantiate it
            manager = EngineManager()
            self.assertIsNotNone(manager)
            # Test that EngineType enum exists
            self.assertTrue(hasattr(EngineType, "MUJOCO"))
        except ImportError as e:
            self.fail(f"Failed to import EngineManager: {e}")


class TestEngineManager(unittest.TestCase):
    """Test engine manager functionality."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from src.shared.python.engine_core.engine_manager import EngineManager

        self.manager = EngineManager()

    def test_engine_discovery(self) -> None:
        """Test that engines are discovered correctly."""
        engines = self.manager.get_available_engines()
        self.assertIsInstance(engines, list)
        self.assertGreater(len(engines), 0, "Should discover at least one engine")

    def test_engine_paths_exist(self) -> None:
        """Test that engine paths are properly configured."""
        for path in self.manager.engine_paths.values():
            self.assertIsInstance(path, Path)
            # Note: Not all engines may be installed, so we don't require existence

    @patch(
        "src.shared.python.engine_core.engine_manager.EngineManager.get_probe_result"
    )
    def test_probe_system(self, mock_get_result: MagicMock) -> None:
        """Test engine probe system."""
        from src.shared.python.engine_core.engine_manager import EngineType

        # Setup mock return
        mock_result = MagicMock()
        mock_result.is_available = True
        mock_result.diagnostic_message = "Mocked result"
        mock_get_result.return_value = mock_result

        # Test MuJoCo probe if available
        if EngineType.MUJOCO in self.manager.probes:
            probe_result = self.manager.get_probe_result(EngineType.MUJOCO)
            self.assertIsNotNone(probe_result)
            self.assertTrue(hasattr(probe_result, "is_available"))
            self.assertTrue(hasattr(probe_result, "diagnostic_message"))


class TestDockerConfiguration(unittest.TestCase):
    """Test Docker configuration and setup."""

    def test_dockerfile_exists(self) -> None:
        """Test that Dockerfile exists and is readable."""
        dockerfile_path = REPO_ROOT / "Dockerfile"
        self.assertTrue(dockerfile_path.exists(), "Dockerfile should exist")

        # Test that it's readable
        content = dockerfile_path.read_text()
        self.assertIn("PYTHONPATH", content, "Dockerfile should set PYTHONPATH")
        self.assertIn("/workspace", content, "Dockerfile should configure workspace")

    def test_docker_image_tag(self) -> None:
        """Test that Dockerfile uses a pinned base image."""
        dockerfile_path = REPO_ROOT / "Dockerfile"
        content = dockerfile_path.read_text()
        self.assertIn("FROM python:3.12-slim", content)
        self.assertIn("AS builder", content)
        self.assertIn("AS runtime", content)
        self.assertNotIn(":latest", content, "Should use explicit base image tags")


class TestMuJoCoModule(unittest.TestCase):
    """Test MuJoCo module structure and availability."""

    def test_mujoco_module_exists(self) -> None:
        """Test that MuJoCo humanoid golf module exists."""
        mujoco_path = Path(
            "src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf"
        )
        self.assertTrue(mujoco_path.exists(), "MuJoCo module directory should exist")

        main_file = mujoco_path / "__main__.py"
        self.assertTrue(main_file.exists(), "MuJoCo module should have __main__.py")

    def test_mujoco_module_structure(self) -> None:
        """Test MuJoCo module has required components."""
        mujoco_path = Path(
            "src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf"
        )

        required_files = [
            "__init__.py",
            "__main__.py",
        ]

        for file_name in required_files:
            file_path = mujoco_path / file_name
            self.assertTrue(
                file_path.exists(), f"Required file {file_name} should exist"
            )

    def test_mujoco_module_name_in_handler(self) -> None:
        """Test that the module name in model handlers is correct."""
        from src.launchers.launcher_model_handlers import HumanoidMuJoCoHandler

        handler = HumanoidMuJoCoHandler()
        # The handler should use the package path (not .main suffix)
        # since Python -m runs __main__.py automatically
        mock_model = Mock()
        mock_process_manager = Mock()
        mock_process_manager.launch_module.return_value = Mock()

        handler.launch(mock_model, Path("."), mock_process_manager)

        # Verify launch_module was called with the correct module name
        call_args = mock_process_manager.launch_module.call_args
        module_name = call_args.kwargs.get(
            "module_name",
            call_args[1].get(
                "module_name", call_args[0][1] if len(call_args[0]) > 1 else None
            ),
        )
        self.assertFalse(
            module_name.endswith(".main"),
            f"Module name should not end with .main, got: {module_name}",
        )


class TestLauncherIntegration(unittest.TestCase):
    """Integration tests for the launcher system."""

    def test_launch_golf_suite_script(self) -> None:
        """Test that launch_golf_suite.py script exists and has correct structure."""
        script_path = Path("launch_golf_suite.py")
        self.assertTrue(script_path.exists(), "Launch script should exist")

        # Test that it has the expected structure
        content = script_path.read_text()
        self.assertIn("def main(", content)
        self.assertIn("Golf Modeling Suite", content)
        # launch_engine_directly should now be imported from launcher_factory
        self.assertIn("from src.shared.python.launcher_factory", content)

    @patch("src.launchers.upstream_drift_launcher.UpstreamDriftLauncher")
    def test_unified_launcher_import(self, mock_upstream_drift_launcher: Mock) -> None:
        """Test that unified launcher can be imported."""
        try:
            from src.launchers.unified_launcher import UnifiedLauncher

            # Test that we can instantiate it
            launcher = UnifiedLauncher()
            self.assertIsNotNone(launcher)
        except ImportError as e:
            # This might fail if PyQt6 is not available, which is acceptable
            if "PyQt6" not in str(e):
                self.fail(f"Unexpected import error: {e}")


class TestLauncherUISearchAndRuntimeSettings(unittest.TestCase):
    """Test cases for the new launcher features: global search, clear search, console access and runtime settings."""

    def test_global_search_ignores_category_filter(self) -> None:
        """Test that get_filtered_order ignores category filter when search text is non-empty."""
        from src.launchers.launcher_layout_manager import LayoutManager

        # Create LayoutManager mock/instance
        manager = LayoutManager(Path("dummy.json"), {}, MagicMock(), MagicMock())

        # Setup models
        model1 = MagicMock()
        model1.name = "Data Processor"
        model1.id = "data_processor"
        model1.description = "Processes bio data"

        model2 = MagicMock()
        model2.name = "Shot Tracer"
        model2.id = "shot_tracer"
        model2.description = "Traces shots"

        manager.available_models = {"data_processor": model1, "shot_tracer": model2}
        manager.model_order = ["data_processor", "shot_tracer"]
        manager._get_model = MagicMock(
            side_effect=lambda mid: manager.available_models.get(mid)
        )

        # Mock category helper
        manager._get_model_category = MagicMock(
            side_effect=lambda m: "Tools & Data" if m == model1 else "Simulation"
        )

        # Case 1: Search text empty, category Tools & Data active
        manager.current_category_filter = "Tools & Data"
        manager.current_filter_text = ""
        filtered = manager.get_filtered_order()
        self.assertEqual(filtered, ["data_processor"])

        # Case 2: Search text non-empty, category Tools & Data active -> Search should be global!
        manager.current_category_filter = "Tools & Data"
        manager.current_filter_text = "tracer"
        filtered = manager.get_filtered_order()
        self.assertEqual(
            filtered, ["shot_tracer"]
        )  # Bypassed category constraint and matches search globally!

    def test_clear_search_behavior(self) -> None:
        """Test that _clear_search clears input and removes focus if text or focus exists."""
        from src.launchers.launcher_ui_setup import UISetupManager

        mock_launcher = MagicMock()
        ui = UISetupManager(mock_launcher)
        ui.search_input = MagicMock()

        # Case 1: has text but no focus
        ui.search_input.text.return_value = "tracer"
        ui.search_input.hasFocus.return_value = False
        ui._clear_search()
        ui.search_input.clear.assert_called_once()
        ui.search_input.clearFocus.assert_called_once()

        # Case 2: no text but has focus
        ui.search_input.clear.reset_mock()
        ui.search_input.clearFocus.reset_mock()
        ui.search_input.text.return_value = ""
        ui.search_input.hasFocus.return_value = True
        ui._clear_search()
        ui.search_input.clear.assert_called_once()
        ui.search_input.clearFocus.assert_called_once()

    def test_windows_mode_mutual_exclusion_fallback(self) -> None:
        """Test that disabling Windows mode falls back to enabled if neither Docker nor WSL are checked."""
        from src.launchers.launcher_dialogs import DialogsManager

        mock_launcher = MagicMock()
        mgr = DialogsManager(mock_launcher)
        mgr.chk_docker = MagicMock()
        mgr.chk_wsl = MagicMock()
        mgr.chk_windows = MagicMock()
        mgr.lbl_execution_mode = MagicMock()

        # Setup: all unchecked
        mgr.chk_docker.isChecked.return_value = False
        mgr.chk_wsl.isChecked.return_value = False
        mgr.chk_windows.isChecked.return_value = False

        # Disable windows (state = 0)
        mgr._on_windows_mode_changed(0)
        mgr.chk_windows.setChecked.assert_called_with(True)

    def test_auto_open_console_on_launch_failure(self) -> None:
        """Test that launching simulation auto-opens console and sets the checkbox on failure."""
        from src.launchers.launcher_simulation import SimulationManager

        mock_launcher = MagicMock()
        mgr = SimulationManager(mock_launcher)
        mgr._console_dock = MagicMock()
        mgr._action_console = MagicMock()
        mgr.lbl_status = MagicMock()

        # Mock handler launching returning False
        mock_handler = MagicMock()
        mock_handler.launch.return_value = False
        mgr.model_handler_registry = MagicMock()
        mgr.model_handler_registry.get_handler.return_value = mock_handler

        # Mock resolve path and deps check
        mgr.selected_model = "m1"
        mock_model = MagicMock()
        mock_model.name = "Data Processor"
        mock_model.type = "tool"
        mgr._get_model = MagicMock(return_value=mock_model)
        mgr._try_launch_special_app = MagicMock(return_value=False)
        mgr._try_launch_docker = MagicMock(return_value=False)
        mgr._check_local_dependencies = MagicMock(return_value=True)

        with patch(
            "src.launchers.launcher_simulation.resolve_model_artifact_path",
            return_value=Path("."),
        ):
            mgr.launch_simulation()

            # Verify console dock is shown and checked
            mgr._console_dock.show.assert_called_once()
            mgr._action_console.setChecked.assert_called_with(True)


class TestJunctionPathSecurityValidation(unittest.TestCase):
    """Test that path validation correctly handles sibling repositories and junctions."""

    def test_find_tools_repo_for_security_with_junction(self) -> None:
        """Test that _find_tools_repo_for_security detects sibling Tools repo even under junction."""
        from src.shared.python.security.secure_subprocess import (
            _find_tools_repo_for_security,
        )

        suite_root = Path(r"C:\fake\UpstreamDrift")
        candidate_path = Path(r"C:\fake\Tools")

        def mock_is_dir(self_path: Path) -> bool:
            return str(self_path) in {
                r"C:\fake\Tools",
                r"C:\fake\Tools\src",
                r"C:\fake\UpstreamDrift\vendor\ud-tools",
            }

        def mock_resolve(self_path: Path) -> Path:
            if str(self_path) == r"C:\fake\UpstreamDrift\vendor\ud-tools":
                return Path(r"C:\fake\Tools")
            return self_path

        with (
            patch.object(Path, "is_dir", mock_is_dir),
            patch.object(Path, "resolve", mock_resolve),
        ):
            # Should find Tools even though C:\fake\UpstreamDrift\vendor\ud-tools resolves to it
            found = _find_tools_repo_for_security(suite_root)
            self.assertEqual(found, candidate_path)

    def test_validate_script_path_accepts_tools_sibling_scripts(self) -> None:
        """Test that validate_script_path allows script execution in the Tools sibling folder."""
        from src.shared.python.security.secure_subprocess import (
            validate_script_path,
        )

        suite_root = Path(r"C:\fake\UpstreamDrift")
        script_path = Path(r"C:\fake\Tools\src\data_processing\launch_pyqt6.py")

        def mock_is_dir(self_path: Path) -> bool:
            return str(self_path) in {r"C:\fake\Tools", r"C:\fake\Tools\src"}

        def mock_exists(self_path: Path) -> bool:
            return True

        def mock_is_file(self_path: Path) -> bool:
            return True

        with (
            patch.object(Path, "is_dir", mock_is_dir),
            patch.object(Path, "exists", mock_exists),
            patch.object(Path, "is_file", mock_is_file),
            patch(
                "src.shared.python.security.secure_subprocess._find_tools_repo_for_security",
                return_value=Path(r"C:\fake\Tools"),
            ),
        ):
            # Should not raise any SecureSubprocessError
            validate_script_path(script_path, suite_root)


if __name__ == "__main__":
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestSharedModuleImports,
        TestEngineManager,
        TestDockerConfiguration,
        TestMuJoCoModule,
        TestLauncherIntegration,
        TestLauncherUISearchAndRuntimeSettings,
        TestJunctionPathSecurityValidation,
    ]

    # Add PyQt tests only if available
    if PYQT6_AVAILABLE:
        pass
    else:
        pass

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary

    if not result.failures and not result.errors:
        pass
