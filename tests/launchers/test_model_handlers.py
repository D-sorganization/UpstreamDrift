"""TDD Tests for Model Launch Handlers.

Tests the SpecialAppHandler and PuttingGreenHandler to ensure
motion_capture, model_explorer, matlab_unified, and putting_green
tiles can be launched correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.launchers.launcher_model_handlers import (
    ModelHandlerRegistry,
    PuttingGreenHandler,
    SpecialAppHandler,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockModel:
    """Minimal mock model matching the ModelConfig protocol."""

    id: str
    name: str
    path: str
    type: str


@pytest.fixture
def repo_path(tmp_path: Path) -> Path:
    """Create a temporary repo root with mock scripts."""
    # Create mock script files
    scripts = [
        "src/launchers/motion_capture_launcher.py",
        "src/launchers/matlab_launcher_unified.py",
        "src/tools/urdf_generator/launch_urdf_generator.py",
        "src/engines/physics_engines/putting_green/python/simulator.py",
    ]
    for script in scripts:
        script_path = tmp_path / script
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("# mock script\n")
    return tmp_path


@pytest.fixture
def process_manager() -> MagicMock:
    """Mock ProcessManager that returns a mock process."""
    pm = MagicMock()
    pm.launch_script.return_value = MagicMock()
    pm.launch_module.return_value = MagicMock()
    return pm


# =============================================================================
# SpecialAppHandler Tests
# =============================================================================


class TestSpecialAppHandler:
    """Test SpecialAppHandler for special_app model types."""

    def test_can_handle_special_app(self) -> None:
        """Handler accepts 'special_app' type."""
        handler = SpecialAppHandler()
        assert handler.can_handle("special_app")

    def test_cannot_handle_physics_engine(self) -> None:
        """Handler rejects physics engine types."""
        handler = SpecialAppHandler()
        assert not handler.can_handle("mujoco")
        assert not handler.can_handle("drake")
        assert not handler.can_handle("pinocchio")

    def test_launch_motion_capture(
        self, repo_path: Path, process_manager: MagicMock
    ) -> None:
        """Motion capture tile launches successfully."""
        model = MockModel(
            id="motion_capture",
            name="Motion Capture",
            path="src/launchers/motion_capture_launcher.py",
            type="special_app",
        )
        handler = SpecialAppHandler()
        result = handler.launch(model, repo_path, process_manager)

        assert result is True
        process_manager.launch_script.assert_called_once()
        call_kwargs = process_manager.launch_script.call_args
        assert call_kwargs.kwargs["name"] == "Motion Capture"

    def test_launch_model_explorer(
        self, repo_path: Path, process_manager: MagicMock
    ) -> None:
        """Model explorer tile launches successfully."""
        model = MockModel(
            id="model_explorer",
            name="Model Explorer",
            path="src/tools/urdf_generator/launch_urdf_generator.py",
            type="special_app",
        )
        handler = SpecialAppHandler()
        result = handler.launch(model, repo_path, process_manager)

        assert result is True
        process_manager.launch_script.assert_called_once()

    def test_launch_matlab_unified(
        self, repo_path: Path, process_manager: MagicMock
    ) -> None:
        """Matlab unified tile launches successfully."""
        model = MockModel(
            id="matlab_unified",
            name="Matlab Models",
            path="src/launchers/matlab_launcher_unified.py",
            type="special_app",
        )
        handler = SpecialAppHandler()
        result = handler.launch(model, repo_path, process_manager)

        assert result is True

    def test_launch_fails_for_missing_path(
        self, repo_path: Path, process_manager: MagicMock
    ) -> None:
        """DBC: Launch fails when model has no path."""
        model = MockModel(id="bad", name="Bad", path="", type="special_app")
        handler = SpecialAppHandler()
        result = handler.launch(model, repo_path, process_manager)

        assert result is False
        process_manager.launch_script.assert_not_called()

    def test_launch_fails_for_missing_script(
        self, repo_path: Path, process_manager: MagicMock
    ) -> None:
        """DBC: Launch fails when script file doesn't exist."""
        model = MockModel(
            id="missing",
            name="Missing",
            path="src/does_not_exist.py",
            type="special_app",
        )
        handler = SpecialAppHandler()
        result = handler.launch(model, repo_path, process_manager)

        assert result is False
        process_manager.launch_script.assert_not_called()


# =============================================================================
# PuttingGreenHandler Tests
# =============================================================================


class TestPuttingGreenHandler:
    """Test PuttingGreenHandler for putting_green model type."""

    def test_can_handle_putting_green(self) -> None:
        """Handler accepts 'putting_green' type."""
        handler = PuttingGreenHandler()
        assert handler.can_handle("putting_green")

    def test_cannot_handle_other_types(self) -> None:
        """Handler rejects non-putting-green types."""
        handler = PuttingGreenHandler()
        assert not handler.can_handle("special_app")
        assert not handler.can_handle("mujoco")

    def test_launch_putting_green(
        self, repo_path: Path, process_manager: MagicMock
    ) -> None:
        """Putting green simulator launches successfully."""
        model = MockModel(
            id="putting_green",
            name="Putting Green",
            path="src/engines/physics_engines/putting_green/python/simulator.py",
            type="putting_green",
        )
        handler = PuttingGreenHandler()
        result = handler.launch(model, repo_path, process_manager)

        assert result is True
        process_manager.launch_script.assert_called_once()
        call_kwargs = process_manager.launch_script.call_args
        assert call_kwargs.kwargs["name"] == "Putting Green Simulator"

    def test_launch_fails_no_path(
        self, repo_path: Path, process_manager: MagicMock
    ) -> None:
        """DBC: Launch fails when model has no path."""
        model = MockModel(id="pg", name="PG", path="", type="putting_green")
        handler = PuttingGreenHandler()
        result = handler.launch(model, repo_path, process_manager)

        assert result is False


# =============================================================================
# Registry Integration Tests
# =============================================================================


class TestRegistryIntegration:
    """Test that the ModelHandlerRegistry includes all handlers."""

    def test_registry_handles_special_app(self) -> None:
        """Registry finds handler for special_app type."""
        registry = ModelHandlerRegistry()
        handler = registry.get_handler("special_app")
        assert handler is not None
        assert isinstance(handler, SpecialAppHandler)

    def test_registry_handles_putting_green(self) -> None:
        """Registry finds handler for putting_green type."""
        registry = ModelHandlerRegistry()
        handler = registry.get_handler("putting_green")
        assert handler is not None
        assert isinstance(handler, PuttingGreenHandler)

    def test_registry_still_handles_mujoco(self) -> None:
        """Regression: existing handlers still work."""
        registry = ModelHandlerRegistry()
        assert registry.get_handler("custom_humanoid") is not None

    def test_registry_still_handles_drake(self) -> None:
        """Regression: Drake handler still works."""
        registry = ModelHandlerRegistry()
        assert registry.get_handler("drake") is not None

    def test_registry_still_handles_pinocchio(self) -> None:
        """Regression: Pinocchio handler still works."""
        registry = ModelHandlerRegistry()
        assert registry.get_handler("pinocchio") is not None

    def test_registry_still_handles_opensim(self) -> None:
        """Regression: OpenSim handler still works."""
        registry = ModelHandlerRegistry()
        assert registry.get_handler("opensim") is not None

    def test_registry_still_handles_myosim(self) -> None:
        """Regression: MyoSim handler still works."""
        registry = ModelHandlerRegistry()
        assert registry.get_handler("myosim") is not None

    def test_registry_returns_none_for_unknown(self) -> None:
        """Registry returns None for unknown types."""
        registry = ModelHandlerRegistry()
        assert registry.get_handler("nonexistent") is None
