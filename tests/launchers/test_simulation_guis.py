"""Tests for the new simulation GUI modules and external tool adapters.

Validates that all new GUI modules expose the get_dockable_ui protocol
and that external tool adapters degrade gracefully when repos are unavailable.

Design by Contract:
    - Every GUI module must expose get_dockable_ui() -> QMainWindow
    - External adapters must never crash, even with missing repos
    - All widgets must have cleanup() methods
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

# ===========================================================================
# Swing-to-Flight Pipeline GUI tests
# ===========================================================================


class TestSwingFlightGUI:
    """Test the swing-to-flight pipeline GUI module."""

    def test_module_importable(self) -> None:
        """The gui module must be importable."""
        mod = importlib.import_module("src.tools.swing_flight_pipeline.gui")
        assert mod is not None

    def test_has_get_dockable_ui(self) -> None:
        """Module must expose get_dockable_ui function."""
        mod = importlib.import_module("src.tools.swing_flight_pipeline.gui")
        assert hasattr(mod, "get_dockable_ui")
        assert callable(mod.get_dockable_ui)

    def test_has_window_class(self) -> None:
        """Module must define SwingFlightWindow."""
        mod = importlib.import_module("src.tools.swing_flight_pipeline.gui")
        assert hasattr(mod, "SwingFlightWindow")

    def test_has_widget_class(self) -> None:
        """Module must define SwingFlightWidget."""
        mod = importlib.import_module("src.tools.swing_flight_pipeline.gui")
        assert hasattr(mod, "SwingFlightWidget")


# ===========================================================================
# Ball Flight GUI tests
# ===========================================================================


class TestBallFlightGUI:
    """Test the ball flight simulator GUI module."""

    def test_module_importable(self) -> None:
        mod = importlib.import_module("src.tools.ball_flight_gui.gui")
        assert mod is not None

    def test_has_get_dockable_ui(self) -> None:
        mod = importlib.import_module("src.tools.ball_flight_gui.gui")
        assert hasattr(mod, "get_dockable_ui")
        assert callable(mod.get_dockable_ui)

    def test_has_window_class(self) -> None:
        mod = importlib.import_module("src.tools.ball_flight_gui.gui")
        assert hasattr(mod, "BallFlightWindow")


# ===========================================================================
# Putting Green GUI tests
# ===========================================================================


class TestPuttingGreenGUI:
    """Test the putting green simulator GUI module."""

    def test_module_importable(self) -> None:
        mod = importlib.import_module("src.tools.putting_green_gui.gui")
        assert mod is not None

    def test_has_get_dockable_ui(self) -> None:
        mod = importlib.import_module("src.tools.putting_green_gui.gui")
        assert hasattr(mod, "get_dockable_ui")
        assert callable(mod.get_dockable_ui)

    def test_has_window_class(self) -> None:
        mod = importlib.import_module("src.tools.putting_green_gui.gui")
        assert hasattr(mod, "PuttingGreenWindow")


# ===========================================================================
# External Tools Adapter tests
# ===========================================================================


class TestExternalToolsAdapter:
    """Test the external tools adapter module."""

    def test_module_importable(self) -> None:
        mod = importlib.import_module("src.launchers.external_tools_adapter")
        assert mod is not None

    def test_has_all_adapter_functions(self) -> None:
        """Adapter must expose factory functions for all 3 external tools."""
        mod = importlib.import_module("src.launchers.external_tools_adapter")
        assert hasattr(mod, "get_video_analyzer_dockable_ui")
        assert hasattr(mod, "get_data_explorer_dockable_ui")
        assert hasattr(mod, "get_data_processor_dockable_ui")

    def test_external_tools_registry(self) -> None:
        """EXTERNAL_TOOLS dict must have entries for all 3 tools."""
        mod = importlib.import_module("src.launchers.external_tools_adapter")
        assert "video_analyzer" in mod.EXTERNAL_TOOLS
        assert "data_explorer" in mod.EXTERNAL_TOOLS
        assert "data_processor" in mod.EXTERNAL_TOOLS

    def test_find_tools_repo_returns_path_or_none(self) -> None:
        """_find_tools_repo must return Path or None (never crash)."""
        mod = importlib.import_module("src.launchers.external_tools_adapter")
        # Reset cached value
        mod._TOOLS_REPO = None
        result = mod._find_tools_repo()
        assert result is None or isinstance(result, Path)

    def test_ensure_tools_on_path_returns_bool(self) -> None:
        """_ensure_tools_on_path must return a boolean."""
        mod = importlib.import_module("src.launchers.external_tools_adapter")
        mod._TOOLS_REPO = None
        result = mod._ensure_tools_on_path()
        assert isinstance(result, bool)


# ===========================================================================
# SwingBallFlightPipeline source tests
# ===========================================================================


class TestSwingBallFlightPipelineSource:
    """Test that the pipeline module is available on the current branch."""

    def test_pipeline_module_exists(self) -> None:
        """swing_ball_flight_pipeline.py must exist."""
        p = Path("src/shared/python/physics/swing_ball_flight_pipeline.py")
        assert p.exists(), (
            f"Pipeline file missing: {p}. "
            "Cherry-pick from feat/5337-swing-ball-flight-pipeline."
        )

    def test_pipeline_importable(self) -> None:
        """The pipeline module must import without errors."""
        try:
            mod = importlib.import_module(
                "src.shared.python.physics.swing_ball_flight_pipeline"
            )
            assert hasattr(mod, "SwingBallFlightPipeline")
            assert hasattr(mod, "SwingState")
            assert hasattr(mod, "PipelineResult")
        except ImportError:
            pytest.skip("Pipeline dependencies not available")


# ===========================================================================
# models.yaml tile registration tests
# ===========================================================================


class TestTileRegistration:
    """Test that all new tiles are registered in models.yaml."""

    @pytest.fixture()
    def yaml_content(self) -> str:
        p = Path("src/config/models.yaml")
        return p.read_text(encoding="utf-8")

    def test_swing_flight_pipeline_tile(self, yaml_content: str) -> None:
        assert "swing_flight_pipeline" in yaml_content

    def test_ball_flight_simulator_tile(self, yaml_content: str) -> None:
        assert "ball_flight_simulator" in yaml_content

    def test_putting_green_gui_tile(self, yaml_content: str) -> None:
        assert "putting_green_gui" in yaml_content

    def test_embed_adapter_references(self, yaml_content: str) -> None:
        """External tiles must reference embed_adapter."""
        assert "external_tools_adapter.py" in yaml_content
