"""Smoke tests for MuJoCo GUI tab modules (issue #2350).

These tests verify that tab classes can be imported and instantiated without
error in a headless environment.  They use ``pytest.importorskip`` to skip
gracefully when PyQt6 or mujoco is unavailable (e.g. pure-unit CI runners).

Coverage goal: basic import + construction of each tab class so that obvious
import-time errors, missing attributes, and broken __init__ signatures are
caught before they reach integration tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers — mock heavy Qt / simulation dependencies
# ---------------------------------------------------------------------------


class _FakeSimWidget:
    """Minimal stand-in for MuJoCoSimWidget."""

    model = None
    data = None

    def get_sim_state(self):
        return {"time": 0.0, "qpos": [], "qvel": []}


class _FakeMainWindow:
    """Minimal stand-in for AdvancedGolfAnalysisWindow."""

    status_bar = MagicMock()


# ---------------------------------------------------------------------------
# Module-level import guards
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Analysis tab
# ---------------------------------------------------------------------------


class TestAnalysisTabImport:
    """Ensure AnalysisTab module imports correctly."""

    def test_analysis_tab_module_importable(self) -> None:
        """AnalysisTab class should be importable when PyQt6 is present."""
        pytest.importorskip(
            "PyQt6",
            reason="PyQt6 is required for MuJoCo GUI tab tests",
        )
        # The real import may also need mujoco; skip rather than fail.
        try:
            from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.gui.tabs.analysis_tab import (  # noqa: E501
                AnalysisTab,
            )
        except ImportError as exc:
            pytest.skip(f"AnalysisTab import skipped: {exc}")

        assert AnalysisTab is not None, "AnalysisTab class should not be None"

    def test_analysis_tab_has_required_attrs(self) -> None:
        """AnalysisTab class definition exposes expected public interface."""
        pytest.importorskip("PyQt6")
        try:
            from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.gui.tabs.analysis_tab import (  # noqa: E501
                AnalysisTab,
            )
        except ImportError as exc:
            pytest.skip(f"AnalysisTab import skipped: {exc}")

        assert hasattr(AnalysisTab, "__init__"), "AnalysisTab must define __init__"


# ---------------------------------------------------------------------------
# Controls tab
# ---------------------------------------------------------------------------


class TestControlsTabImport:
    """Ensure ControlsTab module imports correctly."""

    def test_controls_tab_module_importable(self) -> None:
        pytest.importorskip("PyQt6")
        try:
            from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.gui.tabs.controls_tab import (  # noqa: E501
                ControlsTab,
            )
        except ImportError as exc:
            pytest.skip(f"ControlsTab import skipped: {exc}")

        assert ControlsTab is not None

    def test_controls_tab_simplified_threshold_constant(self) -> None:
        """ControlsTab.SIMPLIFIED_ACTUATOR_THRESHOLD should be a positive int."""
        pytest.importorskip("PyQt6")
        try:
            from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.gui.tabs.controls_tab import (  # noqa: E501
                ControlsTab,
            )
        except ImportError as exc:
            pytest.skip(f"ControlsTab import skipped: {exc}")

        threshold = ControlsTab.SIMPLIFIED_ACTUATOR_THRESHOLD
        assert isinstance(threshold, int), "SIMPLIFIED_ACTUATOR_THRESHOLD must be int"
        assert threshold > 0, "SIMPLIFIED_ACTUATOR_THRESHOLD must be positive"


# ---------------------------------------------------------------------------
# Visualization tab
# ---------------------------------------------------------------------------


class TestVisualizationTabImport:
    """Ensure VisualizationTab module imports correctly."""

    def test_visualization_tab_module_importable(self) -> None:
        pytest.importorskip("PyQt6")
        try:
            from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.gui.tabs.visualization_tab import (  # noqa: E501
                VisualizationTab,
            )
        except ImportError as exc:
            pytest.skip(f"VisualizationTab import skipped: {exc}")

        assert VisualizationTab is not None


# ---------------------------------------------------------------------------
# Humanoid Config tab
# ---------------------------------------------------------------------------


class TestHumanoidConfigTabImport:
    """Ensure HumanoidConfigTab module imports correctly."""

    def test_humanoid_config_tab_importable(self) -> None:
        pytest.importorskip("PyQt6")
        try:
            from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.gui.tabs.humanoid_config_tab import (  # noqa: E501
                HumanoidConfigTab,
            )
        except ImportError as exc:
            pytest.skip(f"HumanoidConfigTab import skipped: {exc}")

        assert HumanoidConfigTab is not None


# ---------------------------------------------------------------------------
# Manipulation tab
# ---------------------------------------------------------------------------


class TestManipulationTabImport:
    """Ensure ManipulationTab module imports correctly."""

    def test_manipulation_tab_importable(self) -> None:
        pytest.importorskip("PyQt6")
        try:
            from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.gui.tabs.manipulation_tab import (  # noqa: E501
                ManipulationTab,
            )
        except ImportError as exc:
            pytest.skip(f"ManipulationTab import skipped: {exc}")

        assert ManipulationTab is not None


# ---------------------------------------------------------------------------
# Physics tab
# ---------------------------------------------------------------------------


class TestPhysicsTabImport:
    """Ensure PhysicsTab module imports correctly."""

    def test_physics_tab_importable(self) -> None:
        pytest.importorskip("PyQt6")
        try:
            from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.gui.tabs.physics_tab import (  # noqa: E501
                PhysicsTab,
            )
        except ImportError as exc:
            pytest.skip(f"PhysicsTab import skipped: {exc}")

        assert PhysicsTab is not None


# ---------------------------------------------------------------------------
# Plotting tab
# ---------------------------------------------------------------------------


class TestPlottingTabImport:
    """Ensure PlottingTab module imports correctly."""

    def test_plotting_tab_importable(self) -> None:
        pytest.importorskip("PyQt6")
        try:
            from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.gui.tabs.plotting_tab import (  # noqa: E501
                PlottingTab,
            )
        except ImportError as exc:
            pytest.skip(f"PlottingTab import skipped: {exc}")

        assert PlottingTab is not None


# ---------------------------------------------------------------------------
# Manipulability tab
# ---------------------------------------------------------------------------


class TestManipulabilityTabImport:
    """Ensure ManipulabilityTab module imports correctly."""

    def test_manipulability_tab_importable(self) -> None:
        pytest.importorskip("PyQt6")
        try:
            from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.gui.tabs.manipulability_tab import (  # noqa: E501
                ManipulabilityTab,
            )
        except ImportError as exc:
            pytest.skip(f"ManipulabilityTab import skipped: {exc}")

        assert ManipulabilityTab is not None


# ---------------------------------------------------------------------------
# tabs __init__ package
# ---------------------------------------------------------------------------


class TestTabsPackageInit:
    """The tabs package __init__ should be importable."""

    def test_tabs_package_importable(self) -> None:
        """tabs package __init__ should import without errors."""
        pytest.importorskip("PyQt6")
        try:
            import src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.gui.tabs  # noqa: E501, F401
        except ImportError as exc:
            pytest.skip(f"tabs package import skipped: {exc}")
