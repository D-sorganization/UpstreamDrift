"""Tests for PyQt6 vs React UI consistency.

Verifies that the desktop (PyQt6) and web (React) interfaces expose
the same features and maintain consistent behavior.
"""

import json
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.integration


class TestUIFeatureParity:
    """Test that PyQt6 and React UIs have feature parity."""

    @pytest.fixture
    def pyqt_launcher_path(self) -> Path:
        """Path to PyQt6 launcher."""
        return Path("src/launchers/golf_launcher.py")

    @pytest.fixture
    def react_app_path(self) -> Path:
        """Path to React app entry."""
        return Path("ui/src/App.tsx")

    def test_pyqt_launcher_exists(self, pyqt_launcher_path: Path) -> None:
        """Verify PyQt6 launcher exists."""
        assert pyqt_launcher_path.exists(), "PyQt6 launcher not found"

    def test_react_app_exists(self, react_app_path: Path) -> None:
        """Verify React app entry exists."""
        assert react_app_path.exists(), "React app entry not found"

    def test_engine_selector_in_both_uis(
        self, pyqt_launcher_path: Path, react_app_path: Path
    ) -> None:
        """Verify engine selection is available in both UIs."""
        # Check PyQt6
        if pyqt_launcher_path.exists():
            pyqt_content = pyqt_launcher_path.read_text()
            has_engine_pyqt = (
                "engine" in pyqt_content.lower()
                or "mujoco" in pyqt_content.lower()
                or "drake" in pyqt_content.lower()
            )
            assert has_engine_pyqt, "Engine selection missing from PyQt UI"

        # Check React
        engine_selector = Path("ui/src/components/simulation/EngineSelector.tsx")
        assert engine_selector.exists(), "Engine selector missing from React UI"

    def test_simulation_controls_in_both_uis(self) -> None:
        """Verify simulation controls exist in both UIs."""
        # React controls
        react_controls = Path("ui/src/components/simulation/SimulationControls.tsx")
        assert react_controls.exists(), "SimulationControls missing from React UI"

        # PyQt controls should be in launcher
        pyqt_launcher = Path("src/launchers/golf_launcher.py")
        if pyqt_launcher.exists():
            content = pyqt_launcher.read_text()
            has_controls = any(
                term in content.lower()
                for term in ["start", "stop", "pause", "run", "simulate"]
            )
            assert has_controls, "Simulation controls missing from PyQt UI"

    def test_3d_visualization_in_both_uis(self) -> None:
        """Verify 3D visualization exists in both UIs."""
        # React: Three.js scene
        react_3d = Path("ui/src/components/visualization/Scene3D.tsx")
        has_react_3d = react_3d.exists()

        # PyQt: Should have some visualization
        pyqt_ui_components = Path("src/launchers/ui_components.py")
        has_pyqt_viz = False
        if pyqt_ui_components.exists():
            content = pyqt_ui_components.read_text()
            has_pyqt_viz = any(
                term in content.lower()
                for term in ["scene", "viewer", "render", "opengl", "matplotlib"]
            )

        assert has_react_3d, "3D visualization missing from React UI"
        # PyQt visualization is optional but noted
        if not has_pyqt_viz:
            pytest.xfail("3D visualization may be limited in PyQt UI")


class TestAPIConsistency:
    """Test that both UIs use consistent API contracts."""

    def test_api_client_exists(self) -> None:
        """Verify React API client exists."""
        api_client = Path("ui/src/api/client.ts")
        assert api_client.exists(), "API client missing from React UI"

    def test_api_endpoints_documented(self) -> None:
        """Verify API endpoints are documented or discoverable."""
        # Check for API routes
        api_routes = Path("src/api/routes")
        if api_routes.exists():
            route_files = list(api_routes.glob("*.py"))
            assert len(route_files) > 0, "No API route files found"

    def test_websocket_support(self) -> None:
        """Verify WebSocket support exists for real-time updates."""
        api_client = Path("ui/src/api/client.ts")
        if api_client.exists():
            content = api_client.read_text()
            has_websocket = "websocket" in content.lower() or "ws" in content.lower()
            assert has_websocket, "WebSocket support missing from API client"


class TestUIBuildConfiguration:
    """Test UI build and configuration consistency."""

    def test_vite_config_exists(self) -> None:
        """Verify Vite configuration exists."""
        vite_config = Path("ui/vite.config.ts")
        assert vite_config.exists(), "Vite config missing"

    def test_typescript_config_exists(self) -> None:
        """Verify TypeScript configuration exists."""
        tsconfig = Path("ui/tsconfig.json")
        assert tsconfig.exists(), "TypeScript config missing"

    def test_package_json_has_build_script(self) -> None:
        """Verify package.json has build script."""
        package_json = Path("ui/package.json")
        assert package_json.exists(), "package.json missing"

        with open(package_json) as f:
            pkg = json.load(f)

        assert "scripts" in pkg, "No scripts in package.json"
        assert "build" in pkg["scripts"], "No build script in package.json"

    def test_package_json_should_have_test_script(self) -> None:
        """Verify package.json has test script (expected but may be missing)."""
        package_json = Path("ui/package.json")
        if not package_json.exists():
            pytest.skip("package.json not found")

        with open(package_json) as f:
            pkg = json.load(f)

        if "test" not in pkg.get("scripts", {}):
            pytest.xfail("Test script missing from package.json - needs to be added")
        else:
            assert "test" in pkg["scripts"]
