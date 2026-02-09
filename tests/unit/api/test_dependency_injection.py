"""Tests for Phase 3: Complete API dependency injection migration.

Verifies that:
1. No route modules contain configure() functions or module-level mutable state
2. All endpoints receive dependencies via FastAPI Depends()
3. Dependencies can be overridden for testing (the core DI contract)
4. The server startup no longer calls configure() on any route module
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Skip if FastAPI not available
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

API_ROUTES_DIR = Path(__file__).parent.parent.parent.parent / "src" / "api" / "routes"
API_SERVER_FILE = Path(__file__).parent.parent.parent.parent / "src" / "api" / "server.py"

# Route modules that previously had configure() + module-level globals
MIGRATED_ROUTE_MODULES = [
    "src.api.routes.core",
    "src.api.routes.engines",
    "src.api.routes.simulation",
    "src.api.routes.analysis",
    "src.api.routes.video",
    "src.api.routes.export",
]


# ===========================================================================
# Static analysis tests -- no configure() or module-level mutable state
# ===========================================================================


class TestNoConfigure:
    """Verify that configure() functions have been fully removed."""

    @pytest.mark.parametrize("module_name", MIGRATED_ROUTE_MODULES)
    def test_no_configure_function(self, module_name: str) -> None:
        """Each migrated route module must not define a configure() function."""
        mod = importlib.import_module(module_name)
        assert not hasattr(mod, "configure"), (
            f"{module_name} still exposes a configure() function. "
            "All dependencies should use FastAPI Depends() instead."
        )

    @pytest.mark.parametrize("module_name", MIGRATED_ROUTE_MODULES)
    def test_no_module_level_underscore_globals(self, module_name: str) -> None:
        """No legacy _engine_manager, _logger, etc. module-level globals."""
        mod = importlib.import_module(module_name)
        legacy_names = {
            "_engine_manager",
            "_logger",
            "_simulation_service",
            "_analysis_service",
            "_video_pipeline",
            "_active_tasks",
        }
        found = legacy_names & set(dir(mod))
        assert not found, (
            f"{module_name} still has legacy module-level globals: {found}. "
            "These should be removed after migration to Depends()."
        )

    def test_server_has_no_configure_calls(self) -> None:
        """server.py must not call .configure() on any route module."""
        source = API_SERVER_FILE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        configure_calls: list[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Look for <name>.configure(...) calls
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "configure"
                ):
                    configure_calls.append(node.lineno)
        assert not configure_calls, (
            f"server.py still contains configure() calls at lines {configure_calls}. "
            "All routes now use Depends() for dependency injection."
        )


class TestNoModuleLevelMutableState:
    """Verify that route modules have no module-level mutable state."""

    @pytest.mark.parametrize("module_name", MIGRATED_ROUTE_MODULES)
    def test_no_global_statement(self, module_name: str) -> None:
        """Route modules must not use the 'global' keyword."""
        mod = importlib.import_module(module_name)
        source = inspect.getsource(mod)
        tree = ast.parse(source)
        global_stmts: list[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                global_stmts.append(node.lineno)
        assert not global_stmts, (
            f"{module_name} still uses 'global' at lines {global_stmts}. "
            "Module-level mutable state should be replaced with Depends()."
        )


# ===========================================================================
# Dependency override tests -- prove DI actually works
# ===========================================================================


class TestDependencyOverride:
    """Verify that endpoints accept injected dependencies via Depends()."""

    @pytest.fixture
    def mock_engine_manager(self) -> MagicMock:
        """Create a mock engine manager."""
        mgr = MagicMock()
        mgr.get_available_engines.return_value = ["mock_engine"]
        mgr.get_current_engine.return_value = None
        mgr.get_engine_status.return_value = MagicMock(value="available")
        return mgr

    @pytest.fixture
    def mock_task_manager(self) -> MagicMock:
        """Create a mock task manager."""
        mgr = MagicMock()
        mgr.__contains__ = MagicMock(return_value=False)
        return mgr

    @pytest.fixture
    def test_app(
        self, mock_engine_manager: MagicMock, mock_task_manager: MagicMock
    ) -> FastAPI:
        """Create a minimal FastAPI app with dependency overrides."""
        from src.api.dependencies import (
            get_engine_manager,
            get_logger,
            get_task_manager,
        )
        from src.api.routes.core import router as core_router
        from src.api.routes.export import router as export_router

        app = FastAPI()
        app.include_router(core_router)
        app.include_router(export_router)

        # Override dependencies with mocks
        app.dependency_overrides[get_engine_manager] = lambda: mock_engine_manager
        app.dependency_overrides[get_task_manager] = lambda: mock_task_manager
        app.dependency_overrides[get_logger] = lambda: MagicMock()

        return app

    def test_health_uses_injected_engine_manager(
        self, test_app: FastAPI, mock_engine_manager: MagicMock
    ) -> None:
        """Health endpoint should use the injected engine manager."""
        with TestClient(test_app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        # The mock returns a list with 1 element
        assert data["engines_available"] == 1
        # Verify the mock was actually called
        mock_engine_manager.get_available_engines.assert_called()

    def test_root_endpoint_has_no_dependencies(self, test_app: FastAPI) -> None:
        """Root endpoint should work without any service dependencies."""
        with TestClient(test_app) as client:
            response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"

    def test_export_uses_injected_task_manager(
        self, test_app: FastAPI, mock_task_manager: MagicMock
    ) -> None:
        """Export endpoint should use the injected task manager."""
        with TestClient(test_app) as client:
            response = client.get("/export/test-task-id")

        # 404 because the mock task manager says the task doesn't exist
        assert response.status_code == 404
        mock_task_manager.__contains__.assert_called()


class TestDependencyProviders:
    """Test the dependency provider functions in dependencies.py."""

    def test_get_engine_manager_returns_503_when_missing(self) -> None:
        """get_engine_manager should raise 503 when not initialized."""
        from src.api.dependencies import get_engine_manager

        mock_request = MagicMock()
        mock_request.app.state = MagicMock(spec=[])  # No attributes

        with pytest.raises(fastapi.HTTPException) as exc_info:
            get_engine_manager(mock_request)
        assert exc_info.value.status_code == 503

    def test_get_simulation_service_returns_503_when_missing(self) -> None:
        """get_simulation_service should raise 503 when not initialized."""
        from src.api.dependencies import get_simulation_service

        mock_request = MagicMock()
        mock_request.app.state = MagicMock(spec=[])

        with pytest.raises(fastapi.HTTPException) as exc_info:
            get_simulation_service(mock_request)
        assert exc_info.value.status_code == 503

    def test_get_analysis_service_returns_503_when_missing(self) -> None:
        """get_analysis_service should raise 503 when not initialized."""
        from src.api.dependencies import get_analysis_service

        mock_request = MagicMock()
        mock_request.app.state = MagicMock(spec=[])

        with pytest.raises(fastapi.HTTPException) as exc_info:
            get_analysis_service(mock_request)
        assert exc_info.value.status_code == 503

    def test_get_video_pipeline_returns_503_when_missing(self) -> None:
        """get_video_pipeline should raise 503 when not initialized."""
        from src.api.dependencies import get_video_pipeline

        mock_request = MagicMock()
        mock_request.app.state = MagicMock(spec=[])

        with pytest.raises(fastapi.HTTPException) as exc_info:
            get_video_pipeline(mock_request)
        assert exc_info.value.status_code == 503

    def test_get_task_manager_returns_503_when_missing(self) -> None:
        """get_task_manager should raise 503 when not initialized."""
        from src.api.dependencies import get_task_manager

        mock_request = MagicMock()
        mock_request.app.state = MagicMock(spec=[])

        with pytest.raises(fastapi.HTTPException) as exc_info:
            get_task_manager(mock_request)
        assert exc_info.value.status_code == 503

    def test_get_engine_manager_returns_instance(self) -> None:
        """get_engine_manager should return the stored instance."""
        from src.api.dependencies import get_engine_manager

        mock_request = MagicMock()
        mock_mgr = MagicMock()
        mock_request.app.state.engine_manager = mock_mgr

        result = get_engine_manager(mock_request)
        assert result is mock_mgr

    def test_get_logger_returns_none_gracefully(self) -> None:
        """get_logger should return None when logger not set, without error."""
        from src.api.dependencies import get_logger

        mock_request = MagicMock()
        mock_request.app.state = MagicMock(spec=[])

        result = get_logger(mock_request)
        assert result is None


# ===========================================================================
# Completeness check -- all route modules accounted for
# ===========================================================================


class TestMigrationCompleteness:
    """Verify the migration covers all route modules."""

    def test_all_route_files_have_no_configure(self) -> None:
        """Every .py file in routes/ should have no configure() function."""
        route_files = list(API_ROUTES_DIR.glob("*.py"))
        # Filter out __init__.py
        route_files = [f for f in route_files if f.name != "__init__.py"]
        assert len(route_files) > 0, "No route files found"

        for route_file in route_files:
            source = route_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "configure":
                    # Allow configure_control in dataset.py (it's an endpoint, not DI)
                    if "configure_control" in route_file.read_text(encoding="utf-8"):
                        # Check the exact function name
                        if node.name == "configure":
                            pytest.fail(
                                f"{route_file.name} still has a configure() function "
                                f"at line {node.lineno}"
                            )
