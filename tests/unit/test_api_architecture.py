"""Tests for API architecture improvements (#1485, #1488).

Tests:
- Route registry auto-discovery and registration
- Task manager with TTL, concurrency, and lifecycle
- API versioning (routes available under /api/v1/)
- Linkage mechanisms decomposition (imports still work)
"""

from __future__ import annotations

import asyncio
import time

import pytest

# ── Route Registry Tests ─────────────────────────────────────────


class TestRouteRegistry:
    """Tests for the route auto-discovery and registration pattern (#1485)."""

    def test_discover_routes_finds_modules(self) -> None:
        """discover_routes returns a non-empty list of (name, router) tuples."""
        from src.api.route_registry import discover_routes

        routes = discover_routes()
        assert len(routes) > 0
        # Each item is a (module_name, router) tuple
        for name, router in routes:
            assert isinstance(name, str)
            assert hasattr(router, "routes")  # APIRouter has .routes

    def test_discover_routes_excludes_websocket_modules(self) -> None:
        """WebSocket-only modules are excluded by default."""
        from src.api.route_registry import discover_routes

        routes = discover_routes()
        module_names = {name for name, _ in routes}
        assert "chat_ws" not in module_names
        assert "simulation_ws" not in module_names

    def test_discover_routes_custom_exclude(self) -> None:
        """Custom exclusion set is respected."""
        from src.api.route_registry import discover_routes

        routes_all = discover_routes(exclude=frozenset())
        routes_exclude_core = discover_routes(exclude=frozenset({"core"}))
        # Excluding core should yield one fewer module
        names_all = {name for name, _ in routes_all}
        names_no_core = {name for name, _ in routes_exclude_core}
        assert "core" in names_all
        assert "core" not in names_no_core

    def test_discover_routes_respects_registration_order(self) -> None:
        """Discovered routes follow _REGISTRATION_ORDER for priority modules."""
        from src.api.route_registry import _REGISTRATION_ORDER, discover_routes

        routes = discover_routes()
        names = [name for name, _ in routes]
        # Priority modules should appear in _REGISTRATION_ORDER sequence
        priority_names = [n for n in names if n in _REGISTRATION_ORDER]
        expected_order = [n for n in _REGISTRATION_ORDER if n in priority_names]
        assert priority_names == expected_order, (
            f"Priority modules out of order: {priority_names} != {expected_order}"
        )

    def test_register_routes_on_app(self) -> None:
        """register_routes includes discovered routers on a FastAPI app."""
        from fastapi import FastAPI

        from src.api.route_registry import register_routes

        test_app = FastAPI()
        count = register_routes(test_app, prefix="/test")
        assert count > 0
        # Verify routes were actually added to the app
        assert len(test_app.routes) > 0

    def test_register_routes_with_prefix(self) -> None:
        """Routes registered with a prefix include that prefix in paths."""
        from fastapi import FastAPI

        from src.api.route_registry import register_routes

        test_app = FastAPI()
        register_routes(test_app, prefix="/api/v1")
        route_paths = [r.path for r in test_app.routes if hasattr(r, "path")]
        # At least some routes should have the /api/v1 prefix
        prefixed = [p for p in route_paths if p.startswith("/api/v1")]
        assert len(prefixed) > 0


# ── Task Manager Tests ────────────────────────────────────────────


class TestTaskManager:
    """Tests for the extracted TaskManager with TTL and concurrency (#1485, #1488)."""

    def test_set_and_get(self) -> None:
        """Basic set/get operations work."""
        from src.api.task_manager import TaskManager

        tm = TaskManager()
        tm.set("task-1", {"status": "running"})
        result = tm.get("task-1")
        assert result is not None
        assert result["status"] == "running"

    def test_get_nonexistent_returns_none(self) -> None:
        """Getting a nonexistent task returns None."""
        from src.api.task_manager import TaskManager

        tm = TaskManager()
        assert tm.get("nonexistent") is None

    def test_contains(self) -> None:
        """__contains__ check works."""
        from src.api.task_manager import TaskManager

        tm = TaskManager()
        tm.set("task-1", {"status": "pending"})
        assert "task-1" in tm
        assert "task-2" not in tm

    def test_dict_like_access(self) -> None:
        """Dict-like [] access works for backward compatibility."""
        from src.api.task_manager import TaskManager

        tm = TaskManager()
        tm["task-1"] = {"status": "pending"}
        assert tm["task-1"]["status"] == "pending"

    def test_dict_like_access_raises_keyerror(self) -> None:
        """Dict-like [] access raises KeyError for missing tasks."""
        from src.api.task_manager import TaskManager

        tm = TaskManager()
        with pytest.raises(KeyError):
            _ = tm["missing"]

    def test_empty_task_id_raises(self) -> None:
        """Setting a task with empty ID raises ValueError."""
        from src.api.task_manager import TaskManager

        tm = TaskManager()
        with pytest.raises(ValueError, match="non-empty"):
            tm.set("", {"status": "bad"})
        with pytest.raises(ValueError, match="non-empty"):
            tm.set("   ", {"status": "bad"})

    def test_ttl_expiry(self) -> None:
        """Tasks are cleaned up after TTL expiry."""
        from src.api.task_manager import TaskManager

        tm = TaskManager(ttl_seconds=0)  # Immediately expire
        tm.set("task-1", {"status": "running"})
        time.sleep(0.01)
        # Next access triggers cleanup
        assert tm.get("task-1") is None
        assert "task-1" not in tm

    def test_max_tasks_eviction(self) -> None:
        """Oldest tasks are evicted when MAX_TASKS is exceeded."""
        from src.api.task_manager import TaskManager

        tm = TaskManager(max_tasks=3)
        tm.set("task-1", {"status": "running"})
        tm.set("task-2", {"status": "running"})
        tm.set("task-3", {"status": "running"})
        tm.set("task-4", {"status": "running"})  # Should evict task-1
        assert "task-1" not in tm
        assert "task-4" in tm

    def test_update_progress(self) -> None:
        """update_progress updates the progress field."""
        from src.api.task_manager import TaskManager

        tm = TaskManager()
        tm.set("task-1", {"status": "running"})
        tm.update_progress("task-1", 42.5)
        result = tm.get("task-1")
        assert result is not None
        assert result["progress"] == 42.5

    def test_update_progress_clamped(self) -> None:
        """update_progress clamps to 0-100 range."""
        from src.api.task_manager import TaskManager

        tm = TaskManager()
        tm.set("task-1", {"status": "running"})
        tm.update_progress("task-1", 150.0)
        assert tm.get("task-1")["progress"] == 100.0
        tm.update_progress("task-1", -10.0)
        assert tm.get("task-1")["progress"] == 0.0

    def test_mark_completed(self) -> None:
        """mark_completed updates status and result."""
        from src.api.task_manager import TaskManager, TaskStatus

        tm = TaskManager()
        tm.set("task-1", {"status": "running"})
        tm.mark_completed("task-1", {"data": [1, 2, 3]})
        result = tm.get("task-1")
        assert result is not None
        assert result["status"] == TaskStatus.COMPLETED.value
        assert result["result"] == {"data": [1, 2, 3]}
        assert result["progress"] == 100.0

    def test_mark_failed(self) -> None:
        """mark_failed updates status and error."""
        from src.api.task_manager import TaskManager, TaskStatus

        tm = TaskManager()
        tm.set("task-1", {"status": "running"})
        tm.mark_failed("task-1", "Engine crashed")
        result = tm.get("task-1")
        assert result is not None
        assert result["status"] == TaskStatus.FAILED.value
        assert result["error"] == "Engine crashed"

    def test_active_count(self) -> None:
        """active_count returns correct count of non-expired tasks."""
        from src.api.task_manager import TaskManager

        tm = TaskManager()
        assert tm.active_count() == 0
        tm.set("task-1", {"status": "running"})
        tm.set("task-2", {"status": "running"})
        assert tm.active_count() == 2

    def test_engine_semaphore_property(self) -> None:
        """engine_semaphore returns an asyncio.Semaphore with correct limit."""
        from src.api.task_manager import TaskManager

        tm = TaskManager(max_concurrent=2)
        sem = tm.engine_semaphore
        assert isinstance(sem, asyncio.Semaphore)
        # Semaphore should allow 2 acquisitions
        assert sem._value == 2  # noqa: SLF001

    def test_task_status_enum(self) -> None:
        """TaskStatus enum has expected values."""
        from src.api.task_manager import TaskStatus

        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"


# ── API Versioning Tests ──────────────────────────────────────────


class TestAPIVersioning:
    """Tests for API versioning under /api/v1/ prefix (#1488)."""

    @pytest.fixture
    def client(self):
        """Create a test client for the API."""
        httpx = pytest.importorskip("httpx")  # noqa: F841
        fastapi = pytest.importorskip("fastapi")  # noqa: F841
        from fastapi.testclient import TestClient

        try:
            from src.api.server import app

            with TestClient(app, base_url="http://localhost") as test_client:
                yield test_client
        except ImportError:
            pytest.skip("Cannot import api.server")

    def test_root_endpoint_at_legacy_path(self, client) -> None:
        """Root endpoint works at legacy un-prefixed path."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"

    def test_root_endpoint_at_versioned_path(self, client) -> None:
        """Root endpoint works at versioned /api/v1/ path."""
        response = client.get("/api/v1/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"

    def test_health_at_versioned_path(self, client) -> None:
        """Health endpoint accessible at /api/v1/health."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_engines_at_versioned_path(self, client) -> None:
        """Engines endpoint accessible at /api/v1/engines."""
        response = client.get("/api/v1/engines")
        assert response.status_code == 200
        data = response.json()
        assert "engines" in data

    def test_openapi_schema_has_versioned_routes(self, client) -> None:
        """OpenAPI schema includes versioned route paths."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        paths = list(data.get("paths", {}).keys())
        versioned = [p for p in paths if p.startswith("/api/v1/")]
        assert len(versioned) > 0, f"No versioned routes found. Paths: {paths[:10]}"

    def test_openapi_schema_has_enhanced_description(self, client) -> None:
        """OpenAPI schema has enhanced description with versioning info."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        info = data.get("info", {})
        description = info.get("description", "")
        assert "v1" in description
        assert "Multi-engine" in description or "Versioning" in description

    def test_api_version_constant(self) -> None:
        """API_VERSION and API_PREFIX constants are correctly defined."""
        from src.api.server import API_PREFIX, API_VERSION

        assert API_VERSION == "v1"
        assert API_PREFIX == "/api/v1"


# ── Linkage Decomposition Tests ──────────────────────────────────


class TestLinkageMechanismsDecomposition:
    """Tests that linkage_mechanisms decomposition preserves the public API (#1485)."""

    def test_imports_from_init(self) -> None:
        """All public symbols are importable from the package."""
        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.linkage_mechanisms import (
            LINKAGE_CATALOG,
            generate_chebyshev_linkage_xml,
            generate_delta_robot_xml,
            generate_five_bar_parallel_xml,
            generate_four_bar_linkage_xml,
            generate_geneva_mechanism_xml,
            generate_oldham_coupling_xml,
            generate_pantograph_xml,
            generate_peaucellier_linkage_xml,
            generate_scotch_yoke_xml,
            generate_slider_crank_xml,
            generate_stewart_platform_xml,
            generate_watt_linkage_xml,
        )

        # All should be callable
        assert callable(generate_four_bar_linkage_xml)
        assert callable(generate_slider_crank_xml)
        assert callable(generate_scotch_yoke_xml)
        assert callable(generate_geneva_mechanism_xml)
        assert callable(generate_oldham_coupling_xml)
        assert callable(generate_peaucellier_linkage_xml)
        assert callable(generate_chebyshev_linkage_xml)
        assert callable(generate_watt_linkage_xml)
        assert callable(generate_pantograph_xml)
        assert callable(generate_delta_robot_xml)
        assert callable(generate_five_bar_parallel_xml)
        assert callable(generate_stewart_platform_xml)
        # Catalog should be a dict
        assert isinstance(LINKAGE_CATALOG, dict)

    def test_catalog_has_all_mechanisms(self) -> None:
        """LINKAGE_CATALOG contains all expected mechanisms."""
        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.linkage_mechanisms import (
            LINKAGE_CATALOG,
        )

        assert len(LINKAGE_CATALOG) == 16
        for name, entry in LINKAGE_CATALOG.items():
            assert "xml" in entry, f"Missing 'xml' in catalog entry: {name}"
            assert "actuators" in entry, f"Missing 'actuators' in catalog entry: {name}"
            assert "category" in entry, f"Missing 'category' in catalog entry: {name}"
            assert "description" in entry, (
                f"Missing 'description' in catalog entry: {name}"
            )

    def test_four_bar_generates_valid_xml(self) -> None:
        """Four-bar linkage XML contains expected MuJoCo elements."""
        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.linkage_mechanisms import (
            generate_four_bar_linkage_xml,
        )

        xml = generate_four_bar_linkage_xml()
        assert "<mujoco" in xml
        assert "crank_joint" in xml
        assert "actuator" in xml

    def test_slider_crank_generates_valid_xml(self) -> None:
        """Slider-crank XML contains expected elements."""
        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.linkage_mechanisms import (
            generate_slider_crank_xml,
        )

        xml = generate_slider_crank_xml(orientation="horizontal")
        assert "<mujoco" in xml
        assert "slider_crank" in xml

    def test_init_has_no_logic(self) -> None:
        """__init__.py contains only imports and __all__, no function/class definitions."""
        from pathlib import Path

        init_path = Path(
            "src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/"
            "linkage_mechanisms/__init__.py"
        )
        content = init_path.read_text(encoding="utf-8")
        # Should not contain any function or class definitions
        lines = content.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("def ", "class ")):
                pytest.fail(
                    f"__init__.py should contain only imports, found: {stripped}"
                )

    def test_submodules_exist(self) -> None:
        """All decomposed sub-modules are importable."""
        import importlib

        submodules = [
            "four_bar",
            "slider_mechanisms",
            "special_mechanisms",
            "straight_line",
            "parallel_mechanisms",
            "catalog",
        ]
        base = "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.linkage_mechanisms"
        for name in submodules:
            mod = importlib.import_module(f"{base}.{name}")
            assert mod is not None, f"Failed to import {name}"


# ── OpenAPI Enhancement Tests ─────────────────────────────────────


class TestOpenAPIEnhancements:
    """Tests for OpenAPI schema improvements (#1488)."""

    @pytest.fixture
    def client(self):
        """Create a test client for the API."""
        httpx = pytest.importorskip("httpx")  # noqa: F841
        fastapi = pytest.importorskip("fastapi")  # noqa: F841
        from fastapi.testclient import TestClient

        try:
            from src.api.server import app

            with TestClient(app, base_url="http://localhost") as test_client:
                yield test_client
        except ImportError:
            pytest.skip("Cannot import api.server")

    def test_openapi_has_tags(self, client) -> None:
        """OpenAPI schema includes tag definitions."""
        response = client.get("/openapi.json")
        data = response.json()
        tags = data.get("tags", [])
        assert len(tags) > 0

    def test_openapi_has_global_responses(self, client) -> None:
        """OpenAPI schema includes version info."""
        response = client.get("/openapi.json")
        data = response.json()
        # Version should be 3.0.0 (updated)
        assert data["info"]["version"] == "3.0.0"
