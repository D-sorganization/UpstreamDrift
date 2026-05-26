"""Tests for API architecture improvements (#1485, #1488).

Tests:
- Route registry auto-discovery and registration
- Task manager with TTL, concurrency, and lifecycle
- API versioning (routes available under /api/v1/)
- Linkage mechanisms decomposition (imports still work)
"""

from __future__ import annotations


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
        """WebSocket-only modules are excluded from auto-discovery (handled explicitly)."""
        from src.api.route_registry import discover_routes

        routes = discover_routes()
        module_names = {name for name, _ in routes}
        assert "chat_ws" not in module_names
        assert "simulation_ws" not in module_names

    def test_websocket_modules_registered_explicitly_in_server(self) -> None:
        """WebSocket modules must be explicitly registered in server.py.

        They are excluded from auto-discovery but must still be served —
        this test protects against accidental removal of the explicit registration.
        """
        source = (
            __import__("pathlib").Path("src/api/server.py").read_text(encoding="utf-8")
        )
        assert "chat_ws" in source, (
            "server.py does not register chat_ws router. "
            "WebSocket modules are excluded from auto-discovery and must be "
            "explicitly included in server.py."
        )
        assert "simulation_ws" in source, (
            "server.py does not register simulation_ws router. "
            "WebSocket modules are excluded from auto-discovery and must be "
            "explicitly included in server.py."
        )

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

    def test_tooling_routes_do_not_double_api_prefix(self) -> None:
        """Tooling routers mount once under the versioned API prefix."""
        from fastapi import FastAPI
        from src.api.route_registry import register_routes

        test_app = FastAPI()
        register_routes(test_app, prefix="/api/v1")
        route_paths = {r.path for r in test_app.routes if hasattr(r, "path")}

        expected_paths = {
            "/api/v1/tools/data-explorer/datasets",
            "/api/v1/terrain/presets",
            "/api/v1/tools/putting-green/simulate",
            "/api/v1/tools/motion-capture/sources",
            "/api/v1/launcher/manifest",
        }
        double_prefixed_paths = {
            "/api/v1/api/tools/data-explorer/datasets",
            "/api/v1/api/terrain/presets",
            "/api/v1/api/tools/putting-green/simulate",
            "/api/v1/api/tools/motion-capture/sources",
            "/api/v1/api/launcher/manifest",
        }

        assert expected_paths <= route_paths
        assert route_paths.isdisjoint(double_prefixed_paths)

    def test_expensive_route_modules_receive_quota_dependencies(self) -> None:
        """Simulation and video routes receive quota dependencies at registration."""
        from fastapi import FastAPI
        from src.api.auth.dependencies import CheckSimulationQuota, CheckVideoQuota
        from src.api.route_registry import register_routes

        test_app = FastAPI()
        register_routes(test_app, prefix="/test")

        simulation_route = next(
            r for r in test_app.routes if getattr(r, "path", "") == "/test/simulate"
        )
        video_route = next(
            r
            for r in test_app.routes
            if getattr(r, "path", "") == "/test/analyze/video"
        )

        simulation_dependencies = [
            dep.call for dep in simulation_route.dependant.dependencies
        ]
        video_dependencies = [dep.call for dep in video_route.dependant.dependencies]
        simulation_enforced_dependencies = [
            getattr(dependency, "enforced_dependency", None)
            for dependency in simulation_dependencies
        ]
        video_enforced_dependencies = [
            getattr(dependency, "enforced_dependency", None)
            for dependency in video_dependencies
        ]

        assert CheckSimulationQuota.dependency in simulation_enforced_dependencies
        assert CheckVideoQuota.dependency in video_enforced_dependencies


# ── Task Manager Tests ────────────────────────────────────────────


# ── Dict-like Compatibility Tests (#4843) ────────────────────────


# ── API Versioning Tests ──────────────────────────────────────────


# ── Linkage Decomposition Tests ──────────────────────────────────


# ── OpenAPI Enhancement Tests ─────────────────────────────────────
