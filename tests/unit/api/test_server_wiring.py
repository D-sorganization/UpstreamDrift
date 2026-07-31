"""Tests for server.py route registration — WebSocket routes (issue #2448).

The main server must register chat_ws and simulation_ws routers explicitly
because route_registry.py excludes them from auto-discovery.

These tests read server.py source directly because the server has production
dependencies (slowapi, uvicorn) that are not installed in the unit test
environment; the CI environment has them and will also run the integration
tests that exercise the actual HTTP layer.
"""

from pathlib import Path

_SERVER_PY = Path(__file__).parents[3] / "src" / "api" / "server.py"
_SERVER_SRC = _SERVER_PY.read_text(encoding="utf-8")


class TestWebSocketRoutesRegistered:
    """WebSocket routes are registered on the main server app."""

    def test_server_imports_simulation_ws(self) -> None:
        """server.py imports simulation_ws router module."""
        assert (
            "simulation_ws" in _SERVER_SRC
        ), "server.py must import simulation_ws to register its WebSocket route"

    def test_server_imports_chat_ws(self) -> None:
        """server.py imports chat_ws router module."""
        assert (
            "chat_ws" in _SERVER_SRC
        ), "server.py must import chat_ws to register its WebSocket route"

    def test_server_includes_simulation_ws_router(self) -> None:
        """server.py calls app.include_router for simulation_ws.router."""
        assert "simulation_ws.router" in _SERVER_SRC, (
            "server.py must call app.include_router(simulation_ws.router, ...) "
            "to expose the /ws/simulate WebSocket endpoint"
        )

    def test_server_includes_chat_ws_router(self) -> None:
        """server.py calls app.include_router for chat_ws.router."""
        assert "chat_ws.router" in _SERVER_SRC, (
            "server.py must call app.include_router(chat_ws.router, ...) "
            "to expose the /ws/chat WebSocket endpoint"
        )

    def test_chat_service_initialised_in_lifespan(self) -> None:
        """server.py lifespan initialises ChatService and stores it in app.state.

        Required by the chat_ws route which reads request.app.state.chat_service.
        """
        assert "chat_service" in _SERVER_SRC, (
            "server.py lifespan must initialise ChatService and store it in "
            "app.state.chat_service for the chat WebSocket route to work."
        )
        assert (
            "ChatService" in _SERVER_SRC
        ), "server.py must import and instantiate ChatService"

    def test_route_registry_exclusion_set_contains_websocket_modules(self) -> None:
        """Auto-discovery exclusion list skips chat_ws and simulation_ws."""
        from src.api.route_registry import _EXCLUDED_MODULES

        assert "chat_ws" in _EXCLUDED_MODULES
        assert "simulation_ws" in _EXCLUDED_MODULES
