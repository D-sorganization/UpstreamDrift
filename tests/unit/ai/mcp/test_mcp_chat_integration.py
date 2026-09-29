"""Unit tests for McpChatIntegration (TDD — written before implementation).

These tests verify the four core behaviours mandated by issue #5615:

1. No config file → empty tool list, no exception.
2. Pool with two tools → integration.tools() returns both.
3. call_tool() routes to the pool.
4. A server that fails to connect does not block other servers.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# The module under test will not exist until the implementation step;
# importing it here would cause a collection failure. We guard with a
# try/except so that the test file can be *collected* and the tests
# explicitly fail with a helpful message if the module is missing.
# ---------------------------------------------------------------------------
try:
    from src.shared.python.ai.mcp.mcp_chat_integration import McpChatIntegration

    _IMPORT_OK = True
except ImportError:  # pragma: no cover
    _IMPORT_OK = False
    McpChatIntegration = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_pytestmark = pytest.mark.unit


def _make_descriptor(name: str, server: str = "srv") -> MagicMock:
    """Return a mock McpToolDescriptor-like object."""
    d = MagicMock()
    d.name = name
    d.namespaced_name = f"{server}__{name}"
    d.server_name = server
    return d


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMcpChatIntegration:
    """Tests for McpChatIntegration."""

    def test_import_succeeds(self) -> None:
        """The module must be importable (fails until implementation exists)."""
        assert _IMPORT_OK, (
            "src.shared.python.ai.mcp.mcp_chat_integration could not be "
            "imported — run the implementation step first."
        )

    def test_integration_loads_no_servers_when_config_missing(
        self, tmp_path: Path
    ) -> None:
        """No config file → empty tool list, no exception raised.

        Precondition: config_path points to a non-existent file.
        Postcondition: integration.tools() == [] and no exception.
        """
        if not _IMPORT_OK:
            pytest.skip("McpChatIntegration not yet implemented")

        missing = tmp_path / "does_not_exist" / "mcp_servers.json"
        integration = McpChatIntegration(config_path=missing)
        tools = integration.tools()
        assert tools == [], f"Expected empty list when config is missing, got {tools!r}"

    def test_integration_aggregates_tools_from_pool(self, tmp_path: Path) -> None:
        """Mock pool returning 2 descriptors → integration.tools() returns both.

        Uses a minimal JSON config so the pool construction path is exercised
        without needing live subprocess servers.
        """
        if not _IMPORT_OK:
            pytest.skip("McpChatIntegration not yet implemented")

        config = tmp_path / "mcp_servers.json"
        config.write_text(
            '{"version": 1, "servers": [{"name": "srv1", "command": "echo", '
            '"args": [], "env": {}, "enabled": true}]}',
            encoding="utf-8",
        )

        tool_a = _make_descriptor("tool_alpha", "srv1")
        tool_b = _make_descriptor("tool_beta", "srv1")

        mock_pool = MagicMock()
        mock_pool.tools.return_value = [tool_a, tool_b]
        mock_pool.connected_count = 1
        mock_pool.server_count = 1

        with patch(
            "src.shared.python.ai.mcp.mcp_chat_integration._build_pool",
            return_value=mock_pool,
        ):
            integration = McpChatIntegration(config_path=config)
            tools = integration.tools()

        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "tool_alpha" in names
        assert "tool_beta" in names

    def test_integration_routes_call_to_correct_server(self, tmp_path: Path) -> None:
        """call_tool() must delegate to the pool's call_tool().

        Postcondition: pool.call_tool called exactly once with correct args.
        """
        if not _IMPORT_OK:
            pytest.skip("McpChatIntegration not yet implemented")

        config = tmp_path / "mcp_servers.json"
        config.write_text(
            '{"version": 1, "servers": [{"name": "srv1", "command": "echo", '
            '"args": [], "env": {}, "enabled": true}]}',
            encoding="utf-8",
        )

        mock_pool = MagicMock()
        mock_pool.tools.return_value = []
        mock_pool.connected_count = 1
        mock_pool.server_count = 1
        mock_pool.call_tool.return_value = {"result": "ok"}

        with patch(
            "src.shared.python.ai.mcp.mcp_chat_integration._build_pool",
            return_value=mock_pool,
        ):
            integration = McpChatIntegration(config_path=config)
            result = integration.call_tool("srv1__my_tool", {"arg": "val"})

        mock_pool.call_tool.assert_called_once_with("srv1__my_tool", {"arg": "val"})
        assert result == {"result": "ok"}

    def test_integration_handles_server_failure_gracefully(
        self, tmp_path: Path
    ) -> None:
        """A failing server does not prevent other tools from being available.

        Simulates _build_pool raising an exception for one server while
        still returning a pool with the healthy servers' tools.
        """
        if not _IMPORT_OK:
            pytest.skip("McpChatIntegration not yet implemented")

        config = tmp_path / "mcp_servers.json"
        config.write_text(
            '{"version": 1, "servers": ['
            '{"name": "good_srv", "command": "echo", "args": [], "env": {}, "enabled": true},'
            '{"name": "bad_srv", "command": "broken", "args": [], "env": {}, "enabled": true}'
            "]}",
            encoding="utf-8",
        )

        # Pool is still constructable (it swallows per-server failures internally),
        # but it reports only the healthy server's tools.
        good_tool = _make_descriptor("good_tool", "good_srv")
        mock_pool = MagicMock()
        mock_pool.tools.return_value = [good_tool]
        mock_pool.connected_count = 1
        mock_pool.server_count = 2

        with patch(
            "src.shared.python.ai.mcp.mcp_chat_integration._build_pool",
            return_value=mock_pool,
        ):
            # Must not raise despite bad_srv being unhealthy
            integration = McpChatIntegration(config_path=config)
            tools = integration.tools()

        assert len(tools) == 1
        assert tools[0].name == "good_tool"

    def test_integration_status_string(self, tmp_path: Path) -> None:
        """status() returns a human-readable string like 'MCP: 1/2 connected'."""
        if not _IMPORT_OK:
            pytest.skip("McpChatIntegration not yet implemented")

        config = tmp_path / "mcp_servers.json"
        config.write_text(
            '{"version": 1, "servers": ['
            '{"name": "srv1", "command": "echo", "args": [], "env": {}, "enabled": true},'
            '{"name": "srv2", "command": "echo", "args": [], "env": {}, "enabled": true}'
            "]}",
            encoding="utf-8",
        )

        mock_pool = MagicMock()
        mock_pool.tools.return_value = []
        mock_pool.connected_count = 1
        mock_pool.server_count = 2

        with patch(
            "src.shared.python.ai.mcp.mcp_chat_integration._build_pool",
            return_value=mock_pool,
        ):
            integration = McpChatIntegration(config_path=config)
            status = integration.status()

        assert "MCP" in status
        assert "1" in status
        assert "2" in status

    def test_integration_no_pool_when_no_servers(self, tmp_path: Path) -> None:
        """Empty server list → pool is not built, tools() returns []."""
        if not _IMPORT_OK:
            pytest.skip("McpChatIntegration not yet implemented")

        config = tmp_path / "mcp_servers.json"
        config.write_text(
            '{"version": 1, "servers": []}',
            encoding="utf-8",
        )

        with patch(
            "src.shared.python.ai.mcp.mcp_chat_integration._build_pool"
        ) as mock_build:
            integration = McpChatIntegration(config_path=config)
            tools = integration.tools()

        mock_build.assert_not_called()
        assert tools == []

    def test_integration_call_tool_no_pool_raises(self, tmp_path: Path) -> None:
        """call_tool() raises RuntimeError when no pool is active."""
        if not _IMPORT_OK:
            pytest.skip("McpChatIntegration not yet implemented")

        missing = tmp_path / "mcp_servers.json"
        integration = McpChatIntegration(config_path=missing)

        with pytest.raises(RuntimeError, match="No MCP pool"):
            integration.call_tool("srv__tool", {})
