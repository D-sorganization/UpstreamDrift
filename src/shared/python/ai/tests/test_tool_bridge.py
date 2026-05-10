"""Tests for ChatToolBridge — connects ToolRegistry to chat sessions.

Covers:
- Tool lookup and execution
- Argument validation
- Confirmation gate handling
- Error handling
- Provider format conversion
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.shared.python.ai.tool_bridge import ChatToolBridge


class TestChatToolBridge:
    def test_no_registry_returns_empty_tools(self) -> None:
        bridge = ChatToolBridge(registry=None)
        assert bridge.get_tools_for_provider() == []
        assert bridge.get_tool_names() == []

    def test_set_registry(self) -> None:
        bridge = ChatToolBridge()
        mock_reg = MagicMock()
        bridge.set_registry(mock_reg)
        assert bridge._registry is mock_reg

    def test_get_tool_names(self) -> None:
        mock_tool_a = MagicMock()
        mock_tool_a.name = "alpha"
        mock_tool_b = MagicMock()
        mock_tool_b.name = "beta"

        mock_reg = MagicMock()
        mock_reg.get_all_tools.return_value = [mock_tool_a, mock_tool_b]

        bridge = ChatToolBridge(registry=mock_reg)
        names = bridge.get_tool_names()
        assert names == ["alpha", "beta"]

    def test_get_tools_openai_format(self) -> None:
        mock_tool = MagicMock()
        mock_tool.to_openai_format.return_value = {"type": "function", "function": {}}

        mock_reg = MagicMock()
        mock_reg.get_all_tools.return_value = [mock_tool]

        bridge = ChatToolBridge(registry=mock_reg)
        tools = bridge.get_tools_for_provider("openai")
        assert len(tools) == 1
        assert tools[0]["type"] == "function"

    def test_get_tools_anthropic_format(self) -> None:
        mock_tool = MagicMock()
        mock_tool.to_anthropic_format.return_value = {"name": "test"}

        mock_reg = MagicMock()
        mock_reg.get_all_tools.return_value = [mock_tool]

        bridge = ChatToolBridge(registry=mock_reg)
        tools = bridge.get_tools_for_provider("anthropic")
        assert len(tools) == 1

    def test_get_tool_info(self) -> None:
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.category.name = "ANALYSIS"
        mock_tool.requires_confirmation = False
        mock_tool.to_json_schema.return_value = {"parameters": {"type": "object"}}

        mock_reg = MagicMock()
        mock_reg.get_tool.return_value = mock_tool

        bridge = ChatToolBridge(registry=mock_reg)
        info = bridge.get_tool_info("test_tool")
        assert info is not None
        assert info["name"] == "test_tool"

    def test_get_tool_info_not_found(self) -> None:
        mock_reg = MagicMock()
        mock_reg.get_tool.return_value = None

        bridge = ChatToolBridge(registry=mock_reg)
        info = bridge.get_tool_info("nonexistent")
        assert info is None


@pytest.mark.asyncio
class TestChatToolBridgeExecution:
    async def test_execute_no_registry(self) -> None:
        bridge = ChatToolBridge(registry=None)
        result = await bridge.handle_tool_call("s1", "test", {})
        assert result["success"] is False
        assert "No tool registry" in result["error"]

    async def test_execute_unknown_tool(self) -> None:
        mock_reg = MagicMock()
        mock_reg.get_tool.return_value = None
        mock_reg.get_all_tools.return_value = []

        bridge = ChatToolBridge(registry=mock_reg)
        result = await bridge.handle_tool_call("s1", "unknown", {})
        assert result["success"] is False
        assert "Unknown tool" in result["error"]

    async def test_execute_validation_error(self) -> None:
        mock_tool = MagicMock()
        mock_tool.validate_arguments.return_value = ["Missing required: x"]

        mock_reg = MagicMock()
        mock_reg.get_tool.return_value = mock_tool

        bridge = ChatToolBridge(registry=mock_reg)
        result = await bridge.handle_tool_call("s1", "tool", {})
        assert result["success"] is False
        assert "Validation" in result["error"]

    async def test_execute_success(self) -> None:
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.result = {"data": "value"}
        mock_result.error = None

        mock_tool = MagicMock()
        mock_tool.validate_arguments.return_value = []
        mock_tool.requires_confirmation = False
        mock_tool.execute.return_value = mock_result

        mock_reg = MagicMock()
        mock_reg.get_tool.return_value = mock_tool

        bridge = ChatToolBridge(registry=mock_reg)
        result = await bridge.handle_tool_call("s1", "tool", {"arg": "val"})
        assert result["success"] is True
        assert result["result"] == {"data": "value"}
        assert result["execution_time_s"] >= 0

    async def test_execute_with_confirmation_approved(self) -> None:
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.result = "done"
        mock_result.error = None

        mock_tool = MagicMock()
        mock_tool.validate_arguments.return_value = []
        mock_tool.requires_confirmation = True
        mock_tool.execute.return_value = mock_result

        mock_reg = MagicMock()
        mock_reg.get_tool.return_value = mock_tool

        confirm_cb = AsyncMock(return_value=True)
        bridge = ChatToolBridge(
            registry=mock_reg, require_confirmation_callback=confirm_cb
        )
        result = await bridge.handle_tool_call("s1", "tool", {})
        assert result["success"] is True
        confirm_cb.assert_awaited_once()

    async def test_execute_with_confirmation_denied(self) -> None:
        mock_tool = MagicMock()
        mock_tool.validate_arguments.return_value = []
        mock_tool.requires_confirmation = True

        mock_reg = MagicMock()
        mock_reg.get_tool.return_value = mock_tool

        confirm_cb = AsyncMock(return_value=False)
        bridge = ChatToolBridge(
            registry=mock_reg, require_confirmation_callback=confirm_cb
        )
        result = await bridge.handle_tool_call("s1", "tool", {})
        assert result["success"] is False
        assert "declined" in result["error"]

    async def test_execute_handler_exception(self) -> None:
        mock_tool = MagicMock()
        mock_tool.validate_arguments.return_value = []
        mock_tool.requires_confirmation = False
        mock_tool.execute.side_effect = RuntimeError("kaboom")

        mock_reg = MagicMock()
        mock_reg.get_tool.return_value = mock_tool

        bridge = ChatToolBridge(registry=mock_reg)
        result = await bridge.handle_tool_call("s1", "tool", {})
        assert result["success"] is False
        assert "kaboom" in result["error"]
