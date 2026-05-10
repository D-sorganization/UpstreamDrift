"""Production tool bridge — connects ToolRegistry to ChatServiceBase.

This module provides the bridge between the existing ToolRegistry
(which defines what tools do) and the shared ChatServiceBase
(which handles session management and WebSocket routing).

It enables any application to:
1. Register its domain-specific tools
2. Have them automatically available through the chat WebSocket
3. Execute them with validation and confirmation gates

Usage::

    from src.shared.python.ai.tool_bridge import ChatToolBridge

    bridge = ChatToolBridge(registry=my_registry)
    result = await bridge.handle_tool_call(session_id, tool_call)
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class ChatToolBridge:
    """Bridge between chat sessions and the tool registry.

    Provides tool execution with validation, confirmation gates,
    and result formatting for the chat WebSocket protocol.

    Args:
        registry: The ToolRegistry containing registered tools.
        require_confirmation_callback: Optional async callback that asks the
            user to confirm before executing destructive tools. Returns True
            if confirmed.
    """

    def __init__(
        self,
        registry: Any = None,
        require_confirmation_callback: Any = None,
    ) -> None:
        self._registry = registry
        self._confirmation_callback = require_confirmation_callback
        self._pending_confirmations: dict[str, dict[str, Any]] = {}

    def set_registry(self, registry: Any) -> None:
        """Set or update the tool registry.

        Args:
            registry: ToolRegistry instance.
        """
        self._registry = registry

    def get_tools_for_provider(
        self, provider_format: str = "openai"
    ) -> list[dict[str, Any]]:
        """Get tool definitions formatted for a specific provider.

        Args:
            provider_format: Provider format ('openai' or 'anthropic').

        Returns:
            List of tool definitions in provider format.
        """
        if self._registry is None:
            return []

        tools = self._registry.get_all_tools()
        result = []
        for tool in tools:
            if provider_format == "anthropic":
                result.append(tool.to_anthropic_format())
            else:
                result.append(tool.to_openai_format())
        return result

    def get_tool_names(self) -> list[str]:
        """Get list of registered tool names.

        Returns:
            Sorted list of tool names.
        """
        if self._registry is None:
            return []
        return sorted(t.name for t in self._registry.get_all_tools())

    async def handle_tool_call(
        self,
        session_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a tool call from the chat session.

        Validates arguments, checks confirmation requirements,
        executes the tool, and returns formatted results.

        Args:
            session_id: Chat session ID.
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.

        Returns:
            Dict with keys: success, result, error, execution_time_s.
        """
        if self._registry is None:
            return _result(
                success=False,
                error="No tool registry configured",
            )

        # Look up the tool
        tool = self._registry.get_tool(tool_name)
        if tool is None:
            available = self.get_tool_names()
            return _result(
                success=False,
                error=f"Unknown tool: {tool_name}. Available: {available}",
            )

        # Validate arguments
        validation_errors = tool.validate_arguments(arguments)
        if validation_errors:
            return _result(
                success=False,
                error=f"Validation errors: {'; '.join(validation_errors)}",
            )

        # Check confirmation requirement
        if tool.requires_confirmation:
            if self._confirmation_callback is not None:
                confirmed = await self._confirmation_callback(
                    session_id, tool_name, arguments
                )
                if not confirmed:
                    return _result(
                        success=False,
                        error="User declined confirmation for destructive tool",
                    )
            else:
                logger.warning(
                    "Tool %s requires confirmation but no callback set — "
                    "proceeding without confirmation",
                    tool_name,
                )

        # Execute the tool
        start = time.monotonic()
        try:
            tool_result = tool.execute(arguments)
            elapsed = time.monotonic() - start

            logger.info(
                "Tool %s executed in %.2fs (session=%s, success=%s)",
                tool_name,
                elapsed,
                session_id,
                tool_result.success,
            )

            return {
                "success": tool_result.success,
                "result": tool_result.result,
                "error": tool_result.error,
                "execution_time_s": round(elapsed, 3),
                "tool_name": tool_name,
            }

        except Exception as e:  # noqa: BLE001
            elapsed = time.monotonic() - start
            logger.exception("Tool execution error: %s", tool_name)
            return _result(
                success=False,
                error=f"Tool execution failed: {e}",
                execution_time_s=round(elapsed, 3),
            )

    def get_tool_info(self, tool_name: str) -> dict[str, Any] | None:
        """Get info about a specific tool.

        Args:
            tool_name: Tool name to look up.

        Returns:
            Tool info dict or None.
        """
        if self._registry is None:
            return None

        tool = self._registry.get_tool(tool_name)
        if tool is None:
            return None

        return {
            "name": tool.name,
            "description": tool.description,
            "category": tool.category.name
            if hasattr(tool.category, "name")
            else str(tool.category),
            "requires_confirmation": tool.requires_confirmation,
            "parameters": tool.to_json_schema().get("parameters", {}),
        }


def _result(
    success: bool,
    result: Any = None,
    error: str | None = None,
    execution_time_s: float = 0.0,
) -> dict[str, Any]:
    """Create standardized tool result dict."""
    return {
        "success": success,
        "result": result,
        "error": error,
        "execution_time_s": execution_time_s,
    }
