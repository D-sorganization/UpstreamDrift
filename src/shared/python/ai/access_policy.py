"""Access-mode policy for AI chat tool exposure."""

from __future__ import annotations

from enum import Enum
from typing import Any

from src.shared.python.ai.tool_registry import Tool, ToolRegistry
from src.shared.python.ai.tools.codemap_tools import CODEMAP_TOOL_NAMES


class ChatAccessMode(str, Enum):
    """Explicit chat repository/tool access modes."""

    NO_REPO_ACCESS = "no_repo_access"
    READ_ONLY_DIAGNOSTICS = "read_only_diagnostics"
    AGENT_TOOLS = "agent_tools"


READ_ONLY_REPO_TOOL_NAMES = frozenset(
    {
        "read_file",
        "list_directory",
        "search_knowledge_base",
        *CODEMAP_TOOL_NAMES,
    }
)


def coerce_access_mode(value: str | ChatAccessMode | None) -> ChatAccessMode:
    """Return a valid chat access mode for persisted or UI-provided values."""
    if isinstance(value, ChatAccessMode):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        for mode in ChatAccessMode:
            if normalized in {mode.value, mode.name.lower()}:
                return mode
    return ChatAccessMode.NO_REPO_ACCESS


def allowed_tools_for_access_mode(
    registry: ToolRegistry,
    access_mode: str | ChatAccessMode,
    *,
    rag_enabled: bool = True,
    max_expertise: int = 4,
) -> list[Tool]:
    """Return registry tools allowed by the selected chat access mode."""
    if registry is None:
        raise ValueError("registry must be provided")
    mode = coerce_access_mode(access_mode)
    if mode == ChatAccessMode.NO_REPO_ACCESS:
        return []

    tools = registry.list_tools(max_expertise=max_expertise)
    if mode == ChatAccessMode.AGENT_TOOLS:
        return tools

    allowed_names = set(READ_ONLY_REPO_TOOL_NAMES)
    if not rag_enabled:
        allowed_names.difference_update({"search_knowledge_base", *CODEMAP_TOOL_NAMES})
    return [tool for tool in tools if tool.name in allowed_names]


def tool_declarations_for_access_mode(
    registry: ToolRegistry,
    access_mode: str | ChatAccessMode,
    *,
    provider_format: str = "openai",
    rag_enabled: bool = True,
    max_expertise: int = 4,
) -> list[dict[str, Any]]:
    """Return provider-formatted tool declarations allowed by access mode."""
    tools = allowed_tools_for_access_mode(
        registry,
        access_mode,
        rag_enabled=rag_enabled,
        max_expertise=max_expertise,
    )
    if provider_format == "openai":
        return [tool.to_openai_format() for tool in tools]
    if provider_format == "anthropic":
        return [tool.to_anthropic_format() for tool in tools]
    return [tool.to_json_schema() for tool in tools]


__all__ = [
    "ChatAccessMode",
    "READ_ONLY_REPO_TOOL_NAMES",
    "allowed_tools_for_access_mode",
    "coerce_access_mode",
    "tool_declarations_for_access_mode",
]
