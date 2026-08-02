"""Tool registration helpers for AIAssistantPanel.

Keeps the tool decorators out of the panel module so it stays focused
on coordination.
"""

from __future__ import annotations

from typing import Any

from src.shared.python.ai.tool_registry import ToolCategory


def register_panel_tools(tools_registry: Any, rag_store: Any) -> None:
    """Register CLI shims and the RAG search tool with ``tools_registry``."""

    @tools_registry.register(
        name="claude_cli",
        description="Use Claude CLI to control the application.",
        category=ToolCategory.CONFIGURATION,
    )
    def claude_cli(command: str) -> str:
        return f"Executed Claude CLI: {command}"

    @tools_registry.register(
        name="codex_cli",
        description="Use Codex CLI to control the application.",
        category=ToolCategory.CONFIGURATION,
    )
    def codex_cli(command: str) -> str:
        return f"Executed Codex CLI: {command}"

    @tools_registry.register(
        name="cline_cli",
        description="Use Cline CLI to control the application.",
        category=ToolCategory.CONFIGURATION,
    )
    def cline_cli(command: str) -> str:
        return f"Executed Cline CLI: {command}"

    @tools_registry.register(
        name="search_knowledge_base",
        description="Search the user's resource library/codebase for information.",
        category=ToolCategory.ANALYSIS,
    )
    def search_knowledge_base(query: str) -> str:
        results = rag_store.query(query)
        if not results:
            return "No relevant information found."
        output = ["Found relevant documents:"]
        for doc, score in results:
            output.append(f"--- Document: {doc.id} (Score: {score:.2f}) ---")
            output.append(
                doc.content[:500] + "..." if len(doc.content) > 500 else doc.content
            )
        return "\n\n".join(output)
