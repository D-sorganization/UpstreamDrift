"""Tool registration helpers for AIAssistantPanel.

Keeps the tool decorators out of the panel module so it stays focused
on coordination.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.shared.python.ai.tool_registry import ToolCategory
from src.shared.python.ai.wizards import wizard_for


def register_panel_tools(
    tools_registry: Any, rag_store: Any, project_root: Path | str | None = None
) -> None:
    """Register CLI shims and the knowledge search tool with ``tools_registry``.

    ``search_knowledge_base`` answers from the host product's Wizard pack
    when ``project_root`` has one (Tools#5346), else from ``rag_store``.
    """

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
        wizard = wizard_for(project_root) if project_root else None
        passages = wizard.search(query) if wizard is not None else []
        if passages:
            return "\n\n".join(
                ["Found relevant passages:"]
                + [f"--- {p.citation} ---\n{p.text}" for p in passages]
            )
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
