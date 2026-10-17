"""Agentic skills and workflows manager.

Parses .agents/skills and .agent/workflows directories and provides them
as explicit function-calling tools or slash commands.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SkillDefinition:
    """Definition of an agentic skill or workflow."""

    id: str
    name: str
    description: str
    path: Path
    is_workflow: bool
    requires_consent: bool = True


class SkillsManager:
    """Discovers and manages agentic skills and workflows."""

    def __init__(self, workspace_root: Path | str) -> None:
        """Initialize the SkillsManager and discover skills.

        Args:
            workspace_root: Root directory of the repository/workspace.
        """
        self.workspace_root = Path(workspace_root).resolve()
        self.skills: dict[str, SkillDefinition] = {}
        self.workflows: dict[str, SkillDefinition] = {}
        self._discover()

    def _discover(self) -> None:
        """Scan directories for skills and workflows."""
        # Check .agents/skills and .agent/workflows
        skills_dir = self.workspace_root / ".agents" / "skills"
        workflows_dir = self.workspace_root / ".agent" / "workflows"
        if not workflows_dir.exists():
            workflows_dir = self.workspace_root / ".agents" / "workflows"

        if skills_dir.exists():
            for entry in skills_dir.iterdir():
                if entry.is_dir():
                    skill_md = entry / "SKILL.md"
                    if skill_md.exists():
                        self._parse_and_add(skill_md, is_workflow=False)

        if workflows_dir.exists():
            for entry in workflows_dir.glob("*.md"):
                if entry.is_file():
                    self._parse_and_add(entry, is_workflow=True)

    def _parse_and_add(self, md_path: Path, is_workflow: bool) -> None:
        """Parse a markdown file and add it to the registry.

        Args:
            md_path: Path to the markdown file.
            is_workflow: True if it's a workflow, False if it's a skill.
        """
        try:
            content = md_path.read_text(encoding="utf-8")
            lines = content.splitlines()
            name = md_path.stem if is_workflow else md_path.parent.name
            description = ""
            for line in lines:
                if line.startswith("#"):
                    name = line.lstrip("# ").strip()
                elif line.strip() and not line.startswith("#") and not description:
                    description = line.strip()

            identifier = (
                md_path.stem.lower() if is_workflow else md_path.parent.name.lower()
            )

            # Execution logic usually requires consent
            requires_consent = True

            defn = SkillDefinition(
                id=identifier,
                name=name,
                description=description or f"{name} functionality",
                path=md_path,
                is_workflow=is_workflow,
                requires_consent=requires_consent,
            )

            if is_workflow:
                self.workflows[identifier] = defn
            else:
                self.skills[identifier] = defn
        except Exception as e:  # noqa: BLE001 - skip unreadable or malformed skill files
            logger.warning("Failed to parse %s: %s", md_path, e)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return function calling schemas for the LLM backend.

        Returns:
            A list of OpenAI-compatible function definitions.
        """
        tools: list[dict[str, Any]] = []
        for defn in list(self.skills.values()) + list(self.workflows.values()):
            safe_name = defn.id.replace("-", "_")
            tool = {
                "type": "function",
                "function": {
                    "name": f"execute_{safe_name}",
                    "description": defn.description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            }
            tools.append(tool)
        return tools

    def get_skill_or_workflow(self, identifier: str) -> SkillDefinition | None:
        """Retrieve a specific skill or workflow by ID.

        Args:
            identifier: The ID of the skill or workflow.

        Returns:
            The parsed SkillDefinition or None if not found.
        """
        return self.skills.get(identifier) or self.workflows.get(identifier)
