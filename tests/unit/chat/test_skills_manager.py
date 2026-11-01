"""Tests for SkillsManager workflow discovery (Issue #14225, mirrors Tools#5894)."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.shared.python.chat.skills_manager import SkillsManager

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _write_workflow(directory: Path, stem: str, title: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.md").write_text(f"# {title}\nDescription.", encoding="utf-8")


def test_skills_manager_ignores_legacy_agent_workflows(tmp_path: Path) -> None:
    """Workflows come only from .agents/workflows, never the legacy .agent tree."""
    _write_workflow(tmp_path / ".agents" / "workflows", "sync-all", "Sync All Repos")
    _write_workflow(tmp_path / ".agent" / "workflows", "legacy-only", "Legacy")

    manager = SkillsManager(tmp_path)

    assert set(manager.workflows) == {"sync-all"}
