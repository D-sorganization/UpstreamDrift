"""Tests for the AGENTS.md → system prompt loader (#5373).

Verifies that AGENTS.md is discovered relative to the repo root and
its content is injected into the AI assistant's system prompt at
session start.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.shared.python.ai.agents_md_loader import (
    AgentsMdLoader,
    build_system_prompt_with_agents_md,
    load_agents_md,
)

# ── Fixtures ──────────────────────────────────────────────────────────


SAMPLE_AGENTS_MD = """\
# AGENTS.md — Discovery Workflow & Shared-Infrastructure Directory

## A. Before you write new code — discovery workflow

When you're about to add functionality, run these five steps in order.

1. **Grep `src/shared/python/`** for the concept.
2. **Grep `src/tools/`** and `src/launchers/` for similar tools.
"""


@pytest.fixture()
def agents_md_file(tmp_path: Path) -> Path:
    """Write AGENTS.md to a temp directory (simulated repo root)."""
    agents_md = tmp_path / "AGENTS.md"
    agents_md.write_text(SAMPLE_AGENTS_MD, encoding="utf-8")
    return agents_md


@pytest.fixture()
def repo_root_with_agents_md(tmp_path: Path) -> Path:
    (tmp_path / "AGENTS.md").write_text(SAMPLE_AGENTS_MD, encoding="utf-8")
    return tmp_path


@pytest.fixture()
def repo_root_without_agents_md(tmp_path: Path) -> Path:
    return tmp_path


# ── AgentsMdLoader unit tests ─────────────────────────────────────────


class TestAgentsMdLoader:
    def test_loads_content_when_file_exists(
        self, repo_root_with_agents_md: Path
    ) -> None:
        loader = AgentsMdLoader(repo_root=repo_root_with_agents_md)
        content = loader.load()
        assert content is not None
        assert "AGENTS.md" in content
        assert "discovery workflow" in content.lower()

    def test_returns_none_when_file_missing(
        self, repo_root_without_agents_md: Path
    ) -> None:
        loader = AgentsMdLoader(repo_root=repo_root_without_agents_md)
        assert loader.load() is None

    def test_discovers_file_relative_to_repo_root(
        self, repo_root_with_agents_md: Path
    ) -> None:
        loader = AgentsMdLoader(repo_root=repo_root_with_agents_md)
        assert loader.agents_md_path == repo_root_with_agents_md / "AGENTS.md"

    def test_is_available_true_when_file_exists(
        self, repo_root_with_agents_md: Path
    ) -> None:
        loader = AgentsMdLoader(repo_root=repo_root_with_agents_md)
        assert loader.is_available is True

    def test_is_available_false_when_file_missing(
        self, repo_root_without_agents_md: Path
    ) -> None:
        loader = AgentsMdLoader(repo_root=repo_root_without_agents_md)
        assert loader.is_available is False

    def test_requires_path_type(self) -> None:
        with pytest.raises(TypeError, match="repo_root"):
            AgentsMdLoader(repo_root="not/a/path")  # type: ignore[arg-type]

    def test_load_is_idempotent(self, repo_root_with_agents_md: Path) -> None:
        loader = AgentsMdLoader(repo_root=repo_root_with_agents_md)
        first = loader.load()
        second = loader.load()
        assert first == second

    def test_content_is_stripped(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text(
            "   \n# Title\n\nContent\n   \n", encoding="utf-8"
        )
        loader = AgentsMdLoader(repo_root=tmp_path)
        content = loader.load()
        assert content is not None
        assert not content.startswith(" ")
        assert not content.endswith(" ")

    def test_large_file_is_truncated(self, tmp_path: Path) -> None:
        """Very large AGENTS.md must be truncated to stay within token budget."""
        big_content = "# AGENTS\n" + "x " * 50_000
        (tmp_path / "AGENTS.md").write_text(big_content, encoding="utf-8")
        loader = AgentsMdLoader(repo_root=tmp_path, max_chars=8192)
        content = loader.load()
        assert content is not None
        assert len(content) <= 8192

    def test_max_chars_default_is_reasonable(self, tmp_path: Path) -> None:
        loader = AgentsMdLoader(repo_root=tmp_path)
        assert loader.max_chars >= 4096

    def test_repr_shows_path(self, repo_root_with_agents_md: Path) -> None:
        loader = AgentsMdLoader(repo_root=repo_root_with_agents_md)
        assert "AGENTS.md" in repr(loader)


# ── load_agents_md convenience function ──────────────────────────────


class TestLoadAgentsMd:
    def test_convenience_function(self, repo_root_with_agents_md: Path) -> None:
        content = load_agents_md(repo_root=repo_root_with_agents_md)
        assert content is not None
        assert len(content) > 0

    def test_returns_none_for_missing(self, tmp_path: Path) -> None:
        content = load_agents_md(repo_root=tmp_path)
        assert content is None

    def test_auto_discovers_from_current_package(self) -> None:
        """load_agents_md() with no args should find the real AGENTS.md."""
        content = load_agents_md()
        # Real repo has AGENTS.md — it should be found
        assert content is not None
        assert len(content) > 0


# ── build_system_prompt_with_agents_md ───────────────────────────────


class TestBuildSystemPromptWithAgentsMd:
    def test_agents_md_injected_into_prompt(
        self, repo_root_with_agents_md: Path
    ) -> None:
        prompt = build_system_prompt_with_agents_md(
            app_context="upstream_drift",
            repo_root=repo_root_with_agents_md,
        )
        assert "AGENTS.md" in prompt
        assert "discovery workflow" in prompt.lower()

    def test_base_prompt_also_present(self, repo_root_with_agents_md: Path) -> None:
        prompt = build_system_prompt_with_agents_md(
            app_context="upstream_drift",
            repo_root=repo_root_with_agents_md,
        )
        # The base build_system_prompt contribution should still be present
        assert "UpstreamDrift" in prompt or "upstream" in prompt.lower()

    def test_graceful_without_agents_md(
        self, repo_root_without_agents_md: Path
    ) -> None:
        """Missing AGENTS.md must not crash; base prompt is returned."""
        prompt = build_system_prompt_with_agents_md(
            app_context="upstream_drift",
            repo_root=repo_root_without_agents_md,
        )
        assert len(prompt) > 0

    def test_extra_instructions_still_appended(
        self, repo_root_with_agents_md: Path
    ) -> None:
        prompt = build_system_prompt_with_agents_md(
            app_context="upstream_drift",
            repo_root=repo_root_with_agents_md,
            extra_instructions="Use SI units only.",
        )
        assert "SI units" in prompt

    def test_prompt_is_string(self, repo_root_with_agents_md: Path) -> None:
        prompt = build_system_prompt_with_agents_md(
            app_context="upstream_drift",
            repo_root=repo_root_with_agents_md,
        )
        assert isinstance(prompt, str)
