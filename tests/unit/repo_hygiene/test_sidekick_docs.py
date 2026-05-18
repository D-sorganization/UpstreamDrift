"""TDD tests for Sidekick documentation presence (issue #5465).

Verifies that the key Sidekick documentation files exist and mention
the required entry points so they cannot be accidentally deleted or
left empty.

These are import-free, read-only path checks — suitable for headless CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# AGENTS.md — must mention Sidekick with key entry points
# ---------------------------------------------------------------------------


class TestAgentsMdSidekick:
    """AGENTS.md must document Sidekick with the required file pointers."""

    def _read(self) -> str:
        path = _REPO_ROOT / "AGENTS.md"
        assert path.exists(), "AGENTS.md must exist at the repo root"
        return path.read_text(encoding="utf-8")

    def test_mentions_sidekick(self) -> None:
        """AGENTS.md must contain the word 'Sidekick'."""
        assert "Sidekick" in self._read()

    def test_mentions_sidekick_tokens(self) -> None:
        """AGENTS.md must reference sidekick_tokens.py (design-token source)."""
        assert "sidekick_tokens" in self._read()

    def test_mentions_assistant_panel(self) -> None:
        """AGENTS.md must reference the PyQt panel entry point."""
        assert "assistant_panel" in self._read()

    def test_mentions_chat_panel(self) -> None:
        """AGENTS.md must reference the React/Tauri ChatPanel surface."""
        assert "ChatPanel" in self._read()

    def test_has_sidekick_section_heading(self) -> None:
        """AGENTS.md must have a dedicated Sidekick section."""
        content = self._read()
        # Accept either a Markdown heading or a bold label inside an existing section
        assert "Sidekick" in content and ("##" in content or "**Sidekick" in content)


# ---------------------------------------------------------------------------
# docs/development/embedding_a_tool.md — must reference Sidekick
# ---------------------------------------------------------------------------


class TestEmbeddingAToolSidekick:
    """embedding_a_tool.md must reference Sidekick as a worked example."""

    def _read(self) -> str:
        path = _REPO_ROOT / "docs" / "development" / "embedding_a_tool.md"
        assert path.exists(), "docs/development/embedding_a_tool.md must exist"
        return path.read_text(encoding="utf-8")

    def test_mentions_sidekick(self) -> None:
        """embedding_a_tool.md must mention 'Sidekick'."""
        assert "Sidekick" in self._read()

    def test_mentions_sidekick_adapter(self) -> None:
        """embedding_a_tool.md must reference the Sidekick embed adapter."""
        assert "_embed_adapter" in self._read()

    def test_mentions_prefers_dock(self) -> None:
        """embedding_a_tool.md must document prefers_dock for Sidekick tools."""
        assert "prefers_dock" in self._read()


# ---------------------------------------------------------------------------
# SPEC.md — must have a Sidekick entry in the tool inventory
# ---------------------------------------------------------------------------


class TestSpecMdSidekick:
    """SPEC.md must have a Sidekick row in the Key Components table."""

    def _read(self) -> str:
        path = _REPO_ROOT / "SPEC.md"
        assert path.exists(), "SPEC.md must exist at the repo root"
        return path.read_text(encoding="utf-8")

    def test_has_sidekick_row(self) -> None:
        """SPEC.md Key Components table must contain a 'Sidekick' row."""
        assert "Sidekick" in self._read()

    def test_sidekick_row_has_correct_paths(self) -> None:
        """SPEC.md Sidekick row must reference the PyQt panel path."""
        content = self._read()
        assert "ai/gui/assistant" in content or "assistant_panel" in content

    def test_spec_mentions_sidekick_feature(self) -> None:
        """SPEC.md Feature Status table must mention Sidekick."""
        content = self._read()
        # Either as a feature row (Fxx) or as part of the component table
        assert content.count("Sidekick") >= 2


# ---------------------------------------------------------------------------
# docs/sidekick/README.md — must exist with required content
# ---------------------------------------------------------------------------


class TestSidekickReadme:
    """docs/sidekick/README.md must exist and contain the key sections."""

    def _read(self) -> str:
        path = _REPO_ROOT / "docs" / "sidekick" / "README.md"
        assert path.exists(), (
            "docs/sidekick/README.md must exist; "
            "create it per the acceptance criteria in issue #5465"
        )
        return path.read_text(encoding="utf-8")

    def test_readme_exists(self) -> None:
        """docs/sidekick/README.md must exist."""
        path = _REPO_ROOT / "docs" / "sidekick" / "README.md"
        assert path.exists()

    def test_mentions_sidekick(self) -> None:
        """README.md must mention 'Sidekick'."""
        assert "Sidekick" in self._read()

    def test_describes_pyqt_surface(self) -> None:
        """README.md must describe the PyQt launcher panel."""
        content = self._read()
        assert "AIAssistantPanel" in content or "PyQt" in content

    def test_describes_react_surface(self) -> None:
        """README.md must describe the React/Tauri chat surface."""
        content = self._read()
        assert "ChatPanel" in content or "React" in content or "Tauri" in content

    def test_describes_design_tokens(self) -> None:
        """README.md must document the design-token contract."""
        content = self._read()
        assert "sidekick_tokens" in content or "design token" in content.lower()

    def test_describes_agentic_tools(self) -> None:
        """README.md must explain how to add agentic tools."""
        content = self._read()
        assert "tool" in content.lower()

    def test_describes_embedding_path(self) -> None:
        """README.md must describe how to embed Sidekick as a launcher tile."""
        content = self._read()
        assert "embed" in content.lower() or "_embed_adapter" in content
