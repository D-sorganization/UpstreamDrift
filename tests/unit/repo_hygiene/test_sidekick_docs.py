"""TDD tests for Sidekick documentation presence (issue #5465).

Verifies that the key Sidekick documentation files exist and mention
the required entry points so they cannot be accidentally deleted or
left empty.

These are import-free, read-only path checks — suitable for headless CI.
"""

from __future__ import annotations

import json
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


# ---------------------------------------------------------------------------
# Agent layer docs (epic #5967 / S9 / #5978)
# ---------------------------------------------------------------------------


class TestAgentLayerDocs:
    """Documentation for the Sidekick agent layer must exist and stay
    in sync with the code surface it documents."""

    def test_adr_exists(self) -> None:
        path = _REPO_ROOT / "docs" / "adr" / "0017-sidekick-agentic-action-layer.md"
        assert path.exists(), "ADR-0017 must exist; see epic #5967 / S9"

    def test_adr_status_is_accepted(self) -> None:
        path = _REPO_ROOT / "docs" / "adr" / "0017-sidekick-agentic-action-layer.md"
        assert "Status: Accepted" in path.read_text(encoding="utf-8")

    def test_adr_links_every_sub_issue(self) -> None:
        """ADR-0017 must cross-link every sub-issue of epic #5967."""
        path = _REPO_ROOT / "docs" / "adr" / "0017-sidekick-agentic-action-layer.md"
        content = path.read_text(encoding="utf-8")
        for issue in (
            "5967",
            "5970",
            "5971",
            "5972",
            "5973",
            "5974",
            "5975",
            "5976",
            "5977",
            "5978",
        ):
            assert f"#{issue}" in content, f"ADR-0017 must reference #{issue}"

    def test_developer_guide_exists(self) -> None:
        path = _REPO_ROOT / "docs" / "sidekick" / "agent.md"
        assert path.exists(), "docs/sidekick/agent.md must exist"

    def test_developer_guide_references_every_public_module(self) -> None:
        """The guide must name every public module under sidekick/agent/.

        New modules should land in the guide in the same PR they're
        introduced — that's the drift-prevention invariant for S9.
        """
        path = _REPO_ROOT / "docs" / "sidekick" / "agent.md"
        content = path.read_text(encoding="utf-8")
        for module in (
            "feature_catalog",
            "action_service",
            "subtab_adapter",
            "host_adapter",
            "planner",
            "action_audit",
            "access_policy",
            "workflow_bridge",
            "chat_surface",
        ):
            assert module in content, f"docs/sidekick/agent.md must reference {module}"

    def test_agents_md_mentions_agent_layer(self) -> None:
        """AGENTS.md must point new contributors at the agent layer."""
        path = _REPO_ROOT / "AGENTS.md"
        content = path.read_text(encoding="utf-8")
        assert "sidekick/agent" in content or "SidekickActionService" in content


# ---------------------------------------------------------------------------
# Standalone Sidekick docs (T9 — issue #5987)
# ---------------------------------------------------------------------------


class TestStandaloneSidekickDocs:
    """Standalone Sidekick documentation must exist and stay in sync.

    Added by T9 (#5987).
    """

    def test_adr_standalone_exists(self) -> None:
        adrs = list((_REPO_ROOT / "docs" / "adr").glob("*standalone-sidekick*"))
        assert adrs, (
            "No ADR for standalone-sidekick found; expected docs/adr/0018-standalone-sidekick.md"
        )

    def test_adr_standalone_is_accepted(self) -> None:
        adrs = list((_REPO_ROOT / "docs" / "adr").glob("*standalone-sidekick*"))
        if not adrs:
            pytest.skip("ADR not found")
        assert "Accepted" in adrs[0].read_text(encoding="utf-8")

    def test_adr_standalone_cross_links_issues(self) -> None:
        adrs = list((_REPO_ROOT / "docs" / "adr").glob("*standalone-sidekick*"))
        if not adrs:
            pytest.skip("ADR not found")
        text = adrs[0].read_text(encoding="utf-8")
        for issue in ("5984", "5985", "5986", "5987"):
            assert issue in text, f"ADR must cross-link issue #{issue}"

    def test_standalone_md_exists(self) -> None:
        assert (_REPO_ROOT / "docs" / "sidekick" / "standalone.md").exists(), (
            "docs/sidekick/standalone.md must exist (T9 #5987)"
        )

    def test_standalone_md_has_install_section(self) -> None:
        path = _REPO_ROOT / "docs" / "sidekick" / "standalone.md"
        if not path.exists():
            pytest.skip("standalone.md not found")
        assert "pip install" in path.read_text(encoding="utf-8")

    def test_standalone_md_has_run_examples(self) -> None:
        path = _REPO_ROOT / "docs" / "sidekick" / "standalone.md"
        if not path.exists():
            pytest.skip("standalone.md not found")
        assert "sidekick run" in path.read_text(encoding="utf-8")

    def test_sidekick_readme_cross_links_standalone(self) -> None:
        path = _REPO_ROOT / "docs" / "sidekick" / "README.md"
        if not path.exists():
            pytest.skip("docs/sidekick/README.md not found")
        assert "standalone" in path.read_text(encoding="utf-8").lower(), (
            "docs/sidekick/README.md must cross-link standalone.md"
        )

    def test_agents_md_has_standalone_entry(self) -> None:
        path = _REPO_ROOT / "AGENTS.md"
        assert "sidekick.standalone" in path.read_text(encoding="utf-8"), (
            "AGENTS.md must include a sidekick.standalone entry (T9 #5987)"
        )


# ---------------------------------------------------------------------------
# Chat/Sidekick boundary ADR (issue #6098)
# ---------------------------------------------------------------------------


class TestChatSidekickBoundaryAdr:
    """ADR-0022 must document the legacy chat/Sidekick boundary."""

    def _adr_path(self) -> Path:
        return _REPO_ROOT / "docs" / "adr" / "0022-chat-sidekick-boundary.md"

    def _read_adr(self) -> str:
        path = self._adr_path()
        assert path.exists(), "ADR-0022 must exist for issue #6098"
        return path.read_text(encoding="utf-8")

    def test_adr_exists(self) -> None:
        assert self._adr_path().exists()

    def test_adr_is_accepted(self) -> None:
        assert "Status: Accepted" in self._read_adr()

    def test_adr_cross_links_required_issues(self) -> None:
        content = self._read_adr()
        for issue in ("5922", "5967", "5969", "6098"):
            assert f"#{issue}" in content, f"ADR-0022 must reference #{issue}"

    def test_adr_names_canonical_and_legacy_surfaces(self) -> None:
        content = self._read_adr()
        for token in (
            "_chat_dock_widget_qt.py",
            "AIAssistantPanel",
            "UnifiedToolsSidebar",
            "ChatPanel",
        ):
            assert token in content, f"ADR-0022 must mention {token}"

    def test_adr_index_mentions_0022(self) -> None:
        path = _REPO_ROOT / "docs" / "adr" / "README.md"
        assert "0022-chat-sidekick-boundary.md" in path.read_text(encoding="utf-8")

    def test_file_size_budget_references_adr_0022(self) -> None:
        path = _REPO_ROOT / "scripts" / "config" / "file_size_budget.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        reason = None
        for entry in data["exceptions"]:
            if entry["path"] == "src/shared/python/chat/_chat_dock_widget_qt.py":
                reason = entry["reason"]
                break
        assert reason is not None, "chat dock widget exception must exist"
        assert "ADR-0022" in reason
