"""Regression tests for UpstreamDrift #5620.

The legacy right-panel ``AIAssistantPanel`` spliced into ``content_splitter``
was removed as part of the deprecated-chat sweep.  The canonical chat surface
is the Sidekick dock's "Chat" tab.  These tests assert the old pattern can
never be re-introduced silently.

Tests are static source-inspection checks so they run without PyQt6 and
without a display server — they are always ``headless_safe``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_source(rel_path: str) -> str:
    """Return the source text of a file given its path relative to REPO_ROOT.

    Uses direct file I/O rather than ``importlib`` so the tests remain
    headless-safe even when PyQt6 cannot be loaded (e.g., broken DLL in
    CI environments without a display).
    """
    src_path = REPO_ROOT / rel_path
    assert src_path.exists(), f"Expected source file at {src_path}"
    return src_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# #5620 — deprecated AIAssistantPanel must not be spliced into the splitter
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.headless_safe
class TestNoSplitterAiPanel:
    """launcher_ui_setup must not instantiate AIAssistantPanel in the splitter."""

    def test_no_aiassistantpanel_instantiation_in_ui_setup(self) -> None:
        """AIAssistantPanel() must not be constructed inside launcher_ui_setup.

        The only permissible references are inside comments that explain
        *why* the panel was removed.  An actual ``AIAssistantPanel(`` call
        means the old pattern has been re-introduced — that is the regression.
        """
        src = _read_source("src/launchers/launcher_ui_setup.py")
        # Strip comment lines before checking, so the explanatory NOTE comment
        # that documents the removal does not cause a false positive.
        non_comment_lines = [
            line for line in src.splitlines() if not line.lstrip().startswith("#")
        ]
        non_comment_src = "\n".join(non_comment_lines)
        assert "AIAssistantPanel(" not in non_comment_src, (
            "launcher_ui_setup.py must not instantiate AIAssistantPanel — "
            "the canonical chat surface is the Sidekick dock tab (UD #5620). "
            "Use the Sidekick embed adapter instead."
        )

    def test_no_setup_ai_panel_method_in_ui_setup(self) -> None:
        """_setup_ai_panel must not be defined inside launcher_ui_setup.

        The method was deleted as part of the #5620 cleanup.  A def for it
        would mean the legacy instantiation path has been brought back.
        """
        src = _read_source("src/launchers/launcher_ui_setup.py")
        non_comment_lines = [
            line for line in src.splitlines() if not line.lstrip().startswith("#")
        ]
        non_comment_src = "\n".join(non_comment_lines)
        assert "def _setup_ai_panel" not in non_comment_src, (
            "_setup_ai_panel must not be defined in launcher_ui_setup — "
            "it was removed by UD #5620 to eliminate the duplicate chat panel."
        )

    def test_ai_visible_flag_set_to_false(self) -> None:
        """_ai_visible must be initialised to False (no panel visible on start)."""
        src = _read_source("src/launchers/launcher_ui_setup.py")
        assert "_ai_visible = False" in src, (
            "_ai_visible must be set to False in init_ui() to reflect that "
            "no legacy splitter panel is present (UD #5620)."
        )

    def test_no_aiassistantpanel_imported_at_top_level(self) -> None:
        """AIAssistantPanel must not be imported at module level in launcher_ui_setup.

        Lazy/guarded imports inside methods are acceptable; a bare top-level
        ``from ... import AIAssistantPanel`` or ``import AIAssistantPanel``
        would re-couple the UI setup to the deprecated panel.
        """
        src_raw = _read_source("src/launchers/launcher_ui_setup.py")
        lines = src_raw.splitlines()
        # Consider only lines before the first ``class`` or ``def`` statement
        # that are not inside a comment block — i.e., module-level imports.
        module_level_lines: list[str] = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith(("class ", "def ")):
                break
            if not stripped.startswith("#"):
                module_level_lines.append(line)
        module_top = "\n".join(module_level_lines)
        assert "AIAssistantPanel" not in module_top, (
            "AIAssistantPanel must not appear in module-level imports of "
            "launcher_ui_setup.py (UD #5620)."
        )

    def test_content_splitter_has_no_ai_panel_widget(self) -> None:
        """content_splitter must not add an ai_panel widget.

        Check that the source does not contain a pattern that both references
        ``content_splitter`` and ``ai_panel`` together in non-comment code.
        """
        src = _read_source("src/launchers/launcher_ui_setup.py")
        non_comment_lines = [
            line for line in src.splitlines() if not line.lstrip().startswith("#")
        ]
        for line in non_comment_lines:
            if "content_splitter" in line and "ai_panel" in line:
                pytest.fail(
                    f"Forbidden pattern found in launcher_ui_setup.py — "
                    f"content_splitter must not reference ai_panel (UD #5620):\n  {line}"
                )
