"""Regression: the launcher must NEVER attach a second AIAssistantPanel.

History — closes #5620.

Before this lock, ``LauncherUISetupMixin.init_ui`` instantiated a second
``AIAssistantPanel`` in the right-edge content splitter
(``_setup_ai_panel``). That panel duplicated the canonical Sidekick
chat tab provided by the vendored Tools sidebar, so users saw two
parallel chat surfaces — and the splitter copy was the one that opened
on startup. The duplicate was deleted and these tests pin the
contract so a future agent cannot re-introduce it.

Design rationale:

* TDD: each assertion in this file fails immediately if the deprecated
  code paths are restored — the test would have caught the original
  regression had it existed.
* DbC: the invariant "exactly one chat surface" is a contract the
  launcher exports to its users; the assertions here document and
  enforce it.
* LOD: these tests query only ``GolfLauncher`` and ``launcher_ui_setup``
  module attributes; they do not chain through Sidekick internals.
* DRY: a single ``_source_contents`` helper is reused.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.ui

_LAUNCHER_UI_SETUP = (
    Path(__file__).resolve().parents[3] / "src" / "launchers" / "launcher_ui_setup.py"
)


def _source_contents() -> str:
    return _LAUNCHER_UI_SETUP.read_text(encoding="utf-8")


def _has_method(module_source: str, method_name: str) -> bool:
    tree = ast.parse(module_source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return True
    return False


class TestNoDeprecatedAIPanel:
    """Static guard tests against the deprecated splitter chat surface."""

    def test_setup_ai_panel_method_is_removed(self) -> None:
        """The ``_setup_ai_panel`` method must not exist on the mixin.

        Restoring it would re-create the duplicated chat surface that
        prompted #5620. If a future feature genuinely needs a second
        panel, file a new issue first — do not silently un-delete this
        method.
        """
        assert not _has_method(_source_contents(), "_setup_ai_panel"), (
            "_setup_ai_panel was re-introduced into launcher_ui_setup.py."
            " The canonical chat surface is the Sidekick dock's Chat tab."
            " Do NOT add a second AIAssistantPanel to the content splitter."
        )

    def test_sync_chat_session_method_is_removed(self) -> None:
        """``_sync_chat_session`` only existed to feed the deprecated panel.

        The Sidekick ``ChatDockWidget`` performs its own session
        handshake via the shared ``active_chat_session.txt`` file, so
        this helper has no remaining caller.
        """
        assert not _has_method(_source_contents(), "_sync_chat_session"), (
            "_sync_chat_session was restored. Its only purpose was to"
            " feed the deprecated splitter chat panel — the Sidekick"
            " ChatDockWidget reads the shared session file directly."
        )

    def test_init_ui_does_not_construct_a_splitter_chat_panel(self) -> None:
        """No ``AIAssistantPanel(`` call must appear in ``init_ui`` text.

        We check by text rather than AST because the call would be a
        method-local import + construction. Any restoration of the
        ``AIAssistantPanel(self)`` pattern in this file should fail the
        test and force a conversation.
        """
        source = _source_contents()
        assert "AIAssistantPanel(" not in source, (
            "AIAssistantPanel(...) construction reappeared in"
            " launcher_ui_setup.py. The launcher must surface chat ONLY"
            " through the Sidekick dock's vendored Chat tab."
        )

    def test_self_ai_panel_attribute_not_assigned(self) -> None:
        """No ``self.ai_panel = ...`` assignment may remain.

        Other modules continue to gate on ``hasattr(self, "ai_panel")``
        for backward compatibility (those guards become safe no-ops).
        But this file must no longer materialise the attribute at all,
        so ``hasattr`` everywhere else returns ``False``.
        """
        source = _source_contents()
        assert "self.ai_panel" not in source, (
            "self.ai_panel reference reappeared in launcher_ui_setup.py."
            " Restoring this attribute would silently re-enable the"
            " deprecated chat path on every consumer that uses"
            " ``hasattr(self, 'ai_panel')``."
        )
