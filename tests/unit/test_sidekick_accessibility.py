"""Tests for issue #5506 — sidekick accessibility / hygiene.

Covers:
5506a: assistant_panel.py contains no emoji literals.
5506b: chat_context.py exposes __all__.
5506c: check_tools_sidebar diagnostic uses "info" (not "warning") when not installed.
5506d: ChatPanel.tsx has aria-relevant="additions" and aria-atomic="false".
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock


# Broad emoji regex: Miscellaneous Symbols, Dingbats, Emoticons, etc.
_EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001F64F"
    r"\U0001F680-\U0001F6FF"
    r"\U0001F900-\U0001FAFF"
    r"☀-➿"  # Misc symbols + dingbats
    r"️]"  # Variation selector-16 (emoji presentation)
)

_REPO_ROOT = Path(__file__).parent.parent.parent

# Module path used by check_tools_sidebar's internal import
_SIDEBAR_INTEGRATION = "src.shared.python.gui_launcher.tools_sidebar_integration"


# ---------------------------------------------------------------------------
# 5506a — no emoji literals in assistant_panel.py
# ---------------------------------------------------------------------------


def test_assistant_panel_no_emoji_literals():
    """assistant_panel.py must not contain emoji literal characters (issue #5506a)."""
    panel_path = (
        _REPO_ROOT / "src" / "shared" / "python" / "ai" / "gui" / "assistant_panel.py"
    )
    assert panel_path.exists(), f"File not found: {panel_path}"
    source = panel_path.read_text(encoding="utf-8")
    matches = _EMOJI_RE.findall(source)
    assert not matches, (
        f"assistant_panel.py contains {len(matches)} emoji literal(s): "
        f"{list(set(matches))[:10]}. Replace with text equivalents."
    )


# ---------------------------------------------------------------------------
# 5506b — chat_context.py has __all__
# ---------------------------------------------------------------------------


def test_chat_context_has_all():
    """chat_context must expose __all__ (issue #5506b)."""
    from src.shared.python.ai import chat_context

    assert hasattr(chat_context, "__all__"), "chat_context module is missing __all__"
    assert len(chat_context.__all__) > 0, "__all__ must not be empty"


def test_chat_context_all_contains_record_event():
    """chat_context.__all__ must include the primary public helper (issue #5506b)."""
    from src.shared.python.ai import chat_context

    assert "record_event" in chat_context.__all__, (
        "'record_event' must appear in chat_context.__all__"
    )


# ---------------------------------------------------------------------------
# 5506c — check_tools_sidebar returns "info" when not installed
# ---------------------------------------------------------------------------


def _get_diag():
    """Return a fresh LauncherDiagnostics instance with results list ready."""
    from src.launchers.launcher_diagnostics import LauncherDiagnostics

    diag = LauncherDiagnostics.__new__(LauncherDiagnostics)
    diag.results = []
    return diag


def _make_sidebar_stub(available: bool) -> MagicMock:
    """Build a sys.modules stub for tools_sidebar_integration."""
    stub = MagicMock()
    stub.is_tools_sidebar_available = MagicMock(return_value=available)
    stub._resolved_sidebar_module_name = MagicMock(
        return_value="vendor.ud_tools.shared_sidebar"
    )
    return stub


def test_check_tools_sidebar_not_installed_returns_info():
    """When the Tools sidebar module is absent, status must be 'info' (issue #5506c)."""
    diag = _get_diag()
    stub = _make_sidebar_stub(available=False)
    prev = sys.modules.get(_SIDEBAR_INTEGRATION)
    sys.modules[_SIDEBAR_INTEGRATION] = stub
    try:
        result = diag.check_tools_sidebar()
    finally:
        if prev is None:
            sys.modules.pop(_SIDEBAR_INTEGRATION, None)
        else:
            sys.modules[_SIDEBAR_INTEGRATION] = prev

    assert result.status == "info", (
        f"Expected 'info' when Tools sidebar absent, got '{result.status}'"
    )


def test_check_tools_sidebar_import_error_returns_warning():
    """An ImportError during import must still yield 'warning' (issue #5506c)."""
    diag = _get_diag()

    # Inject a stub whose attribute access raises ImportError so that the
    # `from <mod> import is_tools_sidebar_available` inside check_tools_sidebar
    # raises ImportError during execution.
    class _BrokenModule:
        """Stub module whose attribute access raises ImportError."""

        def __getattr__(self, name: str):
            raise ImportError(f"simulated import failure for {name}")

    prev = sys.modules.get(_SIDEBAR_INTEGRATION)
    sys.modules[_SIDEBAR_INTEGRATION] = _BrokenModule()  # type: ignore[assignment]
    try:
        result = diag.check_tools_sidebar()
    finally:
        if prev is None:
            sys.modules.pop(_SIDEBAR_INTEGRATION, None)
        else:
            sys.modules[_SIDEBAR_INTEGRATION] = prev

    assert result.status == "warning", (
        f"Expected 'warning' on ImportError, got '{result.status}'"
    )


def test_check_tools_sidebar_available_returns_pass():
    """When the sidebar IS importable the status must be 'pass'."""
    diag = _get_diag()
    stub = _make_sidebar_stub(available=True)
    prev = sys.modules.get(_SIDEBAR_INTEGRATION)
    sys.modules[_SIDEBAR_INTEGRATION] = stub
    try:
        result = diag.check_tools_sidebar()
    finally:
        if prev is None:
            sys.modules.pop(_SIDEBAR_INTEGRATION, None)
        else:
            sys.modules[_SIDEBAR_INTEGRATION] = prev

    assert result.status == "pass", (
        f"Expected 'pass' when Tools sidebar available, got '{result.status}'"
    )


# ---------------------------------------------------------------------------
# 5506d — ChatPanel.tsx aria-relevant + aria-atomic
# ---------------------------------------------------------------------------


def test_chat_panel_tsx_aria_relevant_additions():
    """ChatPanel.tsx messages div must have aria-relevant='additions' (issue #5506d)."""
    tsx_path = _REPO_ROOT / "ui" / "src" / "components" / "ui" / "ChatPanel.tsx"
    assert tsx_path.exists(), f"File not found: {tsx_path}"
    source = tsx_path.read_text(encoding="utf-8")
    assert 'aria-relevant="additions"' in source, (
        'ChatPanel.tsx messages div is missing aria-relevant="additions"'
    )


def test_chat_panel_tsx_aria_atomic_false():
    """ChatPanel.tsx messages div must have aria-atomic='false' (issue #5506d)."""
    tsx_path = _REPO_ROOT / "ui" / "src" / "components" / "ui" / "ChatPanel.tsx"
    assert tsx_path.exists(), f"File not found: {tsx_path}"
    source = tsx_path.read_text(encoding="utf-8")
    assert 'aria-atomic="false"' in source, (
        'ChatPanel.tsx messages div is missing aria-atomic="false"'
    )
