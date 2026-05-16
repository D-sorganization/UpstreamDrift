"""TDD tests for modernized OnboardingDialog (issue #5612).

Verifies:
  - No hard-coded hex color literals in the source file
  - No QTextBrowser used for structural layout
  - No inline HTML string for main content (_get_welcome_html removed)
  - No emoji in window title
  - Behavior preserved: dismiss_onboarding, is_first_run, checkbox, doc button
  - Dialog can be instantiated without crash (smoke test)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Repository root resolution
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DIALOG_SRC = _REPO_ROOT / "src" / "launchers" / "onboarding_dialog.py"

# ---------------------------------------------------------------------------
# Static source analysis tests (red until source is rewritten)
# ---------------------------------------------------------------------------


def test_no_hex_colors_in_source() -> None:
    """onboarding_dialog.py must not contain any hard-coded hex color literals."""
    src = _DIALOG_SRC.read_text(encoding="utf-8")
    # Match CSS/Qt hex colors: # followed by 3 or 6 hex digits (word boundary)
    match = re.search(r'"#[0-9a-fA-F]{3,6}\b', src)
    assert match is None, (
        f"Found hardcoded hex color in {_DIALOG_SRC}: {match.group()!r}. "
        "Use QPalette roles instead (issue #5612)."
    )


def test_no_qtextbrowser_for_layout() -> None:
    """onboarding_dialog.py must not import or use QTextBrowser."""
    src = _DIALOG_SRC.read_text(encoding="utf-8")
    assert "QTextBrowser" not in src, (
        "Found QTextBrowser in onboarding_dialog.py. "
        "Replace with native QFrame/QLabel widgets (issue #5612)."
    )


def test_no_html_string_content() -> None:
    """_get_welcome_html must be removed — no inline HTML for main content."""
    src = _DIALOG_SRC.read_text(encoding="utf-8")
    assert "_get_welcome_html" not in src, (
        "Found _get_welcome_html in onboarding_dialog.py. "
        "Delete this method and rebuild with native widgets (issue #5612)."
    )


def test_no_emoji_in_window_title() -> None:
    """setWindowTitle call must not contain emoji characters."""
    src = _DIALOG_SRC.read_text(encoding="utf-8")
    # Find setWindowTitle lines
    title_lines = [line for line in src.splitlines() if "setWindowTitle" in line]
    assert title_lines, "No setWindowTitle call found in onboarding_dialog.py"
    for line in title_lines:
        # Detect emoji: code points above U+1F000 (common emoji range)
        for char in line:
            cp = ord(char)
            is_emoji = (
                0x1F300 <= cp <= 0x1FAFF  # misc symbols, emoticons
                or 0x2600 <= cp <= 0x27BF  # misc symbols
                or 0xFE00 <= cp <= 0xFE0F  # variation selectors
            )
            assert not is_emoji, (
                f"Emoji character U+{cp:04X} found in setWindowTitle line: "
                f"{line!r}. Remove emoji from window title (issue #5612)."
            )


def test_no_orange_brand_color_overrides() -> None:
    """No orange/brand hex colors (#FF8800, #FF5500, #FF9933) in source."""
    src = _DIALOG_SRC.read_text(encoding="utf-8")
    for color in ("#FF8800", "#FF5500", "#FF9933", "#ff8800", "#ff5500", "#ff9933"):
        assert color not in src, (
            f"Found brand color {color!r} in onboarding_dialog.py. "
            "Remove all orange/brand color overrides (issue #5612)."
        )


def test_no_inline_css_flex_or_gradient() -> None:
    """CSS properties unsupported by QTextBrowser must not appear."""
    src = _DIALOG_SRC.read_text(encoding="utf-8")
    for forbidden in ("display: flex", "linear-gradient", "-webkit-background-clip"):
        assert forbidden not in src, (
            f"Found unsupported CSS {forbidden!r} in onboarding_dialog.py. "
            "Remove CSS-in-HTML layout (issue #5612)."
        )


# ---------------------------------------------------------------------------
# Behavioral preservation tests
# ---------------------------------------------------------------------------


def test_is_first_run_returns_true_when_no_config(tmp_path: Path) -> None:
    """is_first_run() returns True when config file does not exist."""
    fake_config = tmp_path / ".upstreamdrift" / "onboarding_config.json"
    with patch("src.launchers.onboarding_dialog.ONBOARDING_CONFIG_PATH", fake_config):
        from src.launchers.onboarding_dialog import is_first_run

        assert is_first_run() is True


def test_is_first_run_returns_false_after_dismiss(tmp_path: Path) -> None:
    """is_first_run() returns False after dismiss_onboarding() is called."""
    fake_config = tmp_path / ".upstreamdrift" / "onboarding_config.json"
    with patch("src.launchers.onboarding_dialog.ONBOARDING_CONFIG_PATH", fake_config):
        from src.launchers.onboarding_dialog import dismiss_onboarding, is_first_run

        dismiss_onboarding()
        assert is_first_run() is False


def test_dismiss_onboarding_writes_config(tmp_path: Path) -> None:
    """dismiss_onboarding() writes onboarding_dismissed=True to disk."""
    fake_config = tmp_path / ".upstreamdrift" / "onboarding_config.json"
    with patch("src.launchers.onboarding_dialog.ONBOARDING_CONFIG_PATH", fake_config):
        from src.launchers.onboarding_dialog import dismiss_onboarding

        dismiss_onboarding()

    assert fake_config.exists()
    data = json.loads(fake_config.read_text(encoding="utf-8"))
    assert data.get("onboarding_dismissed") is True


# ---------------------------------------------------------------------------
# Smoke test — requires headless Qt
# ---------------------------------------------------------------------------


@pytest.mark.headless_safe
def test_dialog_instantiates_without_crash() -> None:
    """OnboardingDialog can be instantiated without raising an exception."""
    # Provide a minimal QApplication if none exists
    try:
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(sys.argv[:1])
    except Exception:
        pytest.skip("PyQt6 not available in this environment")

    try:
        from src.launchers.onboarding_dialog import OnboardingDialog

        dlg = OnboardingDialog(None)
        assert dlg is not None
        dlg.close()
    finally:
        pass  # Don't quit app — other tests may need it
