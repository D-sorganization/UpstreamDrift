"""UI token compliance tests for issues #5480, #5483, #5484, #5485.

Verifies:
  - Sidekick tile uses its own icon, not the Data Explorer icon (#5480)
  - custom_title_bar.py uses theme tokens, not hardcoded hex (#5485)
  - Chat.tsx does not use hardcoded bg-gray-950 class (#5484)
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Repository root resolution
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# #5480 — Sidekick tile uses a unique icon
# ---------------------------------------------------------------------------


def test_sidekick_svg_asset_exists() -> None:
    """The sidekick SVG icon must exist at src/launchers/assets/sidekick.svg."""
    svg_path = _REPO_ROOT / "src" / "launchers" / "assets" / "sidekick.svg"
    assert svg_path.exists(), (
        f"sidekick.svg not found at {svg_path}. "
        "Create a speech-bubble SVG icon for the Sidekick tile (issue #5480)."
    )


def test_models_yaml_sidekick_does_not_use_data_explorer_icon() -> None:
    """models.yaml sidekick entry must not reference data_explorer_modern.png."""
    yaml_path = _REPO_ROOT / "src" / "config" / "models.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    sidekick_entries = [
        m
        for m in data.get("models", [])
        if m.get("id") in ("sidekick", "chat_assistant")
    ]
    assert sidekick_entries, "No sidekick/chat_assistant entry found in models.yaml"

    for entry in sidekick_entries:
        logo = entry.get("launcher", {}).get("logo", "")
        assert "data_explorer_modern" not in logo, (
            f"models.yaml entry '{entry['id']}' still uses data_explorer_modern icon: "
            f"{logo!r}. Fix per issue #5480."
        )


def test_launcher_manifest_sidekick_does_not_use_data_explorer_icon() -> None:
    """launcher_manifest.json sidekick entry must not reference data_explorer* icons."""
    import json

    manifest_path = _REPO_ROOT / "src" / "config" / "launcher_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    sidekick_entries = [
        t
        for t in data.get("tiles", [])
        if t.get("id") in ("sidekick", "chat_assistant")
    ]
    # Manifest may not have a sidekick entry — that is fine.
    for entry in sidekick_entries:
        logo = entry.get("logo", "")
        assert "data_explorer" not in logo, (
            f"launcher_manifest.json entry '{entry['id']}' uses a data_explorer icon: "
            f"{logo!r}. Fix per issue #5480."
        )


# ---------------------------------------------------------------------------
# #5484 — Chat.tsx must not hardcode bg-gray-950
# ---------------------------------------------------------------------------


def test_chat_tsx_no_hardcoded_bg_gray_950() -> None:
    """ui/src/pages/Chat.tsx must not contain the Tailwind class bg-gray-950."""
    chat_path = _REPO_ROOT / "ui" / "src" / "pages" / "Chat.tsx"
    assert chat_path.exists(), f"Chat.tsx not found at {chat_path}"

    content = chat_path.read_text(encoding="utf-8")
    assert "bg-gray-950" not in content, (
        "Chat.tsx still contains hardcoded 'bg-gray-950'. "
        "Remove it so the sidekick-shell CSS rule provides the canvas color (issue #5484)."
    )


# ---------------------------------------------------------------------------
# #5485 — custom_title_bar.py must not contain hardcoded hex colors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden_hex",
    [
        "#1e1e1e",
        "#3a3f4a",
    ],
)
def test_custom_title_bar_no_hardcoded_hex(forbidden_hex: str) -> None:
    """custom_title_bar.py must not contain specific hardcoded hex colors."""
    title_bar_path = _REPO_ROOT / "src" / "launchers" / "custom_title_bar.py"
    assert title_bar_path.exists(), f"custom_title_bar.py not found at {title_bar_path}"

    content = title_bar_path.read_text(encoding="utf-8")
    # Case-insensitive match for the hex value
    pattern = re.compile(re.escape(forbidden_hex), re.IGNORECASE)
    assert not pattern.search(content), (
        f"custom_title_bar.py still contains hardcoded hex color {forbidden_hex!r}. "
        "Replace with theme token lookups (issue #5485)."
    )
