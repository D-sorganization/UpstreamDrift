"""Tests for the Sidekick design-token adapter."""

from __future__ import annotations

from src.shared.python.theme.sidekick_tokens import (
    REQUIRED_SIDEKICK_TOKENS,
    sidekick_tokens_from_theme,
)


def test_sidekick_tokens_map_theme_colors_to_canonical_names() -> None:
    tokens = sidekick_tokens_from_theme(
        {
            "bg": "#000001",
            "group_bg": "#000002",
            "border": "#000003",
            "text": "#000004",
            "text_secondary": "#000005",
            "label": "#000006",
            "focus": "#000007",
            "input_bg": "#000008",
            "accent": "#000009",
            "title_bg": "#00000a",
            "title_border": "#00000b",
            "table_alt": "#00000c",
            "button_hover": "#00000d",
        }
    )

    assert tokens["sidekick.color.canvas"] == "#000001"
    assert tokens["sidekick.color.surface"] == "#000002"
    assert tokens["sidekick.color.border"] == "#000003"
    assert tokens["sidekick.color.text"] == "#000004"
    assert tokens["sidekick.color.focus"] == "#000007"
    assert tokens["sidekick.color.accent.hover"] == "#00000d"


def test_sidekick_tokens_include_spacing_radius_and_font_defaults() -> None:
    tokens = sidekick_tokens_from_theme({})

    assert all(token_name in tokens for token_name in REQUIRED_SIDEKICK_TOKENS)
    assert tokens["sidekick.space.2"] == "8px"
    assert tokens["sidekick.radius.chat"] == "8px"
    assert "system-ui" in tokens["sidekick.font.family"]
