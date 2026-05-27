"""TDD coverage for the 60-token ``ThemeColors`` derivation pipeline.

``ThemeColors.model_post_init`` promotes a 14-base-token theme dict into
a full 60-token semantic palette so every downstream consumer (PyQt6
stylesheets, React, matplotlib) can rely on every modern design token
resolving regardless of how partially specified the input theme was.
"""

from __future__ import annotations

import pytest

# Pydantic model lives in src.shared.python.theme.api but tests import
# through the package's public surface so we don't accidentally bypass
# the post-init derivation.
from src.shared.python.theme.api import ThemeColors


_BASE_14 = {
    "bg": "#ffffff",
    "group_bg": "#f8f9fa",
    "input_bg": "#ffffff",
    "border": "#ced4da",
    "text": "#212529",
    "text_secondary": "#495057",
    "label": "#666e76",
    "focus": "#80bdff",
    "accent": "#5a8fc4",
    "title_bg": "#e3f2fd",
    "title_border": "#90caf9",
    "table_header": "#e9ecef",
    "table_alt": "#f8f9fa",
    "button_hover": "#4a7ba7",
}

_DARK_BASE = {
    **_BASE_14,
    "bg": "#1a1d23",
    "group_bg": "#24272e",
    "input_bg": "#0d1117",
    "text": "#e1e4e8",
}


# ── 60-token surface --------------------------------------------------


def test_construct_with_base_14_fills_all_derived_tokens() -> None:
    tc = ThemeColors(**_BASE_14)
    # Spot-check one token from every derived tier.
    assert tc.bg_base == _BASE_14["bg"]
    assert tc.bg_elevated == _BASE_14["table_header"]
    assert tc.border_default == _BASE_14["border"]
    assert tc.border_focus == _BASE_14["focus"]
    assert tc.text_primary == _BASE_14["text"]
    assert tc.text_link == _BASE_14["accent"]
    assert tc.primary == _BASE_14["accent"]
    assert tc.primary_hover is not None
    assert tc.success is not None
    assert tc.warning is not None
    assert tc.error is not None
    assert tc.info is not None
    assert tc.success_hover is not None
    assert tc.success_muted is not None
    assert tc.chart_blue == _BASE_14["accent"]
    assert tc.chart_green == tc.success
    assert tc.chart_orange == tc.warning
    assert tc.chart_red == tc.error
    assert tc.chart_cyan == tc.info
    assert tc.grid_line is not None
    assert tc.shadow_light is not None


def test_is_dark_inferred_from_bg() -> None:
    light = ThemeColors(**_BASE_14)
    dark = ThemeColors(**_DARK_BASE)
    assert light.is_dark is False
    assert dark.is_dark is True


def test_explicit_overrides_win_over_derivation() -> None:
    """Any field passed explicitly must survive ``model_post_init``."""
    tc = ThemeColors(**_BASE_14, success="#abcdef")
    assert tc.success == "#abcdef"


def test_as_dict_returns_all_resolved_tokens() -> None:
    tc = ThemeColors(**_BASE_14)
    d = tc.as_dict()
    # Every required base must round-trip.
    for k, v in _BASE_14.items():
        assert d[k] == v
    # And the derived ones all populated.
    for required_derived in (
        "bg_base",
        "bg_elevated",
        "border_subtle",
        "text_primary",
        "primary_hover",
        "success",
        "warning",
        "error",
        "info",
        "chart_blue",
        "grid_line",
    ):
        assert d.get(required_derived) is not None, f"{required_derived} missing"


def test_dict_style_access_compat() -> None:
    """``ThemeColors`` supports ``theme["bg"]`` for legacy dict consumers."""
    tc = ThemeColors(**_BASE_14)
    assert tc["bg"] == _BASE_14["bg"]
    assert tc.get("nonexistent", "fallback") == "fallback"
    assert "bg" in tc


def test_missing_base_token_raises() -> None:
    """DbC: the 14 base tokens are required at construction."""
    partial = {k: v for k, v in _BASE_14.items() if k != "bg"}
    with pytest.raises(Exception):  # pydantic ValidationError
        ThemeColors(**partial)


# ── _derive_full_palette compat shim ---------------------------------


def test_derive_full_palette_promotes_partial_dict() -> None:
    from src.shared.python.theme import _derive_full_palette

    partial = {"bg": "#101010", "accent": "#abcdef"}
    full = _derive_full_palette(partial, theme_name="Custom")
    # Defaults fill in for any missing base, derivation populates the rest.
    assert full["bg"] == "#101010"
    assert full["accent"] == "#abcdef"
    assert "bg_elevated" in full
    assert "primary_hover" in full
    assert "success" in full


def test_derive_full_palette_keeps_unknown_keys() -> None:
    """``extra="allow"`` on the model means custom tokens round-trip."""
    from src.shared.python.theme import _derive_full_palette

    partial = {"bg": "#ffffff", "accent": "#5a8fc4", "_brand_x": "#abc123"}
    full = _derive_full_palette(partial)
    assert full.get("_brand_x") == "#abc123"
