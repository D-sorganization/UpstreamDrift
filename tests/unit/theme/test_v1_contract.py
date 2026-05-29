"""Tests for the versioned theme contract (``theme/v1``).

Covers the role-token vocabulary, the ``RolePalette`` validation contract,
the canonical-theme palette builder, and the severity palette. Also proves
the pendulum GUI theme adapter speaks the canonical role tokens and preserves
its historical colour values byte-for-byte.
"""

from __future__ import annotations

import pytest

from src.shared.python.theme.colors import SEMANTIC_COLOR_KEYS, THEME_COLOR_KEYS
from src.shared.python.theme.v1 import (
    CONTRACT_VERSION,
    ROLE_TOKENS,
    SEVERITY_TOKENS,
    RolePalette,
    role_palette,
    severity_palette,
)


def test_contract_version_is_v1() -> None:
    assert CONTRACT_VERSION == "v1"


def test_role_tokens_track_canonical_colour_keys() -> None:
    # The contract vocabulary must stay in lock-step with the canonical
    # colour module so palettes cannot silently drift.
    assert ROLE_TOKENS == THEME_COLOR_KEYS + SEMANTIC_COLOR_KEYS


def test_severity_palette_has_all_severity_tokens() -> None:
    palette = severity_palette()
    assert set(palette) == set(SEVERITY_TOKENS)
    # Returns a fresh copy each call (no shared mutable state).
    palette["info"] = "#000000"
    assert severity_palette()["info"] != "#000000"


def test_role_palette_for_dark_theme_has_required_base_tokens() -> None:
    palette = role_palette("Dark")
    assert palette.name == "Dark"
    for token in THEME_COLOR_KEYS:
        assert palette.get(token) is not None
    # Values come straight from the canonical Dark theme.
    assert palette["bg"] == "#1a1d23"


def test_role_palette_rejects_unknown_theme() -> None:
    with pytest.raises(ValueError, match="unknown theme"):
        role_palette("NoSuchTheme")


def test_rolepalette_rejects_unknown_token() -> None:
    base = dict.fromkeys(THEME_COLOR_KEYS, "#101010")
    base["not_a_token"] = "#202020"
    with pytest.raises(ValueError, match="unknown role token"):
        RolePalette(name="Bad", tokens=base)


def test_rolepalette_rejects_missing_base_token() -> None:
    base = {k: "#101010" for k in THEME_COLOR_KEYS if k != "bg"}
    with pytest.raises(ValueError, match="missing required base tokens"):
        RolePalette(name="Bad", tokens=base)


def test_rolepalette_rejects_invalid_colour() -> None:
    base = dict.fromkeys(THEME_COLOR_KEYS, "#101010")
    base["accent"] = "not-a-colour"
    with pytest.raises(ValueError, match="valid hex colour"):
        RolePalette(name="Bad", tokens=base)


def test_rolepalette_normalises_and_is_frozen() -> None:
    base = dict.fromkeys(THEME_COLOR_KEYS, "#ABCDEF")
    palette = RolePalette(name="Norm", tokens=base)
    assert palette["bg"] == "#abcdef"  # normalised to lower-case
    with pytest.raises((AttributeError, TypeError)):
        palette.name = "mutated"  # type: ignore[misc]


def test_rolepalette_requires_non_empty_name() -> None:
    base = dict.fromkeys(THEME_COLOR_KEYS, "#101010")
    with pytest.raises(ValueError, match="non-empty string"):
        RolePalette(name="", tokens=base)


# ---------------------------------------------------------------------------
# Pendulum GUI theme adapter
# ---------------------------------------------------------------------------


def test_pendulum_theme_palette_conforms_to_v1_contract() -> None:
    from src.shared.python.pendulum_simulator.gui import theme as pend

    assert isinstance(pend.PALETTE, RolePalette)
    assert pend.PALETTE.name == "Pendulum Dark"
    # Every required base role token resolves.
    for token in THEME_COLOR_KEYS:
        assert pend.PALETTE.get(token) is not None


def test_pendulum_constants_delegate_to_canonical_palette() -> None:
    """Public constants must return the canonical role-token values."""
    from src.shared.python.pendulum_simulator.gui import theme as pend

    assert pend.PALETTE["bg"] == pend.BG_DARK
    assert pend.PALETTE["group_bg"] == pend.BG_MEDIUM
    assert pend.PALETTE["input_bg"] == pend.BG_DARKEST
    assert pend.PALETTE["text"] == pend.TEXT_PRIMARY
    assert pend.PALETTE["text_secondary"] == pend.TEXT_SECONDARY
    assert pend.PALETTE["label"] == pend.TEXT_MUTED
    assert pend.PALETTE["accent"] == pend.ACCENT_BLUE
    assert pend.PALETTE["border"] == pend.BORDER_DEFAULT
    assert pend.PALETTE["button_hover"] == pend.BG_HOVER


def test_pendulum_severity_colors_come_from_contract() -> None:
    from src.shared.python.pendulum_simulator.gui import theme as pend

    assert severity_palette() == pend.SEVERITY_COLORS


def test_pendulum_palette_preserves_historical_values() -> None:
    """Visual output must be unchanged from before the unification."""
    from src.shared.python.pendulum_simulator.gui import theme as pend

    assert pend.BG_DARKEST == "#0e0e1e"
    assert pend.BG_DARK == "#12121c"
    assert pend.BG_MEDIUM == "#1a1a2e"
    assert pend.BG_HOVER == "#303060"
    assert pend.TEXT_PRIMARY == "#c0c0e0"
    assert pend.TEXT_SECONDARY == "#9090c8"
    assert pend.TEXT_MUTED == "#808090"
    assert pend.ACCENT_BLUE == "#4888c8"
    assert pend.ACCENT_GREEN == "#50a060"
    assert pend.ACCENT_RED == "#d06060"
    assert pend.ACCENT_AMBER == "#e0a060"
    assert pend.ACCENT_PURPLE == "#8060c0"
    assert pend.BORDER_DEFAULT == "#2a2a4a"
    assert pend.SEVERITY_COLORS == {
        "info": "#6080c0",
        "warning": "#c0a040",
        "error": "#d06060",
        "critical": "#e03030",
    }
