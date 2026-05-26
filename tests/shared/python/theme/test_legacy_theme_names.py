"""TDD coverage for legacy theme-name migration.

When a user upgrades, the QSettings file may still name a theme that
has since been renamed (e.g. ``Office Blue`` → ``Corporate Blue``) or
removed (e.g. ``MS Word`` → ``Light``). ``resolve_legacy_theme_name``
silently maps these to their current canonical name so existing users
don't suddenly land on the default theme on next launch.
"""

from __future__ import annotations

import pytest

from src.shared.python.theme.colors import (
    BUILTIN_THEMES,
    LEGACY_THEME_NAMES,
    resolve_legacy_theme_name,
)


@pytest.mark.parametrize(
    "legacy,canonical",
    [
        ("Office Blue", "Corporate Blue"),
        ("Office Green", "Sage Professional"),
        ("MS Word", "Light"),
        ("MS Excel", "Sage Professional"),
        ("MS PowerPoint", "Corporate Blue"),
        ("Word", "Light"),
        ("Excel", "Sage Professional"),
        ("PowerPoint", "Corporate Blue"),
    ],
)
def test_legacy_aliases_resolve_to_canonical(legacy: str, canonical: str) -> None:
    assert resolve_legacy_theme_name(legacy) == canonical


def test_unknown_name_passes_through() -> None:
    """Names without a mapping must be returned unchanged.

    Otherwise user-defined custom themes (which never appear in the
    map) would resolve to None and the user would lose their theme.
    """
    assert resolve_legacy_theme_name("My Custom Theme") == "My Custom Theme"


def test_none_returns_none() -> None:
    assert resolve_legacy_theme_name(None) is None


def test_every_canonical_target_actually_exists() -> None:
    """Every value in ``LEGACY_THEME_NAMES`` must be a real built-in theme.

    A typo in the map would silently route users to a non-existent
    theme; the theme manager would then fall back to ``Light`` and
    drop the lookup chain we built specifically to preserve their
    preference. This contract test catches that class of regression.
    """
    missing = [v for v in LEGACY_THEME_NAMES.values() if v not in BUILTIN_THEMES]
    assert not missing, f"Legacy aliases pointing at non-existent themes: {missing}"


def test_office_renames_are_present() -> None:
    """Explicit smoke test for today's rename batch."""
    assert "Office Blue" in LEGACY_THEME_NAMES
    assert "Office Green" in LEGACY_THEME_NAMES
    assert LEGACY_THEME_NAMES["Office Blue"] == "Corporate Blue"
    assert LEGACY_THEME_NAMES["Office Green"] == "Sage Professional"
