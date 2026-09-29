"""Tests for :mod:`src.launchers.application_layout`."""

from __future__ import annotations

import pytest

from src.launchers.application_layout import (
    DEFAULT_SIDEBAR_TAB_IDS,
    MENU_ONLY_FEATURES,
    default_sidebar_tab_ids,
    is_menu_only,
)


def test_default_sidebar_tab_ids_returns_list() -> None:
    ids = default_sidebar_tab_ids()
    assert isinstance(ids, list)
    assert ids == list(DEFAULT_SIDEBAR_TAB_IDS)


def test_default_sidebar_tab_ids_is_fresh_copy() -> None:
    """Mutating the returned list must not affect subsequent calls."""
    ids = default_sidebar_tab_ids()
    ids.clear()
    assert default_sidebar_tab_ids() == list(DEFAULT_SIDEBAR_TAB_IDS)


def test_sidekick_features_present_in_default_layout() -> None:
    """Every Tools feature with a sidebar tab is listed in the default layout."""
    required = {"sidekick", "os_terminal", "python_repl", "workspace", "jupyter"}
    assert required <= set(DEFAULT_SIDEBAR_TAB_IDS)


def test_menu_only_features_not_in_sidebar() -> None:
    """Menu-only features must not appear in the default sidebar order."""
    for feature_id in MENU_ONLY_FEATURES:
        assert feature_id not in DEFAULT_SIDEBAR_TAB_IDS


def test_is_menu_only_known() -> None:
    assert is_menu_only("mcp_servers") is True


def test_is_menu_only_unknown_returns_false() -> None:
    assert is_menu_only("os_terminal") is False


def test_is_menu_only_empty_raises() -> None:
    with pytest.raises(ValueError):
        is_menu_only("")
