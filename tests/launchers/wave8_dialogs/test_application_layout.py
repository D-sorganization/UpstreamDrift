"""Tests for src.launchers.application_layout — pure data-only module."""

from __future__ import annotations

import pytest

from src.launchers import application_layout as al


class TestDefaultSidebarTabIds:
    def test_returns_list_copy(self) -> None:
        tabs1 = al.default_sidebar_tab_ids()
        tabs2 = al.default_sidebar_tab_ids()
        assert tabs1 == tabs2
        # Mutating result does not affect subsequent calls.
        tabs1.append("garbage")
        assert "garbage" not in al.default_sidebar_tab_ids()

    def test_returns_list_type(self) -> None:
        tabs = al.default_sidebar_tab_ids()
        assert isinstance(tabs, list)

    def test_matches_constant_order(self) -> None:
        assert al.default_sidebar_tab_ids() == list(al.DEFAULT_SIDEBAR_TAB_IDS)

    def test_contains_expected_features(self) -> None:
        tabs = al.default_sidebar_tab_ids()
        for expected in ("sidekick", "os_terminal", "python_repl", "jupyter"):
            assert expected in tabs

    def test_does_not_contain_menu_only_features(self) -> None:
        tabs = set(al.default_sidebar_tab_ids())
        assert tabs.isdisjoint(al.MENU_ONLY_FEATURES)


class TestIsMenuOnly:
    def test_known_menu_only(self) -> None:
        assert al.is_menu_only("mcp_servers") is True

    @pytest.mark.parametrize("fid", ["sidekick", "jupyter", "workspace", "unknown_x"])
    def test_non_menu_only(self, fid: str) -> None:
        assert al.is_menu_only(fid) is False

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            al.is_menu_only("")


class TestConstants:
    def test_default_tabs_is_tuple(self) -> None:
        assert isinstance(al.DEFAULT_SIDEBAR_TAB_IDS, tuple)
        assert all(isinstance(t, str) for t in al.DEFAULT_SIDEBAR_TAB_IDS)

    def test_menu_only_is_frozenset(self) -> None:
        assert isinstance(al.MENU_ONLY_FEATURES, frozenset)

    def test_no_duplicates_in_defaults(self) -> None:
        assert len(al.DEFAULT_SIDEBAR_TAB_IDS) == len(set(al.DEFAULT_SIDEBAR_TAB_IDS))

    def test_all_exported(self) -> None:
        for name in al.__all__:
            assert hasattr(al, name)
