"""Tests for src.shared.python.ai.glossary_data_core/extended (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.ai.glossary_data_core import get_core_entries

# ---------------------------------------------------------------------------
# get_core_entries
# ---------------------------------------------------------------------------


class TestGetCoreEntries:
    def test_glossary_data_returns_list(self) -> None:
        entries = get_core_entries()
        assert isinstance(entries, list)

    def test_glossary_data_non_empty(self) -> None:
        entries = get_core_entries()
        assert len(entries) > 0

    def test_each_entry_is_dict(self) -> None:
        entries = get_core_entries()
        assert all(isinstance(e, dict) for e in entries)

    def test_entries_have_key_field(self) -> None:
        entries = get_core_entries()
        assert all("key" in e for e in entries)

    def test_entries_have_term_field(self) -> None:
        entries = get_core_entries()
        assert all("term" in e for e in entries)

    def test_entries_have_beginner_explanation(self) -> None:
        entries = get_core_entries()
        assert all("b" in e for e in entries)

    def test_all_keys_are_strings(self) -> None:
        entries = get_core_entries()
        assert all(isinstance(e["key"], str) for e in entries)

    def test_all_terms_are_strings(self) -> None:
        entries = get_core_entries()
        assert all(isinstance(e["term"], str) for e in entries)

    def test_no_duplicate_keys(self) -> None:
        entries = get_core_entries()
        keys = [e["key"] for e in entries]
        assert len(keys) == len(set(keys))
