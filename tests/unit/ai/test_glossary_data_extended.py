"""Tests for src.shared.python.ai.glossary_data_extended (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.ai.glossary_data_extended import get_extended_entries


class TestGetExtendedEntries:
    def test_glossary_data_extended_returns_list(self) -> None:
        entries = get_extended_entries()
        assert isinstance(entries, list)

    def test_glossary_data_extended_nonempty(self) -> None:
        entries = get_extended_entries()
        assert len(entries) > 0

    def test_entries_are_dicts(self) -> None:
        entries = get_extended_entries()
        for entry in entries:
            assert isinstance(entry, dict)

    def test_entries_have_key(self) -> None:
        entries = get_extended_entries()
        for entry in entries:
            assert "key" in entry, f"Entry missing 'key' field: {entry}"

    def test_entries_have_term(self) -> None:
        entries = get_extended_entries()
        for entry in entries:
            assert "term" in entry, f"Entry missing 'term' field: {entry}"

    def test_entries_have_cat(self) -> None:
        entries = get_extended_entries()
        for entry in entries:
            assert "cat" in entry, f"Entry missing 'cat' field: {entry}"

    def test_entries_have_beginner_desc(self) -> None:
        entries = get_extended_entries()
        for entry in entries:
            assert "b" in entry, f"Entry missing beginner description 'b': {entry}"

    def test_keys_are_strings(self) -> None:
        entries = get_extended_entries()
        for entry in entries:
            assert isinstance(entry["key"], str)

    def test_terms_are_strings(self) -> None:
        entries = get_extended_entries()
        for entry in entries:
            assert isinstance(entry["term"], str)

    def test_many_entries(self) -> None:
        entries = get_extended_entries()
        # The docstring mentions ~310 entries
        assert len(entries) >= 50

    def test_multiple_categories_present(self) -> None:
        entries = get_extended_entries()
        cats = {entry["cat"] for entry in entries}
        assert len(cats) > 1  # Multiple categories

    def test_keys_are_unique(self) -> None:
        entries = get_extended_entries()
        keys = [entry["key"] for entry in entries]
        assert len(keys) == len(set(keys)), "Duplicate keys found in glossary"
