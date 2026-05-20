"""Tests for src.shared.python.notes.models."""

from __future__ import annotations

import dataclasses

import pytest

from src.shared.python.notes.models import RecycledNoteItem


@pytest.mark.unit
def test_recycled_note_item_constructs():
    item = RecycledNoteItem(
        item_id="abc",
        reason="manual_delete",
        path="/recycle/abc.json",
        original_path="/notes/abc.md",
        deleted_at="2024-01-01T00:00:00Z",
    )
    assert item.item_id == "abc"
    assert item.reason == "manual_delete"
    assert item.path == "/recycle/abc.json"
    assert item.original_path == "/notes/abc.md"
    assert item.deleted_at == "2024-01-01T00:00:00Z"


@pytest.mark.unit
def test_recycled_note_item_is_frozen():
    item = RecycledNoteItem("a", "r", "p", "o", "d")
    with pytest.raises(dataclasses.FrozenInstanceError):
        item.item_id = "other"  # type: ignore[misc]


@pytest.mark.unit
def test_recycled_note_item_equality():
    a = RecycledNoteItem("a", "r", "p", "o", "d")
    b = RecycledNoteItem("a", "r", "p", "o", "d")
    c = RecycledNoteItem("a", "r", "p", "o", "DIFFERENT")
    assert a == b
    assert a != c


@pytest.mark.unit
def test_recycled_note_item_hashable():
    a = RecycledNoteItem("a", "r", "p", "o", "d")
    b = RecycledNoteItem("a", "r", "p", "o", "d")
    assert hash(a) == hash(b)
    assert {a, b} == {a}


@pytest.mark.unit
def test_recycled_note_item_fields():
    field_names = {f.name for f in dataclasses.fields(RecycledNoteItem)}
    assert field_names == {
        "item_id",
        "reason",
        "path",
        "original_path",
        "deleted_at",
    }
