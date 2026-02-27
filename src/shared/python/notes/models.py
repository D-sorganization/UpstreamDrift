"""Data contracts for the shared notes workspace."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecycledNoteItem:
    """Represents one safely deleted notes snapshot."""

    item_id: str
    reason: str
    path: str
    original_path: str
    deleted_at: str
