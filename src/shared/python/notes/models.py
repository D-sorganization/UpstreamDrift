"""Data contracts for the shared notes workspace."""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_NOTE_COLOR = "#fff7cc"
DEFAULT_BOARD_BACKGROUND_COLOR = "#f7f7f7"
_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
_NOTE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


def normalize_color(value: str, *, field_name: str = "color") -> str:
    """Return a canonical hex color after validating the UI-safe format."""
    if not isinstance(value, str) or not _HEX_COLOR_RE.fullmatch(value.strip()):
        raise ValueError(f"{field_name} must be a #RRGGBB color")
    return value.strip().lower()


@dataclass(frozen=True)
class NoteCard:
    """Markdown-backed visual note card metadata and body."""

    note_id: str
    title: str
    markdown_body: str
    color: str = DEFAULT_NOTE_COLOR
    created_at: str = ""
    updated_at: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.note_id, str) or not _NOTE_ID_RE.fullmatch(self.note_id):
            raise ValueError("note_id must be stable and path-safe")
        if self.markdown_body is None:
            raise ValueError("markdown_body cannot be None")
        object.__setattr__(self, "title", str(self.title))
        object.__setattr__(self, "markdown_body", str(self.markdown_body))
        object.__setattr__(self, "color", normalize_color(self.color))
        object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags))


@dataclass(frozen=True)
class NotesBoardSettings:
    """Visual settings for the notes board/screen."""

    background_color: str = DEFAULT_BOARD_BACKGROUND_COLOR

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "background_color",
            normalize_color(self.background_color, field_name="background_color"),
        )


@dataclass(frozen=True)
class RecycledNoteItem:
    """Represents one safely deleted notes snapshot."""

    item_id: str
    reason: str
    path: str
    original_path: str
    deleted_at: str
