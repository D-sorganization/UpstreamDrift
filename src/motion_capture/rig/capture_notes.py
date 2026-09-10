"""Portable capture identity and notes, shared by UI and headless consumers."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

NOTES_FILE = "capture_notes.json"
MAX_NOTES_BYTES = 200_000


class CaptureNotes(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["capture-notes/1.0.0"] = "capture-notes/1.0.0"
    capture_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    title: str = Field(min_length=1, max_length=200)
    notes: str = Field(default="", max_length=50_000)
    archived: bool = Field(default=False, strict=True)
    created_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    source_capture: str | None = None

    @field_validator("title")
    @classmethod
    def clean_title(cls, value: str) -> str:
        if not value.strip() or any(ord(c) < 32 for c in value):
            raise ValueError("Capture title must contain visible text on one line")
        return value.strip()


def read_notes(root: Path) -> CaptureNotes:
    path = root / NOTES_FILE
    if path.stat().st_size > MAX_NOTES_BYTES:
        raise ValueError("Capture notes file is too large")
    return CaptureNotes.model_validate_json(path.read_text(encoding="utf-8"))
