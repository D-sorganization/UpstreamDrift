"""Capture index with portable notes and reversible file operations (#9861).

SQLite indexes locations only. The session sidecar owns titles, notes and
archive state, so rebuilding the catalog does not lose the instructor's work.
Storage scans run in the library UI's worker, never on its event loop.
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.motion_capture.rig.bundle import (
    MANIFEST_FILE,
    PLAN_FILE,
    RECORDINGS_FILE,
    load_bundle,
)
from src.motion_capture.rig.documents import write_document
from src.motion_capture.rig.edits import EDITS_FILE, has_analysis

NOTES_FILE = "capture_notes.json"
MAX_NOTES_BYTES = 200_000
_UNSAFE_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", re.IGNORECASE)


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


@dataclass(frozen=True)
class LibraryEntry:
    root: Path
    capture_id: str = ""
    title: str = ""
    notes: str = ""
    archived: bool = False
    managed_bytes: int = 0
    external_bytes: int = 0
    problem: str = ""


def _storage(root: Path) -> tuple[int, int]:
    """Logical bytes, excluding symlink targets and counting external media once."""
    managed = 0

    def failed(error: OSError) -> None:
        raise error

    for directory, dirs, files in os.walk(root, followlinks=False, onerror=failed):
        dirs[:] = [d for d in dirs if not (Path(directory) / d).is_symlink()]
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink():
                managed += path.stat().st_size
    sources = {(root / e.file).resolve() for e in load_bundle(root)[1].recordings}
    external = sum(
        path.stat().st_size for path in sources if not path.is_relative_to(root)
    )
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise ValueError(f"Missing recording: {missing[0]}")
    return managed, external


class CaptureLibrary:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._database = self.root / "capture_catalog.sqlite3"
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS captures (path TEXT PRIMARY KEY)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._database, timeout=10)

    def _known(self, root: Path) -> Path:
        root = root.resolve()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT path FROM captures WHERE path = ?", (str(root),)
            ).fetchone()
        if row is None:
            raise ValueError("Register this capture in the library first")
        return root

    def register(self, root: Path) -> CaptureNotes:
        root = root.resolve()
        load_bundle(root)
        if not (root / NOTES_FILE).exists():
            write_document(
                root / NOTES_FILE, CaptureNotes(title=root.name).model_dump(mode="json")
            )
        notes = read_notes(root)
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO captures(path) VALUES (?)", (str(root),)
            )
        return notes

    def update(self, root: Path, **changes: Any) -> CaptureNotes:
        root = self._known(root)
        if set(changes) - {"title", "notes", "archived"}:
            raise ValueError("Only title, notes and archive state are editable")
        payload = {**read_notes(root).model_dump(mode="json"), **changes}
        updated = CaptureNotes.model_validate(payload)
        write_document(root / NOTES_FILE, updated.model_dump(mode="json"))
        return updated

    def list(self, *, query: str = "", archived: bool = False) -> list[LibraryEntry]:
        with self._connect() as connection:
            paths = [
                Path(row[0])
                for row in connection.execute("SELECT path FROM captures ORDER BY path")
            ]
        entries = []
        for root in paths:
            try:
                notes = read_notes(root)
                if (
                    notes.archived != archived
                    or query.casefold()
                    not in f"{notes.title}\n{notes.notes}".casefold()
                ):
                    continue
                managed, external = _storage(root)
                entries.append(
                    LibraryEntry(
                        root,
                        notes.capture_id,
                        notes.title,
                        notes.notes,
                        notes.archived,
                        managed,
                        external,
                    )
                )
            except (ValueError, OSError) as exc:
                entries.append(LibraryEntry(root, title=root.name, problem=str(exc)))
        return entries

    def editable_copy(self, root: Path) -> Path:
        """New lineage; media referenced, calibration/timing retained, results omitted."""
        root = self._known(root)
        plan, index, manifest = load_bundle(root)
        notes = read_notes(root)
        destination = self.root / "captures" / str(uuid4())
        destination.mkdir(parents=True, exist_ok=False)
        try:
            entries = tuple(
                e.model_copy(update={"file": str((root / e.file).resolve())})
                for e in index.recordings
            )
            documents = {
                PLAN_FILE: plan.model_dump(mode="json"),
                RECORDINGS_FILE: index.model_copy(
                    update={"recordings": entries}
                ).model_dump(mode="json"),
                MANIFEST_FILE: manifest.model_dump(mode="json"),
                NOTES_FILE: CaptureNotes(
                    title=f"{notes.title[:185]} · edit copy",
                    notes=notes.notes,
                    source_capture=str(root),
                ).model_dump(mode="json"),
            }
            for filename, payload in documents.items():
                write_document(destination / filename, payload)
            for filename in (EDITS_FILE, "intrinsics.json"):
                if (root / filename).is_file():
                    shutil.copyfile(root / filename, destination / filename)
            self.register(destination)
        except (ValueError, OSError, sqlite3.Error):
            # Only this operation's freshly-created child is removed on failure.
            if destination.resolve().is_relative_to((self.root / "captures").resolve()):
                shutil.rmtree(destination)
            raise
        return destination

    def rename_recording(self, root: Path, view: str, filename: str) -> None:
        """Rename owned, unprocessed media and repair its index; rollback on failure."""
        root = self._known(root)
        if (
            not filename
            or filename in (".", "..")
            or len(filename) > 200
            or filename.endswith((" ", "."))
            or _UNSAFE_NAME.search(filename)
            or _RESERVED.match(filename)
        ):
            raise ValueError("Choose a plain filename without reserved characters")
        if has_analysis(root) or (root / "proxies.json").exists():
            raise ValueError("Create an editable copy before renaming processed media")
        index = load_bundle(root)[1]
        entry = next((e for e in index.recordings if e.view == view), None)
        if entry is None:
            raise ValueError(f"Unknown view: {view}")
        source = (root / entry.file).resolve()
        if not source.is_relative_to(root):
            raise ValueError("Cannot rename an external source recording")
        target = source.with_name(filename)
        if target.suffix.lower() != source.suffix.lower():
            raise ValueError("Keep the recording's existing file extension")
        if target.exists():
            raise ValueError("A file with that name already exists")
        new_file = target.relative_to(root).as_posix()
        entries = tuple(
            e.model_copy(update={"file": new_file}) if e.file == entry.file else e
            for e in index.recordings
        )
        source.rename(target)
        try:
            write_document(
                root / RECORDINGS_FILE,
                index.model_copy(update={"recordings": entries}).model_dump(
                    mode="json"
                ),
            )
        except (ValueError, OSError):
            target.rename(source)
            raise
