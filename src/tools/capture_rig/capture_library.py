"""Capture index with portable notes and reversible file operations (#9861).

SQLite indexes locations only. The session sidecar owns titles, notes and
archive state, so rebuilding the catalog does not lose the instructor's work.
Storage scans run in the library UI's worker, never on its event loop.
"""

from __future__ import annotations

from src.motion_capture.coaching.storage import copy_layers

import os
import builtins
import re
import shutil
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from src.motion_capture.rig.capture_notes import (
    CaptureNotes as CaptureNotes,
    NOTES_FILE as NOTES_FILE,
    MAX_NOTES_BYTES as MAX_NOTES_BYTES,
    read_notes as read_notes,
)

from src.motion_capture.rig.bundle import (
    MANIFEST_FILE,
    PLAN_FILE,
    RECORDINGS_FILE,
    load_bundle,
)
from src.motion_capture.rig.documents import write_document
from src.motion_capture.rig.edits import EDITS_FILE, has_analysis

RENAME_FILE = ".recording-rename.json"
_UNSAFE_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", re.IGNORECASE)


class RenameIntent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["recording-rename/1.0.0"] = "recording-rename/1.0.0"
    view: str
    source: str
    target: str


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


def _storage(root: Path, cancelled: Callable[[], bool]) -> tuple[int, int]:
    """Logical bytes, excluding symlink targets and counting external media once."""
    managed = 0

    def failed(error: OSError) -> None:
        raise error

    for directory, dirs, files in os.walk(root, followlinks=False, onerror=failed):
        if cancelled():
            raise InterruptedError("Library scan cancelled")
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

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._database, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

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
        self.recover_rename(root)
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

    def catalog_entries(
        self,
        *,
        query: str = "",
        archived: bool | None = False,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> builtins.list[LibraryEntry]:
        """Read names, notes and identities without scanning footage or storage.

        Selectors can include archived captures with ``archived=None``. Media may
        be offline while portable calibration evidence remains usable.
        """
        with self._connect() as connection:
            paths = [
                Path(row[0])
                for row in connection.execute("SELECT path FROM captures ORDER BY path")
            ]
        entries = []
        for root in paths:
            if cancelled():
                raise InterruptedError("Library scan cancelled")
            try:
                notes = read_notes(root)
                if (
                    archived is not None and notes.archived != archived
                ) or query.casefold() not in f"{notes.title}\n{notes.notes}".casefold():
                    continue
                entries.append(
                    LibraryEntry(
                        root,
                        notes.capture_id,
                        notes.title,
                        notes.notes,
                        notes.archived,
                    )
                )
            except InterruptedError:
                raise
            except (ValueError, OSError) as exc:
                entries.append(LibraryEntry(root, title=root.name, problem=str(exc)))
        return entries

    def list(
        self,
        *,
        query: str = "",
        archived: bool = False,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> builtins.list[LibraryEntry]:
        """Add storage/recording diagnostics for the existing library manager."""
        entries = []
        for entry in self.catalog_entries(
            query=query, archived=archived, cancelled=cancelled
        ):
            if entry.problem:
                entries.append(entry)
                continue
            try:
                managed, external = _storage(entry.root, cancelled)
                entries.append(
                    replace(entry, managed_bytes=managed, external_bytes=external)
                )
            except InterruptedError:
                raise
            except (ValueError, OSError) as exc:
                entries.append(replace(entry, problem=str(exc)))
        return entries

    def editable_copy(self, root: Path) -> Path:
        """New lineage; media referenced, calibration/timing retained, results omitted."""
        root = self._known(root)
        self.recover_rename(root)
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
            copy_layers(root, destination, tuple(entry.view for entry in entries))
            # Equipment depends on portable notes; rebind after the new ID exists.
            from .equipment import load_capture_club, save_capture_club

            equipment = load_capture_club(root)
            if equipment is not None:
                save_capture_club(destination, equipment.club)
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
        self.recover_rename(root)
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
        intent = RenameIntent(view=view, source=entry.file, target=new_file)
        write_document(root / RENAME_FILE, intent.model_dump(mode="json"))
        try:
            source.rename(target)
            write_document(
                root / RECORDINGS_FILE,
                index.model_copy(update={"recordings": entries}).model_dump(
                    mode="json"
                ),
            )
        except (ValueError, OSError):
            self.recover_rename(root)
            raise
        (root / RENAME_FILE).unlink()

    def recover_rename(self, root: Path) -> None:
        """Finish or roll back an interrupted rename, never overwrite a file."""
        root = root.resolve()
        journal = root / RENAME_FILE
        if not journal.exists():
            return
        if journal.stat().st_size > MAX_NOTES_BYTES:
            raise ValueError("Invalid rename recovery record")
        intent = RenameIntent.model_validate_json(journal.read_text(encoding="utf-8"))
        source, target = (
            (root / intent.source).resolve(),
            (root / intent.target).resolve(),
        )
        if (
            not source.is_relative_to(root)
            or not target.is_relative_to(root)
            or source.parent != target.parent
            or source == target
        ):
            raise ValueError(
                "Rename recovery paths must be different files inside this capture"
            )
        entry = next(
            (e for e in load_bundle(root)[1].recordings if e.view == intent.view), None
        )
        current = (root / entry.file).resolve() if entry else None
        if current not in (source, target) or source.exists() == target.exists():
            raise ValueError("Rename recovery is ambiguous; inspect the capture folder")
        if current == source and target.exists():
            target.rename(source)
        elif current == target and not target.exists():
            raise ValueError("Renamed recording is missing; inspect the capture folder")
        journal.unlink()
