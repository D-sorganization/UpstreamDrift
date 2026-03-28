"""Storage and recycle-bin semantics for project notes."""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

from .models import RecycledNoteItem

UTC = getattr(dt, "UTC", dt.timezone.utc)  # noqa: UP017


class NotesStorage:
    """Persist project notes and provide reversible deletion."""

    def __init__(
        self,
        project_dir: Path | str,
        notes_filename: str = "project.notes.txt",
        recycle_bin_dirname: str = ".notes_recycle_bin",
    ) -> None:
        self.project_dir = Path(project_dir)
        if not self.project_dir.exists() or not self.project_dir.is_dir():
            raise ValueError("project_dir must exist and be a directory")

        if not notes_filename.strip():
            raise ValueError("notes_filename cannot be empty")

        self.notes_path = self.project_dir / notes_filename
        self.recycle_bin_dir = self.project_dir / recycle_bin_dirname
        self.recycle_index_path = self.recycle_bin_dir / "index.json"

    def load_text(self) -> str:
        """Return current notes content or empty string when absent."""
        if not self.notes_path.exists():
            return ""
        return self.notes_path.read_text(encoding="utf-8")

    def save_text(self, text: str) -> Path:
        """Persist notes text to the project notes file."""
        if text is None:
            raise ValueError("text cannot be None")

        self.notes_path.parent.mkdir(parents=True, exist_ok=True)
        self.notes_path.write_text(text, encoding="utf-8")
        return self.notes_path

    def clear(self) -> Path:
        """Clear notes content while preserving the notes file."""
        return self.save_text("")

    def move_to_recycle(self, reason: str = "manual_delete") -> RecycledNoteItem:
        """Move notes file to recycle bin and track it in an index."""
        if not self.notes_path.exists():
            raise FileNotFoundError("notes file does not exist")

        timestamp = dt.datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        item_id = f"{timestamp}_{self.notes_path.stem}"

        self.recycle_bin_dir.mkdir(parents=True, exist_ok=True)
        recycle_path = self.recycle_bin_dir / f"{item_id}.txt"
        shutil.move(str(self.notes_path), str(recycle_path))

        item = RecycledNoteItem(
            item_id=item_id,
            reason=reason,
            path=str(recycle_path),
            original_path=str(self.notes_path),
            deleted_at=timestamp,
        )
        self._append_index(item)
        return item

    def restore(self, item_id: str) -> Path | None:
        """Restore a recycled note item back to the project notes file."""
        if not (item_id is not None):
            raise ValueError("item_id must be provided")
        item = self._find_item(item_id)
        if item is None:
            return None

        source = Path(item.path)
        if not source.exists():
            return None

        self.notes_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(self.notes_path))
        self._remove_index(item_id)
        return self.notes_path

    def purge(self, item_id: str) -> bool:
        """Permanently delete one recycled item by ID."""
        if not (item_id is not None):
            raise ValueError("item_id must be provided")
        item = self._find_item(item_id)
        if item is None:
            return False

        path = Path(item.path)
        if path.exists():
            path.unlink()
        self._remove_index(item_id)
        return True

    def list_recycled(self) -> list[RecycledNoteItem]:
        """Return recycled items, newest first."""
        items = self._read_index()
        items.sort(key=lambda item: item.deleted_at, reverse=True)
        return items

    def latest_recycled_id(self) -> str | None:
        """Return the most recent recycled item id if available."""
        items = self.list_recycled()
        if not items:
            return None
        return items[0].item_id

    def _read_index(self) -> list[RecycledNoteItem]:
        if not self.recycle_index_path.exists():
            return []

        data = json.loads(self.recycle_index_path.read_text(encoding="utf-8"))
        return [RecycledNoteItem(**item) for item in data]

    def _write_index(self, items: list[RecycledNoteItem]) -> None:
        if not (items is not None):
            raise ValueError("items must be provided")
        self.recycle_bin_dir.mkdir(parents=True, exist_ok=True)
        payload = [item.__dict__ for item in items]
        self.recycle_index_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def _append_index(self, item: RecycledNoteItem) -> None:
        if not (item is not None):
            raise ValueError("item must be provided")
        items = self._read_index()
        items.append(item)
        self._write_index(items)

    def _find_item(self, item_id: str) -> RecycledNoteItem | None:
        if not item_id.strip():
            raise ValueError("item_id cannot be empty")

        for item in self._read_index():
            if item.item_id == item_id:
                return item
        return None

    def _remove_index(self, item_id: str) -> None:
        items = [item for item in self._read_index() if item.item_id != item_id]
        self._write_index(items)
