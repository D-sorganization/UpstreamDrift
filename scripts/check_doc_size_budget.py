#!/usr/bin/env python3
"""Enforce a size budget for committed Markdown and Quarto documentation."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "scripts" / "config" / "doc_size_budget.json"
MAX_BYTES = 50 * 1024
DOC_EXTENSIONS = {".md", ".qmd"}
EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    "dist",
    "build",
    "vendor",
}


@dataclass(frozen=True)
class BudgetException:
    """Owned temporary exception for an oversized documentation file."""

    path: str
    owner: str
    expires_on: date

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> BudgetException:
        """Create an exception after validating the required contract fields."""
        path = str(payload.get("path", "")).replace("\\", "/")
        owner = str(payload.get("owner", ""))
        expires_on_text = str(payload.get("expires_on", ""))
        if not path:
            raise ValueError("doc size budget exception is missing path")
        if not owner.startswith("@"):
            raise ValueError(f"{path}: owner must be a GitHub handle or team")
        try:
            expires_on = date.fromisoformat(expires_on_text)
        except ValueError as exc:
            raise ValueError(f"{path}: expires_on must be YYYY-MM-DD") from exc
        return cls(path=path, owner=owner, expires_on=expires_on)

    def is_active(self, today: date) -> bool:
        """Return whether the exception may still be used."""
        return self.expires_on >= today


def _relative_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _is_document(path: Path) -> bool:
    return path.suffix.lower() in DOC_EXTENSIONS


def _is_excluded(path: Path) -> bool:
    return any(
        part in EXCLUDED_PARTS or (part.startswith(".") and part != ".github")
        for part in path.parts
    )


def _iter_documents() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and _is_document(path) and not _is_excluded(path)
    )


def _load_config() -> tuple[int, dict[str, BudgetException]]:
    if not CONFIG_PATH.exists():
        return MAX_BYTES, {}
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    max_bytes = int(data.get("max_bytes", MAX_BYTES))
    exceptions = [
        BudgetException.from_json(item) for item in data.get("exceptions", [])
    ]
    return max_bytes, {item.path: item for item in exceptions}


def main() -> int:
    try:
        max_bytes, exceptions = _load_config()
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        sys.stderr.write(f"invalid doc size budget config: {exc}\n")
        return 1

    today = date.today()
    offenders: list[str] = []
    for document in _iter_documents():
        rel_path = _relative_path(document)
        byte_count = document.stat().st_size
        if byte_count <= max_bytes:
            continue
        exception = exceptions.get(rel_path)
        if exception is not None and exception.is_active(today):
            continue
        offenders.append(f"{rel_path}: {byte_count} bytes")

    if offenders:
        sys.stderr.write(
            f"doc size budget failed (max {max_bytes} bytes):\n- "
            + "\n- ".join(offenders)
            + "\n"
        )
        return 1

    sys.stdout.write(f"doc size budget passed (max {max_bytes} bytes)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
