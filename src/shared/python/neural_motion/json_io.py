"""Shared sorted-JSON file writes for neural_motion receipts."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

__all__ = ["SortedJsonWritableMixin", "write_sorted_json"]


class SortedJsonWritableMixin(ABC):
    """Persist ``as_dict()`` payloads as sorted JSON files."""

    @abstractmethod
    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping for this record."""

    def write_json(self, path: str | Path) -> None:
        write_sorted_json(path, self.as_dict())


def write_sorted_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` as indented, key-sorted JSON with a trailing newline.

    Creates parent directories. Fail-closed on non-mapping payloads via
    ``dict(payload)`` TypeError rather than silent coercion.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
