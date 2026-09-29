"""Atomic replacement for small session sidecars (#9860, #9861)."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def write_document(target: Path, payload: dict[str, Any]) -> None:
    """Replace a JSON sidecar only after serialization and writing succeed."""
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.stem}-",
            suffix=".json",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
