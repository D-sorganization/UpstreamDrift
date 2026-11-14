"""Canonical receipt byte identity for ground-support reproducibility checks (#14771).

Receipts are JSON. Two runs are "the same" for ratchet purposes when their parsed
content matches under a stable canonical serialization (sorted keys, compact
UTF-8). ``elapsed_s`` and similar volatile fields should be stripped or compared
with tolerance by callers before calling :func:`canonical_receipt_bytes`; this
module only guarantees stable *byte* identity for identical parsed objects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_receipt_json(path: Path) -> Any:
    """Load a receipt JSON file (UTF-8)."""
    if path is None:
        raise ValueError("path must be provided")
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def canonical_receipt_bytes(data: Any) -> bytes:
    """Return canonical UTF-8 bytes for *data* (sorted keys, compact separators)."""
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def receipts_canonical_bytes_equal(path_a: Path, path_b: Path) -> bool:
    """Return True if *path_a* and *path_b* decode to JSON-equal documents."""
    a = load_receipt_json(path_a)
    b = load_receipt_json(path_b)
    return canonical_receipt_bytes(a) == canonical_receipt_bytes(b)
