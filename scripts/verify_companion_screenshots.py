"""Verify screenshot manifest SHA-256 digests and pixel dimensions (#9191).

This module validates that companion screenshot metadata accurately reflects
repository image assets. A pending record must carry null asset fields and an
explicit reason. A captured record must resolve within the repository, match
its declared SHA-256 digest, match declared pixel dimensions (parsed directly
from the PNG IHDR header using the standard library), and carry non-empty alt text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


_PENDING_NULL_FIELDS = ("path", "sha256", "width", "height")


def _non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _pending_violations(record_id: str, record: Mapping[str, Any]) -> list[str]:
    """A pending record carries no asset fields and an explicit reason."""
    violations = [
        f"record {record_id!r}: pending screenshot {field} must be null, "
        f"got {record.get(field)!r}"
        for field in _PENDING_NULL_FIELDS
        if record.get(field) is not None
    ]
    if not _non_empty_str(record.get("reason")):
        violations.append(
            f"record {record_id!r}: pending screenshot reason must be a non-empty string"
        )
    return violations


def _resolve_asset(path_val: Any, repo_root: Path) -> tuple[Path | None, str]:
    """Resolve a record path to an existing file inside ``repo_root``, else a reason."""
    if not _non_empty_str(path_val):
        return None, "captured screenshot path must be a non-empty string"
    if (
        path_val.startswith(("/", "\\"))
        or re.match(r"^[A-Za-z]:", path_val)
        or Path(path_val).is_absolute()
    ):
        return None, f"captured screenshot path must not be absolute: {path_val}"
    resolved_repo = repo_root.resolve()
    target = (repo_root / path_val).resolve()
    if not target.is_relative_to(resolved_repo) or target == resolved_repo:
        return None, f"captured screenshot path resolves outside repo_root: {path_val}"
    if not target.is_file():
        return None, f"screenshot file does not exist: {path_val}"
    return target, ""


def _png_size(data: bytes) -> tuple[int, int] | None:
    """Width and height from the PNG IHDR chunk (stdlib only), or None if not a PNG."""
    if not data.startswith(_PNG_SIGNATURE) or len(data) < 24 or data[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def _captured_violations(
    record_id: str, record: Mapping[str, Any], repo_root: Path
) -> list[str]:
    """A captured record needs alt text and bytes matching its hash and pixel size."""
    violations: list[str] = []
    if not _non_empty_str(record.get("alt_text")):
        violations.append(
            f"record {record_id!r}: captured screenshot alt_text must be a non-empty string"
        )
    target, reason = _resolve_asset(record.get("path"), repo_root)
    if target is None:
        return [*violations, f"record {record_id!r}: {reason}"]
    data = target.read_bytes()
    computed = hashlib.sha256(data).hexdigest()
    if computed != record.get("sha256"):
        violations.append(
            f"record {record_id!r}: sha256 mismatch "
            f"(declared {record.get('sha256')!r}, computed {computed!r})"
        )
    size = _png_size(data)
    if size is None:
        return [
            *violations,
            f"record {record_id!r}: unsupported image format: expected a PNG with IHDR",
        ]
    for field, actual in zip(("width", "height"), size, strict=True):
        if record.get(field) != actual:
            violations.append(
                f"record {record_id!r}: {field} mismatch "
                f"(declared {record.get(field)!r}, image is {actual})"
            )
    return violations


def verify_screenshot_records(payload: Mapping[str, Any], repo_root: Path) -> list[str]:
    """Verify screenshot records against repository files.

    Preconditions (DbC): ``payload`` is a Mapping whose ``records`` is a list;
    otherwise ``TypeError``.

    Returns:
        Human-readable violations; an empty list means pass.
    """
    if not isinstance(payload, Mapping):
        raise TypeError(f"payload must be a Mapping, got {type(payload).__name__}")
    records = payload.get("records")
    if not isinstance(records, list):
        raise TypeError(f"records must be a list, got {type(records).__name__}")

    violations: list[str] = []
    for idx, record in enumerate(records):
        if not isinstance(record, Mapping):
            violations.append(
                f"record [{idx}]: screenshot record must be a mapping, "
                f"got {type(record).__name__}"
            )
            continue
        record_id = str(record.get("id") or f"record[{idx}]")
        status = record.get("status")
        if status == "pending":
            violations.extend(_pending_violations(record_id, record))
        elif status == "captured":
            violations.extend(_captured_violations(record_id, record, Path(repo_root)))
        else:
            violations.append(
                f"record {record_id!r}: unknown status {status!r} "
                "(must be 'pending' or 'captured')"
            )
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    """Verify screenshot manifests against repository image assets.

    Exit codes:
        0: Validation passed with zero violations.
        1: Validation failed with violations (printed to stderr).
        2: Manifest was unreadable or contained invalid JSON.
    """
    parser = argparse.ArgumentParser(
        description="Verify companion screenshot manifest bytes and pixel dimensions."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the screenshots JSON manifest.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root path against which screenshot paths resolve.",
    )
    args = parser.parse_args(argv)

    try:
        raw = args.manifest.read_text(encoding="utf-8")
        payload = json.loads(raw)
        violations = verify_screenshot_records(payload, args.repo_root)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        sys.stderr.write(f"unreadable or invalid JSON manifest: {exc}\n")
        return 2

    if violations:
        for violation in violations:
            sys.stderr.write(f"{violation}\n")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
