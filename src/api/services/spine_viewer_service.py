"""Spine Viewer web service (SPN-15c, #12815).

Thin, framework-light layer between the API route and the Qt-free viewer core
(:mod:`src.tools.spine_viewer.core`). It adds only receipt discovery and safe
relative-path handling; verification (SHA-256 of the loads artifact), the
fail-closed scientific status and the view model all come from the core.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

from src.tools.spine_viewer import core

__all__ = [
    "STUDY_ROOT_ENV",
    "list_receipts",
    "load_study_payload",
    "safe_relative_receipt",
    "study_root",
]

STUDY_ROOT_ENV = "UPSTREAMDRIFT_SPINE_STUDY_ROOT"
_RECEIPT_SUFFIX = "_receipt.json"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def study_root() -> Path:
    """Directory under which spine study receipts may be opened."""
    configured = os.environ.get(STUDY_ROOT_ENV)
    root = Path(configured) if configured else _PROJECT_ROOT / "data" / "spine_studies"
    return root.resolve()


def safe_relative_receipt(receipt: str) -> PurePosixPath:
    """Validate a client-supplied receipt path as a plain relative path.

    Raises:
        TypeError: ``receipt`` is not a string.
        ValueError: empty, absolute, drive-qualified, backslashed or with a
            ``..`` component.
    """
    if not isinstance(receipt, str):
        raise TypeError("receipt must be a string")
    if not receipt.strip():
        raise ValueError("receipt must be a non-empty relative path")
    if "\\" in receipt or "\x00" in receipt:
        raise ValueError("receipt must use '/' separators and no control bytes")
    path = PurePosixPath(receipt)
    if path.is_absolute() or ":" in path.parts[0]:
        raise ValueError("receipt must be relative to the study root")
    if ".." in path.parts:
        raise ValueError("receipt must not contain '..' components")
    return path


def list_receipts(root: Path) -> list[str]:
    """Sorted relative POSIX paths of ``*_receipt.json`` files under ``root``.

    Symlinked files and directories are skipped. A missing root lists nothing.
    """
    if not isinstance(root, Path):
        raise TypeError("root must be a Path")
    if not root.is_dir():
        return []
    found: list[str] = []
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(current) / d).is_symlink()]
        for name in files:
            full = Path(current) / name
            if name.endswith(_RECEIPT_SUFFIX) and not full.is_symlink():
                found.append(full.relative_to(root).as_posix())
    return sorted(found)


def load_study_payload(receipt_path: Path) -> dict[str, object]:
    """Open a receipt through the core (hash-verified) and serialise it.

    Raises:
        OSError: the receipt cannot be read.
        AggregationIntegrityError: artifact missing, tampered or inconsistent.
        ValueError: the receipt is malformed.
    """
    if not isinstance(receipt_path, Path):
        raise TypeError("receipt_path must be a Path")
    return core.view_to_dict(core.open_study(receipt_path))
