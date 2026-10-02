"""Atomic JSON and artifact writes for MS-105 run roots."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from .contracts import JOBS_SCHEMA, RunManifest

__all__ = ["atomic_write_bytes", "atomic_write_json", "write_run_manifest"]


_WINDOWS_SHARING_ERRORS = frozenset({5, 32, 33})
_PROMOTION_RETRY_DELAYS_S = (0.01, 0.02, 0.04, 0.08, 0.16)


def _promote_temp(temp_path: Path, destination: Path) -> None:
    """Atomically promote, retrying only bounded verified Windows sharing faults.

    Windows readers can temporarily deny replacement even when the destination
    remains valid. Six attempts and at most 0.31 seconds of backoff preserve the
    old-or-new publication contract; permanent and unrelated errors propagate.
    The owned stage is removed only after the final failed attempt.
    """
    for attempt in range(len(_PROMOTION_RETRY_DELAYS_S) + 1):
        try:
            temp_path.replace(destination)
            return
        except OSError as exc:
            transient = (
                isinstance(exc, PermissionError)
                and getattr(exc, "winerror", None) in _WINDOWS_SHARING_ERRORS
            )
            if transient and attempt < len(_PROMOTION_RETRY_DELAYS_S):
                time.sleep(_PROMOTION_RETRY_DELAYS_S[attempt])
                continue
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON atomically via tempfile + promote."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.stem}-",
        suffix=".tmp",
        delete=False,
    ) as handle:
        staged = Path(handle.name)
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _promote_temp(staged, path)
    return path


def atomic_write_bytes(path: Path, data: bytes) -> Path:
    """Write bytes atomically via tempfile + promote."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.stem}-",
        suffix=".tmp",
        delete=False,
    ) as handle:
        staged = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    _promote_temp(staged, path)
    return path


def write_run_manifest(run_root: Path, manifest: RunManifest) -> Path:
    """Persist ``run_manifest.json`` under ``run_root`` atomically."""
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_dict()
    if payload.get("schema_version") != JOBS_SCHEMA:
        raise ValueError("manifest schema_version mismatch")
    return atomic_write_json(run_root / "run_manifest.json", payload)
