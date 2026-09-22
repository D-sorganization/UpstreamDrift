"""Atomic JSON and artifact writes for MS-105 run roots."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .contracts import JOBS_SCHEMA, RunManifest

__all__ = ["atomic_write_bytes", "atomic_write_json", "write_run_manifest"]


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON atomically via tempfile + ``os.replace``."""
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
        temp_path = Path(handle.name)
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temp_path, path)
    except OSError:
        with contextlib.suppress(OSError):
            temp_path.unlink()
        raise
    return path


def atomic_write_bytes(path: Path, data: bytes) -> Path:
    """Write bytes atomically via tempfile + ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.stem}-",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temp_path, path)
    except OSError:
        with contextlib.suppress(OSError):
            temp_path.unlink()
        raise
    return path


def write_run_manifest(run_root: Path, manifest: RunManifest) -> Path:
    """Persist ``run_manifest.json`` under ``run_root`` atomically."""
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_dict()
    if payload.get("schema_version") != JOBS_SCHEMA:
        raise ValueError("manifest schema_version mismatch")
    return atomic_write_json(run_root / "run_manifest.json", payload)
