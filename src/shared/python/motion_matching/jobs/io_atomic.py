"""Atomic JSON and artifact writes for MS-105 run roots."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

from .contracts import JOBS_SCHEMA, RunManifest

__all__ = ["atomic_write_bytes", "atomic_write_json", "read_text", "write_run_manifest"]


_WINDOWS_SHARING_ERRORS = frozenset({5, 32, 33})
_SHARING_RETRY_DELAYS_S = (0.01, 0.02, 0.04, 0.08, 0.16)
_Result = TypeVar("_Result")


def _sharing_retry(operation: Callable[[], _Result]) -> _Result:
    """Retry verified Windows sharing faults six times, with at most 0.31s delay."""
    for attempt in range(len(_SHARING_RETRY_DELAYS_S) + 1):
        try:
            return operation()
        except OSError as exc:
            transient = (
                isinstance(exc, PermissionError)
                and getattr(exc, "winerror", None) in _WINDOWS_SHARING_ERRORS
            )
            if not transient or attempt == len(_SHARING_RETRY_DELAYS_S):
                raise
            time.sleep(_SHARING_RETRY_DELAYS_S[attempt])
    raise AssertionError("Sharing retry exhausted without returning or raising")


def _promote_temp(temp_path: Path, destination: Path) -> None:
    """Atomically promote; remove only the owned stage after final failure."""
    try:
        _sharing_retry(lambda: temp_path.replace(destination))
    except OSError:
        temp_path.unlink(missing_ok=True)
        raise


def _windows_error_functions() -> (
    tuple[Callable[[int], None], Callable[[], int]] | None
):
    """Bind raw thread-error access before I/O, never ctypes' private copy."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=False)
    setter, getter = kernel.SetLastError, kernel.GetLastError
    setter.argtypes, setter.restype = [wintypes.DWORD], None
    getter.argtypes, getter.restype = [], wintypes.DWORD
    return setter, getter


def _read_attempt(
    path: Path,
    native_errors: tuple[Callable[[int], None], Callable[[], int]] | None,
) -> str:
    if native_errors is None:
        return path.read_text(encoding="utf-8")
    clear_error, capture_error = native_errors
    clear_error(0)
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        native_code = capture_error()
        # CPython's Windows CRT open can omit winerror. Capture immediately;
        # never classify errno 13 alone, stale state, or ctypes' private copy.
        if (
            isinstance(exc, PermissionError)
            and getattr(exc, "winerror", None) is None
            and native_code in _WINDOWS_SHARING_ERRORS
        ):
            exc.winerror = native_code
        raise


def read_text(path: Path) -> str:
    """Read UTF-8 with bounded verified Windows sharing retries.

    Native error functions are bound before I/O, cleared before every attempt,
    and captured immediately on failure. Explicit winerror takes precedence.
    Unknown errors, decoding failures and caller parsing failures propagate.
    """
    if not isinstance(path, Path):
        raise TypeError("Read path must be a pathlib.Path")
    native_errors = _windows_error_functions()
    return _sharing_retry(lambda: _read_attempt(path, native_errors))


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
