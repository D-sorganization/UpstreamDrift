"""Shared capture registry for public and private motion capture datasets (#11162).

Provides a single resolver, used across engines and tools, turning a neutral
capture ID into a verified local file.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from src.shared.python.contracts import postcondition

logger = logging.getLogger(__name__)

__all__ = [
    "CaptureDataUnavailable",
    "CaptureInfo",
    "CaptureIntegrityError",
    "CaptureRegistryError",
    "UnknownCaptureError",
    "capture_info",
    "list_captures",
    "require_capture",
    "resolve_capture",
]


class CaptureRegistryError(Exception):
    """Base exception for capture registry errors."""


class UnknownCaptureError(CaptureRegistryError, KeyError):
    """Raised when a capture ID is not in the registry."""


class CaptureDataUnavailable(CaptureRegistryError, FileNotFoundError):
    """Raised when private capture data is unavailable (unset env var or missing file)."""


class CaptureIntegrityError(CaptureRegistryError, ValueError):
    """Raised when a resolved capture file fails SHA-256 integrity verification."""


@dataclass(frozen=True)
class CaptureInfo:
    """Facts recorded in the public capture manifest."""

    id: str
    where: str
    relative_path: str
    sha256: str
    bytes: int | None
    sample_rate_hz: float | None = None
    frame_count: int | None = None

    @property
    def frame_rate_hz(self) -> float | None:
        """Alias for sample_rate_hz."""
        return self.sample_rate_hz

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaptureInfo:
        """Construct CaptureInfo from manifest dictionary."""
        return cls(
            id=str(data["id"]),
            where=str(data["where"]),
            relative_path=str(data["relative_path"]),
            sha256=str(data["sha256"]),
            bytes=int(data["bytes"]) if data.get("bytes") is not None else None,
            sample_rate_hz=(
                float(data["sample_rate_hz"])
                if data.get("sample_rate_hz") is not None
                else None
            ),
            frame_count=(
                int(data["frame_count"])
                if data.get("frame_count") is not None
                else None
            ),
        )


_HASH_CACHE: dict[tuple[Path, int], str] = {}


def _find_repo_root() -> Path:
    """Find repository root containing data/capture_registry.json."""
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "data" / "capture_registry.json").is_file():
        return candidate
    for p in Path(__file__).resolve().parents:
        if (p / "data" / "capture_registry.json").is_file():
            return p
    return candidate


def _load_registry(repo_root: Path | None = None) -> dict[str, CaptureInfo]:
    """Load capture manifest from data/capture_registry.json."""
    root = repo_root or _find_repo_root()
    manifest_path = root / "data" / "capture_registry.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Capture registry manifest not found at {manifest_path}"
        )

    with manifest_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        entries = {item["id"]: item for item in raw}
    elif (
        isinstance(raw, dict)
        and "captures" in raw
        and isinstance(raw["captures"], dict)
    ):
        entries = raw["captures"]
    elif (
        isinstance(raw, dict)
        and "captures" in raw
        and isinstance(raw["captures"], list)
    ):
        entries = {item["id"]: item for item in raw["captures"]}
    elif isinstance(raw, dict):
        entries = raw
    else:
        raise ValueError(f"Unrecognized registry manifest format in {manifest_path}")

    return {cid: CaptureInfo.from_dict(item) for cid, item in entries.items()}


def _hash_file(path: Path) -> str:
    """Hash file with SHA-256, caching by resolved path and mtime_ns."""
    resolved = path.resolve()
    stat = resolved.stat()
    key = (resolved, stat.st_mtime_ns)
    if key in _HASH_CACHE:
        return _HASH_CACHE[key]

    hasher = hashlib.sha256()
    with resolved.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    _HASH_CACHE[key] = digest
    return digest


def list_captures(*, repo_root: Path | None = None) -> list[str]:
    """List all capture IDs in the registry."""
    registry = _load_registry(repo_root=repo_root)
    return list(registry.keys())


def capture_info(capture_id: str, *, repo_root: Path | None = None) -> CaptureInfo:
    """Return manifest facts for a capture ID."""
    registry = _load_registry(repo_root=repo_root)
    if capture_id not in registry:
        raise UnknownCaptureError(f"Unknown capture ID: {capture_id!r}")
    return registry[capture_id]


@postcondition(lambda result: result.is_file(), "Resolved capture path must exist")
def resolve_capture(
    capture_id: str,
    *,
    data_dir: Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Resolve a capture ID to a verified local file path.

    Parameters
    ----------
    capture_id : str
        Neutral capture ID (e.g. 'capture-A', 'capture-B', 'capture-O').
    data_dir : Path | None
        Directory containing private captures. If omitted, defaults to the
        CAPTURE_DATA_DIR environment variable.
    repo_root : Path | None
        Repository root directory. If omitted, discovers root automatically.

    Returns
    -------
    Path
        Absolute verified path to the capture file.

    Raises
    ------
    UnknownCaptureError
        If capture_id is not registered.
    CaptureDataUnavailable
        If private capture data is missing or CAPTURE_DATA_DIR is unset.
    CaptureIntegrityError
        If the file's SHA-256 hash does not match the registry manifest.
    """
    info = capture_info(capture_id, repo_root=repo_root)

    if info.where == "public":
        root = repo_root or _find_repo_root()
        target_path = (root / info.relative_path).resolve()
        if not target_path.is_file():
            raise CaptureDataUnavailable(
                f"Public capture {capture_id!r} not found at expected path: {target_path}"
            )
    elif info.where == "private":
        resolved_data_dir = data_dir
        if resolved_data_dir is None:
            env_dir = os.environ.get("CAPTURE_DATA_DIR", "").strip()
            if env_dir:
                resolved_data_dir = Path(env_dir)
        if resolved_data_dir is None:
            raise CaptureDataUnavailable(
                f"Capture {capture_id!r} is private and CAPTURE_DATA_DIR environment "
                "variable is not set."
            )
        target_path = (Path(resolved_data_dir) / info.relative_path).resolve()
        if not target_path.is_file():
            raise CaptureDataUnavailable(
                f"Private capture {capture_id!r} not found at {target_path}."
            )
    else:
        raise ValueError(f"Unknown capture 'where' location: {info.where!r}")

    digest = _hash_file(target_path)
    if digest.lower() != info.sha256.lower():
        raise CaptureIntegrityError(
            f"SHA-256 mismatch for capture {capture_id!r} at {target_path}: "
            f"expected {info.sha256}, got {digest}"
        )

    return target_path


def require_capture(
    capture_id: str,
    *,
    data_dir: Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Resolve a capture or skip the running pytest test if data is unavailable."""
    try:
        return resolve_capture(capture_id, data_dir=data_dir, repo_root=repo_root)
    except CaptureDataUnavailable as exc:
        try:
            import pytest

            pytest.skip(f"Capture {capture_id!r} unavailable: {exc}")
        except ImportError:
            raise exc from None
