"""Acquisition receipt builder for capture-O video companion (COV-1, #11269).

Provides provenance tracking, SHA-256 integrity verification, ffprobe metadata
embedding, and fail-closed validation for acquired video assets.
"""

from __future__ import annotations

from collections.abc import Container, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable

__all__ = [
    "AcquisitionEntry",
    "AcquisitionError",
    "AcquisitionOptions",
    "AcquisitionReceipt",
    "EmptyDirectoryError",
    "FFProbeError",
    "HashMismatchError",
    "NonVideoFileError",
    "VideoAcquisitionRejection",
    "build_acquisition_receipt",
    "verify_acquisition_receipt",
]

_HEX_DIGITS = frozenset("0123456789abcdef")
_WINDOWS_ABSOLUTE_PATTERN = re.compile(r"^[a-zA-Z]:[/\\]")
_KNOWN_NON_VIDEO_EXTS = frozenset(
    (
        ".txt",
        ".csv",
        ".tsv",
        ".json",
        ".xml",
        ".yaml",
        ".yml",
        ".md",
        ".rst",
        ".py",
        ".sh",
        ".bat",
        ".ps1",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".zip",
        ".tar",
        ".gz",
        ".7z",
        ".pdf",
        ".docx",
    )
)


class AcquisitionError(Exception):
    """Base exception for video acquisition receipt errors."""


class VideoAcquisitionRejection(AcquisitionError):
    """Base typed rejection naming the rejected file."""

    def __init__(self, message: str, *, filename: str) -> None:
        super().__init__(message)
        self.filename = filename


class NonVideoFileError(VideoAcquisitionRejection, ValueError):
    """Raised when a non-video file is encountered in the acquisition directory."""


class FFProbeError(VideoAcquisitionRejection, RuntimeError):
    """Raised when ffprobe fails on a file or returns invalid metadata."""


class HashMismatchError(AcquisitionError, ValueError):
    """Raised when a file's SHA-256 digest fails integrity check against an existing receipt."""

    def __init__(
        self,
        message: str,
        *,
        filename: str,
        expected_sha256: str,
        actual_sha256: str,
    ) -> None:
        super().__init__(message)
        self.filename = filename
        self.expected_sha256 = expected_sha256
        self.actual_sha256 = actual_sha256


class EmptyDirectoryError(AcquisitionError, ValueError):
    """Raised when the acquisition directory contains no files."""

    def __init__(self, message: str, *, directory: Path | str | None = None) -> None:
        super().__init__(message)
        self.directory = str(directory) if directory is not None else None


def _check_sha256(val: str, field_name: str) -> str:
    """Validate 64 lowercase hex characters."""
    if not isinstance(val, str):
        raise TypeError(f"{field_name} must be a str, got {type(val).__name__}")
    if len(val) != 64 or not all(c in _HEX_DIGITS for c in val):
        raise ValueError(
            f"{field_name} must be exactly 64 lowercase hex characters, got {val!r}"
        )
    return val


def _compute_sha256(path: Path) -> str:
    """Compute standard SHA-256 hex digest of file contents."""
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _sanitize_no_absolute_paths(obj: Any) -> Any:
    """Recursively strip or normalize absolute paths from structured data."""
    if isinstance(obj, dict):
        sanitized = {}
        for k, v in obj.items():
            if k == "filename" and isinstance(v, str):
                sanitized[k] = Path(v).name
            else:
                sanitized[k] = _sanitize_no_absolute_paths(v)
        return sanitized
    if isinstance(obj, list):
        return [_sanitize_no_absolute_paths(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize_no_absolute_paths(item) for item in obj)
    if isinstance(obj, str):
        if _WINDOWS_ABSOLUTE_PATTERN.match(obj) or obj.startswith(("/", "\\")):
            return Path(obj).name
        return obj
    return obj


def _assert_no_absolute_paths(obj: Any, path_prefix: str = "") -> None:
    """Verify that an object contains no absolute local path strings."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_absolute_paths(v, f"{path_prefix}.{k}")
    elif isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            _assert_no_absolute_paths(item, f"{path_prefix}[{idx}]")
    elif isinstance(obj, str):
        if _WINDOWS_ABSOLUTE_PATTERN.match(obj) or obj.startswith(("/", "\\")):
            raise ValueError(f"Absolute local path detected at {path_prefix}: {obj!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class AcquisitionEntry:
    """Provenance record for a single acquired video asset."""

    cov_id: str
    original_filename: str
    sha256: str
    byte_size: int
    source_url: str | None = None
    download_method: str = "album-download-all"
    downloaded_at: str
    lineage: str = "original"
    container_creation_time: str | None = None
    ffprobe: dict[str, Any]
    owner_recollection: str | None = None
    excluded_by_owner: bool = False

    def __post_init__(self) -> None:
        if not re.match(r"^cov-\d+$", self.cov_id):
            raise ValueError(f"cov_id must match 'cov-NN', got {self.cov_id!r}")
        if (
            not self.original_filename
            or Path(self.original_filename).name != self.original_filename
        ):
            raise ValueError(
                "original_filename must be a plain filename without directory components, "
                f"got {self.original_filename!r}"
            )
        _check_sha256(self.sha256, "sha256")
        if self.byte_size <= 0:
            raise ValueError(f"byte_size must be positive, got {self.byte_size}")
        if not self.download_method:
            raise ValueError("download_method cannot be empty")
        if not self.downloaded_at:
            raise ValueError("downloaded_at cannot be empty")
        if self.lineage != "original" and not re.match(
            r"^album-copy-of cov-\d+$", self.lineage
        ):
            raise ValueError(
                f"lineage must be 'original' or 'album-copy-of cov-NN', got {self.lineage!r}"
            )
        _assert_no_absolute_paths(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Serialize entry to dictionary."""
        return {
            "cov_id": self.cov_id,
            "original_filename": self.original_filename,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "source_url": self.source_url,
            "download_method": self.download_method,
            "downloaded_at": self.downloaded_at,
            "lineage": self.lineage,
            "container_creation_time": self.container_creation_time,
            "ffprobe": self.ffprobe,
            "owner_recollection": self.owner_recollection,
            "excluded_by_owner": self.excluded_by_owner,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AcquisitionEntry:
        """Construct entry from dictionary representation."""
        return cls(
            cov_id=str(data["cov_id"]),
            original_filename=str(data["original_filename"]),
            sha256=str(data["sha256"]),
            byte_size=int(data["byte_size"]),
            source_url=(
                str(data["source_url"]) if data.get("source_url") is not None else None
            ),
            download_method=str(data.get("download_method", "album-download-all")),
            downloaded_at=str(data["downloaded_at"]),
            lineage=str(data.get("lineage", "original")),
            container_creation_time=(
                str(data["container_creation_time"])
                if data.get("container_creation_time") is not None
                else None
            ),
            ffprobe=dict(data.get("ffprobe", {})),
            owner_recollection=(
                str(data["owner_recollection"])
                if data.get("owner_recollection") is not None
                else None
            ),
            excluded_by_owner=bool(data.get("excluded_by_owner", False)),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AcquisitionOptions:
    """Configurable options for acquisition receipt generation."""

    source_url: str | None = None
    download_method: str = "album-download-all"
    downloaded_at: str | None = None
    lineage_map: Mapping[str, str] | None = None
    owner_recollections: Mapping[str, str | None] | None = None
    excluded_by_owner: Container[str] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class AcquisitionReceipt:
    """Provenance receipt for the entire acquired video collection."""

    schema_version: str = "capture-o/acquisition-receipt/1.0.0"
    created_at: str
    video_count: int
    total_byte_size: int
    entries: tuple[AcquisitionEntry, ...]

    def __post_init__(self) -> None:
        if self.video_count != len(self.entries):
            raise ValueError(
                f"video_count {self.video_count} does not match len(entries) {len(self.entries)}"
            )
        computed_size = sum(e.byte_size for e in self.entries)
        if self.total_byte_size != computed_size:
            raise ValueError(
                f"total_byte_size {self.total_byte_size} does not match sum of entries {computed_size}"
            )
        _assert_no_absolute_paths(self.to_dict())

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __getitem__(self, idx_or_key: int | str) -> Any:
        if isinstance(idx_or_key, int):
            return self.entries[idx_or_key]
        if isinstance(idx_or_key, str):
            for e in self.entries:
                if e.cov_id == idx_or_key or e.original_filename == idx_or_key:
                    return e
            if idx_or_key in {"entries", "items"}:
                return self.entries
            payload = self.to_dict()
            if idx_or_key in payload:
                return payload[idx_or_key]
            raise KeyError(f"Entry or field not found: {idx_or_key!r}")
        raise TypeError(f"Index must be int or str, got {type(idx_or_key).__name__}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize receipt to dictionary."""
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "video_count": self.video_count,
            "total_byte_size": self.total_byte_size,
            "entries": [e.to_dict() for e in self.entries],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize receipt to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent) + "\n"

    def write_atomic(self, target_path: Path | str) -> Path:
        """Write receipt to path atomically via a temporary file."""
        dest = Path(target_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=dest.parent,
            prefix=f".{dest.name}.",
            delete=False,
        ) as tmp:
            tmp.write(self.to_json())
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, dest)
        return dest

    @classmethod
    def from_dict(
        cls, data: dict[str, Any] | list[dict[str, Any]]
    ) -> AcquisitionReceipt:
        """Construct receipt from dictionary or list."""
        if isinstance(data, list):
            entries = tuple(AcquisitionEntry.from_dict(e) for e in data)
            now = datetime.now(timezone.utc).isoformat()
            total_size = sum(e.byte_size for e in entries)
            return cls(
                created_at=now,
                video_count=len(entries),
                total_byte_size=total_size,
                entries=entries,
            )
        raw_entries = data.get("entries", data.get("items", []))
        entries = tuple(AcquisitionEntry.from_dict(e) for e in raw_entries)
        return cls(
            schema_version=str(
                data.get("schema_version", "capture-o/acquisition-receipt/1.0.0")
            ),
            created_at=str(
                data.get("created_at", datetime.now(timezone.utc).isoformat())
            ),
            video_count=int(data.get("video_count", len(entries))),
            total_byte_size=int(
                data.get("total_byte_size", sum(e.byte_size for e in entries))
            ),
            entries=entries,
        )

    @classmethod
    def from_file(cls, path: Path | str) -> AcquisitionReceipt:
        """Load receipt from a JSON file."""
        file_path = Path(path)
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def _extract_creation_time(ffprobe_data: dict[str, Any]) -> str | None:
    """Extract container or video stream creation timestamp from ffprobe tags."""
    fmt_tags = ffprobe_data.get("format", {}).get("tags", {})
    if isinstance(fmt_tags, dict) and "creation_time" in fmt_tags:
        return str(fmt_tags["creation_time"])
    streams = ffprobe_data.get("streams", [])
    if isinstance(streams, list):
        for s in streams:
            if isinstance(s, dict) and s.get("codec_type") == "video":
                tags = s.get("tags", {})
                if isinstance(tags, dict) and "creation_time" in tags:
                    return str(tags["creation_time"])
    return None


def _probe_via_subprocess(file_path: Path) -> dict[str, Any]:
    """Execute ffprobe subprocess and parse JSON stdout."""
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        raise FFProbeError(
            f"No ffprobe binary or sidecar available to probe {file_path.name}",
            filename=file_path.name,
        )
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(file_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise FFProbeError(
            f"ffprobe failed for {file_path.name} with code {proc.returncode}: {proc.stderr}",
            filename=file_path.name,
        )
    try:
        return json.loads(proc.stdout)
    except Exception as exc:
        raise FFProbeError(
            f"Failed to parse ffprobe json for {file_path.name}: {exc}",
            filename=file_path.name,
        ) from exc


def _probe_file(
    file_path: Path,
    *,
    probe_fn: Callable[[Path], dict[str, Any]] | None,
    ffprobe_dir: Path | None,
) -> dict[str, Any]:
    """Probe video metadata using probe_fn, sidecars, or ffprobe subprocess."""
    if file_path.suffix.lower() in _KNOWN_NON_VIDEO_EXTS:
        raise NonVideoFileError(
            f"Non-video file rejected: {file_path.name}", filename=file_path.name
        )

    if probe_fn is not None:
        try:
            raw = probe_fn(file_path)
        except VideoAcquisitionRejection:
            raise
        except Exception as exc:
            raise FFProbeError(
                f"ffprobe failure for {file_path.name}: {exc}",
                filename=file_path.name,
            ) from exc
    elif ffprobe_dir is not None and (ffprobe_dir / f"{file_path.name}.json").is_file():
        with (ffprobe_dir / f"{file_path.name}.json").open("r", encoding="utf-8") as f:
            raw = json.load(f)
    elif (file_path.parent.parent / "ffprobe" / f"{file_path.name}.json").is_file():
        with (file_path.parent.parent / "ffprobe" / f"{file_path.name}.json").open(
            "r", encoding="utf-8"
        ) as f:
            raw = json.load(f)
    else:
        raw = _probe_via_subprocess(file_path)

    streams = raw.get("streams", [])
    has_video = any(
        isinstance(s, dict) and s.get("codec_type") == "video" for s in streams
    )
    if not has_video:
        raise NonVideoFileError(
            f"No video streams found in {file_path.name}",
            filename=file_path.name,
        )
    return _sanitize_no_absolute_paths(raw)


def verify_acquisition_receipt(
    receipt_or_path: AcquisitionReceipt | Path | str,
    directory: Path | str,
) -> None:
    """Verify that every file in receipt exists in directory with matching SHA-256."""
    if isinstance(receipt_or_path, (str, Path)):
        receipt = AcquisitionReceipt.from_file(receipt_or_path)
    else:
        receipt = receipt_or_path

    dir_path = Path(directory)
    for entry in receipt.entries:
        target = dir_path / entry.original_filename
        if not target.is_file():
            raise FileNotFoundError(f"Missing acquired video: {target}")
        actual_sha = _compute_sha256(target)
        if actual_sha != entry.sha256:
            raise HashMismatchError(
                f"Hash mismatch for {entry.original_filename}: "
                f"expected {entry.sha256}, got {actual_sha}",
                filename=entry.original_filename,
                expected_sha256=entry.sha256,
                actual_sha256=actual_sha,
            )


def _load_existing_receipt(
    receipt_file: Path | None,
    verify_existing: bool,
) -> AcquisitionReceipt | None:
    """Load existing receipt from file if available and requested."""
    if receipt_file and verify_existing and receipt_file.is_file():
        return AcquisitionReceipt.from_file(receipt_file)
    return None


def _check_existing_hashes(
    existing: AcquisitionReceipt,
    files: list[Path],
) -> None:
    """Validate hashes of directory files against existing receipt."""
    entry_map = {e.original_filename: e for e in existing.entries}
    for f in files:
        if f.name in entry_map:
            actual_sha = _compute_sha256(f)
            expected_sha = entry_map[f.name].sha256
            if actual_sha != expected_sha:
                raise HashMismatchError(
                    f"Hash mismatch for {f.name}: expected {expected_sha}, got {actual_sha}",
                    filename=f.name,
                    expected_sha256=expected_sha,
                    actual_sha256=actual_sha,
                )


def _build_entries(
    files: list[Path],
    existing_receipt: AcquisitionReceipt | None,
    options: AcquisitionOptions,
    ffprobe_dir: Path | None,
    probe_fn: Callable[[Path], dict[str, Any]] | None,
) -> list[AcquisitionEntry]:
    """Construct structured entries for all files."""
    existing_map = (
        {e.original_filename: e for e in existing_receipt.entries}
        if existing_receipt
        else {}
    )
    now_iso = options.downloaded_at or datetime.now(timezone.utc).isoformat()
    entries: list[AcquisitionEntry] = []

    for idx, f in enumerate(files, start=1):
        cov_id = f"cov-{idx:02d}"
        sha256 = _compute_sha256(f)
        byte_size = f.stat().st_size
        probe_data = _probe_file(f, probe_fn=probe_fn, ffprobe_dir=ffprobe_dir)
        creation_time = _extract_creation_time(probe_data)

        prior = existing_map.get(f.name)
        item_dl_at = (
            prior.downloaded_at if prior and not options.downloaded_at else now_iso
        )
        item_method = (
            prior.download_method
            if prior and options.download_method == "album-download-all"
            else options.download_method
        )
        item_source_url = (
            options.source_url
            if options.source_url is not None
            else (prior.source_url if prior else None)
        )

        if options.lineage_map and (
            f.name in options.lineage_map or cov_id in options.lineage_map
        ):
            lineage = options.lineage_map.get(
                f.name, options.lineage_map.get(cov_id, "original")
            )
        elif prior:
            lineage = prior.lineage
        else:
            lineage = "original"

        if options.owner_recollections and (
            f.name in options.owner_recollections
            or cov_id in options.owner_recollections
        ):
            recollection = options.owner_recollections.get(
                f.name, options.owner_recollections.get(cov_id)
            )
        elif prior:
            recollection = prior.owner_recollection
        else:
            recollection = None

        if options.excluded_by_owner is not None:
            excluded = (
                f.name in options.excluded_by_owner
                or cov_id in options.excluded_by_owner
            )
        elif prior:
            excluded = prior.excluded_by_owner
        else:
            excluded = False

        entries.append(
            AcquisitionEntry(
                cov_id=cov_id,
                original_filename=f.name,
                sha256=sha256,
                byte_size=byte_size,
                source_url=item_source_url,
                download_method=item_method,
                downloaded_at=item_dl_at,
                lineage=lineage,
                container_creation_time=creation_time,
                ffprobe=probe_data,
                owner_recollection=recollection,
                excluded_by_owner=excluded,
            )
        )
    return entries


def build_acquisition_receipt(
    directory: Path | str,
    *,
    output_path: Path | str | None = None,
    options: AcquisitionOptions | None = None,
    ffprobe_dir: Path | str | None = None,
    probe_fn: Callable[[Path], dict[str, Any]] | None = None,
    verify_existing: bool = True,
    **kwargs: Any,
) -> AcquisitionReceipt:
    """Build or update an acquisition receipt for video assets in directory."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {dir_path}")

    files = sorted(
        [p for p in dir_path.iterdir() if p.is_file() and not p.name.startswith(".")],
        key=lambda p: p.name,
    )
    if not files:
        raise EmptyDirectoryError(
            f"Directory {dir_path} contains no files", directory=dir_path
        )

    receipt_file = Path(output_path) if output_path else None
    existing_receipt = _load_existing_receipt(receipt_file, verify_existing)

    if existing_receipt:
        _check_existing_hashes(existing_receipt, files)

    opts = options or AcquisitionOptions(
        source_url=kwargs.get("source_url"),
        download_method=kwargs.get("download_method", "album-download-all"),
        downloaded_at=kwargs.get("downloaded_at"),
        lineage_map=kwargs.get("lineage_map"),
        owner_recollections=kwargs.get("owner_recollections"),
        excluded_by_owner=kwargs.get("excluded_by_owner"),
    )

    entries = _build_entries(
        files=files,
        existing_receipt=existing_receipt,
        options=opts,
        ffprobe_dir=Path(ffprobe_dir) if ffprobe_dir else None,
        probe_fn=probe_fn,
    )

    created_at = (
        existing_receipt.created_at
        if existing_receipt and not opts.downloaded_at
        else datetime.now(timezone.utc).isoformat()
    )
    receipt = AcquisitionReceipt(
        created_at=created_at,
        video_count=len(entries),
        total_byte_size=sum(e.byte_size for e in entries),
        entries=tuple(entries),
    )

    if receipt_file:
        receipt.write_atomic(receipt_file)
    return receipt
