"""Bundle serialization, atomic persistence, and integrity verification for Shadow Tracker (ST-11).

Defines the durable bundle structure linking:
- `manifest.json`: Asset metadata, camera tracks, uncertainty, assumptions, and SHA-256 hash manifest.
- `observations.json`: Immutable frame observations, native timing authority, and clock evidence.
- `masks.json`: Complete manual mask revisions and parent lineage.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from typing import Any, Final

from ._validation import (
    check_id,
    check_payload_keys,
    check_schema_version,
)
from .contracts import (
    FRAME_OBSERVATION_SCHEMA_VERSION,
    FrameObservation,
)
from .mask_records import MaskFrame
from .source_records import SourceAsset

BUNDLE_SCHEMA_VERSION: str = "1.0.0"

_MANIFEST_REQUIRED_KEYS = frozenset(
    (
        "schema_version",
        "bundle_id",
        "created_at",
        "source_asset",
        "uncertainty",
        "assumptions",
        "evidence_quality",
        "hashes",
    )
)


def _compute_sha256(data: bytes) -> str:
    """Compute hex SHA-256 digest of byte payload."""
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class ShadowTrackerBundle:
    """In-memory representation of a persisted review bundle."""

    bundle_id: str
    schema_version: str = BUNDLE_SCHEMA_VERSION
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source_asset: SourceAsset
    observations: tuple[FrameObservation, ...]
    masks: tuple[MaskFrame, ...]
    uncertainty: dict[str, Any] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    evidence_quality: str = "unreviewed"
    hashes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_schema_version(self.schema_version, BUNDLE_SCHEMA_VERSION)
        check_id(self.bundle_id, "bundle_id")
        if not isinstance(self.source_asset, SourceAsset):
            raise TypeError(
                f"source_asset must be a SourceAsset, got {type(self.source_asset).__name__}"
            )
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "masks", tuple(self.masks))
        object.__setattr__(self, "uncertainty", dict(self.uncertainty))
        object.__setattr__(self, "assumptions", tuple(str(a) for a in self.assumptions))


_RENAME_EXCHANGE_FLAG: Final[int] = 2  # renameat2(2) RENAME_EXCHANGE (Linux)
_AT_FDCWD: Final[int] = -100  # renameat2(2): path is relative to the CWD
_RENAMEAT2_SYSCALL: Final[dict[str, int]] = {
    "x86_64": 316,
    "aarch64": 276,
    "riscv64": 276,
    "ppc64le": 357,
    "s390x": 347,
}


def _rename_exchange(src: Path, dst: Path) -> bool:
    """Atomically swap two existing directory paths via renameat2(RENAME_EXCHANGE).

    Returns True when the kernel performed the single-operation swap; False when the
    platform cannot provide it (the caller falls back to backup-and-rollback renames).
    """
    number = _RENAMEAT2_SYSCALL.get(platform.machine())
    if number is None or not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        syscall = libc.syscall
        syscall.restype = ctypes.c_long
        syscall.argtypes = [
            ctypes.c_long,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        result = syscall(
            number,
            _AT_FDCWD,
            os.fsencode(src),
            _AT_FDCWD,
            os.fsencode(dst),
            _RENAME_EXCHANGE_FLAG,
        )
    except (AttributeError, OSError):
        return False
    return result == 0


def publish_directory(staged: Path, target: Path) -> None:
    """Publish a fully staged bundle directory as the live target.

    The publish is a single atomic kernel operation on Linux (RENAME_EXCHANGE when
    replacing, a plain rename when creating), so readers always observe either the
    previous or the new bundle and never a partially written or missing target.
    On platforms without ``renameat2``, replacement falls back to a backup rename
    with restore-on-failure.
    """
    if not target.exists():
        # First publication: a single rename is already atomic.
        staged.rename(target)
        return

    if _rename_exchange(staged, target):
        # `staged` now holds the replaced (previous) bundle; the caller cleans it up.
        return

    # renameat2 unavailable: two-rename fallback with backup rollback.
    backup_dir = Path(tempfile.mkdtemp(prefix=".tmp_stbackup_", dir=target.parent))
    saved = backup_dir / "saved"
    target.rename(saved)
    try:
        staged.rename(target)
    except BaseException:
        # Restore the original bundle untouched, then re-raise.
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        saved.rename(target)
        shutil.rmtree(backup_dir, ignore_errors=True)
        raise
    shutil.rmtree(backup_dir, ignore_errors=True)


def save_bundle(bundle: ShadowTrackerBundle, target_path: Path | str) -> None:
    """Atomically save a ShadowTrackerBundle to a directory bundle.

    Preconditions:
        - `bundle` must be a valid ShadowTrackerBundle.
        - `target_path` must not contain dangerous path traversal.
    """
    if not isinstance(bundle, ShadowTrackerBundle):
        raise TypeError(f"Expected ShadowTrackerBundle, got {type(bundle).__name__}")

    target_dir = Path(target_path).resolve()
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    obs_payload = [obs.to_dict() for obs in bundle.observations]
    obs_bytes = json.dumps(
        obs_payload,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    obs_sha256 = _compute_sha256(obs_bytes)

    masks_payload = [mask.to_dict() for mask in bundle.masks]
    masks_bytes = json.dumps(
        masks_payload,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    masks_sha256 = _compute_sha256(masks_bytes)

    manifest_payload: dict[str, Any] = {
        "schema_version": bundle.schema_version,
        "bundle_id": bundle.bundle_id,
        "created_at": bundle.created_at,
        "source_asset": bundle.source_asset.to_dict(),
        "uncertainty": dict(bundle.uncertainty),
        "assumptions": list(bundle.assumptions),
        "evidence_quality": bundle.evidence_quality,
        "hashes": {
            "observations.json": obs_sha256,
            "masks.json": masks_sha256,
        },
    }
    manifest_bytes = json.dumps(
        manifest_payload,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")

    # Atomic write using temporary directory in the target parent folder
    temp_dir = Path(tempfile.mkdtemp(prefix=".tmp_stbundle_", dir=target_dir.parent))
    try:
        (temp_dir / "observations.json").write_bytes(obs_bytes)
        (temp_dir / "masks.json").write_bytes(masks_bytes)
        (temp_dir / "manifest.json").write_bytes(manifest_bytes)

        publish_directory(temp_dir, target_dir)
    finally:
        # After a successful exchange `temp_dir` holds the replaced bundle; after a
        # staged-write failure it holds the aborted copy. Never leak either.
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def load_bundle(source_path: Path | str) -> ShadowTrackerBundle:
    """Load a ShadowTrackerBundle from disk and verify tamper integrity.

    Raises:
        FileNotFoundError: If target path or manifest does not exist.
        ValueError: If manifest schema or file checksums fail verification.
    """
    target_dir = Path(source_path).resolve()
    manifest_file = target_dir / "manifest.json"
    obs_file = target_dir / "observations.json"
    masks_file = target_dir / "masks.json"

    if not manifest_file.exists():
        raise FileNotFoundError(f"Missing bundle manifest at {manifest_file}")
    if not obs_file.exists():
        raise FileNotFoundError(f"Missing bundle observations at {obs_file}")
    if not masks_file.exists():
        raise FileNotFoundError(f"Missing bundle masks at {masks_file}")

    manifest_bytes = manifest_file.read_bytes()
    manifest_data = json.loads(manifest_bytes.decode("utf-8"))
    check_payload_keys(manifest_data, _MANIFEST_REQUIRED_KEYS)
    check_schema_version(manifest_data["schema_version"], BUNDLE_SCHEMA_VERSION)

    hashes = manifest_data.get("hashes", {})
    expected_obs_sha = hashes.get("observations.json")
    expected_masks_sha = hashes.get("masks.json")

    obs_bytes = obs_file.read_bytes()
    actual_obs_sha = _compute_sha256(obs_bytes)
    if expected_obs_sha and actual_obs_sha != expected_obs_sha:
        raise ValueError(
            f"Checksum mismatch for observations.json: expected {expected_obs_sha}, got {actual_obs_sha} (tampered or corrupt)"
        )

    masks_bytes = masks_file.read_bytes()
    actual_masks_sha = _compute_sha256(masks_bytes)
    if expected_masks_sha and actual_masks_sha != expected_masks_sha:
        raise ValueError(
            f"Checksum mismatch for masks.json: expected {expected_masks_sha}, got {actual_masks_sha} (tampered or corrupt)"
        )

    raw_source = manifest_data["source_asset"]
    source_asset = SourceAsset.from_dict(raw_source)

    raw_obs_list = json.loads(obs_bytes.decode("utf-8"))
    observations = tuple(
        FrameObservation.from_dict(raw_obs) for raw_obs in raw_obs_list
    )

    raw_masks_list = json.loads(masks_bytes.decode("utf-8"))
    masks = tuple(MaskFrame.from_dict(raw_mask) for raw_mask in raw_masks_list)

    return ShadowTrackerBundle(
        bundle_id=manifest_data["bundle_id"],
        schema_version=manifest_data["schema_version"],
        created_at=manifest_data["created_at"],
        source_asset=source_asset,
        observations=observations,
        masks=masks,
        uncertainty=manifest_data.get("uncertainty", {}),
        assumptions=tuple(manifest_data.get("assumptions", ())),
        evidence_quality=manifest_data.get("evidence_quality", "unreviewed"),
        hashes=hashes,
    )
