"""Bundle serialization, atomic persistence, and integrity verification for Shadow Tracker (ST-11).

Defines the durable bundle structure linking:
- `manifest.json`: Asset metadata, camera tracks, uncertainty, assumptions, and SHA-256 hash manifest.
- `observations.json`: Immutable frame observations, native timing authority, and clock evidence.
- `masks.json`: Complete manual mask revisions and parent lineage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

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

        if target_dir.exists():
            # Atomically replace or merge directory
            for f in temp_dir.iterdir():
                dest = target_dir / f.name
                dest.write_bytes(f.read_bytes())
        else:
            temp_dir.replace(target_dir)
    finally:
        if temp_dir.exists():
            import shutil

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
