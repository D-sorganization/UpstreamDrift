"""Checkpoint persistence and compatible resume for MS-105."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    JOBS_SCHEMA,
    AcceptanceState,
    CheckpointCompatibilityError,
    CorruptPackageError,
    HashBundle,
    JobStage,
)
from .io_atomic import atomic_write_json

__all__ = ["CheckpointRecord", "load_checkpoint", "write_checkpoint"]

_CHECKPOINT_NAME = "checkpoint.json"


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    """Loaded checkpoint with provenance distinct from interrupted output."""

    stage: JobStage
    hashes: HashBundle
    payload: dict[str, Any]
    acceptance: AcceptanceState
    provenance: str = "checkpoint"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": JOBS_SCHEMA,
            "stage": self.stage.value,
            "hashes": self.hashes.to_dict(),
            "payload": dict(self.payload),
            "acceptance": self.acceptance.value,
            "provenance": self.provenance,
        }


def write_checkpoint(
    run_root: Path,
    *,
    stage: JobStage,
    hashes: HashBundle,
    payload: Mapping[str, Any],
    acceptance: AcceptanceState,
) -> Path:
    """Atomically write a stage checkpoint under ``run_root``."""
    if acceptance == AcceptanceState.ACCEPTED:
        raise ValueError("checkpoints must not claim accepted output")
    record = CheckpointRecord(
        stage=stage,
        hashes=hashes,
        payload=dict(payload),
        acceptance=acceptance,
        provenance="checkpoint",
    )
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    return atomic_write_json(run_root / _CHECKPOINT_NAME, record.to_dict())


def load_checkpoint(
    path: Path,
    *,
    expected_hashes: HashBundle,
) -> CheckpointRecord:
    """Load a checkpoint and refuse incompatible hash identities."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorruptPackageError(f"corrupt checkpoint at {path}") from exc
    if not isinstance(raw, dict):
        raise CorruptPackageError(f"checkpoint at {path} is not an object")
    schema = raw.get("schema_version")
    if schema != JOBS_SCHEMA:
        raise CorruptPackageError(
            f"checkpoint schema_version must be {JOBS_SCHEMA!r}; got {schema!r}"
        )
    hashes = HashBundle.from_dict(raw["hashes"])
    if not hashes.matches(expected_hashes):
        raise CheckpointCompatibilityError(
            "checkpoint hashes incompatible with current run identity"
        )
    acceptance = AcceptanceState(str(raw["acceptance"]))
    if acceptance == AcceptanceState.ACCEPTED:
        raise CorruptPackageError("checkpoint must not claim accepted output")
    return CheckpointRecord(
        stage=JobStage(str(raw["stage"])),
        hashes=hashes,
        payload=dict(raw.get("payload", {})),
        acceptance=acceptance,
        provenance=str(raw.get("provenance", "checkpoint")),
    )
