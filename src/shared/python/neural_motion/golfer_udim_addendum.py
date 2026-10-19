"""Dated addendum superseding sealed NM golfer u_dim records (#12670).

Frozen ``nm01_model_roster.json`` and ``checkpoint_matrix.md`` still record
``independent_dof=5`` under intact digests. This addendum is the authoritative
correction after #12608 without editing those sealed artifacts in place.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "ADDENDUM_SCHEMA",
    "GOLFER_MODEL_ID",
    "GolferUdimAddendum",
    "content_digest_for_payload",
    "default_addendum_path",
    "load_golfer_udim_addendum",
]

ADDENDUM_SCHEMA = "neural-sealed-record-addendum/1.0.0"
GOLFER_MODEL_ID = "constrained_upper_body_golfer"

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_ADDENDUM_PATH = (
    _REPO_ROOT
    / "docs"
    / "plans"
    / "neural_motion_matching"
    / "evidence"
    / "nm12670_golfer_independent_dof_addendum.json"
)


def default_addendum_path() -> Path:
    """Return the canonical on-disk addendum path."""
    return _DEFAULT_ADDENDUM_PATH


def content_digest_for_payload(payload: Mapping[str, Any]) -> str:
    """SHA-256 over sorted compact JSON (excludes ``content_digest``)."""
    blob = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GolferUdimAddendum:
    """Parsed addendum with digest verification."""

    schema: str
    content_digest: str
    governing_issue: int
    effective_date: str
    owner_ruling: str
    model_id: str
    corrected_q_dim: int
    corrected_independent_dof: int
    corrected_checkpoint_layout: str
    supersedes_roster_content_digest: str
    supersedes_matrix_digest: str
    checkpoint_invalidation_status: str
    retrain_required: bool
    sealed_records_unchanged: bool

    def __post_init__(self) -> None:
        if self.schema != ADDENDUM_SCHEMA:
            raise ValueError(f"schema must be {ADDENDUM_SCHEMA}")
        if self.model_id != GOLFER_MODEL_ID:
            raise ValueError(f"model_id must be {GOLFER_MODEL_ID}")
        if self.corrected_independent_dof <= 0:
            raise ValueError("corrected_independent_dof must be positive")
        if len(self.content_digest) != 64:
            raise ValueError("content_digest must be a 64-char SHA-256 hex string")


def _parse_addendum(raw: Mapping[str, Any]) -> GolferUdimAddendum:
    recorded_digest = raw.get("content_digest")
    if not isinstance(recorded_digest, str):
        raise ValueError("content_digest must be a string")
    body = {k: v for k, v in raw.items() if k != "content_digest"}
    expected = content_digest_for_payload(body)
    if recorded_digest != expected:
        raise ValueError(
            "addendum content_digest mismatch: "
            f"file has {recorded_digest}, recomputed {expected}"
        )
    invalidation = raw.get("checkpoint_invalidation", {})
    if not isinstance(invalidation, Mapping):
        raise TypeError("checkpoint_invalidation must be a mapping")
    return GolferUdimAddendum(
        schema=str(raw["schema"]),
        content_digest=recorded_digest,
        governing_issue=int(raw["governing_issue"]),
        effective_date=str(raw["effective_date"]),
        owner_ruling=str(raw["owner_ruling"]),
        model_id=str(raw["model_id"]),
        corrected_q_dim=int(raw["corrected_q_dim"]),
        corrected_independent_dof=int(raw["corrected_independent_dof"]),
        corrected_checkpoint_layout=str(raw["corrected_checkpoint_layout"]),
        supersedes_roster_content_digest=str(raw["supersedes_roster_content_digest"]),
        supersedes_matrix_digest=str(raw["supersedes_matrix_digest"]),
        checkpoint_invalidation_status=str(invalidation.get("status", "")),
        retrain_required=bool(invalidation.get("retrain_required", False)),
        sealed_records_unchanged=bool(raw.get("sealed_records_unchanged", False)),
    )


def load_golfer_udim_addendum(path: Path | None = None) -> GolferUdimAddendum:
    """Load and verify the golfer u_dim addendum JSON."""
    target = path if path is not None else default_addendum_path()
    if not target.is_file():
        raise FileNotFoundError(f"addendum not found: {target}")
    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("addendum root must be a JSON object")
    return _parse_addendum(raw)
