"""Frozen, JSON-serialisable receipt records of a spine study (SPN-13a, #12766).

A receipt is the auditable answer to one :class:`SpineStudyRequest`: the SHA-256
of every input, the code version, per-stage provenance, the admission status of
a licensed asset, hashed output artifacts and the headline residual, reserve
and context-band numbers.  It holds basenames and hashes only, never absolute
paths, and carries no timestamp, so the same request yields the same bytes.

``status`` is fail-closed: ``pass`` means only that every stage ran and every
declared software check was met; it never asserts scientific qualification
(``scientific_status`` is always ``unqualified``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = "spine-study-receipt/3"
STATUSES = ("pass", "unqualified", "failed")
STAGE_STATUSES = ("ok", "failed")

__all__ = [
    "SCHEMA_VERSION",
    "FileRecord",
    "StageRecord",
    "SpineStudyReceipt",
    "round_sig",
    "receipt_to_json",
    "write_receipt",
    "load_receipt",
]


def round_sig(value: float, digits: int = 10) -> float:
    """``value`` rounded to ``digits`` significant digits (finite only).

    Postcondition: the result is finite; repeated solves that differ only
    below the rounding level serialise identically.
    """
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"receipt numbers must be finite, got {value!r}")
    return float(f"{number:.{digits}g}")


@dataclass(frozen=True)
class FileRecord:
    """One hashed file: role, basename and lower-case hex SHA-256."""

    role: str
    name: str
    sha256: str

    def __post_init__(self) -> None:
        if len(self.sha256) != 64 or self.sha256 != self.sha256.lower():
            raise ValueError(f"sha256 of {self.name!r} must be 64 lower-case hex")
        if "/" in self.name or "\\" in self.name:
            raise ValueError(f"name must be a basename, got {self.name!r}")

    def as_dict(self) -> dict[str, str]:
        """JSON-compatible form."""
        return {"role": self.role, "name": self.name, "sha256": self.sha256}


@dataclass(frozen=True)
class StageRecord:
    """Outcome of one pipeline stage: provenance strings and scalar metrics."""

    stage: str
    status: Literal["ok", "failed"]
    provenance: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def __post_init__(self) -> None:
        if self.status not in STAGE_STATUSES:
            raise ValueError(f"stage status must be one of {STAGE_STATUSES}")

    def as_dict(self) -> dict[str, Any]:
        """JSON-compatible form."""
        return {
            "stage": self.stage,
            "status": self.status,
            "provenance": dict(self.provenance),
            "metrics": dict(self.metrics),
            "error": self.error,
        }


@dataclass(frozen=True)
class SpineStudyReceipt:
    """Request -> receipt record; see the module docstring.

    Preconditions: ``status`` is one of :data:`STATUSES`; ``request_sha256`` is
    64 hex characters; ``torso_alignment`` is the ``{"matrix", "provenance"}``
    record of the drive alignment and ``pass`` requires a ``calibrated:``
    provenance.  ``asset`` is the :class:`SpineAssetReceipt` dictionary
    or ``None`` when the study names no licensed asset.  ``variant`` is the
    movement-variant provenance (SPN-13d) or ``None``; it is omitted from the
    serialised receipt when ``None`` so variant-free receipts are unchanged.
    """

    study_id: str
    status: Literal["pass", "unqualified", "failed"]
    status_reasons: tuple[str, ...]
    request: dict[str, Any]
    request_sha256: str
    inputs: tuple[FileRecord, ...]
    code_version: str
    opensim_version: str
    scipy_version: str
    numpy_version: str
    law: str
    distribution_source: str
    asset: dict[str, Any] | None
    torso_alignment: dict[str, Any]
    stages: tuple[StageRecord, ...]
    artifacts: tuple[FileRecord, ...]
    residual: dict[str, Any]
    reserve: dict[str, Any]
    context_comparisons: tuple[dict[str, Any], ...]
    scientific_status: str = "unqualified"
    schema_version: str = SCHEMA_VERSION
    variant: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")
        if len(self.request_sha256) != 64:
            raise ValueError("request_sha256 must be 64 hex characters")
        provenance = str(self.torso_alignment.get("provenance", ""))
        if self.status == "pass" and not provenance.startswith("calibrated:"):
            raise ValueError(
                "a receipt without a calibrated torso alignment cannot pass "
                f"(provenance {provenance!r})"
            )
        if self.status == "pass" and (self.asset or {}).get("status", "qualified") != (
            "qualified"
        ):
            raise ValueError("a receipt with a non-qualified asset cannot pass")

    def as_dict(self) -> dict[str, Any]:
        """JSON-compatible form (no paths, no timestamps)."""
        data = {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "status": self.status,
            "status_reasons": list(self.status_reasons),
            "scientific_status": self.scientific_status,
            "request": self.request,
            "request_sha256": self.request_sha256,
            "inputs": [f.as_dict() for f in self.inputs],
            "code_version": self.code_version,
            "opensim_version": self.opensim_version,
            "scipy_version": self.scipy_version,
            "numpy_version": self.numpy_version,
            "law": self.law,
            "distribution_source": self.distribution_source,
            "asset": self.asset,
            "torso_alignment": self.torso_alignment,
            "stages": [s.as_dict() for s in self.stages],
            "artifacts": [a.as_dict() for a in self.artifacts],
            "residual": self.residual,
            "reserve": self.reserve,
            "context_comparisons": list(self.context_comparisons),
        }
        if self.variant is not None:
            data["variant"] = self.variant
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpineStudyReceipt:
        """Rebuild from :meth:`as_dict` output.

        Raises:
            ValueError: unknown ``schema_version`` or a missing key.
        """
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema {data.get('schema_version')!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )
        try:
            return cls(
                study_id=data["study_id"],
                status=data["status"],
                status_reasons=tuple(data["status_reasons"]),
                request=data["request"],
                request_sha256=data["request_sha256"],
                inputs=tuple(FileRecord(**f) for f in data["inputs"]),
                code_version=data["code_version"],
                opensim_version=data["opensim_version"],
                scipy_version=data["scipy_version"],
                numpy_version=data["numpy_version"],
                law=data["law"],
                distribution_source=data["distribution_source"],
                asset=data["asset"],
                torso_alignment=data["torso_alignment"],
                stages=tuple(StageRecord(**s) for s in data["stages"]),
                artifacts=tuple(FileRecord(**a) for a in data["artifacts"]),
                residual=data["residual"],
                reserve=data["reserve"],
                context_comparisons=tuple(data["context_comparisons"]),
                scientific_status=data["scientific_status"],
                variant=data.get("variant"),
            )
        except KeyError as exc:
            raise ValueError(f"receipt lacks key {exc.args[0]!r}") from exc


def receipt_to_json(receipt: SpineStudyReceipt) -> str:
    """Canonical JSON text: sorted keys, 2-space indent, trailing newline."""
    if not isinstance(receipt, SpineStudyReceipt):
        raise TypeError("receipt must be a SpineStudyReceipt")
    return json.dumps(receipt.as_dict(), sort_keys=True, indent=2, allow_nan=False) + (
        "\n"
    )


def write_receipt(receipt: SpineStudyReceipt, path: str | Path) -> Path:
    """Write the canonical JSON to ``path`` (parents created) and return it."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(receipt_to_json(receipt), encoding="utf-8", newline="\n")
    return out


def load_receipt(path: str | Path) -> SpineStudyReceipt:
    """Read a receipt written by :func:`write_receipt`.

    Raises:
        ValueError: malformed JSON content or an unsupported schema.
        OSError: the file cannot be read.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("receipt JSON must be an object")
    return SpineStudyReceipt.from_dict(raw)
