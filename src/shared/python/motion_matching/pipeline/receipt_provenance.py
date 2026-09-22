"""Anthropometric receipt provenance chain validation (#10271).

Hash contract ``receipt-provenance-chain/1``:

* ``base_spec_sha256`` is the *canonical* document digest from
  ``full_body_spec.canonical_sha256`` (semantic identity of the named base).
* ``spec_sha256`` is the *raw file* SHA-256 of the final scaled specification
  bytes as emitted by the producer.
* ``spec_canonical_sha256`` (optional but preferred) is the canonical digest of
  the same final document. When present, a formatting-only reserialization of
  the committed file may change ``spec_sha256`` while preserving semantic
  identity; the chain still passes if the canonical digest matches. A numeric
  change alters the canonical digest and fails closed.
* ``de_leva_table_sha256`` is required on anthropometric hip-calibrated
  receipts and must match the current table, the base document, and the final
  specification.

Intermediate hip-calibrated documents (``full_body_spec_hipcal.json``) are not
fabricated for historical evidence. Pass ``require_hipcal_document=False`` when
only the final scaled specification is retained.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from src.shared.python.motion_matching.anthropometry import de_leva_table_sha256
from src.shared.python.motion_matching.full_body_spec import canonical_sha256

__all__ = [
    "CHAIN_CONTRACT_VERSION",
    "ProvenanceChainInputs",
    "is_sha256_hex",
    "raw_file_sha256",
    "validate_receipt_provenance_chain",
]

CHAIN_CONTRACT_VERSION = "receipt-provenance-chain/1"
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def is_sha256_hex(value: object) -> bool:
    """Return True when ``value`` is a lowercase 64-char hex SHA-256 digest."""
    return isinstance(value, str) and _SHA256_HEX.fullmatch(value) is not None


def raw_file_sha256(path: Path) -> str:
    """SHA-256 of file bytes on disk (raw identity, not canonical JSON)."""
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if not path.is_file():
        raise ValueError(f"File does not exist: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ProvenanceChainInputs:
    """Explicit paths and receipt mapping for chain validation.

    Law of Demeter: callers supply typed paths and the receipt mapping; this
    facade does not reach through pipeline objects or private producers.
    """

    receipt: Mapping[str, Any]
    base_spec_path: Path
    final_spec_path: Path
    hipcal_spec_path: Path | None = None
    require_hipcal_document: bool = False
    require_current_de_leva_table: bool = True
    expected_table_sha256: str | None = None


def _require_sha256_field(receipt: Mapping[str, Any], field: str) -> str:
    value = receipt.get(field)
    if value is None:
        raise ValueError(f"Receipt is missing required '{field}' field")
    if not is_sha256_hex(value):
        raise ValueError(
            f"Receipt field '{field}' must be a lowercase 64-char hex digest"
        )
    return value


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(document, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return document


def _document_table_sha(document: Mapping[str, Any], label: str) -> str:
    table = document.get("de_leva_table_sha256")
    if table is None and isinstance(document.get("subject"), Mapping):
        table = document["subject"].get("de_leva_table_sha256")
    if table is None:
        raise ValueError(f"{label} is missing required 'de_leva_table_sha256'")
    if not is_sha256_hex(table):
        raise ValueError(
            f"{label} de_leva_table_sha256 must be a lowercase 64-char hex digest"
        )
    return table


def _validate_final_spec_identity(
    receipt: Mapping[str, Any],
    final_path: Path,
    final_doc: Mapping[str, Any],
) -> None:
    raw_digest = raw_file_sha256(final_path)
    receipt_raw = _require_sha256_field(receipt, "spec_sha256")
    canonical_digest = canonical_sha256(final_doc)
    receipt_canonical = receipt.get("spec_canonical_sha256")

    if receipt_canonical is None:
        if receipt_raw != raw_digest:
            raise ValueError(
                f"spec_sha256 '{receipt_raw}' does not match raw final spec "
                f"'{raw_digest}' at {final_path}"
            )
        return

    if not is_sha256_hex(receipt_canonical):
        raise ValueError(
            "Receipt field 'spec_canonical_sha256' must be a lowercase 64-char hex digest"
        )
    if receipt_canonical != canonical_digest:
        raise ValueError(
            f"spec_canonical_sha256 '{receipt_canonical}' does not match canonical "
            f"final spec '{canonical_digest}' at {final_path}"
        )
    # Canonical match is semantic acceptance. Raw may differ after formatting-only
    # reserialization (Prettier / indent); producers should still refresh
    # ``spec_sha256`` when rewriting the receipt so byte pins stay current.


def validate_receipt_provenance_chain(inputs: ProvenanceChainInputs) -> None:
    """Validate receipt ↔ base ↔ final (± hipcal) provenance for acceptance.

    Design by Contract (survives ``python -O``; no bare ``assert``):

    - Precondition: ``inputs.receipt`` is a mapping; paths are ``Path`` instances.
    - Precondition: hash fields are lowercase 64-char hex when present.
    - Precondition: base and final documents exist and are JSON objects with
      finite-parseable content (``allow_nan=False`` via ``canonical_sha256``).
    - Postcondition: receipt table hash, base canonical hash, and final raw or
      canonical identity agree with the supplied files and (when requested) the
      current de Leva table.

    Raises:
        TypeError: Wrong input types.
        ValueError: Missing links, stale digests, or unsupported shapes.
    """
    if not isinstance(inputs, ProvenanceChainInputs):
        raise TypeError("inputs must be ProvenanceChainInputs")
    if not isinstance(inputs.receipt, Mapping):
        raise TypeError("receipt must be a mapping")
    if not isinstance(inputs.base_spec_path, Path):
        raise TypeError("base_spec_path must be a pathlib.Path")
    if not isinstance(inputs.final_spec_path, Path):
        raise TypeError("final_spec_path must be a pathlib.Path")
    if inputs.hipcal_spec_path is not None and not isinstance(
        inputs.hipcal_spec_path, Path
    ):
        raise TypeError("hipcal_spec_path must be a pathlib.Path or None")

    expected_table = inputs.expected_table_sha256
    if expected_table is None and inputs.require_current_de_leva_table:
        expected_table = de_leva_table_sha256()
    if expected_table is not None and not is_sha256_hex(expected_table):
        raise ValueError("expected_table_sha256 must be a lowercase 64-char hex digest")

    receipt_table = _require_sha256_field(inputs.receipt, "de_leva_table_sha256")
    if expected_table is not None and receipt_table != expected_table:
        raise ValueError(
            f"Receipt de_leva_table_sha256 '{receipt_table}' does not match "
            f"expected table sha256 '{expected_table}'"
        )

    base_doc = _load_json_object(inputs.base_spec_path, "Base specification")
    base_table = _document_table_sha(base_doc, "Base specification")
    if base_table != receipt_table:
        raise ValueError(
            f"Receipt de_leva_table_sha256 '{receipt_table}' does not match base "
            f"document table '{base_table}' at {inputs.base_spec_path}"
        )

    receipt_base = _require_sha256_field(inputs.receipt, "base_spec_sha256")
    base_canonical = canonical_sha256(base_doc)
    if receipt_base != base_canonical:
        raise ValueError(
            f"base_spec_sha256 '{receipt_base}' does not match canonical base "
            f"'{base_canonical}' at {inputs.base_spec_path}"
        )

    if not inputs.final_spec_path.is_file():
        raise ValueError(f"final spec does not exist: {inputs.final_spec_path}")
    final_doc = _load_json_object(inputs.final_spec_path, "Final specification")
    final_table = _document_table_sha(final_doc, "Final specification")
    if final_table != receipt_table:
        raise ValueError(
            f"Receipt de_leva_table_sha256 '{receipt_table}' does not match final "
            f"specification table '{final_table}' at {inputs.final_spec_path}"
        )
    _validate_final_spec_identity(inputs.receipt, inputs.final_spec_path, final_doc)

    if inputs.hipcal_spec_path is not None and inputs.require_hipcal_document:
        if not inputs.hipcal_spec_path.is_file():
            raise ValueError(
                f"Required hipcal specification is missing: {inputs.hipcal_spec_path}"
            )
        hipcal_doc = _load_json_object(inputs.hipcal_spec_path, "Hipcal specification")
        hipcal_table = _document_table_sha(hipcal_doc, "Hipcal specification")
        if hipcal_table != receipt_table:
            raise ValueError(
                f"Receipt de_leva_table_sha256 '{receipt_table}' does not match "
                f"hipcal table '{hipcal_table}' at {inputs.hipcal_spec_path}"
            )
