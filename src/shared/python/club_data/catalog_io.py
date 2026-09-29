"""Bounded, lossless JSON/CSV exchange for attributed club specifications.

CSV uses one row per claim and flat identity/property columns. Source metadata
is a JSON cell to retain attribution without a second relational file.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from typing import Any, Final, Literal

from pydantic import Field

from .catalog import CatalogModel, ClubIdentity, ClubRecord, PropertyClaim

SCHEMA_VERSION: Final = "upstreamdrift/club-catalog/1"
MAX_EXCHANGE_BYTES = 8 * 1024 * 1024
MAX_RECORDS = 10000
_IDENTITY_FIELDS = tuple(ClubIdentity.model_fields)
_CLAIM_FIELDS = tuple(PropertyClaim.model_fields)
_COLUMNS = (
    "schema_version",
    "catalog_id",
    "swing_weight",
    "record_notes",
    *_IDENTITY_FIELDS,
    *_CLAIM_FIELDS,
)


class _Exchange(CatalogModel):
    schema_version: Literal["upstreamdrift/club-catalog/1"] = SCHEMA_VERSION
    records: tuple[ClubRecord, ...] = Field(max_length=MAX_RECORDS)


def _check_text(text: str) -> None:
    if len(text.encode("utf-8")) > MAX_EXCHANGE_BYTES:
        raise ValueError("Club catalog exceeds the 8 MiB exchange limit")


def _unique(records: Sequence[ClubRecord]) -> tuple[ClubRecord, ...]:
    if len(records) > MAX_RECORDS:
        raise ValueError("Too many club catalog records")
    ids = [record.catalog_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(
            "Catalog contains duplicate build identities; preserve conflicts as claims"
        )
    return tuple(records)


def export_json(records: Sequence[ClubRecord]) -> str:
    """Return a versioned document; refuse duplicates and oversized payloads."""
    result = _Exchange(records=_unique(records)).model_dump_json(indent=2)
    _check_text(result)
    return result


def import_json(text: str) -> tuple[ClubRecord, ...]:
    """Validate the entire document before returning records for a merge."""
    _check_text(text)
    return _unique(_Exchange.model_validate_json(text).records)


def export_csv(records: Sequence[ClubRecord]) -> str:
    """Return one CSV row per claim, retaining empty records and null values."""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for record in _unique(records):
        base = {
            "schema_version": SCHEMA_VERSION,
            "catalog_id": record.catalog_id,
            "swing_weight": record.swing_weight,
            "record_notes": record.notes,
            **record.identity.model_dump(mode="json"),
        }
        for claim in record.claims or (None,):
            fields = claim.model_dump(mode="json") if claim is not None else {}
            if fields:
                fields["source"] = json.dumps(fields["source"], ensure_ascii=False)
            writer.writerow(base | fields)
    result = stream.getvalue()
    _check_text(result)
    return result


def _parse_row(row: dict[str, str]) -> tuple[str, dict[str, Any], PropertyClaim | None]:
    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported club catalog CSV schema version")
    identity: dict[str, Any] = {key: row[key] for key in _IDENTITY_FIELDS}
    identity["release_year"] = row["release_year"] or None
    identity["number"] = row["number"] or None
    validated = ClubIdentity.model_validate(identity)
    base = {
        "identity": validated,
        "swing_weight": row["swing_weight"] or None,
        "notes": row["record_notes"],
    }
    if ClubRecord.model_validate(base).catalog_id != row["catalog_id"]:
        raise ValueError("Catalog ID does not match the row's build identity")
    if not row["property"]:
        if any(row[key] for key in _CLAIM_FIELDS):
            raise ValueError("An empty claim row contains property data")
        return row["catalog_id"], base, None
    fields: dict[str, Any] = {key: row[key] for key in _CLAIM_FIELDS}
    for key in ("value", "confidence", "axis", "frame", "origin"):
        fields[key] = fields[key] or None
    fields["source"] = json.loads(row["source"])
    return row["catalog_id"], base, PropertyClaim.model_validate(fields)


def import_csv(text: str) -> tuple[ClubRecord, ...]:
    """Validate every row and refuse mismatched builds or silently lost columns."""
    _check_text(text)
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != list(_COLUMNS):
        raise ValueError("CSV columns do not match the club catalog schema")
    groups: dict[str, tuple[dict[str, Any], list[PropertyClaim]]] = {}
    empty_ids: set[str] = set()
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("CSV row has the wrong number of columns")
        key, base, claim = _parse_row(row)
        if key in groups:
            if groups[key][0] != base or key in empty_ids or claim is None:
                raise ValueError("Conflicting or duplicate catalog identity rows")
        else:
            groups[key] = (base, [])
        if claim is None:
            empty_ids.add(key)
        else:
            groups[key][1].append(claim)
    return _unique(
        [
            ClubRecord.model_validate(base | {"claims": claims})
            for base, claims in groups.values()
        ]
    )
