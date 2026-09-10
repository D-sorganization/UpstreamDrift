"""Offline public specifications, deterministic review and player overrides."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.resources import files
from typing import Literal

from .catalog import CatalogModel, ClubRecord, PropertyClaim
from .catalog_io import export_json, import_json


class CatalogChange(CatalogModel):
    """A proposed catalog change; applying it to player libraries is never implicit."""

    catalog_id: str
    kind: Literal["added", "changed", "removed"]
    before_revision: str | None
    after_revision: str | None
    before: str
    after: str


def load_public_catalog() -> tuple[ClubRecord, ...]:
    """Load packaged factual examples without network access or player data writes."""
    resource = files(__package__).joinpath("public_clubs.json")
    records = import_json(resource.read_text(encoding="utf-8"))
    for record in records:
        for claim in record.claims:
            source = claim.source
            if (
                claim.status != "published"
                or source.kind != "manufacturer"
                or not source.url
                or source.retrieved_at is None
            ):
                raise ValueError(
                    "Public catalog claims require inspectable manufacturer evidence"
                )
    return records


def _summary_value(record: ClubRecord, property_name: str, component: str) -> str:
    claims = [
        c
        for c in record.claims
        if c.property == property_name
        and c.component == component
        and c.value is not None
    ]
    if not claims:
        return "Unknown"
    values = [f"{c.value:g} {c.unit} ({c.status})" for c in claims]
    return "; ".join(values) + (
        " — Review Conflicting Sources" if len(claims) > 1 else ""
    )


def summarize_record(record: ClubRecord) -> str:
    """Describe reported values without inventing missing physical properties."""
    identity = record.identity
    lines = [
        f"{identity.manufacturer} {identity.model} {identity.number or ''}".strip(),
        f"Build: {identity.build}",
    ]
    for title, prop, component in (
        ("Length", "length", "assembled"),
        ("Head Mass", "mass", "head"),
        ("Head MOI", "moi", "head"),
        ("Loft", "loft", "head"),
    ):
        lines.append(f"{title}: {_summary_value(record, prop, component)}")
    return "\n".join(lines)


def catalog_diff(
    before: Sequence[ClubRecord], after: Sequence[ClubRecord]
) -> tuple[CatalogChange, ...]:
    """Return sorted additions/changes/removals, including full attributed records."""
    # Shared exchange validation enforces record limits and duplicate identities.
    export_json(before)
    export_json(after)
    old = {record.catalog_id: record for record in before}
    new = {record.catalog_id: record for record in after}
    changes = []
    for key in sorted(old.keys() | new.keys()):
        previous, candidate = old.get(key), new.get(key)
        if previous == candidate:
            continue
        changes.append(
            CatalogChange(
                catalog_id=key,
                kind="added"
                if previous is None
                else "removed"
                if candidate is None
                else "changed",
                before_revision=previous.revision if previous else None,
                after_revision=candidate.revision if candidate else None,
                before=previous.model_dump_json(indent=2) if previous else "",
                after=candidate.model_dump_json(indent=2) if candidate else "",
            )
        )
    return tuple(changes)


def with_player_overrides(
    record: ClubRecord, overrides: Sequence[PropertyClaim]
) -> ClubRecord:
    """Project explicit player measurements onto a catalog revision without mutation.

    Persist overrides separately from the public catalog. Reapply them when a new
    catalog revision is reviewed; publishing a source update cannot erase them.
    A null player override deliberately keeps that quantity unknown.
    """
    keys = [(claim.property, claim.component) for claim in overrides]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate player overrides require explicit resolution")
    if any(c.source.kind not in {"player", "measurement"} for c in overrides):
        raise ValueError("Overrides require player or measurement source attribution")
    retained = tuple(c for c in record.claims if (c.property, c.component) not in keys)
    return ClubRecord(
        identity=record.identity,
        claims=retained + tuple(overrides),
        swing_weight=record.swing_weight,
        notes=record.notes,
    )
