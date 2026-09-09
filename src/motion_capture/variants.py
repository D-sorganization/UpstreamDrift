"""Variants: several matches of one session, side by side (#9793).

A variant is a named match of the same recordings: its own ``reconstruct/``
and ``model/`` trees under ``variants/<name>/``, sharing the session's
recordings and observation sets. The default variant (``name == ""``) is
the session root itself, so every existing path keeps working. Every
command that reads or writes reconstruct/model outputs resolves its root
through :func:`variant_dir`; nothing else knows the layout.

``variants/index.json`` (``variants-index/1.0.0``) lists the variants with
the views and observation set they used and how they were matched:
``{"kind": "triangulate"}`` or ``{"kind": "image_space", "cameras_from":
<variant>}``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.shared.python.core.contracts import require

from .provenance import now_utc, stamp, write_json

VARIANTS_DIR = "variants"
INDEX_FILE = "index.json"
INDEX_SCHEMA = "variants-index/1.0.0"
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
DEFAULT = ""


def is_valid_name(name: str) -> bool:
    return name == DEFAULT or bool(NAME_PATTERN.match(name))


def variant_dir(session: Path, name: str = DEFAULT) -> Path:
    """Root of a variant's reconstruct/model trees (the session for ``""``).

    Precondition: a valid name (``[A-Za-z0-9_-]{1,40}`` or empty).
    """
    require(is_valid_name(name), "invalid variant name", name)
    return session if name == DEFAULT else session / VARIANTS_DIR / name


def ensure_variant(session: Path, name: str = DEFAULT) -> Path:
    """:func:`variant_dir`, created on demand."""
    root = variant_dir(session, name)
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass(frozen=True)
class VariantRecord:
    name: str
    views: tuple[str, ...]
    observation_set: str
    source: dict[str, Any]
    created_utc: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "views": list(self.views),
            "observation_set": self.observation_set,
            "source": dict(self.source),
            "created_utc": self.created_utc,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VariantRecord:
        known = {"name", "views", "observation_set", "source", "created_utc"}
        return cls(
            name=str(payload["name"]),
            views=tuple(str(v) for v in payload.get("views", ())),
            observation_set=str(payload.get("observation_set", "observations")),
            source=dict(payload.get("source") or {"kind": "triangulate"}),
            created_utc=payload.get("created_utc"),
            extra={k: v for k, v in payload.items() if k not in known},
        )


def _index_path(session: Path) -> Path:
    return session / VARIANTS_DIR / INDEX_FILE


def _read_index(session: Path) -> dict[str, Any]:
    path = _index_path(session)
    if not path.is_file():
        return {"schema_version": INDEX_SCHEMA, "variants": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(
        payload.get("schema_version") == INDEX_SCHEMA,
        "unknown variants index schema",
        payload.get("schema_version"),
    )
    return payload


def register_variant(
    session: Path,
    name: str,
    *,
    views: Sequence[str],
    observation_set: str = "observations",
    source: Mapping[str, Any] | None = None,
    module: str = __name__,
    **extra: Any,
) -> VariantRecord:
    """Add or replace ``name`` in ``variants/index.json``; returns the record.

    Preconditions: valid name; at least one view; ``source["kind"]`` is
    ``triangulate`` or ``image_space``. Postcondition: the index is stamped
    with provenance and lists the variant.
    """
    require(is_valid_name(name), "invalid variant name", name)
    require(len(views) >= 1, "a variant uses at least one view", views)
    src = dict(source or {"kind": "triangulate"})
    require(
        src.get("kind") in ("triangulate", "image_space"),
        "source kind must be triangulate or image_space",
        src,
    )
    record = VariantRecord(
        name=name,
        views=tuple(views),
        observation_set=observation_set,
        source=src,
        created_utc=now_utc(),
        extra=dict(extra),
    )
    index = _read_index(session)
    index["variants"][name] = record.to_dict()
    stamped = stamp(
        {"schema_version": INDEX_SCHEMA, "variants": index["variants"]},
        schema_version=INDEX_SCHEMA,
        module=module,
        parameters={"count": len(index["variants"])},
        base=session,
    )
    write_json(_index_path(session), stamped)
    return record


def list_variants(session: Path) -> list[VariantRecord]:
    """Registered variants, the default first, then by name."""
    payload = _read_index(session)
    records = [VariantRecord.from_dict(v) for v in payload["variants"].values()]
    return sorted(records, key=lambda r: (r.name != DEFAULT, r.name))


def get_variant(session: Path, name: str) -> VariantRecord | None:
    for record in list_variants(session):
        if record.name == name:
            return record
    return None
