"""Typed metadata for every user-facing input (epic #5968, Phase 0.1).

A :class:`FieldMetadata` describes one input field: its label, tooltip
copy, units, valid range, default, where the default comes from, and
the producer/consumer edges that link it to other fields. Widgets in
both PyQt6 and React consume the same data via a YAML registry so that
help copy lives in one place (DRY) and can be reviewed by non-coders.

Design notes
------------
* Frozen, hashable dataclass — usable as a dict key and as a stable
  identity in coverage ratchets without defensive copies (DRY).
* All public constructors enforce invariants via the shared DbC
  primitives (:func:`src.shared.python.contracts.require`) so bad data
  raises with a descriptive message at the boundary (DbC).
* :class:`FieldRegistry` returns flat tuples of :class:`FieldMetadata`
  objects from its graph queries (LoD: callers do not reach through
  ``registry._fields[id].consumers[0].label``).
* The loader runs the full validation suite (id format, ordered
  numeric range, default within range, no duplicate ids, every
  declared consumer/producer exists, acyclic graph) so the rest of the
  codebase can treat any registry it receives as already-correct.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.shared.python.contracts import require

_ID_PATTERN: re.Pattern[str] = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$")
_SHORT_HELP_MAX_CHARS: int = 80


class FieldMetadataError(ValueError):
    """Raised for any invalid field metadata or registry input.

    Subclasses :class:`ValueError` so that generic input-validation
    handlers in the API layer (which already catch ``ValueError``)
    surface the message without a special-case branch.
    """


# Numeric ranges are ``(min, max)`` tuples; enum ranges are tuples of
# allowed string values.  ``None`` means the field is free-form.
ValidRange = tuple[float, float] | tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class FieldMetadata:
    """Describe one user-facing input field.

    Parameters
    ----------
    id
        Stable dotted identifier (``r"^[a-z][a-z0-9_]*(\\.[a-z0-9_]+)+$"``).
        Example: ``"simulation.timestep"``.  Cannot change once
        shipped — coverage ratchets and translations key off it.
    label
        Short human-readable label shown next to the input.
    short_help
        Tooltip text, ≤ 80 chars.  One sentence.
    long_help
        Markdown body shown in the ``[?]`` popover.  Reading-level
        lint applies (Phase 6).
    units
        Unit symbol (``"s"``, ``"rad"``, ``"kg"``).  ``None`` for
        unitless fields and enums.
    valid_range
        ``(min, max)`` for numerics; tuple of allowed strings for
        enums; ``None`` for free-form text.
    default
        Default value.  Must satisfy ``valid_range`` if one is
        declared.
    default_source
        Free-text attribution for the default (paper, doc URL,
        ``"internal benchmark on …"``).  Surfaced in the tooltip.
    consumers
        IDs of downstream fields that read this value.  Used by the
        cross-sheet linkage UI (Phase 3).
    producers
        IDs of upstream fields that feed this value.
    example
        A representative valid value, rendered in the popover so the
        user has something to anchor on.
    """

    id: str
    label: str
    short_help: str
    long_help: str
    units: str | None
    valid_range: ValidRange
    default: Any
    default_source: str
    consumers: tuple[str, ...] = field(default=())
    producers: tuple[str, ...] = field(default=())
    example: str = ""

    def __post_init__(self) -> None:
        _validate_field_metadata(self)

    # ---- conversion -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-friendly mapping.

        ``valid_range`` is stored as a list (YAML-friendly) and tuples
        of consumers/producers become lists too.  ``from_dict`` is the
        exact inverse.
        """
        return {
            "id": self.id,
            "label": self.label,
            "short_help": self.short_help,
            "long_help": self.long_help,
            "units": self.units,
            "valid_range": (
                list(self.valid_range) if self.valid_range is not None else None
            ),
            "default": self.default,
            "default_source": self.default_source,
            "consumers": list(self.consumers),
            "producers": list(self.producers),
            "example": self.example,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FieldMetadata:
        """Inverse of :meth:`to_dict`.

        Raises :class:`FieldMetadataError` if a required key is
        missing or has the wrong shape (DbC at the I/O boundary).
        """
        require(
            isinstance(payload, Mapping),
            "FieldMetadata.from_dict expects a mapping",
            payload,
        )
        try:
            return cls(
                id=str(payload["id"]),
                label=str(payload["label"]),
                short_help=str(payload["short_help"]),
                long_help=str(payload["long_help"]),
                units=_optional_str(payload.get("units")),
                valid_range=_coerce_valid_range(payload.get("valid_range")),
                default=payload["default"],
                default_source=str(payload["default_source"]),
                consumers=_coerce_str_tuple(payload.get("consumers", ())),
                producers=_coerce_str_tuple(payload.get("producers", ())),
                example=str(payload.get("example", "")),
            )
        except KeyError as exc:
            raise FieldMetadataError(
                f"FieldMetadata missing required key: {exc.args[0]!r}"
            ) from exc


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _coerce_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise FieldMetadataError(
            f"expected a list of strings, got the string {value!r}"
        )
    if not isinstance(value, Iterable):
        raise FieldMetadataError(f"expected an iterable, got {type(value).__name__}")
    return tuple(str(v) for v in value)


def _coerce_valid_range(value: Any) -> ValidRange:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise FieldMetadataError(
            f"valid_range must be a sequence, got {type(value).__name__}"
        )
    items = list(value)
    if not items:
        raise FieldMetadataError("valid_range may not be an empty sequence")
    if all(isinstance(v, str) for v in items):
        return tuple(items)
    if len(items) != 2:
        raise FieldMetadataError(
            "numeric valid_range must have exactly two entries (min, max)"
        )
    try:
        lo, hi = float(items[0]), float(items[1])
    except (TypeError, ValueError) as exc:
        raise FieldMetadataError(
            f"numeric valid_range entries must be numeric: {items!r}"
        ) from exc
    return (lo, hi)


def _validate_field_metadata(fm: FieldMetadata) -> None:
    """Single source of truth for FieldMetadata invariants (DbC, DRY).

    Centralising validation here means every construction path
    (direct, ``from_dict``, YAML loader) gets the same checks.
    """
    if not isinstance(fm.id, str) or not _ID_PATTERN.match(fm.id):
        raise FieldMetadataError(
            f"id must match {_ID_PATTERN.pattern!r}, got {fm.id!r}"
        )
    if not fm.label:
        raise FieldMetadataError(f"{fm.id}: label must be non-empty")
    if not fm.short_help:
        raise FieldMetadataError(f"{fm.id}: short_help must be non-empty")
    if len(fm.short_help) > _SHORT_HELP_MAX_CHARS:
        raise FieldMetadataError(
            f"{fm.id}: short_help exceeds {_SHORT_HELP_MAX_CHARS} chars "
            f"(got {len(fm.short_help)})"
        )
    if not fm.long_help:
        raise FieldMetadataError(f"{fm.id}: long_help must be non-empty")
    if not fm.default_source:
        raise FieldMetadataError(f"{fm.id}: default_source must be non-empty")
    if fm.units is not None and not fm.units:
        raise FieldMetadataError(f"{fm.id}: units must be a non-empty string or None")
    _validate_range_and_default(fm)
    _validate_id_tuple(fm.id, "consumers", fm.consumers)
    _validate_id_tuple(fm.id, "producers", fm.producers)
    if fm.id in fm.consumers or fm.id in fm.producers:
        raise FieldMetadataError(f"{fm.id}: field cannot consume/produce itself")


def _validate_range_and_default(fm: FieldMetadata) -> None:
    rng = fm.valid_range
    if rng is None:
        return
    if _is_enum_range(rng):
        if fm.default not in rng:
            raise FieldMetadataError(
                f"{fm.id}: default {fm.default!r} not in enum {rng!r}"
            )
        return
    lo, hi = rng  # type: ignore[misc]
    if lo > hi:
        raise FieldMetadataError(
            f"{fm.id}: numeric valid_range is reversed: ({lo}, {hi})"
        )
    if not isinstance(fm.default, (int, float)) or isinstance(fm.default, bool):
        raise FieldMetadataError(
            f"{fm.id}: numeric range requires numeric default, "
            f"got {type(fm.default).__name__}"
        )
    if not lo <= float(fm.default) <= hi:
        raise FieldMetadataError(
            f"{fm.id}: default {fm.default} outside range [{lo}, {hi}]"
        )


def _is_enum_range(rng: ValidRange) -> bool:
    return rng is not None and bool(rng) and all(isinstance(v, str) for v in rng)


def _validate_id_tuple(owner_id: str, label: str, ids: tuple[str, ...]) -> None:
    seen: set[str] = set()
    for other in ids:
        if not _ID_PATTERN.match(other):
            raise FieldMetadataError(
                f"{owner_id}: {label} contains malformed id {other!r}"
            )
        if other in seen:
            raise FieldMetadataError(
                f"{owner_id}: {label} contains duplicate {other!r}"
            )
        seen.add(other)


# ---- registry --------------------------------------------------------


class FieldRegistry:
    """A validated collection of :class:`FieldMetadata` objects.

    Construction performs full cross-field validation: no duplicate
    ids, every declared consumer/producer points at a real field, and
    the consumer graph is acyclic.  After construction, lookups are
    O(1) and graph queries return tuples of full
    :class:`FieldMetadata` objects (LoD).
    """

    __slots__ = ("_fields", "_consumers_by", "_producers_by")

    def __init__(self, fields: Iterable[FieldMetadata]) -> None:
        as_tuple = tuple(fields)
        require(
            all(isinstance(f, FieldMetadata) for f in as_tuple),
            "FieldRegistry only accepts FieldMetadata instances",
            as_tuple,
        )
        by_id: dict[str, FieldMetadata] = {}
        for fm in as_tuple:
            if fm.id in by_id:
                raise FieldMetadataError(f"duplicate field id: {fm.id!r}")
            by_id[fm.id] = fm
        _validate_graph(by_id)
        self._fields: dict[str, FieldMetadata] = dict(
            sorted(by_id.items(), key=lambda kv: kv[0])
        )
        self._consumers_by: dict[str, tuple[FieldMetadata, ...]] = {
            fid: tuple(self._fields[c] for c in fm.consumers)
            for fid, fm in self._fields.items()
        }
        self._producers_by: dict[str, tuple[FieldMetadata, ...]] = {
            fid: tuple(self._fields[p] for p in fm.producers)
            for fid, fm in self._fields.items()
        }

    def get(self, field_id: str) -> FieldMetadata:
        """Return the field with ``field_id``; raise :class:`KeyError`."""
        try:
            return self._fields[field_id]
        except KeyError as exc:
            raise KeyError(f"unknown field id: {field_id!r}") from exc

    def iter_fields(self) -> Iterator[FieldMetadata]:
        """Yield fields in id-sorted order (deterministic)."""
        return iter(self._fields.values())

    def consumers_of(self, field_id: str) -> tuple[FieldMetadata, ...]:
        """Return the downstream fields that read ``field_id``."""
        self.get(field_id)  # raise if unknown
        return self._consumers_by[field_id]

    def producers_of(self, field_id: str) -> tuple[FieldMetadata, ...]:
        """Return the upstream fields that feed ``field_id``."""
        self.get(field_id)
        return self._producers_by[field_id]

    def __contains__(self, field_id: object) -> bool:
        return isinstance(field_id, str) and field_id in self._fields

    def __len__(self) -> int:
        return len(self._fields)

    def __iter__(self) -> Iterator[FieldMetadata]:
        return self.iter_fields()


def _validate_graph(by_id: dict[str, FieldMetadata]) -> None:
    """Cross-field validation: id refs exist and graph is acyclic."""
    for fm in by_id.values():
        for other in fm.consumers:
            if other not in by_id:
                raise FieldMetadataError(
                    f"{fm.id}: declared consumer {other!r} does not exist"
                )
        for other in fm.producers:
            if other not in by_id:
                raise FieldMetadataError(
                    f"{fm.id}: declared producer {other!r} does not exist"
                )
    _ensure_acyclic(by_id)
    _ensure_consumer_producer_symmetry(by_id)


def _ensure_acyclic(by_id: dict[str, FieldMetadata]) -> None:
    """DFS-based cycle check across the consumer graph."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = dict.fromkeys(by_id, WHITE)

    def visit(node: str, stack: tuple[str, ...]) -> None:
        if color[node] == GRAY:
            raise FieldMetadataError(
                f"cycle detected in field graph: {' -> '.join((*stack, node))}"
            )
        if color[node] == BLACK:
            return
        color[node] = GRAY
        for child in by_id[node].consumers:
            visit(child, (*stack, node))
        color[node] = BLACK

    for start in by_id:
        if color[start] == WHITE:
            visit(start, ())


def _ensure_consumer_producer_symmetry(by_id: dict[str, FieldMetadata]) -> None:
    """If A.consumers contains B, B.producers must contain A.

    Symmetry isn't required (an author can declare only one side and
    we infer the other), but if both are declared they must agree —
    silent mismatch hides cross-sheet links from the UI.
    """
    for fm in by_id.values():
        for child_id in fm.consumers:
            child = by_id[child_id]
            if child.producers and fm.id not in child.producers:
                raise FieldMetadataError(
                    f"{fm.id} lists {child_id} as a consumer, but "
                    f"{child_id}.producers does not include {fm.id}"
                )


# ---- YAML loader -----------------------------------------------------


def load_registry(path: str | Path) -> FieldRegistry:
    """Load ``path`` (a YAML file) and return a validated registry.

    Schema::

        fields:
          - id: <dotted.lower.id>
            label: ...
            short_help: ...
            long_help: ...
            units: <symbol or null>
            valid_range: [min, max] | [enum, values, ...] | null
            default: <value>
            default_source: ...
            consumers: [<id>, ...]
            producers: [<id>, ...]
            example: ...
    """
    path = Path(path)
    require(path.is_file(), f"field metadata YAML not found: {path}", path)
    raw_text = path.read_text(encoding="utf-8")
    payload = yaml.safe_load(raw_text) or {}
    if not isinstance(payload, Mapping):
        raise FieldMetadataError(
            f"{path}: top-level YAML must be a mapping, got {type(payload).__name__}"
        )
    entries = payload.get("fields", [])
    if not isinstance(entries, Sequence):
        raise FieldMetadataError(
            f"{path}: 'fields' must be a sequence, got {type(entries).__name__}"
        )
    fields_built = tuple(FieldMetadata.from_dict(entry) for entry in entries)
    return FieldRegistry(fields_built)


__all__ = [
    "FieldMetadata",
    "FieldMetadataError",
    "FieldRegistry",
    "ValidRange",
    "load_registry",
]
