"""Provenance records for displayed values (epic #5968, Phase 0.4).

When any number, label, or computed display string is rendered in the
UI, it should carry a :class:`ProvenanceRecord` describing where it
came from: the formula or source, the field ids that fed it, the
engine + run id that produced it, and when. Widgets in
``src.shared.python.ui`` and ``ui/src/components/ux`` wrap raw values
in :class:`ProvenanceValue` so a single hover or right-click can
answer "why does this say 500?" without leaving the screen.

This module is pure data + validation; no Qt or React import lives
here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.shared.python.contracts import require

_FIELD_ID: re.Pattern[str] = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$")


class ProvenanceError(ValueError):
    """Raised for any invalid provenance input."""


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """Where a displayed value came from.

    Attributes
    ----------
    formula
        Human-readable derivation, e.g. ``"fps = 1.0 / timestep"`` or
        ``"sum of joint kinetic energies"`` or ``"constant 9.81"``.
    inputs
        Tuple of field ids consumed by the formula, in canonical
        ``simulation.timestep`` form.  Empty for true constants.
    source
        Identifier that uniquely names the producing computation,
        typically ``"<engine>:<run_id>"`` or
        ``"<engine>:<step_index>"``.  Echoed in the popover so the
        user can trace which run a stale number came from.
    computed_at
        Timezone-aware timestamp.
    engine
        Engine name (``"mujoco"``, ``"drake"``, ``"pinocchio"``, …).
    run_id
        Run identifier; may equal the suffix of ``source``.
    """

    formula: str
    inputs: tuple[str, ...]
    source: str
    computed_at: datetime
    engine: str
    run_id: str

    def __post_init__(self) -> None:
        _validate_record(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "formula": self.formula,
            "inputs": list(self.inputs),
            "source": self.source,
            "computed_at": self.computed_at.isoformat(),
            "engine": self.engine,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProvenanceRecord:
        require(
            isinstance(payload, Mapping),
            "ProvenanceRecord.from_dict expects a mapping",
            payload,
        )
        try:
            return cls(
                formula=str(payload["formula"]),
                inputs=_coerce_ids(payload.get("inputs", ())),
                source=str(payload["source"]),
                computed_at=_parse_dt(payload["computed_at"]),
                engine=str(payload["engine"]),
                run_id=str(payload["run_id"]),
            )
        except KeyError as exc:
            raise ProvenanceError(
                f"ProvenanceRecord missing required key: {exc.args[0]!r}"
            ) from exc


def _coerce_ids(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise ProvenanceError(
            f"inputs must be a sequence of ids, got {type(value).__name__}"
        )
    return tuple(str(v) for v in value)


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError as exc:
            raise ProvenanceError(f"computed_at not ISO 8601: {value!r}") from exc
    raise ProvenanceError(
        f"computed_at must be datetime or ISO string, got {type(value).__name__}"
    )


def _validate_record(rec: ProvenanceRecord) -> None:
    if not rec.formula:
        raise ProvenanceError("formula must be non-empty")
    if not rec.source:
        raise ProvenanceError("source must be non-empty")
    if not rec.engine:
        raise ProvenanceError("engine must be non-empty")
    if not rec.run_id:
        raise ProvenanceError("run_id must be non-empty")
    for input_id in rec.inputs:
        if not _FIELD_ID.match(input_id):
            raise ProvenanceError(f"inputs contains non-dotted-id {input_id!r}")
    if rec.computed_at.tzinfo is None:
        raise ProvenanceError("computed_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class ProvenanceValue:
    """A displayable value bundled with its :class:`ProvenanceRecord`.

    ``value`` may be any JSON-serialisable scalar.  ``display_units``
    is the unit the value is presented in (which may differ from the
    canonical unit — e.g. degrees for display, radians internally).
    """

    value: Any
    record: ProvenanceRecord
    display_units: str = ""
    label: str = field(default="")

    def __post_init__(self) -> None:
        if not isinstance(self.record, ProvenanceRecord):
            raise ProvenanceError(
                f"record must be a ProvenanceRecord, got {type(self.record).__name__}"
            )

    def describe(self) -> str:
        """Return a single multi-line string for tooltips / popovers."""
        if self.record.inputs:
            inputs_line = "inputs: " + ", ".join(self.record.inputs)
        else:
            inputs_line = "(no inputs)"
        unit = f" {self.display_units}" if self.display_units else ""
        return (
            f"value: {self.value}{unit}\n"
            f"formula: {self.record.formula}\n"
            f"{inputs_line}\n"
            f"source: {self.record.source}\n"
            f"computed at: {self.record.computed_at.isoformat()}"
        )


__all__ = ["ProvenanceError", "ProvenanceRecord", "ProvenanceValue"]
