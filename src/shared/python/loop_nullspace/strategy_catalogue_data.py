"""Serialise the NSX-95 strategy catalogue as JSON-safe data (#12724).

The authoritative registry remains :data:`~.strategy_spec.CATALOGUE`; this module
exports it for ``src/config/nsx_strategy_catalogue.json`` and contract tests.
No golfer measurements or published effect sizes are added here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from src.shared.python.loop_nullspace.strategy_spec import (
    CATALOGUE,
    StrategyEntry,
    StrategyId,
)

__all__ = [
    "CONFIG_PATH",
    "export_catalogue",
    "load_catalogue_json",
    "validate_catalogue_payload",
]

_REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = _REPO_ROOT / "src" / "config" / "nsx_strategy_catalogue.json"


def export_catalogue() -> list[dict[str, object]]:
    """One row per S1–S12 for JSON export."""
    return [_entry_dict(CATALOGUE[sid]) for sid in StrategyId]


def _entry_dict(entry: StrategyEntry) -> dict[str, object]:
    return {
        "id": entry.id.value,
        "name": entry.name,
        "mechanism": entry.mechanism,
        "mechanism_ids": list(entry.mechanism_ids),
        "availability": entry.availability.value,
        "cost": {"basis": entry.cost.basis, "description": entry.cost.description},
        "evaluators": list(entry.evaluators),
        "adapter": entry.adapter,
        "representations": [
            {
                "model_class": r.model_class.value,
                "availability": r.availability.value,
                "reason": r.reason,
            }
            for r in entry.representations
        ],
    }


def validate_catalogue_payload(payload: object) -> None:
    """``ValueError`` when ``payload`` is not twelve well-formed catalogue rows."""
    if not isinstance(payload, list) or len(payload) != 12:
        raise ValueError("catalogue payload must be a list of twelve entries")
    ids = [row.get("id") for row in payload if isinstance(row, dict)]
    if ids != [f"S{i}" for i in range(1, 13)]:
        raise ValueError("catalogue ids must be S1..S12 in order")


def load_catalogue_json(path: Path | None = None) -> list[dict[str, object]]:
    """Read and validate the committed JSON catalogue."""
    target = path or CONFIG_PATH
    raw = json.loads(target.read_text(encoding="utf-8"))
    validate_catalogue_payload(raw)
    return raw


def catalogue_matches_config(
    payload: Mapping[str, object] | list[dict[str, object]],
) -> bool:
    """True when every exported field matches ``payload`` (by id)."""
    exported = {str(row["id"]): row for row in export_catalogue()}
    if isinstance(payload, list):
        by_id = {str(row["id"]): row for row in payload if isinstance(row, dict)}
    else:
        by_id = {str(k): v for k, v in payload.items() if isinstance(v, dict)}
    if set(by_id) != set(exported):
        return False
    for sid, row in exported.items():
        other = by_id[sid]
        if row != other:
            return False
    return True
