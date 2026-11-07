"""Fleet grip-roll guard (SHOW-A13, #14738).

Keeps every model and binding on the single A7 grip-roll convention
(``motion_matching.grip_roll``; 0 deg = back of the wrist on top of the shaft,
+ toward the target, default 15 deg). The guard only *reads* specs and
bindings; it never writes them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.motion_matching.grip_roll import (
    address_grip_roll_deg,
    grip_pose_state_from_spec,
)

TOLERANCE_DEG = 1.0
SPEC_GLOB = "full_body_spec*.json"
BINDING_GLOB = "*.json"


@dataclass(frozen=True)
class Violation:
    """One guard failure: the item id and a human-readable reason."""

    item: str
    reason: str


def discover_specs(models_dir: Path) -> list[str]:
    """Names of every built full-body spec in ``models_dir`` (hands required)."""
    require(models_dir.is_dir(), f"models_dir not found: {models_dir}")
    return [
        p.name
        for p in sorted(models_dir.glob(SPEC_GLOB))
        if str(
            json.loads(p.read_text(encoding="utf-8")).get("schema_version", "")
        ).startswith("full-body")
    ]


def discover_bindings(bindings_dir: Path) -> list[str]:
    """Names of every active grip binding manifest in ``bindings_dir``."""
    require(bindings_dir.is_dir(), f"bindings_dir not found: {bindings_dir}")
    return sorted(p.name for p in bindings_dir.glob(BINDING_GLOB))


def measure_spec(
    spec_document: Mapping[str, Any], club_rotation: Any
) -> tuple[float, float]:
    """``(lead, trail)`` address roll of a spec at the recorded club rotation."""
    return address_grip_roll_deg(
        grip_pose_state_from_spec(spec_document, club_rotation)
    )


def _entry_problem(entry: Mapping[str, Any]) -> str | None:
    if not str(entry.get("reason", "")).strip():
        return "allowlist entry has no reason"
    return (
        None
        if isinstance(entry.get("issue"), int)
        else "allowlist entry has no owner issue number"
    )


def audit(
    *,
    models_dir: Path,
    bindings_dir: Path,
    guard: Mapping[str, Any],
    target_deg: float,
    tolerance_deg: float = TOLERANCE_DEG,
) -> list[Violation]:
    """Every violation of the 15 +- 1 deg rule across specs and bindings."""
    records, allow = guard.get("records", {}), guard.get("allowlist", {})
    found: list[Violation] = []
    items = [f"spec/{n}" for n in discover_specs(models_dir)] + [
        f"binding/{n}" for n in discover_bindings(bindings_dir)
    ]
    for name, entry in allow.items():
        if problem := _entry_problem(entry):
            found.append(Violation(name, problem))
        if name not in items:
            found.append(Violation(name, "allowlist entry matches no spec or binding"))
    for name in items:
        if name not in records and name not in allow:
            found.append(Violation(name, "no roll record and not allowlisted"))
        if name not in records:
            continue
        doc = json.loads(
            (models_dir / name.removeprefix("spec/")).read_text(encoding="utf-8")
        )
        roll = measure_spec(doc, np.asarray(records[name]["club_rotation"], float))[0]
        in_band = abs(roll - target_deg) <= tolerance_deg
        if name in allow and in_band:
            found.append(Violation(name, "reads in band: remove it from the allowlist"))
        elif name not in allow and not in_band:
            found.append(
                Violation(
                    name,
                    f"lead roll {roll:.2f} deg is outside {target_deg} +- {tolerance_deg}",
                )
            )
    return found
