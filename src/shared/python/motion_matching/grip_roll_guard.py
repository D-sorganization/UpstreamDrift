"""Fleet grip-roll guard (SHOW-A13, #14738).

Keeps every model and binding on the single A7 grip-roll convention
(``motion_matching.grip_roll``; 0 deg = back of the wrist on top of the shaft,
+ toward the target, default 15 deg). The guard only *reads* specs and
bindings; it never writes them.

A full-body spec is measured from its own wrist dorsal axes plus the native club
rotation recorded at address (``records``). Anything that cannot be measured yet
must sit on the explicit ``allowlist`` with a reason and an owner issue, so the
list shrinks as the owning issues land and cannot grow silently.
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
    names = []
    for path in sorted(models_dir.glob(SPEC_GLOB)):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if str(doc.get("schema_version", "")).startswith("full-body"):
            names.append(path.name)
    return names


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
    if not isinstance(entry.get("issue"), int):
        return "allowlist entry has no owner issue number"
    return None


def audit(
    *,
    models_dir: Path,
    bindings_dir: Path,
    guard: Mapping[str, Any],
    target_deg: float,
    tolerance_deg: float = TOLERANCE_DEG,
) -> list[Violation]:
    """Every violation of the 15 +- 1 deg rule across specs and bindings.

    ``guard`` holds ``records`` (``spec/<file>`` -> recorded ``club_rotation``) and
    ``allowlist`` (``spec/<file>`` or ``binding/<file>`` -> ``{reason, issue}``). A measured item outside the
    band fails unless allowlisted; an allowlisted item that now reads in-band
    fails as stale, so the allowlist can only shrink; an item with neither a
    record nor an allowlist entry fails as unaccounted.
    """
    records: Mapping[str, Any] = guard.get("records", {})
    allow: Mapping[str, Any] = guard.get("allowlist", {})
    found: list[Violation] = []
    items = [
        *(f"spec/{n}" for n in discover_specs(models_dir)),
        *(f"binding/{n}" for n in discover_bindings(bindings_dir)),
    ]
    for name, entry in allow.items():
        problem = _entry_problem(entry)
        if problem:
            found.append(Violation(name, problem))
        if name not in items:
            found.append(Violation(name, "allowlist entry matches no spec or binding"))
    for name in items:
        if name not in records:
            if name not in allow:
                found.append(Violation(name, "no roll record and not allowlisted"))
            continue
        doc = json.loads(
            (models_dir / name.removeprefix("spec/")).read_text(encoding="utf-8")
        )
        values = measure_spec(doc, np.asarray(records[name]["club_rotation"], float))
        in_band = all(abs(v - target_deg) <= tolerance_deg for v in values)
        if name in allow and in_band:
            found.append(Violation(name, "reads in band: remove it from the allowlist"))
        elif name not in allow and not in_band:
            found.append(
                Violation(
                    name,
                    f"roll {values[0]:.2f}/{values[1]:.2f} deg is outside "
                    f"{target_deg} +- {tolerance_deg}",
                )
            )
    return found
