"""Core glossary entries: dynamics, kinematics, biomechanics, golf, simulation, forces.

Part of the expanded glossary (Issue #764). Contains ~200 entries across
6 foundational categories for the golf biomechanics education system.

Data is stored in ``data/glossary_core.json`` and loaded at call time.
"""

from __future__ import annotations

import json
from pathlib import Path

_DATA_FILE = Path(__file__).parent / "data" / "glossary_core.json"


def get_core_entries() -> list[dict]:
    """Return core glossary entries as compact dicts.

    Each dict has: key, term, cat, b (beginner), i (intermediate).
    Optional: a (advanced), f (formula), u (units), r (related keys list).
    """
    with open(_DATA_FILE, encoding="utf-8") as fh:
        entries: list[dict] = json.load(fh)
    return entries
