"""Extended glossary: analysis, validation, control, robotics, math, data, ai,
visualization, materials, injury, anatomy, optimization, signal processing, muscle.

Part of the expanded glossary (Issue #764). ~310 entries across 14 categories.

Data is stored in ``data/glossary_extended.json`` and loaded at call time.
"""

from __future__ import annotations

import json
from pathlib import Path

_DATA_FILE = Path(__file__).parent / "data" / "glossary_extended.json"


def get_extended_entries() -> list[dict]:
    """Return extended glossary entries as compact dicts.

    Each dict has: key, term, cat, b (beginner), i (intermediate).
    Optional: a (advanced), f (formula), u (units), r (related keys list).
    """
    with open(_DATA_FILE, encoding="utf-8") as fh:
        entries: list[dict] = json.load(fh)
    return entries
