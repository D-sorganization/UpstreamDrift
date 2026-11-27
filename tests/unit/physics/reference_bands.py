"""Acceptance bands of the published reference shots (#15443 F7).

One place reads ``reference_shots_15443.json`` for the older benchmark tests,
so they hold the same sourced row and band as the reference-shot regression
instead of their own wider literals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_FIXTURE = Path(__file__).with_name("reference_shots_15443.json")


def reference_row(row_id: str) -> dict[str, Any]:
    """The reference shot ``row_id``; ``KeyError`` when the fixture has none."""
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    for row in data["shots"]:
        if row["id"] == row_id:
            return row
    raise KeyError(f"no reference shot {row_id!r} in {_FIXTURE.name}")


def band_m(row_id: str, metric: str) -> tuple[float, float]:
    """``(low, high)`` in metres of ``metric`` (``carry`` or ``apex``) of a row.

    The half width is the fixture's ``<metric>_band_fraction`` of the
    reference value. ``KeyError`` for an unknown row or metric.
    """
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    settings = data["settings"]
    value = reference_row(row_id)[f"{metric}_yd"] * settings["yard_m"]
    half = value * settings[f"{metric}_band_fraction"]
    return value - half, value + half
