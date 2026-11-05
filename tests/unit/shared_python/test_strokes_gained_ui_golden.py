"""``ui/src/utils/strokesGained.golden.json`` matches the shared core (#14468).

``strokesGained.test.ts`` holds the React side to the same file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.putting_analytics import service_sg
from src.shared.python.putting_analytics.aggregate import DEFAULT_GRID_FT

pytestmark = pytest.mark.unit

_GOLDEN = Path(__file__).resolve().parents[3] / "ui/src/utils/strokesGained.golden.json"
GOLDEN = json.loads(_GOLDEN.read_text(encoding="utf-8"))
CASE = GOLDEN["case"]


def test_study_defaults_are_the_core_defaults() -> None:
    assert GOLDEN["study"] == {
        "n": service_sg.DEFAULT_SG_N,
        "paces_ft": list(service_sg.DEFAULT_SG_PACES_FT),
        "distances_ft": list(DEFAULT_GRID_FT),
    }


def test_case_report_is_the_core_study() -> None:
    """The panel's arguments, cut to a small study, give the pinned report."""
    report = service_sg.sg_study({**CASE["args"], **CASE["small_study"]})
    curve = CASE["report"]["curve"]
    assert report["curve"]["distances_ft"] == curve["distances_ft"]
    assert report["curve"]["expected_putts"] == pytest.approx(curve["expected_putts"])
    assert report["expected_putts"] == pytest.approx(CASE["report"]["expected_putts"])
    assert report["sg_player"] == pytest.approx(CASE["report"]["sg_player"])
    assert report["distance_ft"] == CASE["report"]["distance_ft"]
