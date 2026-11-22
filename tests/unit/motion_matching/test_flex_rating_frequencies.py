"""Review of ``SHAFT_FLEX_DATA`` cpm values against a cited source (#15150 C19).

Source and measurement convention are recorded in the fixture. The repo table
sits inside every published graphite driver range, but its R, S and X values are
14, 22 and 29 cpm above the published means. Moving them would change the
calibrated shaft-chain stiffness and force a refit of the driver bundles, so
the discrepancy is recorded as a strict xfail rather than a table change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.club_configurations import (  # noqa: E501
    SHAFT_FLEX_DATA,
)

pytestmark = pytest.mark.unit

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "shaft_flex"
    / "driver_frequency_by_flex.json"
)
SOURCE = json.loads(FIXTURE.read_text(encoding="utf-8"))
FLEXES = sorted(SOURCE["repo_flex_to_source_flex"])
# Half of the repo's own 20 cpm step between adjacent flexes.
MEAN_TOLERANCE_CPM = 10.0


def _row(repo_flex: str) -> dict[str, float]:
    return SOURCE["graphite_driver"][SOURCE["repo_flex_to_source_flex"][repo_flex]]


def test_fixture_records_citation_and_convention() -> None:
    for key in ("citation", "url", "accessed", "convention", "slope_assumption"):
        assert SOURCE[key]
    assert SOURCE["resolution_cpm"] == 1


@pytest.mark.parametrize("flex", FLEXES)
def test_repo_frequency_inside_published_range(flex: str) -> None:
    row = _row(flex)
    assert row["low"] <= SHAFT_FLEX_DATA[flex]["stiffness_cpm"] <= row["high"]


def test_repo_frequencies_increase_with_flex_like_source() -> None:
    order = ["Ladies", "Senior", "Regular", "Stiff", "X-Stiff"]
    repo = [SHAFT_FLEX_DATA[f]["stiffness_cpm"] for f in order]
    src = [_row(f)["mean"] for f in order]
    assert repo == sorted(repo) and src == sorted(src)


@pytest.mark.parametrize(
    "flex",
    [
        "Ladies",
        "Senior",
        pytest.param(
            "Regular",
            marks=pytest.mark.xfail(
                strict=True, reason="260 cpm is 14 above the 246 cpm mean (#15150)"
            ),
        ),
        pytest.param(
            "Stiff",
            marks=pytest.mark.xfail(
                strict=True, reason="280 cpm is 22 above the 258 cpm mean (#15150)"
            ),
        ),
        pytest.param(
            "X-Stiff",
            marks=pytest.mark.xfail(
                strict=True, reason="300 cpm is 29 above the 271 cpm mean (#15150)"
            ),
        ),
    ],
)
def test_repo_frequency_near_published_mean(flex: str) -> None:
    gap = abs(SHAFT_FLEX_DATA[flex]["stiffness_cpm"] - _row(flex)["mean"])
    assert gap <= MEAN_TOLERANCE_CPM
