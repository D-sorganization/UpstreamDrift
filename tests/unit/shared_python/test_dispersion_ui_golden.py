"""Golden parity: dispersion.golden.json vs the shared putting core (#14460).

The React dispersion panel reads its noise channels, batch sizes and the
argument mapping from the same golden file (ui/src/utils/dispersion.test.ts).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.putting_analytics import service_dispersion
from src.shared.python.putting_analytics.service_variance import (
    _READ_BOUNDS,
    _STROKE_BOUNDS,
)

pytestmark = pytest.mark.unit

GOLDEN = json.loads(
    (
        Path(__file__).resolve().parents[3] / "ui/src/utils/dispersion.golden.json"
    ).read_text(encoding="utf-8")
)
CASES = GOLDEN["cases"]


def test_channels_are_the_core_noise_channels() -> None:
    core = {**_STROKE_BOUNDS, **_READ_BOUNDS}
    for key, spec in GOLDEN["channels"].items():
        assert tuple(spec) == core[key], key
    assert set(GOLDEN["channels"]) == {k for k in core if "_bias" not in k}, (
        "every SD channel of the core is shown"
    )


def test_spread_grid_and_batch_sizes_match_the_core() -> None:
    assert (
        tuple(GOLDEN["spread_distances_ft"])
        == service_dispersion.DEFAULT_SPREAD_GRID_FT
    )
    assert max(GOLDEN["batch_sizes"]) <= service_dispersion.MAX_DISPERSION_N
    assert GOLDEN["spread_n"] <= service_dispersion.MAX_SPREAD_N


@pytest.mark.parametrize("case", CASES, ids=[str(i) for i in range(len(CASES))])
def test_mapped_args_are_valid_core_input(case: dict) -> None:
    green = service_dispersion.green_from_args(case["args"])
    assert green.grade_pct == case["args"]["grade_pct"]
    assert green.aspect_deg == case["args"]["aspect_deg"]
    assert green.stimp == case["args"]["stimp"]


def test_core_batch_reproduces_the_golden_values() -> None:
    case = CASES[0]
    channels = {k: spec[0] for k, spec in GOLDEN["channels"].items()}
    report = service_dispersion.dispersion_study(
        {**case["args"], **channels, "n": case["n"]}
    )
    assert report["make_rate"] == case["make_rate"]
    assert sum(report["holed"]) == case["holed"]
    assert report["mean_leave_ft"] == pytest.approx(case["mean_leave_ft"], rel=1e-12)
