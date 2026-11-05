"""``ui/src/utils/counterTorque.golden.json`` matches the shared core (#14470).

Its series are the core's stroke series for ``request`` and its ranges the
core's; ``counterTorque.test.ts`` reads the summary off those series.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from src.shared.python.putting_dynamics import counter_torque as ct
from src.shared.python.putting_dynamics.putter_presets import balance_preset

pytestmark = pytest.mark.unit

_GOLDEN = Path(__file__).resolve().parents[3] / "ui/src/utils/counterTorque.golden.json"


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(_GOLDEN.read_text(encoding="utf-8"))


def test_response_is_the_core_stroke_series(golden: dict) -> None:
    settings = dict(golden["request"])
    n_points = settings.pop("n_points")
    inputs = ct.CounterTorqueInputs(**settings)
    series = [asdict(s) for s in ct.stroke_series(inputs, n_points=n_points)]
    response = golden["response"]
    assert response["putter_name"] == balance_preset(inputs.putter).name
    assert set(response["samples"]) <= set(series[0])
    for key, values in response["samples"].items():
        assert values == pytest.approx([s[key] for s in series], abs=1e-12), key


def test_ranges_are_the_core_ranges(golden: dict) -> None:
    assert golden["ranges"] == {
        "preload_nm": list(ct.PRELOAD_RANGE_NM),
        "off_preload_nm": list(ct.PRELOAD_RANGE_NM),
        "noise_scale": list(ct.NOISE_RANGE),
        "gust_impulse_nm": list(ct.GUST_RANGE_NM),
        "lie_deg": list(ct.LIE_RANGE_DEG),
    }
