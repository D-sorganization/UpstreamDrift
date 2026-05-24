from __future__ import annotations

import json
from pathlib import Path

import pytest

from sidekick.standalone import runner

pytestmark = pytest.mark.integration


def test_run_wgs_reactor_fixture_outputs_calculation_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = Path(__file__).resolve().parents[2]
    inputs = root / "fixtures" / "wgs.json"

    code = runner.run_calculator("wgs_reactor", str(inputs), fmt="json")

    assert code == runner.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["values"]["temperature_c"] == 350.0
    assert payload["values"]["co_conversion_fraction"] == pytest.approx(
        0.8134184631238724
    )
    assert payload["units"]["temperature_c"] == "degC"
    assert payload["metadata"]["calculator"] == "wgs_reactor"
