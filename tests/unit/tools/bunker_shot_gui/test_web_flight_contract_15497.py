"""Exercise the web display guard with the real adapter's wire representation."""

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from src.tools.bunker_shot_gui.flight import BunkerFlight

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def record() -> dict:
    trace = BunkerFlight(
        ((0.0, 0.0, 0.0), (4.0, 1.0, 3.0), (8.0, 0.0, 0.0)),
        ((8.0, 0.0, 0.0), (10.0, -1.0, 0.0)),
        8.0,
    )
    return {"refused": False, "carry": {"carry_m": 8.0}, "flight": asdict(trace)}


def display(record: dict) -> dict | None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node >=22.6 is required for the web contract test")
    version = subprocess.check_output([node, "--version"], text=True, timeout=10)
    if tuple(map(int, version.strip().lstrip("v").split(".")[:2])) < (22, 6):
        pytest.skip("Node >=22.6 is required for native TypeScript execution")
    module = ROOT / "ui/src/components/bunker/flightTrace.ts"
    script = (
        f"import {{ displayTrace }} from {json.dumps(module.as_uri())};"
        "let input = ''; for await (const chunk of process.stdin) input += chunk;"
        "process.stdout.write(JSON.stringify(displayTrace(JSON.parse(input))));"
    )
    result = subprocess.run(
        [node, "--experimental-strip-types", "--input-type=module", "-e", script],
        input=json.dumps(record),
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    return json.loads(result.stdout)


def test_adapter_wire_trace_keeps_coordinates_and_assumption(record: dict) -> None:
    assert display(record) == json.loads(json.dumps(record["flight"]))


@pytest.mark.parametrize(
    "change",
    [
        "refused",
        "missing",
        "null",
        "carry",
        "nan",
        "gap",
        "empty",
        "shape",
        "assumption",
    ],
)
def test_unavailable_or_invalid_trace_never_reaches_plot(
    record: dict, change: str
) -> None:
    if change == "refused":
        record["refused"] = True
    elif change == "missing":
        del record["flight"]
    elif change == "null":
        record["flight"] = None
    elif change == "carry":
        record["carry"]["carry_m"] = 9.0
    elif change == "nan":
        # JSON cannot transport NaN; a null coordinate is invalid too.
        record["flight"]["flight_m"] = [[0, 0, None], [8, 0, 0]]
    elif change == "gap":
        record["flight"]["rollout_m"] = [[9, 0, 0], [10, 0, 0]]
    elif change == "empty":
        record["flight"]["flight_m"] = []
    elif change == "shape":
        record["flight"]["flight_m"] = [[0, 0], [8, 0]]
    else:
        record["flight"]["assumption"] = ""
    assert display(record) is None
