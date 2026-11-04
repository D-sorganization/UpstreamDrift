"""Saved scenarios, comparison and JSON setup import/export (#14471, epic #14452).

Golden parity: a setup is the green and player the shared services read
(:func:`service_dispersion.green_from_args`,
:func:`service_variance.profile_from_args`), a scenario's numbers are exactly
:func:`service_dispersion.dispersion_study`'s for its setup, and the document
carries the studio export's ``schema_version`` (4.0.0) so a studio file loads.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from src.shared.python.putting_analytics import service_dispersion, service_setup

pytestmark = pytest.mark.unit

SETUP = {
    "distance_ft": 12.0,
    "grade_pct": 2.0,
    "aspect_deg": -90.0,
    "stimp": 11.0,
    "face_sd_deg": 1.2,
    "speed_cv": 0.07,
}


def test_setup_fills_the_service_defaults_and_keeps_the_dialled_values() -> None:
    setup = service_setup.setup_from_args(SETUP)
    assert set(setup) == set(service_setup.SETUP_KEYS)
    assert {k: setup[k] for k in SETUP} == SETUP
    assert setup["path_sd_deg"] == 2.0  # profile_from_args default
    assert setup["grade_rel_sd"] == 0.0


def test_flat_green_drops_its_aspect() -> None:
    setup = service_setup.setup_from_args({**SETUP, "grade_pct": 0.0})
    assert setup["aspect_deg"] == 0.0  # green_from_args: no fall line


def test_scenario_records_the_dispersion_study_for_its_setup() -> None:
    scenario = service_setup.scenario_study(
        {"name": "Firm", "setup": SETUP, "n": 40, "seed": 3}
    )
    report = service_dispersion.dispersion_study({**SETUP, "n": 40, "seed": 3})
    assert scenario["name"] == "Firm"
    assert scenario["setup"] == service_setup.setup_from_args(SETUP)
    assert scenario["result"] == {
        "make_rate": report["make_rate"],
        "mean_leave_ft": report["mean_leave_ft"],
        "n": 40,
        "seed": 3,
    }


def _scenario(name: str, make: float, leave: float | None) -> dict[str, Any]:
    result = {"make_rate": make, "mean_leave_ft": leave, "n": 40, "seed": 0}
    return {
        "name": name,
        "setup": service_setup.setup_from_args(SETUP),
        "result": result,
    }


def test_comparison_reports_each_scenario_against_the_first() -> None:
    rows = service_setup.compare_scenarios(
        {
            "scenarios": [
                _scenario("Base", 0.40, 2.0),
                _scenario("Fast", 0.55, 2.6),
                _scenario("All in", 1.0, None),
            ]
        }
    )["rows"]
    assert [r["name"] for r in rows] == ["Base", "Fast", "All in"]
    assert rows[0]["make_rate_delta"] == 0.0
    assert rows[1]["make_rate_delta"] == pytest.approx(0.15)
    assert rows[1]["mean_leave_delta_ft"] == pytest.approx(0.6)
    assert rows[2]["mean_leave_delta_ft"] is None  # nothing missed


def test_document_round_trips_through_json() -> None:
    scenarios = [_scenario("Base", 0.4, 2.0)]
    doc = service_setup.export_setup({"setup": SETUP, "scenarios": scenarios})
    assert doc["schema_version"] == service_setup.SCHEMA_VERSION == "4.0.0"
    back = service_setup.import_setup({"document": json.loads(json.dumps(doc))})
    assert back["setup"] == service_setup.setup_from_args(SETUP)
    assert back["scenarios"] == scenarios


def test_a_studio_export_loads_its_green_and_distance() -> None:
    studio = {
        "schema_version": "4.0.0",
        "suite": "UpstreamDrift_3D_Putting_Studio_v4",
        "state": {
            "puttDistFt": 15.0,
            "stimpFt": 12.0,
            "uphillSlopePct": 1.0,
            "crossSlopePct": 2.0,
        },
        "savedScenarios": [{"name": "Studio row", "makePct": "41.0%"}],
    }
    back = service_setup.import_setup({"document": studio})
    setup = back["setup"]
    assert (setup["distance_ft"], setup["stimp"]) == (15.0, 12.0)
    # The studio's green rises along +x (uphill) and +y (left): it falls back
    # toward the ball and to the right.
    assert setup["grade_pct"] == pytest.approx(math.hypot(1.0, 2.0))
    assert setup["aspect_deg"] == pytest.approx(math.degrees(math.atan2(-2.0, -1.0)))
    assert back["scenarios"] == []  # studio rows hold formatted text, not setups
    assert back["studio_scenarios"] == 1


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"document": {"schema_version": "5.0.0", "setup": SETUP}}, "schema_version"),
        ({"document": {"schema_version": "4.0.0"}}, "setup"),
        ({"document": "not a dict"}, "document"),
        ({"document": {"schema_version": "4.0.0", "setup": {"stimp": 30.0}}}, "stimp"),
    ],
)
def test_bad_documents_are_rejected(args: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        service_setup.import_setup(args)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"scenarios": []}, "scenarios"),
        ({"scenarios": [{"name": "x"}]}, "result"),
        ({"setup": SETUP, "scenarios": "x"}, "scenarios"),
    ],
)
def test_bad_scenarios_are_rejected(args: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        if "setup" in args:
            service_setup.export_setup(args)
        else:
            service_setup.compare_scenarios(args)
