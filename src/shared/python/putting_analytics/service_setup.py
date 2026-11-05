"""JSON service for putting setups, saved scenarios and comparison (#14471, epic #14452).

One versioned setup document for every putting surface (desktop simulator,
web page, studio), carrying the studio export's ``schema_version`` so a file
saved on one surface loads on another:

* **Setup.** The green and the player the shared services read: the putt
  distance, :func:`.service_dispersion.green_from_args` (grade, fall aspect,
  Stimp) and :func:`.service_variance.profile_from_args` (stroke and read
  dispersion), with the services' defaults filled in.
* **Scenario.** A named setup with the result of
  :func:`.service_dispersion.dispersion_study` for it (make rate, mean leave).
* **Comparison.** Each scenario's make rate and leave against the first.
* **Import.** A document of this schema, or a studio export, whose ``state``
  gives the distance, Stimp and the planar slope (``uphillSlopePct`` raises
  the green along the line, ``crossSlopePct`` to the left). Studio scenario
  rows are formatted text, so only their count is reported.

Arguments are untrusted tool input, checked with :func:`.service._check`.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

from src.shared.python.putting_analytics import service_dispersion
from src.shared.python.putting_analytics.service import MAX_DISTANCE_FT, _check, _number
from src.shared.python.putting_analytics.service_dispersion import green_from_args
from src.shared.python.putting_analytics.service_variance import profile_from_args

SCHEMA_VERSION = "4.0.0"  # the studio export's (js/io.js)
SUITE = "UpstreamDrift_Putting_Setup"
SETUP_KEYS = (
    "distance_ft",
    "grade_pct",
    "aspect_deg",
    "stimp",
    "face_sd_deg",
    "path_sd_deg",
    "speed_cv",
    "face_bias_deg",
    "speed_bias",
    "grade_rel_sd",
    "aspect_sd_deg",
    "stimp_rel_sd",
    "grade_rel_bias",
    "face_weight",
)
MAX_SCENARIOS = 50
_STUDIO_STATE = {"puttDistFt": "distance_ft", "stimpFt": "stimp"}


def setup_from_args(args: Mapping[str, Any]) -> dict[str, float]:
    """The validated setup, every key of :data:`SETUP_KEYS` filled in."""
    distance = _number(args, "distance_ft", 10.0)
    _check(
        0.0 < distance <= MAX_DISTANCE_FT,
        f"'distance_ft' must be in (0, {MAX_DISTANCE_FT}]",
    )
    green = green_from_args(args)
    profile = profile_from_args(args)
    setup = {
        "distance_ft": distance,
        "grade_pct": green.grade_pct,
        "aspect_deg": green.aspect_deg,
        "stimp": green.stimp,
        **dataclasses.asdict(profile.stroke),
        **dataclasses.asdict(profile.read),
        "face_weight": profile.face_weight,
    }
    return {key: float(setup[key]) for key in SETUP_KEYS}


def scenario_study(args: Mapping[str, Any]) -> dict[str, Any]:
    """A named setup with its :func:`.service_dispersion.dispersion_study` result."""
    name = args.get("name")
    _check(isinstance(name, str) and 0 < len(name) <= 80, "'name' needs 1-80 chars")
    setup = setup_from_args(_mapping(args, "setup"))
    run = {key: args[key] for key in ("n", "seed") if key in args}
    report = service_dispersion.dispersion_study({**setup, **run})
    result = {
        "make_rate": report["make_rate"],
        "mean_leave_ft": report["mean_leave_ft"],
        "n": report["n"],
        "seed": int(_number(args, "seed", 0)),
    }
    return {"name": name, "setup": setup, "result": result}


def compare_scenarios(args: Mapping[str, Any]) -> dict[str, Any]:
    """Each scenario's make rate and mean leave against the first scenario's."""
    scenarios = _scenarios(args, required=True)
    base = scenarios[0]["result"]
    rows = []
    for scenario in scenarios:
        result = scenario["result"]
        leaves = (result["mean_leave_ft"], base["mean_leave_ft"])
        rows.append(
            {
                "name": scenario["name"],
                "make_rate": result["make_rate"],
                "make_rate_delta": result["make_rate"] - base["make_rate"],
                "mean_leave_ft": result["mean_leave_ft"],
                "mean_leave_delta_ft": None
                if None in leaves
                else leaves[0] - leaves[1],
            }
        )
    return {"schema_version": 1, "baseline": scenarios[0]["name"], "rows": rows}


def export_setup(args: Mapping[str, Any]) -> dict[str, Any]:
    """The setup document: ``schema_version``, suite, setup and scenarios."""
    return {
        "schema_version": SCHEMA_VERSION,
        "suite": SUITE,
        "setup": setup_from_args(_mapping(args, "setup")),
        "scenarios": _scenarios(args, required=False),
    }


def import_setup(args: Mapping[str, Any]) -> dict[str, Any]:
    """Setup and scenarios from a document of this schema or a studio export."""
    doc = _mapping(args, "document")
    version = str(doc.get("schema_version", ""))
    major = SCHEMA_VERSION.split(".")[0]
    _check(
        version.split(".")[0] == major,
        f"'schema_version' {version!r} is not {major}.x",
    )
    if "state" in doc:
        studio_rows = doc.get("savedScenarios", [])
        return {
            "setup": setup_from_args(_studio_setup(_mapping(doc, "state"))),
            "scenarios": [],
            "studio_scenarios": len(studio_rows)
            if isinstance(studio_rows, list)
            else 0,
        }
    return {
        "setup": setup_from_args(_mapping(doc, "setup")),
        "scenarios": _scenarios(doc, required=False),
        "studio_scenarios": 0,
    }


def _studio_setup(state: Mapping[str, Any]) -> dict[str, float]:
    """The studio state's distance, Stimp and planar slope as setup arguments."""
    setup = {ours: _number(state, theirs) for theirs, ours in _STUDIO_STATE.items()}
    up = _number(state, "uphillSlopePct", 0.0)
    cross = _number(state, "crossSlopePct", 0.0)
    # Height rises by (up, cross) % along the line and to its left: the green
    # falls the opposite way, CCW of the line.
    setup["grade_pct"] = math.hypot(up, cross)
    setup["aspect_deg"] = (
        math.degrees(math.atan2(-cross, -up)) if setup["grade_pct"] else 0.0
    )
    return setup


def _mapping(args: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = args.get(key)
    _check(isinstance(value, Mapping), f"'{key}' must be an object")
    return value  # type: ignore[return-value]


def _scenarios(args: Mapping[str, Any], *, required: bool) -> list[dict[str, Any]]:
    raw = args.get("scenarios", [])
    low = 1 if required else 0
    _check(
        isinstance(raw, list) and low <= len(raw) <= MAX_SCENARIOS,
        f"'scenarios' needs {low} to {MAX_SCENARIOS} entries",
    )
    out = []
    for scenario in raw:
        _check(isinstance(scenario, Mapping), "each scenario must be an object")
        result = _mapping(scenario, "result")
        leave = result.get("mean_leave_ft")
        out.append(
            {
                "name": str(scenario.get("name", "")),
                "setup": setup_from_args(_mapping(scenario, "setup")),
                "result": {
                    "make_rate": _number(result, "make_rate"),
                    "mean_leave_ft": None
                    if leave is None
                    else _number(result, "mean_leave_ft"),
                    "n": int(_number(result, "n")),
                    "seed": int(_number(result, "seed", 0)),
                },
            }
        )
    return out


__all__ = [
    "MAX_SCENARIOS",
    "SCHEMA_VERSION",
    "SETUP_KEYS",
    "SUITE",
    "compare_scenarios",
    "export_setup",
    "import_setup",
    "scenario_study",
    "setup_from_args",
]
