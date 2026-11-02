"""Variance-attribution tool: which error source costs the most strokes (#14177)."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from src.shared.python.putting_analytics import service_variance
from src.sim_run.mcp import SecurityPolicy, SimulationMcpServer

pytestmark = pytest.mark.unit

BASELINE = {
    "distances_ft": [0.5, 3.0, 10.0, 30.0, 60.0],
    "expected_putts": [1.0, 1.1, 1.5, 1.9, 2.2],
}
CLASSES = [
    {"distance_ft": 6.0, "count": 3.0},
    {"distance_ft": 12.0, "slope_pct": 2.0, "orientation": "right_to_left"},
]


def _args(**overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "baseline": BASELINE,
        "classes": CLASSES,
        "face_sd_deg": 1.5,
        "path_sd_deg": 2.0,
        "speed_cv": 0.08,
        "grade_rel_sd": 0.25,
        "factors": ["face", "speed", "read_grade", "read_aspect"],
        "base_samples": 16,
        "n_samples": 64,
        "n_bootstrap": 20,
        "seed": 3,
    }
    return {**args, **overrides}


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    return service_variance.variance_study(_args())


def test_report_is_json_with_one_entry_per_class(report: dict[str, Any]) -> None:
    json.loads(json.dumps(report, allow_nan=False))
    assert report["schema_version"] == 1
    assert report["factors"] == ["face", "speed", "read_grade", "read_aspect"]
    assert [c["weight"] for c in report["classes"]] == pytest.approx([0.75, 0.25])
    for entry in report["classes"]:
        assert 1.0 <= entry["baseline_strokes"] < 3.0
        assert [row["factor"] for row in entry["eliminations"]] == report["factors"]
        assert entry["sobol"]["base_samples"] == 16


def test_an_error_source_with_no_spread_gains_nothing(report: dict[str, Any]) -> None:
    for entry in report["classes"]:
        aspect = next(r for r in entry["eliminations"] if r["factor"] == "read_aspect")
        assert aspect["gain"] == 0.0 and aspect["gain_reduced"] == 0.0


def test_round_totals_weight_each_class_by_first_putts(report: dict[str, Any]) -> None:
    per_round = report["per_round"]
    assert per_round["first_putts"] == 18.0
    gains = [row["gain_removed"] for row in per_round["factors"]]
    assert gains == sorted(gains, reverse=True)
    face = next(row for row in per_round["factors"] if row["factor"] == "face")
    want = sum(
        18.0
        * c["weight"]
        * next(r["gain"] for r in c["eliminations"] if r["factor"] == "face")
        for c in report["classes"]
    )
    assert face["gain_removed"] == pytest.approx(want)


def test_study_is_seeded(report: dict[str, Any]) -> None:
    assert service_variance.variance_study(_args()) == report


def test_default_factors_are_the_modelled_channels() -> None:
    assert "strike" not in service_variance.MODELLED_FACTORS
    assert service_variance.factors_from_args({}) == service_variance.MODELLED_FACTORS


def test_non_finite_numbers_become_null() -> None:
    cleaned = service_variance.finite_json({"a": [1.0, math.nan], "b": math.inf})
    assert cleaned == {"a": [1.0, None], "b": None}


@pytest.mark.parametrize(
    "overrides",
    [
        {"face_sd_deg": -1.0},
        {"speed_cv": "fast"},
        {"speed_bias": -1.0},
        {"grade_rel_bias": -1.0},
        {"face_weight": 1.5},
        {"pace_ft_past": 20.0},
        {"factors": []},
        {"factors": ["strike"]},
        {"factors": ["face", "face"]},
        {"factors": ["wind"]},
        {"base_samples": 1},
        {"n_samples": 1},
        {"n_bootstrap": 5},
        {"reduction": 0.0},
        {"first_putts_per_round": -1.0},
        {"classes": []},
        {"baseline": None},
    ],
)
def test_bad_arguments_raise_value_error(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        service_variance.variance_study(_args(**overrides))


def test_work_over_the_cap_is_refused() -> None:
    with pytest.raises(ValueError, match="work"):
        service_variance.variance_study(_args(n_samples=10_000))


def _call(server: SimulationMcpServer, name: str, args: dict[str, Any]) -> Any:
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": args},
    }
    return server.handle_request(request)


def test_mcp_lists_and_runs_the_variance_tool() -> None:
    server = SimulationMcpServer(security_policy=SecurityPolicy(auth_token=None))
    listing = server.handle_request({"jsonrpc": "2.0", "id": 0, "method": "tools/list"})
    specs = {tool["name"]: tool for tool in listing["result"]["tools"]}
    schema = specs["putting_variance_attribution"]["inputSchema"]
    assert {"classes", "baseline"} <= set(schema["required"])
    args = _args(classes=CLASSES[:1], factors=["face"])
    response = _call(server, "putting_variance_attribution", args)
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["per_round"]["factors"][0]["factor"] == "face"
    assert _call(server, "putting_variance_attribution", {})["error"]["code"] == -32602
