"""JSON putting studies for tool adapters (#14177)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.shared.python.putting_analytics import service
from src.sim_run.mcp import SecurityPolicy, SimulationMcpServer

pytestmark = pytest.mark.unit

# Test-only miss pricing, shaped like a putting curve; not reference data.
BASELINE = {
    "distances_ft": [1.0, 3.0, 6.0, 10.0, 20.0, 40.0],
    "expected_putts": [1.0, 1.1, 1.4, 1.6, 1.85, 2.1],
}


def _args(**overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "distance_ft": 8.0,
        "baseline": BASELINE,
        "aims_deg": [0.0, 3.0],
        "paces_ft": [0.5, 2.0],
        "line_sd_deg": 0.5,
        "speed_sd": 0.03,
        "n": 12,
        "seed": 4,
    }
    return {**args, **overrides}


def test_optimize_putt_returns_ranked_json_floats() -> None:
    result = service.optimize_putt(_args())
    json.dumps(result)  # plain, serialisable values only
    assert result["candidates"] == 4 and result["n"] == 12
    top = result["top"]
    assert top[0] == result["best"]
    strokes = [entry["expected_strokes"] for entry in top]
    assert strokes == sorted(strokes)
    assert service.optimize_putt(_args()) == result  # seeded


def test_flat_green_prefers_the_straight_line() -> None:
    result = service.optimize_putt(_args(aims_deg=[-4.0, 0.0, 4.0], paces_ft=[1.0]))
    assert result["best"]["action"]["aim_offset_deg"] == 0.0


def test_must_make_ranks_by_make_rate() -> None:
    result = service.optimize_putt(_args(objective="must_make"))
    rates = [entry["make_rate"] for entry in result["top"]]
    assert rates == sorted(rates, reverse=True)


def test_misread_cost_is_zero_for_perfect_reads() -> None:
    sloped = _args(grade_pct=1.0, aspect_deg=270.0, aims_deg=[0.0, 2.0, 4.0])
    exact = service.misread_cost({**sloped, "reads": [{}, {}]})
    assert exact["cost"] == 0.0 and exact["reads"] == 2
    misread = service.misread_cost({**sloped, "reads": [{"slope_scale": 0.3}]})
    assert misread["expected_strokes"] == pytest.approx(
        misread["perfect_read_strokes"] + misread["cost"]
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"distance_ft": 0.0},
        {"distance_ft": 500.0},
        {"grade_pct": 12.0},
        {"stimp": 2.0},
        {"aims_deg": []},
        {"aims_deg": [90.0]},
        {"paces_ft": [-1.0]},
        {"objective": "longest"},
        {"n": 1},
        {"n": True},
        {"line_sd_deg": -1.0},
        {"baseline": None},
        {"n": service.MAX_ROLLOUTS},  # 4 candidates x n is over the cap
    ],
)
def test_bad_arguments_raise_value_error(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        service.optimize_putt(_args(**overrides))


@pytest.mark.parametrize(
    "reads", [None, [], ["x"], [{"slope_scale": 0.0}], [{}] * 2000]
)
def test_bad_reads_raise_value_error(reads: Any) -> None:
    with pytest.raises(ValueError):
        service.misread_cost(_args(reads=reads))


CLASSES = [
    {"distance_ft": 4.0, "count": 3.0},
    {"distance_ft": 12.0, "slope_pct": 2.0, "orientation": "right_to_left"},
]


def _library_args(**overrides: Any) -> dict[str, Any]:
    args = {**_args(n=8), "classes": CLASSES}
    del args["distance_ft"]
    return {**args, **overrides}


def test_library_study_weights_classes_by_count() -> None:
    report = service.library_study(_library_args())
    json.dumps(report)
    assert report["schema_version"] == 1 and report["samples_per_action"] == 8
    assert [c["weight"] for c in report["classes"]] == pytest.approx([0.75, 0.25])
    for entry in report["classes"]:
        assert entry["expected_putts"]["se"] >= 0.0
    assert service.library_study(_library_args()) == report  # seeded


@pytest.mark.parametrize(
    ("orientation", "aspect_deg"),
    [("uphill", 180.0), ("left_to_right", 270.0), ("right_to_left", 90.0)],
)
def test_class_surface_falls_along_the_orientation(
    orientation: str, aspect_deg: float
) -> None:
    putt_class = service.library_from_args(
        {
            "classes": [
                {"distance_ft": 10.0, "slope_pct": 2.0, "orientation": orientation}
            ]
        }
    ).classes[0]
    slope = service.class_surface(putt_class).height.gradient(0.0, 0.0)
    want = service.surface_from_args({"grade_pct": 2.0, "aspect_deg": aspect_deg})
    assert slope == pytest.approx(want.height.gradient(0.0, 0.0))


def test_library_grid_stops_at_the_longest_class() -> None:
    assert service._grid_covering(12.0)[-1] == 15.0
    assert service._grid_covering(60.0) == service.LIBRARY_GRID_FT


@pytest.mark.parametrize(
    "classes",
    [
        None,
        [],
        ["x"],
        [{"distance_ft": 1.0}],
        [{"distance_ft": 5.0, "slope_pct": 2.0}],  # sloped but "flat"
        [{"distance_ft": 5.0, "orientation": "uphill"}],  # flat but "uphill"
        [{"distance_ft": 5.0, "count": 0.0}],
        [{"distance_ft": 5.0}, {"distance_ft": 5.0}],
        [{"distance_ft": 5.0}] * (service.MAX_LIBRARY_CLASSES + 1),
    ],
)
def test_bad_library_raises_value_error(classes: Any) -> None:
    with pytest.raises(ValueError):
        service.library_study(_library_args(classes=classes))


def test_library_study_refuses_work_over_the_cap() -> None:
    with pytest.raises(ValueError, match="grid"):
        service.library_study(_library_args(n=service.MAX_LIBRARY_ROLLOUTS // 4))


def _call(server: SimulationMcpServer, name: str, args: dict[str, Any]) -> Any:
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": args},
    }
    return server.handle_request(request)


def _server(token: str | None = None) -> SimulationMcpServer:
    return SimulationMcpServer(security_policy=SecurityPolicy(auth_token=token))


def test_mcp_lists_and_runs_the_putting_tools() -> None:
    server = _server()
    listing = server.handle_request({"jsonrpc": "2.0", "id": 0, "method": "tools/list"})
    names = {tool["name"] for tool in listing["result"]["tools"]}
    assert {"optimize_putt", "putting_misread_cost", "putting_library_study"} <= names
    response = _call(server, "optimize_putt", _args())
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["best"]["expected_strokes"] >= 1.0
    assert 0.0 <= payload["best"]["make_rate"] <= 1.0


def test_mcp_maps_bad_arguments_and_auth_to_json_rpc_errors() -> None:
    assert _call(_server(), "optimize_putt", {})["error"]["code"] == -32602
    denied = _call(_server("secret"), "putting_misread_cost", _args())
    assert denied["error"]["code"] == -32000


def test_mcp_runs_the_library_study() -> None:
    response = _call(_server(), "putting_library_study", _library_args())
    report = json.loads(response["result"]["content"][0]["text"])
    assert report["totals"]["expected_putts_per_round"]["mean"] > 0.0
