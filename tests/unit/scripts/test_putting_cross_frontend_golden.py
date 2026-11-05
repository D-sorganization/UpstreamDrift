"""Cross-frontend golden parity (#14449): the fixture stays the core's output,
PyQt6 matches the route within its tolerances and the report stays current."""

from __future__ import annotations

import json

import pytest

from scripts import putting_cross_frontend_golden as gen

pytestmark = pytest.mark.unit

GOLDEN = json.loads(gen.GOLDEN_PATH.read_text(encoding="utf-8"))
TOLERANCE = GOLDEN["tolerance"]


def _leaves(node: object, path: str = "") -> dict[str, object]:
    if isinstance(node, dict):
        return {
            key: value
            for name, child in node.items()
            for key, value in _leaves(child, f"{path}.{name}").items()
        }
    if isinstance(node, list):
        return {
            key: value
            for i, child in enumerate(node)
            for key, value in _leaves(child, f"{path}[{i}]").items()
        }
    return {path: node}


def test_fixture_is_the_cores_current_output() -> None:
    fresh, frozen = _leaves(gen.build_golden()), _leaves(GOLDEN)
    assert fresh.keys() == frozen.keys()
    for key, value in frozen.items():
        if isinstance(value, float):
            assert fresh[key] == pytest.approx(value, abs=TOLERANCE["distance_m"]), key
        else:
            assert fresh[key] == value, key


@pytest.mark.parametrize("putt", GOLDEN["cases"], ids=lambda c: c["name"])
def test_pyqt6_scene_matches_the_route(putt: dict) -> None:
    route, pyqt6 = gen.route_values(putt["response"]), putt["pyqt6"]
    assert route["holed"] == pyqt6["holed"]
    for key in ("total_roll_m", "leave_m", "rest_dy_m"):
        assert route[key] == pytest.approx(pyqt6[key], abs=TOLERANCE["distance_m"])
    assert route["launch_speed_mps"] == pytest.approx(
        pyqt6["launch_speed_mps"], rel=TOLERANCE["launch_speed_rel"]
    )


def test_parity_report_is_current() -> None:
    assert gen.REPORT_PATH.read_text(encoding="utf-8") == gen.render_report(GOLDEN)
