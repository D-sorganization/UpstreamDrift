"""``ui/src/utils/autoSolve.golden.json`` matches the shared core (#14466).

``autoSolve.test.ts`` holds the React side to the same file: the solve is
``/solve-aim-3d``'s read of ``request`` and the reads are the desktop text.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.api.models.putting_green_requests import Putt3DSimulationRequest
from src.api.routes.putting_green import _build_putting_3d_surface
from src.shared.python.putting_dynamics import solve_aim

pytestmark = pytest.mark.unit

_GOLDEN = Path(__file__).resolve().parents[3] / "ui/src/utils/autoSolve.golden.json"


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(_GOLDEN.read_text(encoding="utf-8"))


def test_solve_is_the_core_read_of_the_request(golden: dict) -> None:
    payload = Putt3DSimulationRequest(**golden["request"])
    surface, _, putter = _build_putting_3d_surface(payload)
    aim = solve_aim(surface, (0.0, 0.0), (payload.hole_x_m, payload.hole_y_m), putter)
    solve = golden["solve"]
    assert solve["putter_speed_mps"] == pytest.approx(aim.putter_speed_mps, abs=1e-9)
    assert solve["aim_heading_deg"] == pytest.approx(
        math.degrees(aim.heading_rad), abs=1e-9
    )
    assert solve["break_m"] == pytest.approx(aim.break_m, abs=1e-9)
    assert (solve["holed"], solve["converged"]) == (aim.holed, aim.converged)


def test_reads_are_the_desktop_text(golden: dict) -> None:
    builder = pytest.importorskip("src.tools.putting_green_gui._scene_builder")
    text = pytest.importorskip("src.tools.putting_green_gui._scene_text")
    from src.shared.python.ui.units import UnitSystem

    solve = golden["solve"]
    solution = builder.PuttSolution(
        putter_speed_ms=solve["putter_speed_mps"],
        aim_deg=solve["aim_heading_deg"],
        break_m=solve["break_m"],
        holed=solve["holed"],
        converged=solve["converged"],
    )
    for key, system in (
        ("metric", UnitSystem.METRIC),
        ("imperial", UnitSystem.IMPERIAL),
    ):
        assert golden["reads"][key] == text.format_read(solution, system).rstrip("\n")
