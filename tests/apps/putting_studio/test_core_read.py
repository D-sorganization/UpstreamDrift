"""Core optimum read in the putting studio (#14649, epic #14452).

The studio's Auto-Solve searched aim only, from one putt at a fixed head
speed. ``PuttingPhysics.solveAimCore`` ports the core
``green_reading.solve_aim`` onto the studio putt: the head speed whose
cup-free roll stops 0.3 m past the hole, and the heading whose path crosses
the hole. Golden parity: on the core greens the studio read matches the core
read in speed, aim and verdict.

Two core cases are left out of the parity set, each for a stated reason:

- Severe Slopes falls 5% toward the hole, steeper than the rolling friction
  at these green speeds, so neither model finds a putt that stops.
- Long downhill putts (Undulating Championship at 35 ft): the core ball's
  launch hop carries over the falling green for several metres, and the
  studio ball has no airborne phase. Both reads hole, at different speeds.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.generate_studio_greens import EXTENT_M
from src.shared.python.putting_dynamics import PutterState, green_preset
from src.shared.python.putting_dynamics.green_reading import solve_aim

pytestmark = pytest.mark.unit

_STUDIO = Path(__file__).resolve().parents[3] / "src" / "apps" / "putting_studio"
_PHYSICS = _STUDIO / "js" / "putting_physics.js"
_NODE = shutil.which("node")
_requires_node = pytest.mark.skipif(_NODE is None, reason="node is not on PATH")
_FT = 0.3048


def _js(name: str) -> str:
    return (_STUDIO / "js" / name).read_text(encoding="utf-8")


def _node(expression: str) -> object:
    script = (
        "const window = globalThis;\n"
        + _js("core_impact.js")
        + _js("core_greens.js")
        + f"\nconst PuttingPhysics = require({json.dumps(str(_PHYSICS))});"
        + f"\nprocess.stdout.write(JSON.stringify({expression}));"
    )
    assert _NODE is not None
    out = subprocess.run(
        [_NODE, "-e", script], capture_output=True, text=True, check=True, timeout=60
    )
    return json.loads(out.stdout)


def _studio_read(green: str, dist_ft: float, stimp: float) -> dict[str, object]:
    got = _node(
        f"PuttingPhysics.solveAimCore({{stimpFt: {stimp!r}, distFt: {dist_ft!r}, "
        "skidTable: window.CORE_IMPACT.skid, "
        f"coreGreen: window.CORE_GREENS.presets[{json.dumps(green)}]}})"
    )
    assert isinstance(got, dict)
    return got


@_requires_node
@pytest.mark.parametrize(
    "green", ["flat_practice", "tiered", "undulating_championship"]
)
@pytest.mark.parametrize(("dist_ft", "stimp"), [(8.0, 11.5), (20.0, 10.0)])
def test_studio_read_matches_the_core_read(
    green: str, dist_ft: float, stimp: float
) -> None:
    half = dist_ft * _FT / 2.0
    core = solve_aim(
        green_preset(green).surface(stimp_ft=stimp, extent_m=EXTENT_M),
        (-half, 0.0),
        (half, 0.0),
        PutterState.typical(1.0),
    )
    studio = _studio_read(green, dist_ft, stimp)
    assert studio["clubSpeedMps"] == pytest.approx(core.putter_speed_mps, rel=0.015)
    assert studio["aimDeg"] == pytest.approx(math.degrees(core.heading_rad), abs=0.1)
    assert studio["holed"] is core.holed is True
    assert studio["converged"] is True


@_requires_node
def test_slider_green_read_is_straight_when_flat() -> None:
    got = _node(
        "PuttingPhysics.solveAimCore({stimpFt: 11, distFt: 12, upPct: 0, "
        "crossPct: 0, skidTable: window.CORE_IMPACT.skid})"
    )
    assert isinstance(got, dict)
    assert got["aimDeg"] == pytest.approx(0.0, abs=1e-6)
    assert got["holed"] is True


@_requires_node
def test_read_breaks_against_a_cross_slope() -> None:
    got = _node(
        "PuttingPhysics.solveAimCore({stimpFt: 11, distFt: 12, upPct: 0, "
        "crossPct: 2, skidTable: window.CORE_IMPACT.skid})"
    )
    assert isinstance(got, dict)
    assert got["holed"] is True
    assert abs(got["aimDeg"]) > 0.5


@_requires_node
def test_read_refuses_a_non_positive_overrun() -> None:
    with pytest.raises(subprocess.CalledProcessError):
        _node(
            "PuttingPhysics.solveAimCore({stimpFt: 11, distFt: 12, "
            "skidTable: window.CORE_IMPACT.skid}, 0)"
        )


def test_auto_solve_uses_the_core_read() -> None:
    controls = _js("controls_c.js")
    assert "PuttingPhysics.solveAimCore(" in controls
    assert "for (let cand" not in controls
    # The batch strikes at the read's head speed while the read is current.
    assert "function currentReadKey()" in _js("state.js")
    assert "state.read.key === currentReadKey()" in _js("green3d.js")


def test_matrix_studio_optimal_aim_cell_is_present() -> None:
    matrix = _STUDIO.parents[2] / "src" / "config" / "putting_parity_matrix.json"
    features = json.loads(matrix.read_text(encoding="utf-8"))["features"]
    assert features["optimal_aim"]["cells"]["studio"].startswith("present |")
