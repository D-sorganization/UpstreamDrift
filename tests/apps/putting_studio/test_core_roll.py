"""Core skid-to-roll speed in the putting studio (#14649, epic #14452).

The core ball leaves the skid at about 5/7 of its launch speed, so it rolls
little more than half as far as pure roll would carry it. The studio kept
almost all its speed through the skid and hid the gap with an uncited 0.96
derate in its speed solve. ``js/core_impact.js`` now gives the core's
roll-start speed for each ball speed; the studio putt takes it at the end of
the skid and the speed solve inverts the same table. Golden parity: on a
flat green the studio rolls as far as the core.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import generate_studio_impact as gen
from src.shared.python.putting_dynamics import PutterState, simulate_strike
from src.shared.python.putting_dynamics.surfaces import SurfaceSpec

pytestmark = pytest.mark.unit

_STUDIO = Path(__file__).resolve().parents[3] / "src" / "apps" / "putting_studio"
_FIXTURE = _STUDIO / "js" / "core_impact.js"
_PHYSICS = _STUDIO / "js" / "putting_physics.js"
_NODE = shutil.which("node")
_requires_node = pytest.mark.skipif(_NODE is None, reason="node is not on PATH")


def _node(expression: str) -> object:
    script = (
        "const window = globalThis;\n"
        + _FIXTURE.read_text(encoding="utf-8")
        + f"\nconst PuttingPhysics = require({json.dumps(str(_PHYSICS))});"
        + f"\nprocess.stdout.write(JSON.stringify({expression}));"
    )
    assert _NODE is not None
    out = subprocess.run(
        [_NODE, "-e", script], capture_output=True, text=True, check=True, timeout=30
    )
    return json.loads(out.stdout)


def _core_rest_m(club_mps: float, stimp_ft: float) -> float:
    """Rest distance of the core typical putter on a flat green, no hole."""
    putt = simulate_strike(
        PutterState.typical(club_mps), SurfaceSpec.flat_uniform(stimp_ft)
    )
    return putt.rest_x_m


def _studio_rest_m(club_mps: float, stimp_ft: float) -> float:
    got = _node(
        f"PuttingPhysics.simulateSinglePuttRK4({{vClubMps: {club_mps!r}, aimDeg: 0, "
        f"pathErrDeg: 0, faceErrDeg: 0, stimpFt: {stimp_ft!r}, distFt: 190, "
        "roughnessMm: 0, skidTable: window.CORE_IMPACT.skid}).finalX"
    )
    assert isinstance(got, float)
    return got


def test_rows_carry_the_core_roll_start_speed() -> None:
    rows = gen.parse(_FIXTURE.read_text(encoding="utf-8"))["skid"]
    assert rows[0] == [0.0, 0.0, 0.0]
    for club in (0.6, 1.4, 2.6):
        assert gen.skid_row(club) in rows
    # A sliding sphere keeps about 5/7 of its speed once it rolls.
    for ball, _, roll in rows[1:]:
        assert 0.6 < roll / ball < 0.8


@pytest.mark.parametrize("stimp", [8.0, 13.0])
def test_roll_start_speed_does_not_depend_on_green_speed(stimp: float) -> None:
    # One table serves every Stimp, like the skid distance.
    assert gen.roll_start_speed_mps(1.4, stimp) == pytest.approx(
        gen.roll_start_speed_mps(1.4, 10.0), abs=1e-6
    )


@_requires_node
@pytest.mark.parametrize("club", [0.8, 1.5, 2.5])
@pytest.mark.parametrize("stimp", [9.0, 12.0])
def test_studio_rolls_as_far_as_the_core_on_a_flat_green(
    club: float, stimp: float
) -> None:
    assert _studio_rest_m(club, stimp) == pytest.approx(
        _core_rest_m(club, stimp), rel=0.03
    )


@_requires_node
@pytest.mark.parametrize("dist_ft", [6.0, 15.0, 30.0])
def test_speed_solve_arrives_at_the_entry_speed(dist_ft: float) -> None:
    entry, stimp = 0.58, 11.0
    club = _node(
        f"PuttingPhysics.computeClubSpeedForCapture({{distFt: {dist_ft!r}, "
        f"entrySpeedMps: {entry!r}, stimpFt: {stimp!r}, "
        "skidTable: window.CORE_IMPACT.skid})"
    )
    assert isinstance(club, float)
    mu = _node(f"PuttingPhysics.stimpToRollingMu({stimp})")
    assert isinstance(mu, float)
    mu_g = 9.80665 * mu
    expected = dist_ft * 0.3048 + entry**2 / (2.0 * mu_g)
    assert _studio_rest_m(club, stimp) == pytest.approx(expected, rel=0.03)


@_requires_node
def test_speed_solve_without_the_table_is_refused() -> None:
    with pytest.raises(subprocess.CalledProcessError):
        _node(
            "PuttingPhysics.computeClubSpeedForCapture({distFt: 10, "
            "entrySpeedMps: 0.58, stimpFt: 11})"
        )


def test_page_drops_the_uncited_derate() -> None:
    physics = _PHYSICS.read_text(encoding="utf-8")
    assert "_CLUB_SPEED_EFFICIENCY_DERATE" not in physics
    green = (_STUDIO / "js" / "green3d.js").read_text(encoding="utf-8")
    assert green.count("skidTable: window.CORE_IMPACT.skid") == 3
