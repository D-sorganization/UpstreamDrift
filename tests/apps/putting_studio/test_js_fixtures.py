"""Node-evaluated parity checks for the temporary JS putting physics.

Epic #14141 step 8a. ``js/putting_physics.js`` is a temporary
hand-kept copy (see ``src/apps/putting_studio/SUNSET.md``); the Python
``putting_dynamics`` core is authoritative. Runs the JS under node
against the same shared fixtures as
``tests/unit/putting_dynamics/test_reference_fixtures.py``, reusing
its fixture loader (DRY). Also checks a Stimpmeter-definition
invariant and cross-language parity vs. live Python values -- parity
only, not independent validation.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from src.shared.python.core.physics_constants import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_RADIUS_M,
    GRAVITY_M_S2,
    PUTTER_FACE_COR,
)
from src.shared.python.putting_dynamics import (
    HOLE_RADIUS_M,
    STIMP_RELEASE_SPEED_MPS,
    STIMPMETER_RELEASE_ANGLE_DEG,
    capture_speed_mps,
    load_reference_fixtures,
    stimp_to_rolling_mu,
)

_PY_CONSTS = {
    "GRAVITY_M_S2": float(GRAVITY_M_S2),
    "STIMP_RELEASE_SPEED_MPS": STIMP_RELEASE_SPEED_MPS,
    "STIMPMETER_RELEASE_ANGLE_DEG": STIMPMETER_RELEASE_ANGLE_DEG,
    "HOLE_RADIUS_M": HOLE_RADIUS_M,
    "GOLF_BALL_RADIUS_M": float(GOLF_BALL_RADIUS_M),
    "GOLF_BALL_MASS_KG": float(GOLF_BALL_MASS_KG),
    "PUTTER_FACE_COR": float(PUTTER_FACE_COR),
}

pytestmark = pytest.mark.unit

_JS_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "apps"
    / "putting_studio"
    / "js"
    / "putting_physics.js"
)
_NODE = shutil.which("node")
_requires_node = pytest.mark.skipif(_NODE is None, reason="node is not on PATH")

#: maps_to -> JS expression builder. A js fixture with no entry here
#: must fail loudly, not silently skip (see test below).
_JS_CALLS: dict[str, Callable[[dict], str]] = {
    "putting_dynamics.stimp_to_rolling_mu": (
        lambda i: f"PuttingPhysics.stimpToRollingMu({i['stimp_ft']})"
    ),
    "putting_dynamics.STIMP_RELEASE_SPEED_MPS": (
        lambda i: "PuttingPhysics.STIMP_RELEASE_SPEED_MPS"
    ),
    "putting_dynamics.STIMPMETER_RELEASE_ANGLE_DEG": (
        lambda i: "PuttingPhysics.STIMPMETER_RELEASE_ANGLE_DEG"
    ),
    "putting_dynamics.capture_speed_mps": lambda i: "PuttingPhysics.captureSpeedMps(0)",
}


def _run_node(expression: str) -> float:
    script = (
        f"const PuttingPhysics = require({json.dumps(str(_JS_PATH))});"
        f"process.stdout.write(JSON.stringify({expression}));"
    )
    result = subprocess.run(
        [_NODE, "-e", script], capture_output=True, text=True, timeout=30, check=True
    )
    return float(json.loads(result.stdout))


def _js_fixtures() -> list[pytest.param]:
    return [
        pytest.param(fixture, id=fixture.fixture_id)
        for fixture in load_reference_fixtures()
        if "js" in fixture.applies_to and fixture.pending_reason is None
    ]


@_requires_node
@pytest.mark.parametrize("fixture", _js_fixtures())
def test_js_matches_external_fixture(fixture) -> None:
    builder = _JS_CALLS.get(fixture.maps_to)
    if builder is None:
        pytest.fail(
            f"{fixture.fixture_id}: maps_to={fixture.maps_to!r} has no entry in "
            "_JS_CALLS -- add one instead of silently skipping this js fixture"
        )
    actual = _run_node(builder(fixture.inputs))
    expected = fixture.expected
    assert expected.value is not None
    assert actual == pytest.approx(
        expected.value, rel=expected.tolerance_rel, abs=expected.tolerance_abs or 0.0
    ), expected.tolerance_basis


@_requires_node
@pytest.mark.parametrize("stimp_ft", [6.0, 10.0, 13.0])
def test_flat_green_rollout_matches_stimpmeter_definition(stimp_ft: float) -> None:
    """A 1.83 m/s release must roll stimp_ft * 0.3048 m on a flat green.

    This restates the Stimpmeter's own definition; it is not a model
    snapshot.
    """
    final_x = _run_node(
        "PuttingPhysics.simulateSinglePuttRK4("
        "{vClubMps: 1.83, aimDeg: 0, pathErrDeg: 0, faceErrDeg: 0, "
        f"stimpFt: {stimp_ft}, distFt: 60, roughnessMm: 0, isStimpRamp: true}})"
        ".finalX"
    )
    expected_m = stimp_ft * 0.3048
    assert final_x == pytest.approx(expected_m, rel=0.03)


def test_js_file_has_no_cdn_reference_or_global_state() -> None:
    text = _JS_PATH.read_text(encoding="utf-8")
    assert "https://" not in text
    assert "state." not in text


@_requires_node
@pytest.mark.parametrize("js_name, python_value", sorted(_PY_CONSTS.items()))
def test_js_constants_match_python_exactly(js_name: str, python_value: float) -> None:
    """Parity only: every exported JS constant equals the Python value it mirrors."""
    assert _run_node(f"PuttingPhysics.{js_name}") == pytest.approx(
        python_value, rel=1e-9
    )


@_requires_node
def test_js_functions_match_python_exactly() -> None:
    """Parity only: JS functions match the live Python functions they mirror."""
    assert _run_node("PuttingPhysics.stimpToRollingMu(10.0)") == pytest.approx(
        stimp_to_rolling_mu(10.0), rel=1e-9
    )
    assert _run_node("PuttingPhysics.captureSpeedMps(0)") == pytest.approx(
        capture_speed_mps(), rel=1e-9
    )
