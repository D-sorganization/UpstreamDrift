"""Heel and toe strikes in the putting studio (#14647, epic #14452).

The studio's strike slider was toe-only (0 to 15 mm), its gear twist closed
the face on a toe strike, and the batch strikes ignored the slider. The
slider now spans the desktop's heel (-) to toe (+) range, carried in
``js/core_impact.js``, and offsets every batch strike. Golden parity: the
range is the desktop's and a toe strike opens the face, as in the core
``strike()``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import generate_studio_impact as gen
from src.shared.python.putting_dynamics import PutterState, strike
from src.tools.putting_green_gui._scene_builder import IMPACT_TOE_RANGE_M

pytestmark = pytest.mark.unit

_STUDIO = Path(__file__).resolve().parents[3] / "src" / "apps" / "putting_studio"
_FIXTURE = _STUDIO / "js" / "core_impact.js"
_PHYSICS = _STUDIO / "js" / "putting_physics.js"
_NODE = shutil.which("node")


def _meta() -> dict:
    return gen.parse(_FIXTURE.read_text(encoding="utf-8"))["meta"]


def test_fixture_carries_the_desktop_range_and_the_core_sign() -> None:
    meta = _meta()
    assert meta["strike_range_mm"] == [1e3 * m for m in IMPACT_TOE_RANGE_M]
    toe = strike(PutterState.typical(1.5), impact_toe_m=0.01).face_twist_rad_s
    assert meta["toe_strike_opens_face"] is (toe > 0.0)


@pytest.mark.skipif(_NODE is None, reason="node is not on PATH")
def test_studio_gear_twist_follows_the_core_sign() -> None:
    # Studio angles are positive to the left (y+), so an opened face starts
    # the ball right: a toe strike lowers the launch, a heel strike raises it.
    calls = ",".join(
        "P.simulateSinglePuttRK4({vClubMps: 1.4, aimDeg: 0, pathErrDeg: 0, "
        f"faceErrDeg: 0, strikeMm: {mm}, stimpFt: 10, distFt: 10, "
        "skidTable: window.CORE_IMPACT.skid}).launchDeg"
        for mm in (-10, 0, 10)
    )
    script = (
        "const window = globalThis;\n"
        + _FIXTURE.read_text(encoding="utf-8")
        + f"\nconst P = require({json.dumps(str(_PHYSICS))});"
        + f"\nprocess.stdout.write(JSON.stringify([{calls}]));"
    )
    assert _NODE is not None
    out = subprocess.run(
        [_NODE, "-e", script], capture_output=True, text=True, check=True, timeout=30
    )
    heel, centre, toe = json.loads(out.stdout)
    assert centre == pytest.approx(0.0, abs=1e-12)
    opens = _meta()["toe_strike_opens_face"]
    assert (toe < centre < heel) is opens


def test_slider_spans_heel_to_toe() -> None:
    html = (_STUDIO / "index.html").read_text(encoding="utf-8")
    tag = re.search(r'<input id="toeOffsetSlider"[^>]*>', html)
    assert tag is not None
    lo, hi = _meta()["strike_range_mm"]
    assert f'min="{lo:.1f}"' in tag.group(0) and f'max="{hi:.1f}"' in tag.group(0)
    assert 'value="0.0"' in tag.group(0)
    state = (_STUDIO / "js" / "state.js").read_text(encoding="utf-8")
    assert "toeOffsetMm: 0.0," in state


def test_batch_strikes_start_from_the_slider() -> None:
    green = (_STUDIO / "js" / "green3d.js").read_text(encoding="utf-8")
    assert "strikeMm: state.toeOffsetMm + dxCB" in green
    assert "strikeMm: state.toeOffsetMm + dxUN" in green


def test_heel_strikes_are_labelled_and_costed_like_toe_strikes() -> None:
    io_js = (_STUDIO / "js" / "io.js").read_text(encoding="utf-8")
    assert "'+' + coreUnitText('strike'" not in io_js
    for name in ("controls_a.js", "green3d.js"):
        text = (_STUDIO / "js" / name).read_text(encoding="utf-8")
        assert "Math.abs(state.toeOffsetMm" in text, name


def test_matrix_marks_the_studio_strike_location_present() -> None:
    matrix = _STUDIO.parents[2] / "src" / "config" / "putting_parity_matrix.json"
    features = json.loads(matrix.read_text(encoding="utf-8"))["features"]
    assert features["strike_location"]["cells"]["studio"].startswith("present |")
