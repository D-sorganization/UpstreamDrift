"""Body section tables for the lofts: normalisation and landmarks (#15299)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ASSETS = Path(__file__).resolve().parents[3] / "assets"
sys.path.insert(0, str(ASSETS / "makehuman_cc0"))
sys.path.insert(0, str(ASSETS / "body_cc0"))

import build_body_sections as build  # noqa: E402
from mh_rig_io import MakeHumanRig  # noqa: E402
from mh_sections import Section, SegmentTable  # noqa: E402

pytestmark = pytest.mark.unit

FRAME = np.eye(3)  # lateral x, up y, front z
SHOULDER = {
    "neck_base": "arm/head",
    "joint": "arm/tail",
    "arm_end": "elbow/tail",
    "bones": ["tr*"],
    "cap_bones": ["cap"],
    "min_weight": 0.5,
    "slope_from": 0.5,
    "steps": 3,
    "window": 0.05,
}


def _rig() -> MakeHumanRig:
    """Neck base at the origin, shoulder joint 2 out, the arm hanging down."""
    top = [[-1.0, 0.5, 0.0], [-1.5, 0.25, 0.0], [-2.0, 0.0, 0.0], [-1.5, -3.0, 0.0]]
    cap = [[-2.0, 0.6, 0.0], [-2.8, 0.0, 0.0], [-2.0, -0.9, 0.0]]
    head = [[0.0, 1.0, 0.0], [0.4, 2.2, 0.0], [-0.4, 2.2, 0.3], [0.0, 3.0, -0.5]]
    joints = [[0.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [-2.0, -2.0, 0.0]]
    vertices = np.array(top + cap + head + joints, dtype=float)
    return MakeHumanRig(
        vertices=vertices,
        body_quads=np.array([[0, 1, 2, 3], [4, 5, 6, 7], [7, 8, 9, 10]]),
        bones={
            "arm": {"head": "neck", "tail": "joint", "parent": None},
            "elbow": {"head": "joint", "tail": "elbow", "parent": "arm"},
        },
        joints={"neck": [11], "joint": [12], "elbow": [13]},
        weights={
            "trap": [(i, 1.0) for i in range(4)],
            "cap": [(i, 1.0) for i in (4, 5, 6)],
            "jaw": [(7, 1.0)],
        },
        skeleton_licence="CC0",
        weights_licence="CC0",
    )


def test_rows_are_loft_fractions_then_millimetres() -> None:
    table = SegmentTable(
        2.0,
        (
            Section(0.0, 1.0, 0.0, 0.9, 0.5, 6.0, 2.5),
            Section(1.0, 0.5, 0.0, 0.3, 0.3, 3.0, 2.0),
        ),
    )
    out = build.normalised_rows(table, mm=100.0)
    assert out["rows"] == [
        [0.0, 1.0, 1.0, 0.1, 2.5, 100.0, 90.0, 50.0, 600.0],
        [1.0, 0.5, 0.4286, 0.0, 2.0, 50.0, 30.0, 30.0, 300.0],
    ]
    assert (out["length_mm"], out["belly_along"]) == (200.0, 0.0)
    assert (out["half_width_hint_mm"], out["half_depth_hint_mm"]) == (100.0, 70.0)
    assert out["max_radius_mm"] == pytest.approx(600.0 / (2.0 * np.pi), abs=0.05)
    assert len(out["rows"][0]) == len(build.LOFT_COLUMNS) + len(build.SIZE_COLUMNS)


def test_shoulder_slope_and_cap() -> None:
    out = build.shoulder_measures(_rig(), SHOULDER, FRAME, mm=100.0)
    assert out["neck_base_to_joint_mm"] == 200.0
    assert out["top_rows"] == [[0.5, -50.0], [0.75, -25.0], [1.0, -0.0]]
    assert out["trapezius_slope_deg"] == pytest.approx(26.6, abs=0.05)
    # The vertex below the joint is on the arm's side of it: not part of the cap.
    assert out["deltoid_cap_radius_mm"] == pytest.approx(70.0)
    assert (out["skin_above_joint_mm"], out["skin_outside_joint_mm"]) == (60.0, 80.0)
    with pytest.raises(ValueError, match="no shoulder skin"):
        build.shoulder_measures(
            _rig(), {**SHOULDER, "window": 0.001, "steps": 4}, FRAME, 1
        )


def test_head_landmarks() -> None:
    head = {"jaw_bones": ["jaw"], "min_weight": 0.5, "skull_from": 0.5}
    out = build.head_landmarks(_rig(), head, FRAME, mm=100.0)
    assert out == {"menton_to_vertex_mm": 200.0, "breadth_mm": 80.0, "length_mm": 80.0}


def test_json_is_written_the_way_prettier_keeps_it() -> None:
    record = {"rows": [[0.1, 2.0], [0.2, 3.0]], "names": ["x" * 30, "y" * 30, "z" * 30]}
    text = build.dumps(record)
    assert json.loads(text) == record
    assert "    [0.1, 2.0],\n    [0.2, 3.0]\n" in text
    assert text.count("\n") == 11 and text.endswith("}\n")
