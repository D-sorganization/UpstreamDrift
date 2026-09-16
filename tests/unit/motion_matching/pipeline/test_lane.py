"""Unit tests for pipeline lane configuration and stance detection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.contact_law import GroundPlane

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"


def test_stance_spheres_detects_grounded_spheres() -> None:
    from src.shared.python.motion_matching.pipeline.lane import stance_spheres

    labels = ("RAnkleOut", "LAnkleOut", "RToeIn", "RToeOut", "LToeIn", "LToeOut")
    # Frame 0 flat, frame 1 right ankle lifts by 0.05m (> tolerance 0.02m)
    points = np.zeros((2, len(labels), 3))
    points[1, labels.index("RAnkleOut"), 2] = 0.05
    valid = np.ones((2, len(labels)), dtype=bool)

    stance = stance_spheres(points, valid, labels)
    assert len(stance) == 2
    # Frame 0: both feet flat
    assert "heel_r" in stance[0]
    assert "forefoot_r" in stance[0]
    assert "toe_r" in stance[0]
    assert "heel_l" in stance[0]

    # Frame 1: right heel lifted
    assert "heel_r" not in stance[1]
    assert "heel_l" in stance[1]


def test_stance_spheres_validates_shapes() -> None:
    from src.shared.python.motion_matching.pipeline.lane import stance_spheres

    with pytest.raises(ValueError, match="3D array"):
        stance_spheres(np.zeros((2, 3)), np.ones((2, 3), dtype=bool), ("a", "b", "c"))

    with pytest.raises(ValueError, match="match points"):
        stance_spheres(
            np.zeros((2, 3, 3)), np.ones((2, 2), dtype=bool), ("a", "b", "c")
        )


def test_add_toe_spheres_appends_toe_contacts() -> None:
    from src.shared.python.motion_matching.pipeline.lane import add_toe_spheres

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    updated = add_toe_spheres(spec)

    names = {s["name"] for s in updated["contact"]["spheres"]}
    assert "toe_r" in names
    assert "toe_l" in names


def test_document_bounds_and_wrist_bounds() -> None:
    from src.shared.python.motion_matching.pipeline.lane import (
        document_bounds,
        wrist_bounds,
    )

    wb = wrist_bounds()
    assert "LWInputX" in wb
    assert "RWInputX" in wb
    assert wb["LWInputX"][0] < wb["LWInputX"][1]

    doc = {
        "coordinate_ranges_deg": {
            "SpineInputX": [-10.0, 10.0],
            "LWInputX": [-20.0, 20.0],  # in IK_UNBOUNDED, should be skipped
            "hip_flexion_r": [-30.0, 60.0],  # ends with _r, should be skipped
        }
    }
    db = document_bounds(doc)
    assert "SpineInputX" in db
    assert "LWInputX" not in db
    assert "hip_flexion_r" not in db
    assert db["SpineInputX"][0] < db["SpineInputX"][1]

    with pytest.raises(ValueError, match="ranges low < high"):
        document_bounds({"coordinate_ranges_deg": {"BadJoint": [10.0, -10.0]}})
