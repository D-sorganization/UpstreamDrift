"""`_grip_offset` grip-roll investigation (HRG-GR1, #13202).

Argon's reported defect: the roll is a ``R_y`` left-multiplied on the 4x4
``child_to_follower``; ``y`` is the shaft only in the lead (club) child frame,
not in the trail ``RHandStandoff`` frame, so a non-zero roll turns the trail
hand about the wrong axis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.motion_matching import anthropometric_geometry as geometry

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
NATIVE = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
SPECS = [
    "full_body_spec_anthro_driver.json",
    "full_body_spec_anthro_iron7.json",
]


def _wrist(doc: dict, suffix: str) -> np.ndarray:
    for joint in doc["joints"]:
        if joint["child"].endswith(suffix):
            return np.asarray(joint["child_to_follower"], float)
    raise AssertionError(suffix)


def _club_frame_rotation(doc: dict, side: str) -> np.ndarray:
    """Rotation of the follower frame in the club (shaft) frame."""
    closure = np.asarray(
        json.loads(NATIVE.read_text())["closure"]["placement_b"], float
    )[:3, :3]
    if side == "L":
        return _wrist(doc, "Clubface Vector")[:3, :3]
    return closure @ _wrist(doc, "RHandStandoff")[:3, :3]


@pytest.mark.parametrize("side", ["L", "R"])
def test_roll_turns_each_hand_about_the_shaft_by_the_roll(side: str) -> None:
    native = json.loads(NATIVE.read_text())
    plain = geometry.build_upper_body(native, stature_m=1.71, mass_kg=78.0)
    rolled = geometry.build_upper_body(
        native, stature_m=1.71, mass_kg=78.0, grip_roll_deg=40.0
    )
    twist = _club_frame_rotation(rolled, side) @ _club_frame_rotation(plain, side).T
    rotvec = np.degrees(Rotation.from_matrix(twist).as_rotvec())
    assert rotvec == pytest.approx([0.0, 40.0, 0.0], abs=1e-9)


def test_roll_keeps_the_hands_rigid_about_the_shaft() -> None:
    """The wrist-to-hand standoff turns about the shaft with the hand."""
    native = json.loads(NATIVE.read_text())
    plain = geometry.build_upper_body(native, stature_m=1.71, mass_kg=78.0)
    rolled = geometry.build_upper_body(
        native, stature_m=1.71, mass_kg=78.0, grip_roll_deg=40.0
    )
    closure_b = np.asarray(native["closure"]["placement_b"], float)
    closure_a = np.asarray(native["closure"]["placement_a"], float)
    for side, suffix in (("L", "Clubface Vector"), ("R", "RHandStandoff")):
        # club <- child frame at the closure weld
        x = np.eye(4) if side == "L" else closure_b @ np.linalg.inv(closure_a)
        # follower position expressed in the club frame
        p0 = (x @ _wrist(plain, suffix))[:3, 3]
        p1 = (x @ _wrist(rolled, suffix))[:3, 3]
        assert abs(p0[1] - p1[1]) < 1e-12, side  # no shift along the shaft
        assert np.hypot(p0[0], p0[2]) == pytest.approx(
            np.hypot(p1[0], p1[2]), abs=1e-12
        )


@pytest.mark.parametrize("name", SPECS)
def test_committed_specs_carry_zero_roll_so_a_fix_cannot_change_them(name: str) -> None:
    doc = json.loads((ROOT / "docs/development/full_body_models" / name).read_text())
    assert doc["subject"]["grip_roll_deg"] == 0.0


def test_zero_roll_default_build_is_unchanged() -> None:
    native = json.loads(NATIVE.read_text())
    a = geometry.build_upper_body(native, stature_m=1.71, mass_kg=78.0)
    b = geometry.build_upper_body(
        native, stature_m=1.71, mass_kg=78.0, grip_roll_deg=0.0
    )
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
