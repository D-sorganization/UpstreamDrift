"""The drawn grip hands follow the physics wrist frames (SHOW-A15, #14747).

``grip_hand_frame`` used to put the wrist follower's dorsal axis on the mesh
frame's x axis, but the baked hand has its dorsum shell at the grip_pose V
azimuth from x (about -35 deg lead, +75 deg trail). So what was drawn sat 35 and
75 deg away from where the physics attaches the back of the hand. The contract
here is the drawn dorsum itself, not the frame axis.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import club_assembly as ca
from src.shared.python.model_appearance import grip_hands as gh
from src.shared.python.model_appearance.grip_hand_mesh import build_grip_hand_mesh
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.hand_spec import (
    resolved_hand_spec,
    wrist_follower_dorsal_axis,
)

pytestmark = pytest.mark.unit

MODELS = Path(__file__).resolve().parents[3] / "docs/development/full_body_models"
SPECS = {
    "driver": MODELS / "full_body_spec_anthro_driver.json",
    "iron7": MODELS / "full_body_spec_anthro_iron7.json",
    "driver_v2": MODELS / "full_body_spec_anthro_driver_v2.json",
}
DORSUM_TOL_DEG = 2.0
GRIP_AXIS = np.array([0.0, 1.0, 0.0])


def _drawn_dorsum_direction(spec: dict, hand: Hand) -> np.ndarray:
    """Unit direction, off the shaft, of the placed dorsum shell's centroid."""
    club = ca.assembly_from_spec(spec)
    hands = resolved_hand_spec(spec)
    depth = hands.pose.distance_below_butt_m(hand)
    baked = build_grip_hand_mesh(
        hand,
        grip_radius_m=gh.grip_radius_at(club, depth),
        pose=hands.pose,
        params=gh._mesh_params(hands.hand_length_m),
    )
    dorsum = next(c for c in baked.components if c.finger == "dorsum")
    frame = gh.grip_hand_frame(club, hand, hands.pose, hands=hands)
    centroid = (dorsum.mesh.vertices @ frame[:3, :3].T).mean(axis=0)
    centroid -= float(centroid @ GRIP_AXIS) * GRIP_AXIS
    return centroid / np.linalg.norm(centroid)


@pytest.mark.parametrize("key", sorted(SPECS))
@pytest.mark.parametrize("hand", list(Hand))
def test_drawn_dorsum_is_the_wrist_frame_dorsal_axis(key: str, hand: Hand) -> None:
    spec = json.loads(SPECS[key].read_text(encoding="utf-8"))
    expected = wrist_follower_dorsal_axis(spec, hand)
    drawn = _drawn_dorsum_direction(spec, hand)
    angle = math.degrees(math.acos(float(np.clip(drawn @ expected, -1.0, 1.0))))
    assert angle <= DORSUM_TOL_DEG, (key, hand, angle)


@pytest.mark.parametrize("key", sorted(SPECS))
def test_placed_hands_stay_proper_rotations_and_keep_chirality(key: str) -> None:
    spec = json.loads(SPECS[key].read_text(encoding="utf-8"))
    club = ca.assembly_from_spec(spec)
    hands = resolved_hand_spec(spec)
    for hand in Hand:
        rot = gh.grip_hand_frame(club, hand, hands.pose, hands=hands)[:3, :3]
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(rot) == pytest.approx(1.0)
        np.testing.assert_allclose(rot[:, 2], GRIP_AXIS, atol=1e-12)
