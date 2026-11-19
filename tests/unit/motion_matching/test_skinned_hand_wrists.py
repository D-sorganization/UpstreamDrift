"""The golfer's wrist joints moved to the skinned hands' wrists (#15253).

``apply_skinned_hand_wrists`` returns a spec revision whose two wrist joints
sit where the posed skinned hands have theirs on the club, so a forearm ends
at a hand. Only joint placements move: masses and inertias stay as they were.
The joints are turned as well as moved (#15297): with the model wrists at the
angles the skinned hands are posed with, each forearm runs along its hand's.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.model_appearance import club_assembly as ca
from src.shared.python.model_appearance import skinned_club_hands as sch
from src.shared.python.model_appearance.hand_backends.skinned_settings import (
    default_settings,
)
from src.shared.python.model_appearance.schema import physics_spec_sha256
from src.shared.python.model_appearance.skinned_body import spec_body_poses
from src.shared.python.motion_matching.execution.skinned_hand_wrists import (
    apply_skinned_hand_wrists,
    skinned_wrist_coordinates_rad,
    wrist_joints_in_club_m,
)
from src.shared.python.motion_matching.full_body_spec import canonical_sha256

pytestmark = [pytest.mark.unit, pytest.mark.slow, pytest.mark.timeout(600)]
ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver_v2.json"
#: (hand, wrist coordinate prefix, suffix of the hand body's name).
HANDS = (("lead", "L", "Clubface Vector"), ("trail", "R", "RHandStandoff"))
#: A model forearm and its hand's forearm run this close (deg) at the hand's
#: own wrist angles; flexing the elbow moves the wrist this squarely thumbward.
ALONG_DEG, THUMBWARD = 1.0, 0.99


@pytest.fixture(scope="module")
def base() -> dict[str, Any]:
    return json.loads(SPEC.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def moved(base: dict[str, Any]) -> dict[str, Any]:
    return apply_skinned_hand_wrists(base)


def _rotations(document: dict[str, Any]) -> list[np.ndarray]:
    closure = document["closure"]
    pa, pb = (
        np.asarray(closure[k], dtype=float) for k in ("placement_a", "placement_b")
    )
    joints = [
        np.asarray(j["child_to_follower"], dtype=float)
        for j in document["joints"]
        if j["child"].endswith(("Clubface Vector", "RHandStandoff"))
    ]
    return [pa[:3, :3], pb[:3, :3], *(j[:3, :3] for j in joints)]


def test_the_wrist_joints_sit_at_the_skinned_wrists(base, moved) -> None:
    club = ca.assembly_from_spec(base)
    assert club is not None
    target = sch.wrist_joints_m(club)
    before, after = wrist_joints_in_club_m(base), wrist_joints_in_club_m(moved)
    for side in ("lead", "trail"):
        np.testing.assert_allclose(after[side], target[side], atol=1e-9)
        # The joints really moved: they were centimetres from the skinned wrists.
        assert float(np.linalg.norm(before[side] - target[side])) > 0.02


def test_only_placements_change_and_the_hands_turn_with_them(base, moved) -> None:
    untouched = copy.deepcopy(base)
    apply_skinned_hand_wrists(base)
    assert base == untouched  # the input is never mutated
    assert moved["bodies"] == base["bodies"]  # masses and inertias as they were
    np.testing.assert_allclose(
        moved["closure"]["placement_b"], base["closure"]["placement_b"]
    )
    for turn in _rotations(moved):  # still rigid placements
        np.testing.assert_allclose(turn @ turn.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(turn) > 0.0
    # The capsule grip's hand turn is gone (#15297): both hands turned on the club.
    was, now = _rotations(base), _rotations(moved)
    assert not np.allclose(now[0], was[0], atol=0.1)  # trail, through the weld
    assert not np.allclose(now[2], was[2], atol=0.1)  # lead wrist joint


def _wrist(document: dict[str, Any], suffix: str) -> dict[str, Any]:
    return next(j for j in document["joints"] if j["child"].endswith(suffix))


def _forearms(
    document: dict[str, Any], coordinates: dict[str, float]
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per hand: wrist point, unit line to the elbow, club turn (world)."""
    poses = spec_body_poses(document, coordinates)
    weld = np.asarray(document["closure"]["placement_a"]) @ np.linalg.inv(
        np.asarray(document["closure"]["placement_b"])
    )
    out = {}
    for hand, _, suffix in HANDS:
        joint = _wrist(document, suffix)
        forearm = poses[joint["parent"]]  # its origin lies on the forearm's line
        wrist = (forearm @ np.asarray(joint["parent_to_base"])[:, 3])[:3]
        line = forearm[:3, 3] - wrist
        club = poses[joint["child"]] @ (weld if hand == "trail" else np.eye(4))
        out[hand] = (wrist, line / np.linalg.norm(line), club[:3, :3])
    return out


def test_the_forearms_run_along_the_hands_at_the_hands_own_wrist_angles(
    base, moved
) -> None:
    """#15297: moved but not turned, the forearms were 44 and 130 degrees off."""
    club = ca.assembly_from_spec(base)
    assert club is not None
    at = skinned_wrist_coordinates_rad(moved)
    posed = dict(default_settings().wrist_deg)
    reach = sch.forearm_reach_m(club)
    forearms = _forearms(moved, at)
    for hand, prefix, _ in HANDS:
        extension, ulnar = posed[hand]
        # The angles are the hands' settings, radial deviation positive.
        assert np.degrees(at[f"{prefix}WInputX"]) == pytest.approx(-ulnar)
        assert abs(np.degrees(at[f"{prefix}WInputY"])) == pytest.approx(extension)
        _, line, club_turn = forearms[hand]
        off = np.degrees(np.arccos(np.clip(line @ (club_turn @ reach[hand][1]), -1, 1)))
        assert off <= ALONG_DEG, (hand, off)


def test_flexing_the_elbow_carries_the_wrist_toward_the_thumb(base, moved) -> None:
    """The model's neutral forearm has the thumb leading as the elbow flexes."""
    club = ca.assembly_from_spec(base)
    assert club is not None
    at = skinned_wrist_coordinates_rad(moved)
    axes = sch.forearm_axes(club)
    for hand, prefix, _ in HANDS:
        low, high = moved["coordinate_ranges_deg"][f"{prefix}EInput"]
        flexed = np.radians(0.5 * (low if abs(low) > abs(high) else high))
        bent = {**at, f"{prefix}EInput": flexed}
        more = {**at, f"{prefix}EInput": flexed * 1.02}
        start, _, club_turn = _forearms(moved, bent)[hand]
        step = _forearms(moved, more)[hand][0] - start
        radial = club_turn @ axes[hand][:, 0]
        assert step @ radial / np.linalg.norm(step) >= THUMBWARD, hand


def test_the_revision_draws_the_skinned_hands_and_records_its_parent(
    base, moved
) -> None:
    assert moved["subject"]["hand_visual_backend"] == "skinned_human"
    revision = moved["spec_revision"]
    assert revision["configuration"] == "skinned_hand_wrists"
    assert (
        revision["version"] == int(base.get("spec_revision", {}).get("version", 0)) + 1
    )
    assert revision["parent_spec_canonical_sha256"] == canonical_sha256(base)
    assert revision["parent_physics_spec_sha256"] == physics_spec_sha256(base)
    assert physics_spec_sha256(moved) != physics_spec_sha256(base)


def test_applying_it_twice_changes_nothing(moved) -> None:
    again = apply_skinned_hand_wrists(moved)
    assert again == moved


def test_a_left_handed_document_is_refused(base) -> None:
    left = copy.deepcopy(base)
    left["subject"]["handedness"] = "left"
    with pytest.raises(ValueError, match="left-handed"):
        apply_skinned_hand_wrists(left)


def test_the_hand_meshes_are_cut_at_the_wrist(base) -> None:
    club = ca.assembly_from_spec(base)
    assert club is not None
    cuff = sch.forearm_cuff_m()
    whole = sch.skinned_hand_meshes(club, cut_at_wrist=False)
    cut = sch.skinned_hand_meshes(club)
    reach = sch.forearm_reach_m(club)
    for label, side in (("lead_hand", "lead"), ("trail_hand", "trail")):
        wrist, along = reach[side]
        far = (cut[label].vertices[np.unique(cut[label].faces)] - wrist) @ along
        # Nothing drawn runs further up the forearm than the cuff; the whole
        # mesh carried a stub several centimetres long.
        assert float(far.max()) <= cuff + 1e-9
        assert float(((whole[label].vertices - wrist) @ along).max()) > cuff + 0.03
        assert len(cut[label].faces) < len(whole[label].faces)
