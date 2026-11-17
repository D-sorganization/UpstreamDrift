"""The golfer's wrist joints moved to the skinned hands' wrists (#15253).

``apply_skinned_hand_wrists`` returns a spec revision whose two wrist joints
sit where the posed skinned hands have theirs on the club, so a forearm ends
at a hand. Only joint placements move: masses and inertias stay as they were.
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
from src.shared.python.model_appearance.schema import physics_spec_sha256
from src.shared.python.motion_matching.execution.skinned_hand_wrists import (
    apply_skinned_hand_wrists,
    wrist_joints_in_club_m,
)
from src.shared.python.motion_matching.full_body_spec import canonical_sha256

pytestmark = [pytest.mark.unit, pytest.mark.slow, pytest.mark.timeout(600)]
ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver_v2.json"


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


def test_only_placements_move(base, moved) -> None:
    untouched = copy.deepcopy(base)
    apply_skinned_hand_wrists(base)
    assert base == untouched  # the input is never mutated
    assert moved["bodies"] == base["bodies"]  # masses and inertias as they were
    for was, now in zip(_rotations(base), _rotations(moved), strict=True):
        np.testing.assert_allclose(now, was, atol=1e-12)
    np.testing.assert_allclose(
        moved["closure"]["placement_b"], base["closure"]["placement_b"]
    )


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
