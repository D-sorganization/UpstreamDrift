"""Skinned human hands on the golfer's club, squared to the clubface (#15251).

The capsule hands stay the default. A spec that sets
``subject.hand_visual_backend`` to ``skinned_human`` draws the posed pair of
:mod:`hand_backends.skinned_pair` instead, in a frame taken from the clubface:
the top of the grip under the hands is at right angles to the face normal, so
a square face and a neutral grip agree.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import club_assembly as ca
from src.shared.python.model_appearance import grip_hands as gh
from src.shared.python.model_appearance import skinned_club_hands as sch
from src.shared.python.model_appearance.hand_backends.qa import (
    load_hand_visual_quality,
)
from src.shared.python.model_appearance.hand_spec import (
    ResolvedHandSpec,
    resolved_hand_spec,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SPEC = json.loads(
    (
        ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
    ).read_text(encoding="utf-8")
)
_AXIS = np.array([0.0, 1.0, 0.0])  # butt -> head in the club-body frame


def _club() -> ca.ClubAssembly:
    club = ca.assembly_from_spec(SPEC)
    assert club is not None
    return club


def _skinned_spec() -> dict:
    spec = copy.deepcopy(SPEC)
    spec["subject"]["hand_visual_backend"] = "skinned_human"
    return spec


def test_capsule_hands_stay_the_default() -> None:
    assert resolved_hand_spec(SPEC).visual_backend == "capsule"
    assert ResolvedHandSpec().visual_backend == "capsule"


def test_the_spec_selects_the_skinned_hands() -> None:
    assert resolved_hand_spec(_skinned_spec()).visual_backend == "skinned_human"


def test_an_unknown_hand_backend_is_refused() -> None:
    spec = _skinned_spec()
    spec["subject"]["hand_visual_backend"] = "mitten"
    with pytest.raises(ValueError, match="hand_visual_backend"):
        resolved_hand_spec(spec)
    with pytest.raises(ValueError, match="hand_visual_backend"):
        ResolvedHandSpec(visual_backend="mitten")


def test_the_hand_frame_is_squared_to_the_clubface() -> None:
    club = _club()
    frame = sch.skinned_hand_frame(club)
    rotation = frame[:3, :3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(rotation[:, 2], _AXIS, atol=1e-9)
    # The top of the grip is at right angles to the face normal.
    assert float(rotation[:, 0] @ ca.clubface_vector(club)) == pytest.approx(
        0.0, abs=1e-9
    )
    # The origin is on the grip axis, the lead hand's distance below the butt.
    butt = np.array([0.0, -club.length_m, club.axis_offset_m])
    below = sch.owner_grip_spec().lead_below_butt_m
    np.testing.assert_allclose(frame[:3, 3], butt + below * _AXIS, atol=1e-9)


@pytest.fixture(scope="module")
def placed() -> tuple[ca.ClubAssembly, dict, dict]:
    club = _club()
    hands = resolved_hand_spec(_skinned_spec())
    return club, gh.grip_hand_meshes(club, hands=hands), sch.wrist_joints_m(club)


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_the_skinned_pair_replaces_the_capsule_hands(placed) -> None:
    club, meshes, _ = placed
    capsule = gh.grip_hand_meshes(club)
    assert set(meshes) == set(capsule) == {"lead_hand", "trail_hand"}
    for label, mesh in meshes.items():
        # A different mesh, closed and wound outward like the capsule hand.
        assert len(mesh.vertices) != len(capsule[label].vertices)
        assert mesh.volume() > 0.5 * capsule[label].volume()
        edges = np.sort(mesh.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
        _, uses = np.unique(edges, axis=0, return_counts=True)
        assert (uses == 2).all()


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_the_hands_hold_the_club_grip(placed) -> None:
    club, meshes, _ = placed
    butt = np.array([0.0, -club.length_m, club.axis_offset_m])
    limit = float(load_hand_visual_quality()["max_club_grip_sink_m"])
    for mesh in meshes.values():
        rel = mesh.vertices - butt
        along = rel @ _AXIS
        radial = np.linalg.norm(rel - np.outer(along, _AXIS), axis=1)
        on_grip = (along >= 0.0) & (along <= club.grip_length_m)
        tube = np.array([gh.grip_radius_at(club, d) for d in along[on_grip]])
        # The hand is posed on the standard golf grip, slimmer than the club's
        # drawn grip tube, so the skin sinks into the tube (measured 0.97 mm
        # lead, 1.89 mm trail). Solving the pair takes about a minute.
        assert float((tube - radial[on_grip]).max()) <= limit
        assert radial.min() < 0.02  # it is on the grip, not beside it


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_the_wrists_sit_over_the_shaft_squared_to_the_face(placed) -> None:
    club, _, wrists = placed
    quality = load_hand_visual_quality()
    frame = sch.skinned_hand_frame(club)
    top, trail_side = frame[:3, 0], frame[:3, 1]
    lead = wrists["lead"] - frame[:3, 3]
    trail = wrists["trail"] - frame[:3, 3]
    assert abs(float(lead @ trail_side)) <= float(quality["max_lead_wrist_offset_m"])
    low, high = quality["trail_wrist_offset_range_m"]
    assert float(low) <= float(trail @ trail_side) <= float(high)
    # Both wrists are on the top side of the grip, above the shaft at address.
    assert float(lead @ top) > 0.0
    assert float(trail @ top) > 0.0
