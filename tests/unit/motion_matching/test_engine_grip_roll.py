"""Engine-agnostic grip roll from adapter ``BodyStates`` (SHOW-A12, #14737).

The shared metric (``grip_roll.address_grip_roll_deg``) is the only formula; the
engine hook only turns world poses of the club and hand bodies, plus each
binding's hand dorsal axis, into its ``GripPoseState``. No assertion here pins
the A7 zero (#14731): these tests prove parity with the shared metric.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.contracts import ContractViolationError
from src.shared.python.grip_contact.binding import BodyStates
from src.shared.python.grip_contact.contact_topology import contact_topology
from src.shared.python.grip_contact.topology_contact import palm_pad_direction
from src.shared.python.motion_matching.engine_grip_roll import (
    EngineGripAxes,
    engine_address_grip_roll_deg,
    engine_grip_axes_from_grip_frame,
    engine_grip_axes_from_manifest,
    engine_grip_axes_from_topology,
    grip_pose_state_from_body_states,
    grip_pose_state_from_club_grip_frames,
)
from src.shared.python.motion_matching.grip_contract import GripSpec
from src.shared.python.motion_matching.grip_roll import (
    GripPoseState,
    address_grip_roll_deg,
)

pytestmark = pytest.mark.unit

BINDING = SimpleNamespace(
    club_body="club", lead_hand_body="lead", trail_hand_body="trail"
)
AXES = EngineGripAxes(
    club_shaft_local=(0.0, 0.0, 1.0),
    lead_dorsal_local=(1.0, 0.0, 0.0),
    trail_dorsal_local=(0.0, 1.0, 0.0),
)


def _pose(rot: Rotation, pos=(0.0, 0.0, 0.0)) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rot.as_matrix()
    out[:3, 3] = pos
    return out


def _club_rotation() -> Rotation:
    return Rotation.from_euler("xyz", [-120.0, 5.0, 20.0], degrees=True)


def _states(spin_deg: float = 0.0) -> BodyStates:
    club = _club_rotation()
    shaft_world = club.apply([0.0, 0.0, 1.0])
    spin = Rotation.from_rotvec(np.radians(spin_deg) * shaft_world)
    hand_lead = spin * club * Rotation.from_euler("z", 30.0, degrees=True)
    hand_trail = spin * club * Rotation.from_euler("z", -40.0, degrees=True)
    return BodyStates(
        {
            "club": _pose(club),
            "lead": _pose(hand_lead, (0.0, 0.0, 0.1)),
            "trail": _pose(hand_trail, (0.0, 0.0, 0.2)),
        }
    )


def test_state_matches_the_shared_metric_input() -> None:
    state = grip_pose_state_from_body_states(_states(), BINDING, AXES)
    assert isinstance(state, GripPoseState)
    club = _club_rotation().as_matrix()
    world_lead = state.club_rotation @ state.lead_dorsal_club
    expected = (_club_rotation() * Rotation.from_euler("z", 30.0, degrees=True)).apply(
        AXES.lead_dorsal_local
    )
    np.testing.assert_allclose(world_lead, expected, atol=1e-9)
    np.testing.assert_allclose(
        state.club_rotation[:, 1], club @ np.array([0.0, 0.0, 1.0]), atol=1e-9
    )


def test_roll_equals_the_shared_metric_on_the_built_state() -> None:
    states = _states()
    state = grip_pose_state_from_body_states(states, BINDING, AXES)
    assert engine_address_grip_roll_deg(states, BINDING, AXES) == pytest.approx(
        address_grip_roll_deg(state)
    )


def test_spin_about_the_shaft_changes_the_roll_one_for_one() -> None:
    base = engine_address_grip_roll_deg(_states(), BINDING, AXES)
    spun = engine_address_grip_roll_deg(_states(spin_deg=7.0), BINDING, AXES)
    assert spun[0] - base[0] == pytest.approx(-7.0, abs=1e-6)
    assert spun[1] - base[1] == pytest.approx(-7.0, abs=1e-6)


def test_missing_body_is_a_precondition_failure() -> None:
    states = BodyStates({"club": _states()["club"], "lead": _states()["lead"]})
    with pytest.raises(ContractViolationError, match="trail"):
        grip_pose_state_from_body_states(states, BINDING, AXES)


def test_axes_must_be_finite_nonzero_unit_able_vectors() -> None:
    for invalid in ((0, 0, 0), (np.nan, 0, 0)):
        with pytest.raises(ContractViolationError):
            EngineGripAxes((0, 0, 1), invalid, (0, 1, 0))


def test_manifest_axes_round_trip_and_default_the_shaft() -> None:
    d = {"lead": [1, 0, 0], "trail": [0, 1, 0]}
    axes = engine_grip_axes_from_manifest(
        {"model_id": "m", "hand_dorsal_axis_local": d}
    )
    assert axes.club_shaft_local == (0.0, 1.0, 0.0)
    assert axes.lead_dorsal_local == (1.0, 0.0, 0.0)
    shaft = engine_grip_axes_from_manifest(
        {
            "model_id": "m",
            "club_shaft_axis_local": [0, 0, 1],
            "hand_dorsal_axis_local": d,
        }
    )
    assert shaft.club_shaft_local == (0.0, 0.0, 1.0)


def test_manifest_without_dorsal_axes_names_the_model() -> None:
    with pytest.raises(ValueError, match="model_x.*hand_dorsal_axis_local"):
        engine_grip_axes_from_manifest({"model_id": "model_x"})


def _spec_with_alphas(lead_deg: float, trail_deg: float) -> GripSpec:
    base = GripSpec()
    return replace(
        base,
        lead=replace(base.lead, alpha_deg=lead_deg),
        trail=replace(base.trail, alpha_deg=trail_deg),
    )


def test_palm_direction_follows_alpha_plus_pi_in_the_grip_plane() -> None:
    topo = contact_topology(_spec_with_alphas(20.0, 45.0))
    for hand, alpha in (("lead", 20.0), ("trail", 45.0)):
        x, y, z = palm_pad_direction(topo, hand)
        assert x == 0.0
        deg = (np.degrees(np.arctan2(z, y)) + 360.0) % 360.0
        assert deg == pytest.approx((alpha + 180.0) % 360.0, abs=1e-6)


def test_dorsal_is_the_outward_palm_pad_direction_for_both_hands() -> None:
    topo = contact_topology(_spec_with_alphas(20.0, 45.0))
    axes = engine_grip_axes_from_topology(topo)
    assert axes.lead_dorsal_local == pytest.approx(palm_pad_direction(topo, "lead"))
    assert axes.trail_dorsal_local == pytest.approx(palm_pad_direction(topo, "trail"))
    assert axes.club_shaft_local == (0.0, 1.0, 0.0)


def test_grip_frame_engines_take_the_shaft_from_the_hand_x_axis() -> None:
    club = _club_rotation()
    hand = club * Rotation.from_euler("y", 90.0, degrees=True)
    states = BodyStates(
        {"club": _pose(club), "lead": _pose(hand), "trail": _pose(hand)}
    )
    axes = engine_grip_axes_from_grip_frame(states, BINDING, contact_topology())
    assert axes.club_shaft_local == pytest.approx((0.0, 0.0, -1.0), abs=1e-9)


@pytest.mark.requires_pinocchio
def test_pinocchio_native_golfer_reads_equal_roll_for_equal_alpha() -> None:
    adapter_mod = pytest.importorskip(
        "src.engines.physics_engines.pinocchio.python.native_golfer_grip_adapter"
    )
    try:
        adapter = adapter_mod.NativePinocchioGolferGripAdapter()
    except ImportError as exc:
        pytest.skip(str(exc))
    topo = contact_topology()
    binding = adapter.resolve_binding(topo)
    states = adapter.sample_states(binding)
    axes = engine_grip_axes_from_grip_frame(states, binding, topo)
    lead, trail = engine_address_grip_roll_deg(states, binding, axes)
    assert -180.0 < lead <= 180.0
    assert lead == pytest.approx(trail, abs=1e-6)


def test_club_grip_frames_give_the_same_state_as_hand_poses() -> None:
    topo = contact_topology(_spec_with_alphas(20.0, 45.0))
    club = _club_rotation()
    g = Rotation.from_euler("y", -90.0, degrees=True)
    states = BodyStates(
        {"club": _pose(club), "lead": _pose(club * g), "trail": _pose(club * g)}
    )
    via_hands = engine_address_grip_roll_deg(
        states, BINDING, engine_grip_axes_from_grip_frame(states, BINDING, topo)
    )
    via_frames = address_grip_roll_deg(
        grip_pose_state_from_club_grip_frames(
            club.as_matrix(), {"L": _pose(g), "R": _pose(g)}, topo
        )
    )
    assert via_frames == pytest.approx(via_hands, abs=1e-9)


def test_club_grip_frames_need_both_sides() -> None:
    with pytest.raises(ContractViolationError, match="R"):
        grip_pose_state_from_club_grip_frames(
            np.eye(3), {"L": np.eye(4)}, contact_topology()
        )
