"""Engine-agnostic grip roll from adapter ``BodyStates`` (SHOW-A12, #14737).

The shared metric (``grip_roll.address_grip_roll_deg``) is the only formula; the
engine hook only turns world poses of the club and hand bodies, plus each
binding's hand dorsal axis, into its ``GripPoseState``. No assertion here pins
the A7 zero (#14731): these tests prove parity with the shared metric.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.contracts import ContractViolationError
from src.shared.python.grip_contact.binding import BodyStates
from src.shared.python.grip_contact.contact_topology import contact_topology
from src.shared.python.grip_contact.topology_contact import palm_pad_direction
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.hand_spec import wrist_follower_dorsal_axis
from src.shared.python.motion_matching.grip_contract import GripSpec
from src.shared.python.motion_matching.engine_grip_roll import (
    EngineGripAxes,
    engine_address_grip_roll_deg,
    engine_grip_axes_from_grip_frame,
    engine_grip_axes_from_topology,
    engine_grip_axes_from_manifest,
    grip_pose_state_from_body_states,
)
from src.shared.python.motion_matching.grip_roll import (
    GripPoseState,
    address_grip_roll_deg,
)

pytestmark = pytest.mark.unit

BINDING = SimpleNamespace(
    club_body="club", lead_hand_body="lead", trail_hand_body="trail"
)
# Engine shaft axis is local +z (not the metric's +y): the hook must reconcile.
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
    # Shaft (local z) leans 60 deg off vertical toward the golfer, a real lie.
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
    # Re-derive the metric from the pieces: dorsal in the (shaft-aligned) club
    # frame must give the same world dorsal as the hand's own pose does.
    club = _club_rotation().as_matrix()
    world_lead = state.club_rotation @ state.lead_dorsal_club
    expected = (_club_rotation() * Rotation.from_euler("z", 30.0, degrees=True)).apply(
        AXES.lead_dorsal_local
    )
    np.testing.assert_allclose(world_lead, expected, atol=1e-9)
    # column 1 of the metric's club frame is the engine shaft in the world
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
    assert spun[0] - base[0] == pytest.approx(7.0, abs=1e-6)
    assert spun[1] - base[1] == pytest.approx(7.0, abs=1e-6)


def test_missing_body_is_a_precondition_failure() -> None:
    states = BodyStates({"club": _states()["club"], "lead": _states()["lead"]})
    with pytest.raises(ContractViolationError, match="trail"):
        grip_pose_state_from_body_states(states, BINDING, AXES)


def test_axes_must_be_finite_nonzero_unit_able_vectors() -> None:
    with pytest.raises(ContractViolationError):
        EngineGripAxes((0, 0, 0), (1, 0, 0), (0, 1, 0))
    with pytest.raises(ContractViolationError):
        EngineGripAxes((0, 0, 1), (np.nan, 0, 0), (0, 1, 0))


def test_manifest_axes_round_trip_and_default_the_shaft() -> None:
    axes = engine_grip_axes_from_manifest(
        {
            "model_id": "m",
            "hand_dorsal_axis_local": {"lead": [1, 0, 0], "trail": [0, 1, 0]},
        }
    )
    assert axes.club_shaft_local == (0.0, 1.0, 0.0)
    assert axes.lead_dorsal_local == (1.0, 0.0, 0.0)
    shaft = engine_grip_axes_from_manifest(
        {
            "model_id": "m",
            "club_shaft_axis_local": [0, 0, 1],
            "hand_dorsal_axis_local": {"lead": [1, 0, 0], "trail": [0, 1, 0]},
        }
    )
    assert shaft.club_shaft_local == (0.0, 0.0, 1.0)


def test_manifest_without_dorsal_axes_names_the_model() -> None:
    with pytest.raises(ValueError, match="model_x.*hand_dorsal_axis_local"):
        engine_grip_axes_from_manifest({"model_id": "model_x"})


SPEC_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)


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
        assert np.degrees(np.arctan2(z, y)) == pytest.approx(
            (alpha + 180.0 + 180.0) % 360.0 - 180.0, abs=1e-6
        )


def test_palm_directions_agree_with_the_spec_dorsals_on_the_hand_offset() -> None:
    # Frame-independent check against the MuJoCo spec: the angle between the
    # two hands' directions about the shaft. Default grip: 0 deg in both.
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    lead = wrist_follower_dorsal_axis(spec, Hand.LEAD)
    trail = wrist_follower_dorsal_axis(spec, Hand.TRAIL)
    topo = contact_topology()
    p_lead = np.asarray(palm_pad_direction(topo, "lead"))
    p_trail = np.asarray(palm_pad_direction(topo, "trail"))
    assert float(lead @ trail) == pytest.approx(float(p_lead @ p_trail), abs=1e-6)


def test_dorsal_is_the_outward_palm_pad_direction_for_both_hands() -> None:
    # Ruling (A12): dorsal = +u, outward from the shaft through the palm pad.
    topo = contact_topology(_spec_with_alphas(20.0, 45.0))
    axes = engine_grip_axes_from_topology(topo)
    assert axes.lead_dorsal_local == pytest.approx(palm_pad_direction(topo, "lead"))
    assert axes.trail_dorsal_local == pytest.approx(palm_pad_direction(topo, "trail"))
    assert axes.club_shaft_local == (0.0, 1.0, 0.0)


def test_grip_frame_engines_take_the_shaft_from_the_hand_x_axis() -> None:
    # Hand frames whose +x is the shaft (grip-frame convention, as Pinocchio's
    # R_grip): the shaft in the club frame is the hand x column seen from it.
    club = _club_rotation()
    hand = club * Rotation.from_euler("y", 90.0, degrees=True)  # x -> club -z
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
    except ImportError as exc:  # pinocchio runtime absent
        pytest.skip(str(exc))
    topo = contact_topology()
    binding = adapter.resolve_binding(topo)
    states = adapter.sample_states(binding)
    axes = engine_grip_axes_from_grip_frame(states, binding, topo)
    lead, trail = engine_address_grip_roll_deg(states, binding, axes)
    assert -180.0 < lead <= 180.0
    assert lead == pytest.approx(trail, abs=1e-6)  # same alpha, same R_grip
