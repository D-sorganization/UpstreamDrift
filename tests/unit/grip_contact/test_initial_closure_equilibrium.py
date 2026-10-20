"""The neutral-pose closure preload is self-equilibrated on the club.

The shared preload is an internal hand squeeze, so at the neutral (t = 0) pose
the physical hand-grip pads must exert no net force or moment on the club.
Every hand-grip pad sits at the shared preload; each pad without a ring
partner (the lead heel pad, the ten-finger trail little finger and the trail
index trigger) is opposed by an antagonist pad diametrically across the grip
at the same axial station (#13555).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_contact import GripInterface
from src.shared.python.grip_contact.contact_topology import (
    DEFAULT_TOPOLOGY_PRELOAD_M,
    GripStyle,
    contact_topology,
    pair_surface_gap_m,
)
from src.shared.python.grip_contact.bushing_law import BushingState
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    build_topology_model,
    topology_side_wrench,
)
from src.shared.python.motion_matching.grip_contract import GripSpec

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
SQUEEZE = 1100.0
#: Antagonist pair -> (pair it opposes, active when (style, trigger_finger)).
ANTAGONISTS = {
    "lead_butt_fingers_grip": ("lead_heel_pad_butt", lambda s, t: True),
    "trail_thumb_pad_grip": ("trail_index_trigger_grip", lambda s, t: t),
    "trail_hypothenar_grip": (
        "ten_finger_little_finger_grip",
        lambda s, t: s is GripStyle.TEN_FINGER,
    ),
}
CASES = [(s, t) for s in GripStyle for t in (False, True)]


def _model(style: GripStyle, trigger_finger: bool) -> TopologyContactModel:
    interface = GripInterface.from_spec(json.loads(SPEC_PATH.read_bytes()))
    topo = contact_topology(GripSpec(style=style, trigger_finger=trigger_finger))
    return build_topology_model(interface, topo, squeeze_per_hand_n=SQUEEZE)


def _neutral_club_wrench(model: TopologyContactModel) -> tuple[np.ndarray, np.ndarray]:
    """Net hand-grip force and moment on the club at the neutral pose.

    The moment is taken about the club axis point at the right grip frame's
    axial origin, so both hands' contacts share one reference point.
    """
    rest = BushingState(
        rotation=np.eye(3),
        position_m=np.zeros(3),
        velocity_m_s=np.zeros(3),
        omega_rad_s=np.zeros(3),
    )
    force = np.zeros(3)
    moment = np.zeros(3)
    for side in ("L", "R"):
        axis_z = model.grip_capsule(side).p0_m[2]
        to_club = np.array([model.cylinder.origin_axial_m(side), 0.0, -axis_z])
        for pw in topology_side_wrench(model, side, rest, rest).pair_wrenches.values():
            force += pw.force_b_n
            moment += np.cross(np.asarray(pw.contact_point_m) + to_club, pw.force_b_n)
    return force, moment


@pytest.mark.parametrize(("style", "trigger_finger"), CASES)
def test_neutral_pose_net_club_wrench_is_zero(
    style: GripStyle, trigger_finger: bool
) -> None:
    """At t = 0 the physical pads exert no net force or moment on the club."""
    force, moment = _neutral_club_wrench(_model(style, trigger_finger))
    np.testing.assert_allclose(force, 0.0, atol=1e-9)
    np.testing.assert_allclose(moment, 0.0, atol=1e-9)


@pytest.mark.parametrize(("style", "trigger_finger"), CASES)
def test_every_hand_grip_pad_carries_the_shared_preload(
    style: GripStyle, trigger_finger: bool
) -> None:
    """Every hand-grip pad, antagonists included, sits at the shared preload."""
    topo = contact_topology(GripSpec(style=style, trigger_finger=trigger_finger))
    for pair in topo.hand_grip_pairs():
        gap = pair_surface_gap_m(topo, pair)
        assert gap == pytest.approx(-DEFAULT_TOPOLOGY_PRELOAD_M, abs=1e-12), pair.name


@pytest.mark.parametrize(("style", "trigger_finger"), CASES)
def test_antagonist_opposes_each_unpaired_pad(
    style: GripStyle, trigger_finger: bool
) -> None:
    """An antagonist exists exactly when the pad it opposes does."""
    topo = contact_topology(GripSpec(style=style, trigger_finger=trigger_finger))
    names = {pair.name for pair in topo.pairs}
    for antagonist, (opposed, active) in ANTAGONISTS.items():
        assert (antagonist in names) is active(style, trigger_finger), antagonist
        assert (opposed in names) is active(style, trigger_finger), opposed
