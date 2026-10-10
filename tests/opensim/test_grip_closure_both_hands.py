"""Both hands are constrained in every OpenSim golf model (OSV-2, #11728)."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python import msk_club
from src.engines.physics_engines.opensim.python.grip_closure import (
    constrained_hands,
    constrained_hands_from_text,
    weld_closure_series,
)
from src.engines.physics_engines.opensim.python.tour_matching.address import (
    FROZEN_ADDRESS_TOLERANCE_PROFILE,
)
from src.shared.python.grip_contact import GripInterface, load_coordinate_swing
from src.shared.python.model_appearance.club_assembly import assembly_from_spec
from src.shared.python.model_appearance.grip_pose import DEFAULT_GRIP_POSE, Hand

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "src/engines/physics_engines/opensim/models"
FIXTURES = ROOT / "tests/fixtures/club_face"
SPECS = ROOT / "docs/development/full_body_models"
RAJAGOPAL_MODELS = ("golf_humanoid.osim", "golf_humanoid_scaled.osim")
CLUBS = ("driver", "iron7")
GENERATED = {
    "driver": MODELS / "generated/full_body_anthro_driver.osim",
    "iron7": MODELS / "generated/full_body_anthro_iron7.osim",
}


@pytest.mark.parametrize("model", RAJAGOPAL_MODELS)
def test_rajagopal_models_constrain_both_hands(model: str) -> None:
    """OSV-9 topology: lead ``WeldJoint`` carries the club, trail is closed."""
    hands = constrained_hands(MODELS / model)
    assert hands == {
        "lead": f"WeldJoint:{msk_club.LEAD_JOINT}",
        "trail": f"WeldConstraint:{msk_club.TRAIL_CONSTRAINT}",
    }


@pytest.mark.parametrize("club", sorted(GENERATED))
def test_generated_full_body_models_constrain_both_hands(club: str) -> None:
    hands = constrained_hands(GENERATED[club])
    assert set(hands) == {"lead", "trail"}


def test_a_model_with_only_the_trail_hand_is_reported_as_such(tmp_path: Path) -> None:
    text = (MODELS / "golf_humanoid.osim").read_text()
    start = text.index(f'<WeldJoint name="{msk_club.LEAD_JOINT}">')
    end = text.index("</WeldJoint>", start) + len("</WeldJoint>")
    stripped = tmp_path / "one_hand.osim"
    stripped.write_text(text[:start] + text[end:])
    assert set(constrained_hands(stripped)) == {"trail"}


def test_bushing_grip_holds_both_hands_through_forces() -> None:
    model = ET.Element("Model")
    for set_tag in ("BodySet", "JointSet", "ConstraintSet", "ForceSet"):
        ET.SubElement(ET.SubElement(model, set_tag), "objects")
    bodies = model.find("BodySet/objects")
    assert bodies is not None
    for hand in msk_club.HAND_BODIES.values():
        ET.SubElement(bodies, "Body", {"name": hand})
    calibration = msk_club.load_calibration("golf_humanoid", "driver")
    msk_club.attach_club(
        model, msk_club.load_msk_club("driver"), calibration, grip_model="bushing"
    )
    text = ET.tostring(model, encoding="unicode")
    hands = constrained_hands_from_text(text)
    assert hands == {
        "lead": "BushingForce:grip_bushing_left",
        "trail": "BushingForce:grip_bushing_right",
    }


@pytest.mark.parametrize("club", CLUBS)
def test_grip_pose_agrees_with_the_spec_grip_interface(club: str) -> None:
    """The shared grip pose mirrors the spec, the single source of truth.

    The spec club frame has its origin at the sole and the shaft toward the
    grip along -y, so the butt is at ``y = -length`` and a hand at grip point
    ``y`` sits ``y + length`` below the butt.
    """
    spec = json.loads((SPECS / f"full_body_spec_anthro_{club}.json").read_bytes())
    interface = GripInterface.from_spec(spec)
    assembly = assembly_from_spec(spec)
    assert assembly is not None
    for side, hand in (("L", Hand.LEAD), ("R", Hand.TRAIL)):
        below_butt = float(interface.frame(side).position_m[1]) + assembly.length_m
        assert DEFAULT_GRIP_POSE.distance_below_butt_m(hand) == pytest.approx(
            below_butt, abs=1e-9
        )


@pytest.mark.parametrize("club", CLUBS)
def test_msk_club_grip_points_follow_the_grip_pose(club: str) -> None:
    """The OpenSim club frames carry the shared lead and trail positions."""
    mclub = msk_club.load_msk_club(club)
    butt_y = -mclub.assembly.length_m
    for side, hand in (("L", Hand.LEAD), ("R", Hand.TRAIL)):
        below_butt = float(mclub.grip_points[side][1]) - butt_y
        assert below_butt == pytest.approx(
            DEFAULT_GRIP_POSE.distance_below_butt_m(hand), abs=1e-9
        )


@pytest.mark.parametrize("model", RAJAGOPAL_MODELS)
def test_rajagopal_models_load_with_the_trail_constraint(model: str) -> None:
    osim = pytest.importorskip("opensim")
    loaded = osim.Model(str(MODELS / model))
    loaded.initSystem()
    constraints = loaded.getConstraintSet()
    assert constraints.get(msk_club.TRAIL_CONSTRAINT).get_isEnforced()


@pytest.mark.parametrize("club", sorted(GENERATED))
def test_generated_weld_closure_stays_within_tolerance_on_the_canned_swing(
    club: str,
) -> None:
    pytest.importorskip("opensim")
    spec = json.loads((SPECS / f"full_body_spec_anthro_{club}.json").read_bytes())
    swing = load_coordinate_swing(
        FIXTURES / f"swing_q_{club}.npz",
        FIXTURES / "address_poses.json",
        club,
        spec["coordinate_order"],
    )
    series = weld_closure_series(GENERATED[club], swing.names, swing.q)
    assert series.available, series.reason
    assert series.residual_m is not None and series.residual_m.size == swing.q.shape[0]
    assert series.within(FROZEN_ADDRESS_TOLERANCE_PROFILE.max_grip_closure_m) is True
    assert np.isfinite(series.max_m)
