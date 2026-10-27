"""Per-hand ``grip_force_limit_n`` field, validation and fail-closed flag (HRG-HX4, #13761).

``grip_force_limit_n`` is a force in newtons. It is not
``GripPose.grip_strength_deg``, which is a visual roll. Per the option (c)
ruling the cap is only ever enforced in the Python bushing law; MuJoCo pad
contact (and every engine without enforcement) must raise on a finite limit.
"""

from __future__ import annotations

import math

import pytest

from src.shared.python.grip_contact.capability_inventory import (
    FORCE_LIMIT_MECHANISMS,
    load_grip_capability_inventory,
)
from src.shared.python.grip_contact.grip_force_limit import UNLIMITED
from src.shared.python.model_appearance.grip_pose import (
    DEFAULT_GRIP_POSE,
    GripPose,
    Hand,
)

pytestmark = pytest.mark.unit

ENGINES = (
    "mujoco",
    "drake",
    "opensim",
    "pinocchio",
    "simscape",
    "jaxsim",
    "myosuite",
)


def test_grip_pose_exposes_per_hand_grip_force_limit_default_unlimited() -> None:
    for hand in Hand:
        assert DEFAULT_GRIP_POSE.grip_force_limit_n(hand) == UNLIMITED
    assert UNLIMITED == "unlimited"
    pose = GripPose(lead_grip_force_limit_n=300.0, trail_grip_force_limit_n=250.0)
    assert pose.grip_force_limit_n(Hand.LEAD) == 300.0
    assert pose.grip_force_limit_n(Hand.TRAIL) == 250.0


@pytest.mark.parametrize(
    "bad", [0.0, -1.0, math.nan, math.inf, "fast", "", True, None, [5.0]]
)
@pytest.mark.parametrize(
    "field", ["lead_grip_force_limit_n", "trail_grip_force_limit_n"]
)
def test_grip_force_limit_rejects_nonpositive_values(field: str, bad: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        GripPose(**{field: bad})


def test_finite_grip_limit_raises_on_engine_without_enforcement() -> None:
    inventory = load_grip_capability_inventory()
    report = inventory.force_limit_enforcement_report()
    assert report["default"] == UNLIMITED
    engines = report["engines"]
    assert set(ENGINES) <= set(engines)
    for engine in ENGINES:
        assert engines[engine] == dict.fromkeys(FORCE_LIMIT_MECHANISMS, False)
        for mechanism in FORCE_LIMIT_MECHANISMS:
            with pytest.raises(ValueError, match=engine):
                inventory.require_grip_force_limit_enforceable(engine, mechanism, 100.0)
    with pytest.raises(ValueError, match="unknown_engine"):
        inventory.require_grip_force_limit_enforceable(
            "unknown_engine", "pad_contact", 1.0
        )


def test_default_unlimited_leaves_grip_pose_and_engines_unchanged() -> None:
    """Regression: the explicit ``unlimited`` default changes nothing."""
    explicit = GripPose(
        lead_grip_force_limit_n=UNLIMITED, trail_grip_force_limit_n=UNLIMITED
    )
    assert explicit == DEFAULT_GRIP_POSE == GripPose()
    for hand in Hand:
        assert explicit.distance_below_butt_m(hand) == (
            DEFAULT_GRIP_POSE.distance_below_butt_m(hand)
        )
    inventory = load_grip_capability_inventory()
    for engine in (*ENGINES, "unknown_engine"):
        for mechanism in FORCE_LIMIT_MECHANISMS:
            inventory.require_grip_force_limit_enforceable(engine, mechanism, UNLIMITED)
