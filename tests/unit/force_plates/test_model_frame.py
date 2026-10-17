"""OF-03 (#13083): plate placement from a model's declared target frame.

Hand-derived goldens.  open-fp stance frame: +x from plate L toward plate R,
+y toward the ball (the golfer faces +y), so the target is stance -x for a
right-handed golfer and +x for a left-handed one.  The placement yaw maps that
stance direction onto the target frame's ``target_dir``; it never mirrors.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.force_plates import to_ground_reaction_series
from src.shared.python.force_plates.model_frame import (
    TargetedPlacement,
    placement_for_target,
)
from src.shared.python.impact_parameters.target_frame import TargetFrame
from tests.unit.force_plates import _trial_factory as tf

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

ORIGIN = np.array([0.4, -0.2, 0.022])
NATIVE_RH = TargetFrame()  # ADR-0041: target -Y, golfer faces -X
NATIVE_LH = TargetFrame(handedness="left")
SPEC_RH = TargetFrame(target_dir=(0.0, 1.0, 0.0), frame_id="spec:target_+y")


def _yaw(placed: TargetedPlacement) -> float:
    rot = placed.placement.rotation
    return math.atan2(rot[1, 0], rot[0, 0])


def _world(placed: TargetedPlacement, xy: tuple[float, float]) -> np.ndarray:
    return placed.placement.to_world(np.array([xy]))[0]


@pytest.mark.parametrize(
    ("handedness", "target", "yaw_deg", "faces"),
    [
        ("RH", NATIVE_RH, 90.0, (-1.0, 0.0)),
        ("LH", NATIVE_LH, -90.0, (1.0, 0.0)),
        ("RH", SPEC_RH, -90.0, (1.0, 0.0)),
    ],
)
def test_golden_yaw_and_facing(handedness, target, yaw_deg, faces) -> None:
    placed = placement_for_target(handedness, target, origin_m=ORIGIN)
    assert math.degrees(_yaw(placed)) == pytest.approx(yaw_deg)
    facing = _world(placed, (0.0, 1.0)) - _world(placed, (0.0, 0.0))
    np.testing.assert_allclose(facing[:2], faces, atol=1e-12)
    assert np.linalg.det(placed.placement.rotation) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("handedness", "target", "stance", "world_xy"),
    [
        # RH native, yaw +90: (x, y) -> (-y, x) about the origin.
        ("RH", NATIVE_RH, (0.0, 0.0), (0.4, -0.2)),
        ("RH", NATIVE_RH, (-0.15, 0.0), (0.4, -0.35)),
        ("RH", NATIVE_RH, (0.15, 0.1), (0.3, -0.05)),
        # LH native, yaw -90: (x, y) -> (y, -x).
        ("LH", NATIVE_LH, (0.15, 0.0), (0.4, -0.35)),
        ("LH", NATIVE_LH, (-0.15, 0.1), (0.5, -0.05)),
    ],
)
def test_points_map_to_hand_computed_world_points(
    handedness, target, stance, world_xy
) -> None:
    placed = placement_for_target(handedness, target, origin_m=ORIGIN)
    point = _world(placed, stance)
    np.testing.assert_allclose(point, [*world_xy, ORIGIN[2]], atol=1e-12)


@pytest.mark.parametrize(
    ("handedness", "target", "lead_xy"),
    [("RH", NATIVE_RH, (-0.15, 0.0)), ("LH", NATIVE_LH, (0.15, 0.0))],
)
def test_lead_plate_lies_on_the_target_side(handedness, target, lead_xy) -> None:
    # RH lead is the left foot (plate L, stance -x); LH lead is the right foot.
    placed = placement_for_target(handedness, target, origin_m=ORIGIN)
    toward_target = _world(placed, lead_xy) - ORIGIN
    assert float(toward_target @ target.x_t) == pytest.approx(0.15)


def test_result_records_the_frame_and_handedness() -> None:
    placed = placement_for_target("RH", SPEC_RH, origin_m=ORIGIN)
    assert placed.frame_id == "spec:target_+y"
    assert placed.handedness == "RH"


def test_handedness_mismatch_is_refused_not_mirrored() -> None:
    with pytest.raises(ValueError, match="handedness"):
        placement_for_target("LH", NATIVE_RH, origin_m=ORIGIN)
    with pytest.raises(ValueError, match="handedness"):
        placement_for_target("RH", NATIVE_LH, origin_m=ORIGIN)


def test_unknown_handedness_and_wrong_target_type_are_refused() -> None:
    with pytest.raises(ValueError, match="handedness"):
        placement_for_target("right", NATIVE_RH, origin_m=ORIGIN)
    with pytest.raises(TypeError, match="TargetFrame"):
        placement_for_target("RH", {"target_dir": (0, -1, 0)}, origin_m=ORIGIN)


def test_tilted_up_is_refused() -> None:
    # A valid TargetFrame whose up is not world +Z (GCV-1 ground is Z-up).
    tilted = TargetFrame(target_dir=(0.0, 0.0, 1.0), up=(1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match=r"\+Z"):
        placement_for_target("RH", tilted, origin_m=ORIGIN)


def test_origin_is_required_and_finite() -> None:
    with pytest.raises(ValueError, match="origin_m"):
        placement_for_target("RH", NATIVE_RH, origin_m=[0.0, float("nan"), 0.0])


def test_left_handed_series_keeps_anatomical_feet() -> None:
    trial = tf.make_trial(handedness="LH")
    placed = placement_for_target("LH", NATIVE_LH, origin_m=ORIGIN)
    series = to_ground_reaction_series(trial, placed.placement).series
    for foot in ("left", "right"):
        np.testing.assert_allclose(
            series.force_n[foot][:, 2], trial.load_n[foot], equal_nan=True
        )
    # The left plate (stance -x) lands on the far side of the target for LH.
    left_cop = series.cop_m["left"]
    loaded = ~np.isnan(left_cop[:, 0])
    assert loaded.any()
    assert np.all(left_cop[loaded] @ NATIVE_LH.x_t < ORIGIN @ NATIVE_LH.x_t)
