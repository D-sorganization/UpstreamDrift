"""Back-squat bar geometry: torso shelf, spec grip width, ML bar axis (#12347).

The golf humanoid's hands sit 0.08 m apart on a club axis in front of the
body, so a bar carried there moves the centre of mass ahead of the feet
(#12345).  A back squat carries the bar on the upper back with the standard's
grip width and the bar perpendicular to the sagittal plane.  Every expected
value below is read from the parity standard through :mod:`lift_spec`; none is
copied into this file.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pytest

from src.shared.python.lifting import lift_spec
from src.shared.python.lifting.back_bar import (
    BAR_CENTRE_FRAME,
    BAR_LONG_AXIS_LOCAL,
    LEFT_HAND_FRAME,
    RIGHT_HAND_FRAME,
    BackBarGeometry,
    apply_grip_width,
    bar_axis_world,
    fit_bar_shelf,
    standard_bar_world,
)
from src.shared.python.lifting.bundles import (
    BACK_SQUAT_BUNDLE,
    LIFT_SPEC,
    lift_to_full_body,
    load_seed_pose,
    solve_back_bar_standing,
    stance_frame,
    stance_mid_foot,
)
from src.shared.python.lifting.sagittal_retarget import (
    SagittalFrame,
    SagittalSolution,
)
from src.shared.python.motion_matching.same_input import InputBundle

pytestmark = pytest.mark.unit

LIFT = "squat"
HEIGHT_M = 1.78
BAR_TOLERANCE_M = 0.05
AXIS_TOLERANCE_DEG = 5.0
GRIP_TOLERANCE_M = 1e-3


@dataclass(frozen=True)
class StartPose:
    """What the issue asks of the start pose, measured from joint angles."""

    bar_error_m: float
    grip_width_m: float
    axis_angle_deg: float
    left_hand_is_left: bool


def measure_start_pose(spec_bytes: bytes, q: np.ndarray, height_m: float) -> StartPose:
    """Bar, hands and axis of pose ``q`` against the standard's back-bar spec."""
    from src.engines.physics_engines.mujoco.python.full_body_ik import (
        FullBodyMarkerKinematics,
    )
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )
    from src.shared.python.motion_matching.contact_law import GroundPlane

    kin = FullBodyMarkerKinematics(
        NativeMujocoFullBodyModel(spec_bytes), {"probe": (BAR_CENTRE_FRAME, (0, 0, 0))}
    )
    ground = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)
    frame = stance_frame(kin, q, ground)
    mid_foot = stance_mid_foot(kin, q, ground)
    geometry = BackBarGeometry.from_standard(LIFT, height_m=height_m)
    poses = kin.body_poses(q, [BAR_CENTRE_FRAME, LEFT_HAND_FRAME, RIGHT_HAND_FRAME])
    rotation, centre = poses[BAR_CENTRE_FRAME]
    axis = rotation @ np.asarray(BAR_LONG_AXIS_LOCAL)
    bar_axis = bar_axis_world(geometry, frame)
    hands = poses[LEFT_HAND_FRAME][1] - poses[RIGHT_HAND_FRAME][1]
    cosine = min(1.0, abs(float(axis @ bar_axis)))
    return StartPose(
        bar_error_m=float(
            np.linalg.norm(centre - standard_bar_world(geometry, mid_foot, frame))
        ),
        grip_width_m=abs(float(hands @ axis)),
        axis_angle_deg=math.degrees(math.acos(cosine)),
        left_hand_is_left=bool(hands @ bar_axis > 0.0),
    )


def assert_matches_the_standard(pose: StartPose) -> None:
    assert pose.bar_error_m <= BAR_TOLERANCE_M
    grip = lift_spec.lift_grip(LIFT)
    assert pose.grip_width_m == pytest.approx(grip.width_m, abs=GRIP_TOLERANCE_M)
    assert pose.axis_angle_deg <= AXIS_TOLERANCE_DEG
    assert pose.left_hand_is_left


@pytest.fixture(scope="module")
def spec_bytes() -> bytes:
    return LIFT_SPEC.read_bytes()


# -- the standard supplies the numbers (no copies here) -----------------------


def test_grip_is_read_from_the_standard() -> None:
    raw = json.loads(lift_spec.STANDARD_PATH.read_text(encoding="utf-8"))
    block = raw["exercises"][LIFT]["grip"]
    grip = lift_spec.lift_grip(LIFT)
    assert grip.width_m == block["width_m"]
    assert grip.width_range_m == tuple(block["width_range_m"])
    assert lift_spec.bar_axis() == tuple(raw["grip_point"]["bar_axis"])
    assert lift_spec.standard_height_m() == raw["anthropometrics"]["height_m"]


def test_geometry_scales_the_start_bar_position_with_height() -> None:
    start = lift_spec.lift_via_pose(LIFT, "start").bar_position_m
    scale = HEIGHT_M / lift_spec.standard_height_m()
    geometry = BackBarGeometry.from_standard(LIFT, height_m=HEIGHT_M)
    assert geometry.grip_width_m == lift_spec.lift_grip(LIFT).width_m
    assert geometry.bar_axis == lift_spec.bar_axis()
    np.testing.assert_allclose(geometry.start_bar_position_m, np.array(start) * scale)


@pytest.mark.parametrize("height", [0.0, -1.7, float("nan"), float("inf")])
def test_geometry_rejects_a_bad_height(height: float) -> None:
    with pytest.raises(ValueError, match="height_m"):
        BackBarGeometry.from_standard(LIFT, height_m=height)


@pytest.mark.parametrize("lift", ["deadlift", "bench_press", "not_a_lift"])
def test_geometry_is_only_defined_for_the_back_squat(lift: str) -> None:
    with pytest.raises(ValueError, match="back-bar"):
        BackBarGeometry.from_standard(lift, height_m=HEIGHT_M)


def test_standard_bar_world_places_the_bar_in_the_lifter_frame() -> None:
    geometry = BackBarGeometry.from_standard(LIFT, height_m=HEIGHT_M)
    frame = SagittalFrame(forward=(0.0, 1.0, 0.0), up=(0.0, 0.0, 1.0))
    mid_foot = np.array([2.0, 3.0, 0.0])
    x, y, z = geometry.start_bar_position_m
    expected = (
        mid_foot
        + x * np.array([0, 1.0, 0])
        + y * frame.left
        + z * np.array([0, 0, 1.0])
    )
    np.testing.assert_allclose(
        standard_bar_world(geometry, mid_foot, frame), expected, atol=1e-12
    )


# -- the derived spec carries the grip width -----------------------------------


def _right_hand_y(spec: dict) -> float:
    """Right palm along the club: the closure holds it, it is not a club frame."""
    return float(spec["closure"]["placement_b"][1][3])


def _frame_y(spec: dict, name: str) -> float:
    (frame,) = (f for f in spec["frames"] if f["name"] == name)
    return float(frame["placement"][1][3])


def test_apply_grip_width_separates_the_hands_by_the_width(spec_bytes: bytes) -> None:
    width = lift_spec.lift_grip(LIFT).width_m
    derived = json.loads(apply_grip_width(spec_bytes, width))
    gap = abs(_frame_y(derived, LEFT_HAND_FRAME) - _right_hand_y(derived))
    assert gap == pytest.approx(width, abs=1e-9)
    # the bar centre stays midway between the hands
    mid = 0.5 * (_frame_y(derived, LEFT_HAND_FRAME) + _right_hand_y(derived))
    assert _frame_y(derived, BAR_CENTRE_FRAME) == pytest.approx(mid, abs=1e-9)


def test_apply_grip_width_depends_only_on_the_requested_width(
    spec_bytes: bytes,
) -> None:
    low, high = lift_spec.lift_grip(LIFT).width_range_m
    twice = json.loads(apply_grip_width(apply_grip_width(spec_bytes, high), low))
    once = json.loads(apply_grip_width(spec_bytes, low))
    for name in (LEFT_HAND_FRAME, BAR_CENTRE_FRAME):
        assert _frame_y(twice, name) == pytest.approx(_frame_y(once, name), abs=1e-12)


def test_apply_grip_width_leaves_the_rest_of_the_spec_alone(spec_bytes: bytes) -> None:
    derived = json.loads(apply_grip_width(spec_bytes, 0.6))
    original = json.loads(spec_bytes)
    changed = {key for key in original if original[key] != derived[key]}
    assert changed == {"frames", "bodies"}
    assert derived["closure"] == original["closure"]
    assert derived["coordinate_order"] == original["coordinate_order"]


@pytest.mark.parametrize("width", [0.0, -0.6, float("nan"), float("inf")])
def test_apply_grip_width_rejects_a_bad_width(spec_bytes: bytes, width: float) -> None:
    with pytest.raises(ValueError, match="width_m"):
        apply_grip_width(spec_bytes, width)


def test_apply_grip_width_rejects_a_spec_without_the_hand_frames() -> None:
    with pytest.raises(ValueError, match="frame"):
        apply_grip_width(json.dumps({"frames": [], "bodies": []}).encode(), 0.6)
    with pytest.raises(TypeError, match="spec_bytes"):
        apply_grip_width("not bytes", 0.6)  # type: ignore[arg-type]


# -- the shelf is a torso frame ------------------------------------------------


def test_fit_bar_shelf_round_trips_through_the_torso_frame() -> None:
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    origin = np.array([1.0, 2.0, 3.0])
    shelf = fit_bar_shelf(
        "Hub", rotation, origin, origin + rotation @ [0.0, -0.1, 0.05]
    )
    assert shelf.frame == "Hub"
    np.testing.assert_allclose(shelf.offset_m, [0.0, -0.1, 0.05], atol=1e-12)
    np.testing.assert_allclose(
        shelf.world_position(rotation, origin), origin + rotation @ shelf.offset_m
    )


def test_fit_bar_shelf_rejects_a_bar_off_the_torso() -> None:
    eye, origin = np.eye(3), np.zeros(3)
    with pytest.raises(ValueError, match="shelf"):
        fit_bar_shelf("Hub", eye, origin, np.array([0.0, 0.0, 0.6]))
    with pytest.raises(ValueError, match="finite"):
        fit_bar_shelf("Hub", eye, origin, np.array([0.0, np.nan, 0.0]))


# -- the start pose (the issue's first acceptance item) -------------------------


@pytest.mark.integration
@pytest.mark.requires_mujoco
def test_start_pose_carries_the_bar_on_the_back_per_the_standard(
    spec_bytes: bytes,
) -> None:
    pytest.importorskip("mujoco")
    geometry = BackBarGeometry.from_standard(LIFT, height_m=HEIGHT_M)
    start = solve_back_bar_standing(spec_bytes, load_seed_pose(), geometry)
    assert_matches_the_standard(measure_start_pose(start.spec_bytes, start.q, HEIGHT_M))


@pytest.mark.integration
@pytest.mark.requires_mujoco
@pytest.mark.xfail(
    strict=True,
    reason=(
        "#12347 slice 1: the committed npz predates the back-bar geometry; "
        "regenerating it changes the bar-path, kinetics and barbell evidence "
        "inputs (#12340, #12341, #12345), so it lands with that evidence"
    ),
)
def test_committed_bundle_start_pose_matches_the_standard() -> None:
    """The issue's acceptance item, on the committed bundle (open until slice 2).

    ``strict`` turns the XPASS after regeneration into a failure, so the marker
    cannot outlive the gap it documents.
    """
    pytest.importorskip("mujoco")
    bundle = InputBundle.load(BACK_SQUAT_BUNDLE)
    q = np.asarray(bundle.provenance["ik"]["q_standing"], dtype=float)
    height = float(bundle.provenance["optimizer"]["height_m"])
    assert_matches_the_standard(measure_start_pose(bundle.spec_bytes, q, height))


@pytest.mark.integration
@pytest.mark.requires_mujoco
def test_trajectory_keeps_the_grip_and_the_axis_through_the_descent(
    spec_bytes: bytes,
) -> None:
    pytest.importorskip("mujoco")
    bundle = InputBundle.load(BACK_SQUAT_BUNDLE)
    record = dict(bundle.provenance["sagittal_solution"])
    keep = 6
    solution = SagittalSolution(
        np.asarray(record["time_s"])[:keep],
        np.asarray(record["segment_angles_rad"])[:keep],
        np.asarray(record["bar_m"])[:keep],
        record["source"],
    )
    geometry = BackBarGeometry.from_standard(LIFT, height_m=HEIGHT_M)
    lift = lift_to_full_body(spec_bytes, solution, load_seed_pose(), geometry=geometry)
    assert lift.back_bar is not None
    for q in lift.q:
        pose = measure_start_pose(lift.back_bar.spec_bytes, q, HEIGHT_M)
        assert pose.grip_width_m == pytest.approx(
            lift_spec.lift_grip(LIFT).width_m, abs=GRIP_TOLERANCE_M
        )
        assert pose.axis_angle_deg <= AXIS_TOLERANCE_DEG
        assert pose.left_hand_is_left
    assert float(lift.closure_m.max()) <= 1e-3
