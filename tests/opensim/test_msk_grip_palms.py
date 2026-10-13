"""OSV-9 grip orientation (#12306): palms face each other across the grip.

At the calibrated address pose of every Rajagopal golf model and club, the
trail palm faces the target and the lead palm faces away from it, so the two
palms are parallel and opposed across the grip, as in a golf grip. The hand
frame axes (palm normal +x, fingers -y, thumb +z right / -z left) are those
of the Rajagopal hand meshes. The OpenSim checks skip without OpenSim.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python import club_visuals
from src.engines.physics_engines.opensim.python import msk_club as mc

pytestmark = pytest.mark.unit

MODELS = ("golf_humanoid", "golf_humanoid_scaled")
#: Largest angle (deg) between the trail palm normal and the lead palm's
#: opposed normal, and between the trail palm normal and the target.
PALM_TOL_DEG = 30.0
#: Closest an address coordinate may come to a range bound (rad or m).
BOUND_MARGIN = 0.01


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = float(a @ b) / float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


@pytest.fixture(scope="module")
def cal():  # noqa: ANN201
    osim = pytest.importorskip("opensim")
    if not hasattr(osim, "Model"):
        pytest.skip("opensim in sys.modules is a test double")
    club_visuals.register_geometry_path()
    from src.engines.physics_engines.opensim.python import (
        msk_club_calibration as module,
    )

    return module


@pytest.fixture(scope="module")
def probes(cal):  # noqa: ANN001, ANN201
    return {name: cal.PoseProbe(mc.MODELS_DIR / f"{name}.osim") for name in MODELS}


def test_palm_and_thumb_axes_match_the_hand_meshes(cal) -> None:  # noqa: ANN001
    """The palm normal is +x and the thumb axis is radial for both hands."""
    for side in mc.HAND_BODIES:
        assert cal.PALM_AXIS @ cal.THUMB_AXIS[side] == pytest.approx(0.0)
    # thumb tip is radial: +z on the right hand, -z on the left
    assert cal.THUMB_AXIS["R"][2] > 0 > cal.THUMB_AXIS["L"][2]
    np.testing.assert_allclose(cal.PALM_AXIS, [1.0, 0.0, 0.0])


@pytest.mark.parametrize("club", mc.CLUBS)
@pytest.mark.parametrize("name", MODELS)
def test_trail_palm_faces_the_target_opposite_the_lead_palm(
    cal,
    probes,
    name: str,
    club: str,  # noqa: ANN001
) -> None:
    probe = probes[name]
    probe.set(mc.load_calibration(name, club).address_q)
    trail = probe.body(mc.HAND_BODIES["R"])[:3, :3] @ cal.PALM_AXIS
    lead = probe.body(mc.HAND_BODIES["L"])[:3, :3] @ cal.PALM_AXIS
    assert _angle_deg(trail, -lead) < PALM_TOL_DEG, (name, club)
    assert _angle_deg(trail, cal.TARGET_DIRECTION) < PALM_TOL_DEG, (name, club)


@pytest.mark.parametrize("club", mc.CLUBS)
@pytest.mark.parametrize("name", MODELS)
def test_trail_forearm_is_not_at_a_range_bound(
    cal,
    probes,
    name: str,
    club: str,  # noqa: ANN001
) -> None:
    value = mc.load_calibration(name, club).address_q["pro_sup_r"]
    low, high = probes[name].coordinate_range("pro_sup_r")
    assert low + BOUND_MARGIN < value < high - BOUND_MARGIN, (name, club, value)


@pytest.mark.parametrize("club", mc.CLUBS)
@pytest.mark.parametrize("name", MODELS)
def test_no_address_joint_sits_at_an_address_range_bound(
    cal,
    name: str,
    club: str,  # noqa: ANN001
) -> None:
    calibration = mc.load_calibration(name, club)
    recorded = cal.ADDRESS_BOUND_REASONS.get(name, {})
    for coord, (low, high) in cal.ADDRESS_RANGES.items():
        value = calibration.address_q[coord]
        if coord in recorded:
            assert recorded[coord].strip(), (name, coord)
            continue
        assert low + BOUND_MARGIN < value < high - BOUND_MARGIN, (name, club, coord)
        assert coord not in calibration.report["coordinates_at_bounds"]


# ------------------------------------------------------------ pure numpy
def _spec():  # noqa: ANN202
    from src.engines.physics_engines.opensim.python import (
        msk_club_calibration as module,
    )

    return module


@pytest.mark.parametrize("side", sorted(mc.HAND_BODIES))
def test_specified_grip_rotation_lays_thumb_down_the_shaft_and_palm_to_target(
    side: str,
) -> None:
    module = _spec()
    shaft = np.array([0.72, -0.69, 0.03])
    rot = module.grip_hand_rotation(side, shaft)
    np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-12)
    assert np.linalg.det(rot) == pytest.approx(1.0)
    unit = shaft / np.linalg.norm(shaft)
    np.testing.assert_allclose(rot @ module.THUMB_AXIS[side], unit, atol=1e-12)
    palm = rot @ module.PALM_AXIS
    facing = module.TARGET_DIRECTION if side == "R" else -module.TARGET_DIRECTION
    assert palm @ unit == pytest.approx(0.0, abs=1e-12)
    assert _angle_deg(palm, facing) < 5.0  # the shaft is nearly square to target
    # the trail fingers wrap under the shaft: they point down, toward the ground
    fingers = rot @ np.array([0.0, -1.0, 0.0])
    assert fingers[1] < 0.0


def test_specified_grip_rotation_contracts() -> None:
    module = _spec()
    with pytest.raises(ValueError, match="side"):
        module.grip_hand_rotation("X", np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="parallel"):
        module.grip_hand_rotation("R", module.TARGET_DIRECTION)
    with pytest.raises(ValueError, match="finite"):
        module.grip_hand_rotation("R", np.array([np.nan, 0.0, 0.0]))


@pytest.mark.parametrize("side", sorted(mc.HAND_BODIES))
def test_opensim_hand_orientation_comes_from_the_shared_grip_pose(side: str) -> None:
    """No OpenSim-only grip angle: the shaft diagonal and palm facing are the
    shared :mod:`grip_pose` definition, in the Rajagopal hand-frame axes."""
    from src.shared.python.model_appearance.grip_pose import DEFAULT_GRIP_POSE

    module = _spec()
    hand = module.SIDE_HANDS[side]
    expected = DEFAULT_GRIP_POSE.shaft_direction_in_hand(
        hand, module.RADIAL_AXIS[side], module.DISTAL_AXIS
    )
    np.testing.assert_allclose(module.THUMB_AXIS[side], expected, atol=1e-12)
    assert module.PALM_FACING[side] == DEFAULT_GRIP_POSE.palm_target_sign(hand)
    assert not hasattr(module, "GRIP_DIAGONAL_DEG")
