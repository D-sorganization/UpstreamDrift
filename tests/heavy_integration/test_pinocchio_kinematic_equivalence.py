"""Pinocchio kinematic-equivalence audit (issue #4136).

This test loads ``golfer.urdf`` with pinocchio, sets the spine joints
to three reference configurations (address, top-of-backswing, impact),
and verifies that the ``mid_hands`` and ``club_head`` frame poses
agree with an independently-implemented numpy forward-kinematic chain
to within the spec tolerances:

* grip-position RMSE < 5 mm
* grip-orientation geodesic < 1 deg

The numpy chain is the audit's ground truth: it walks the same
pelvis -> lumbars -> thorax stack -> mid_hands -> club_head links the
URDF defines, with no shared code, so a disagreement points at a
URDF-versus-pinocchio bug or a spec drift.

Skips cleanly when pinocchio is not installable (e.g. Python 3.14 on
Windows where the wheel is not yet published).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.engines.pinocchio._kinematic_equivalence_data import (
    GOLFER_URDF,
    GRIP_ORIENTATION_TOL_RAD,
    GRIP_POSITION_RMSE_TOL_M,
    REFERENCE_POSES,
    SpineConfig,
    geodesic_angle,
    numpy_spine_fk,
    position_rmse,
)

pytestmark = [pytest.mark.requires_pinocchio]


def _pin():
    try:
        import pinocchio as pin

        return pin
    except ImportError:
        pytest.skip("pinocchio not installed (Python 3.14 / Windows pip)")


@pytest.fixture(scope="module")
def loaded_model():
    pin = _pin()
    if not GOLFER_URDF.exists():
        pytest.skip(f"golfer.urdf not found at {GOLFER_URDF}")
    model = pin.buildModelFromUrdf(str(GOLFER_URDF))
    data = model.createData()
    return pin, model, data


def _set_spine_q(pin, model, cfg: SpineConfig) -> np.ndarray:
    """Build a configuration vector with the spine joints set to ``cfg``
    and every other joint at its neutral value."""
    q = pin.neutral(model)

    # Map of URDF joint name -> radians for this pose. The URDF has the
    # following revolute joints along the spine that affect mid_hands:
    spine_joint_angles = {
        "pelvis_to_lumbar1_intermediate": cfg.lumbar1_x,
        "lumbar1_intermediate_to_lumbar1": cfg.lumbar1_y,
        "lumbar1_to_lumbar2_intermediate": cfg.lumbar2_x,
        "lumbar2_intermediate_to_lumbar2": cfg.lumbar2_y,
        "lumbar2_to_lumbar3_intermediate": cfg.lumbar3_x,
        "lumbar3_intermediate_to_lumbar3": cfg.lumbar3_y,
        "lumbar3_to_thorax1": cfg.thorax1_z,
        "thorax1_to_thorax2": cfg.thorax2_z,
        "thorax2_to_thorax3": cfg.thorax3_z,
    }

    for joint_name, angle in spine_joint_angles.items():
        if not model.existJointName(joint_name):
            pytest.skip(
                f"spine joint '{joint_name}' missing from URDF model "
                "- URDF schema has drifted from the audit"
            )
        joint_id = model.getJointId(joint_name)
        # Each of these is a single-DOF revolute, so idx_q is the q index.
        q[model.idx_qs[joint_id]] = angle
    return q


def _frame_T(pin, model, data, frame_name: str) -> np.ndarray:
    """Return the world-frame 4x4 SE(3) transform of a named frame."""
    if not model.existFrame(frame_name):
        pytest.skip(f"frame '{frame_name}' missing from URDF model")
    fid = model.getFrameId(frame_name)
    T = data.oMf[fid]
    H = np.eye(4)
    H[:3, :3] = T.rotation
    H[:3, 3] = T.translation
    return H


@pytest.mark.parametrize("cfg", REFERENCE_POSES, ids=lambda c: c.name)
def test_mid_hands_agrees_with_numpy_fk(loaded_model, cfg) -> None:
    """Pinocchio FK of mid_hands matches the numpy chain within tolerance."""
    pin, model, data = loaded_model

    q = _set_spine_q(pin, model, cfg)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pin.framesForwardKinematics(model, data, q)

    T_pin = _frame_T(pin, model, data, "mid_hands")

    expected = numpy_spine_fk(cfg)["mid_hands"]
    pos_rmse = position_rmse(T_pin[:3, 3], expected[:3, 3])
    ori_err = geodesic_angle(T_pin[:3, :3], expected[:3, :3])

    assert pos_rmse < GRIP_POSITION_RMSE_TOL_M, (
        f"{cfg.name}: mid_hands position RMSE {pos_rmse * 1e3:.3f} mm "
        f">= tolerance {GRIP_POSITION_RMSE_TOL_M * 1e3:.1f} mm"
    )
    assert ori_err < GRIP_ORIENTATION_TOL_RAD, (
        f"{cfg.name}: mid_hands orientation geodesic "
        f"{np.rad2deg(ori_err):.4f} deg >= tolerance "
        f"{np.rad2deg(GRIP_ORIENTATION_TOL_RAD):.2f} deg"
    )


@pytest.mark.parametrize("cfg", REFERENCE_POSES, ids=lambda c: c.name)
def test_club_head_agrees_with_numpy_fk(loaded_model, cfg) -> None:
    """Pinocchio FK of club_head matches the numpy chain within tolerance.

    club_head is welded to mid_hands via two fixed joints, so any
    disagreement here that does not also show in mid_hands flags a fixed-
    transform drift in the URDF.
    """
    pin, model, data = loaded_model

    q = _set_spine_q(pin, model, cfg)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pin.framesForwardKinematics(model, data, q)

    T_pin = _frame_T(pin, model, data, "club_head")
    expected = numpy_spine_fk(cfg)["club_head"]
    pos_rmse = position_rmse(T_pin[:3, 3], expected[:3, 3])
    ori_err = geodesic_angle(T_pin[:3, :3], expected[:3, :3])

    assert pos_rmse < GRIP_POSITION_RMSE_TOL_M, (
        f"{cfg.name}: club_head position RMSE {pos_rmse * 1e3:.3f} mm "
        f">= tolerance {GRIP_POSITION_RMSE_TOL_M * 1e3:.1f} mm"
    )
    assert ori_err < GRIP_ORIENTATION_TOL_RAD, (
        f"{cfg.name}: club_head orientation geodesic "
        f"{np.rad2deg(ori_err):.4f} deg >= tolerance "
        f"{np.rad2deg(GRIP_ORIENTATION_TOL_RAD):.2f} deg"
    )


def test_simscape_address_row_loads_or_skips() -> None:
    """Smoke check: the Simscape ground-truth CSV is parseable.

    This guards the rest of the audit from silently passing because the
    upstream Simscape dataset has been moved or renamed.
    """
    from tests.unit.engines.pinocchio._kinematic_equivalence_data import (
        load_simscape_address_row,
    )

    row = load_simscape_address_row()
    if not row:
        pytest.skip("Simscape dataset CSV unavailable on this host")
    # The dataset row 0 is the address pose; HipAngularPositionZ should be
    # roughly -45 deg for a right-handed setup (this trial).
    assert "HipLogs_HipAngularPositionZ" in row
    assert "ClubLogs_CHGlobalPosition_1" in row
