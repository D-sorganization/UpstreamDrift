"""Heavy integration tests for the `mid_hands` frame and floating base on
``golfer.urdf`` (issue #4112, PIN-MODEL-GRIP-FRAME).

These tests validate the cross-engine parity-spec §2.6 contract: the club
is welded to a virtual ``mid_hands`` frame at the geometric centre of the
grip, and the pelvis is exposed as a 6-DOF floating base.

All tests skip gracefully when pinocchio is not installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parents[2]
GOLFER_URDF = (
    REPO_ROOT / "src/engines/physics_engines/pinocchio/models/generated/golfer.urdf"
)
GOLFER_IK_URDF = (
    REPO_ROOT / "src/engines/physics_engines/pinocchio/models/generated/golfer_ik.urdf"
)


def _pin():
    """Import pinocchio or skip the test."""
    try:
        import pinocchio as pin
    except ImportError:
        pytest.skip("pinocchio not installed")
        return None  # pragma: no cover — pytest.skip raises
    return pin


@pytest.fixture(scope="module")
def golfer_with_floating_base():
    """Load ``golfer.urdf`` with a free-flyer root joint."""
    pin = _pin()
    if not GOLFER_URDF.exists():
        pytest.skip(f"Golfer URDF not found at {GOLFER_URDF}")
    model = pin.buildModelFromUrdf(str(GOLFER_URDF), pin.JointModelFreeFlyer())
    data = model.createData()
    return pin, model, data


@pytest.fixture(scope="module")
def golfer_ik_with_floating_base():
    """Load ``golfer_ik.urdf`` with a free-flyer root joint."""
    pin = _pin()
    if not GOLFER_IK_URDF.exists():
        pytest.skip(f"Golfer IK URDF not found at {GOLFER_IK_URDF}")
    model = pin.buildModelFromUrdf(str(GOLFER_IK_URDF), pin.JointModelFreeFlyer())
    data = model.createData()
    return pin, model, data


class TestGolferUrdfLoadsCleanly:
    """Acceptance criterion 1: URDF loads via ``pin.buildModelFromUrdf``."""

    def test_golfer_urdf_loads(self, golfer_with_floating_base) -> None:
        _pin, model, _data = golfer_with_floating_base
        assert model.nq > 0
        assert model.nv > 0
        assert model.nframes > 0

    def test_golfer_ik_urdf_loads(self, golfer_ik_with_floating_base) -> None:
        _pin, model, _data = golfer_ik_with_floating_base
        assert model.nq > 0
        assert model.nv > 0


class TestFloatingBase:
    """Acceptance criterion 3: floating base at the pelvis."""

    def test_root_joint_is_free_flyer(self, golfer_with_floating_base) -> None:
        """``model.joints[1]`` is the free-flyer (joint 0 is the universe)."""
        _pin, model, _data = golfer_with_floating_base
        # Pinocchio: joints[0] is "universe"; the first real joint is at [1].
        assert model.joints[1].shortname() == "JointModelFreeFlyer", (
            f"Expected JointModelFreeFlyer at index 1, got "
            f"{model.joints[1].shortname()}"
        )

    def test_dof_count_matches_spec(self, golfer_with_floating_base) -> None:
        """6 base DOFs + 23 internal DOFs = 29; nq = 30 (quaternion adds w)."""
        _pin, model, _data = golfer_with_floating_base
        assert (
            model.nv == 29
        ), f"Expected nv=29 (6 base + 23 internal), got nv={model.nv}"
        assert model.nq == 30, f"Expected nq=30 (29 + quaternion w), got nq={model.nq}"

    def test_ik_urdf_floating_base(self, golfer_ik_with_floating_base) -> None:
        _pin, model, _data = golfer_ik_with_floating_base
        assert model.joints[1].shortname() == "JointModelFreeFlyer"


class TestMidHandsFrame:
    """Acceptance criterion 2: ``mid_hands`` frame exists at the grip centre."""

    def test_mid_hands_frame_exists(self, golfer_with_floating_base) -> None:
        pin, model, _data = golfer_with_floating_base
        fid = model.getFrameId("mid_hands")
        assert fid != model.nframes, "`mid_hands` frame missing from model"

    def test_mid_hands_at_geometric_centre(self, golfer_with_floating_base) -> None:
        """At the address pose, ``mid_hands`` is within 1 cm of the geometric
        mean of ``hand_left`` and ``hand_right``.

        ``golfer.urdf`` does not carry the ``hand_left_tip`` / ``hand_right_tip``
        markers (those live on ``golfer_ik.urdf``), so we use the hand link
        frames themselves as anchors. The two hands are symmetric across the
        sagittal plane in the neutral configuration, so the geometric mean
        reduces to a point on the body midline at the same height as the hands.
        """
        pin, model, data = golfer_with_floating_base
        q0 = pin.neutral(model)
        pin.framesForwardKinematics(model, data, q0)

        mid_id = model.getFrameId("mid_hands")
        left_id = model.getFrameId("hand_left")
        right_id = model.getFrameId("hand_right")
        assert mid_id != model.nframes
        assert left_id != model.nframes
        assert right_id != model.nframes

        p_mid = data.oMf[mid_id].translation
        p_left = data.oMf[left_id].translation
        p_right = data.oMf[right_id].translation
        midpoint = 0.5 * (p_left + p_right)

        # Spec tolerance: <= 1 cm.
        err = float(np.linalg.norm(p_mid - midpoint))
        assert err <= 0.01, (
            f"mid_hands ({p_mid}) is {err * 1e3:.2f} mm from the geometric "
            f"mean of hand_left and hand_right ({midpoint}); tolerance is "
            f"10 mm per parity-spec §2.6"
        )

    def test_mid_hands_anatomically_between_hands(
        self, golfer_with_floating_base
    ) -> None:
        """``mid_hands`` lies on the segment between the two hand links."""
        pin, model, data = golfer_with_floating_base
        q0 = pin.neutral(model)
        pin.framesForwardKinematics(model, data, q0)

        p_mid = data.oMf[model.getFrameId("mid_hands")].translation
        p_left = data.oMf[model.getFrameId("hand_left")].translation
        p_right = data.oMf[model.getFrameId("hand_right")].translation

        # mid_hands y must lie between the two hand y coordinates.
        y_lo, y_hi = sorted([p_left[1], p_right[1]])
        assert (
            y_lo - 1e-6 <= p_mid[1] <= y_hi + 1e-6
        ), f"mid_hands.y={p_mid[1]} not between hand y range [{y_lo}, {y_hi}]"


class TestClubAttachment:
    """The club is now welded to ``mid_hands``, not ``hand_left``."""

    def test_club_shaft_frame_present(self, golfer_with_floating_base) -> None:
        pin, model, _data = golfer_with_floating_base
        assert model.getFrameId("club_shaft") != model.nframes
        assert model.getFrameId("club_head") != model.nframes

    def test_mid_hands_to_club_shaft_joint_exists(
        self, golfer_with_floating_base
    ) -> None:
        """Joints become frames in pinocchio; the fixed-joint frame name
        is preserved as a frame name."""
        pin, model, _data = golfer_with_floating_base
        # The fixed joint becomes a frame with the joint's child link name.
        # The fact that club_shaft frame exists and its kinematic ancestor
        # is mid_hands is the contract. We verify by checking that at the
        # neutral pose the club_shaft origin is offset from mid_hands by
        # exactly the value declared in the URDF (0, 0, -0.05).
        q0 = pin.neutral(model)
        pin.framesForwardKinematics(model, data := model.createData(), q0)
        p_mid = data.oMf[model.getFrameId("mid_hands")].translation
        p_shaft = data.oMf[model.getFrameId("club_shaft")].translation
        offset = p_shaft - p_mid
        np.testing.assert_allclose(
            offset,
            [0.0, 0.0, -0.05],
            atol=1e-9,
            err_msg="club_shaft must hang 5 cm below mid_hands at neutral pose",
        )


class TestNeutralPoseStillValid:
    """Sanity: the FK pipeline runs end-to-end on the modified model."""

    def test_frames_forward_kinematics_runs(self, golfer_with_floating_base) -> None:
        pin, model, data = golfer_with_floating_base
        q0 = pin.neutral(model)
        pin.framesForwardKinematics(model, data, q0)
        for fid in range(model.nframes):
            T = data.oMf[fid]
            R = T.rotation
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)
            assert abs(np.linalg.det(R) - 1.0) < 1e-9
            assert np.all(np.isfinite(T.translation))
