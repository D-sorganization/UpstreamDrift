# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for the 3D Bilateral kinematic model (issue #225)."""

from __future__ import annotations

import numpy as np
import pytest

from movement_optimizer.models import (
    Bilateral3DModel,
    Bilateral3DPose,
    BodyModel,
    LagrangianDynamics,
    make_squat_config,
)


@pytest.fixture()
def body() -> BodyModel:
    return BodyModel(75.0, 1.75)


@pytest.fixture()
def model(body: BodyModel) -> Bilateral3DModel:
    return Bilateral3DModel(body)


class TestBilateral3DPose:
    def test_from_sagittal_broadcasts_to_both_legs(self) -> None:
        q = np.array([0.1, -0.2, 0.3])
        pose = Bilateral3DPose.from_sagittal(q)
        assert pose.left_leg == pose.right_leg == (0.1, -0.2, 0.3)
        assert pose.torso == pytest.approx(0.3)

    def test_from_sagittal_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            Bilateral3DPose.from_sagittal(np.array([0.1, 0.2]))


class TestBilateral3DModelConstruction:
    def test_requires_body_model(self) -> None:
        with pytest.raises(TypeError, match="BodyModel"):
            Bilateral3DModel("not a body")  # type: ignore[arg-type]

    def test_negative_stance_rejected(self, body: BodyModel) -> None:
        with pytest.raises(ValueError, match="stance_width_m"):
            Bilateral3DModel(body, stance_width_m=-0.1)

    def test_default_stance_is_visible(self, body: BodyModel) -> None:
        m = Bilateral3DModel(body)
        assert m.stance_width_m >= 0.20

    def test_explicit_stance_respected(self, body: BodyModel) -> None:
        m = Bilateral3DModel(body, stance_width_m=0.35)
        assert m.stance_width_m == pytest.approx(0.35)


class TestTPose:
    """All-zero angles must produce the canonical upright T-pose."""

    def test_t_pose_ankles_on_ground(self, model: Bilateral3DModel) -> None:
        fk = model.forward_kinematics(model.t_pose())
        assert fk["left_ankle"][2] == pytest.approx(0.0)
        assert fk["right_ankle"][2] == pytest.approx(0.0)

    def test_t_pose_shoulder_height_equals_sum_of_segments(self, model: Bilateral3DModel) -> None:
        fk = model.forward_kinematics(model.t_pose())
        expected_height = model.L_shin + model.L_thigh + model.L_torso
        assert fk["shoulder"][2] == pytest.approx(expected_height)

    def test_t_pose_is_bilaterally_symmetric(self, model: Bilateral3DModel) -> None:
        fk = model.forward_kinematics(model.t_pose())
        # Left side is at +y, right at -y; mirroring should match.
        assert fk["left_hip"][1] == pytest.approx(-fk["right_hip"][1])
        assert fk["left_hip"][0] == pytest.approx(fk["right_hip"][0])
        assert fk["left_hip"][2] == pytest.approx(fk["right_hip"][2])

    def test_t_pose_pelvis_midway(self, model: Bilateral3DModel) -> None:
        fk = model.forward_kinematics(model.t_pose())
        pelvis = fk["pelvis"]
        assert pelvis[1] == pytest.approx(0.0)  # midline
        assert pelvis[2] == pytest.approx(model.L_shin + model.L_thigh)


class TestKneeFlexion:
    """Flexing only the knee should produce a known-position check."""

    def test_90deg_knee_flex_drops_hip_by_thigh_length(self, model: Bilateral3DModel) -> None:
        # Flex the left knee 90deg forward: ankle stays, shin stays vertical,
        # thigh now horizontal (pointing +x).  So left_hip should be at
        # (L_thigh, +half_w, L_shin) -- the thigh rotated from "up" to "forward".
        q_knee = np.pi / 2.0
        pose = Bilateral3DPose(
            left_leg=(0.0, q_knee, 0.0),
            right_leg=(0.0, 0.0, 0.0),
            torso=0.0,
        )
        fk = model.forward_kinematics(pose)

        expected = np.array([model.L_thigh, 0.5 * model.stance_width_m, model.L_shin])
        np.testing.assert_allclose(fk["left_hip"], expected, atol=1e-10)

        # Right hip untouched
        expected_right = np.array([0.0, -0.5 * model.stance_width_m, model.L_shin + model.L_thigh])
        np.testing.assert_allclose(fk["right_hip"], expected_right, atol=1e-10)


class TestSagittal2DParity:
    """Symmetric 3D pose must reproduce the 2D sagittal model exactly."""

    @pytest.mark.parametrize(
        "q2d",
        [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.1, -0.2, 0.05]),
            np.array([0.4, -1.5, 0.8]),
            np.array([-0.1, 0.05, -0.05]),
        ],
    )
    def test_projection_matches_2d_chain(
        self, body: BodyModel, model: Bilateral3DModel, q2d: np.ndarray
    ) -> None:
        dyn, *_ = make_squat_config(body, 60.0)
        assert isinstance(dyn, LagrangianDynamics)

        fk2d = dyn.forward_kinematics(q2d)
        pose3d = Bilateral3DPose.from_sagittal(q2d)
        fk3d = model.forward_kinematics(pose3d)

        # Sagittal plane = (x, z) in 3D == (x, y) in 2D.
        def xz(p3: np.ndarray) -> np.ndarray:
            return np.array([p3[0], p3[2]])

        np.testing.assert_allclose(xz(fk3d["left_ankle"]), fk2d["ankle"], atol=1e-12)
        np.testing.assert_allclose(xz(fk3d["right_ankle"]), fk2d["ankle"], atol=1e-12)
        np.testing.assert_allclose(xz(fk3d["left_knee"]), fk2d["knee"], atol=1e-12)
        np.testing.assert_allclose(xz(fk3d["right_knee"]), fk2d["knee"], atol=1e-12)
        np.testing.assert_allclose(xz(fk3d["left_hip"]), fk2d["hip"], atol=1e-12)
        np.testing.assert_allclose(xz(fk3d["right_hip"]), fk2d["hip"], atol=1e-12)
        np.testing.assert_allclose(xz(fk3d["pelvis"]), fk2d["hip"], atol=1e-12)
        np.testing.assert_allclose(xz(fk3d["shoulder"]), fk2d["shoulder"], atol=1e-12)


class TestInputValidation:
    def test_forward_kinematics_rejects_raw_tuple(self, model: Bilateral3DModel) -> None:
        with pytest.raises(TypeError, match="Bilateral3DPose"):
            model.forward_kinematics((0.0, 0.0, 0.0))  # type: ignore[arg-type]


class TestSegmentPairs:
    def test_segment_pairs_reference_valid_joints(self, model: Bilateral3DModel) -> None:
        fk = model.forward_kinematics(model.t_pose())
        for a, b in model.segment_pairs():
            assert a in fk, f"unknown joint {a}"
            assert b in fk, f"unknown joint {b}"
