"""Wave 6 coverage: src.learning.retargeting.retargeter."""

from __future__ import annotations

import numpy as np
import pytest

from src.learning.retargeting.retargeter import (
    MotionRetargeter,
    SkeletonConfig,
)


def _simple_skel(name: str = "s") -> SkeletonConfig:
    return SkeletonConfig(
        name=name,
        joint_names=["root", "shoulder", "elbow", "hand"],
        parent_indices=[-1, 0, 1, 2],
        joint_offsets=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ]
        ),
        semantic_labels={
            "root": "root",
            "left_shoulder": "shoulder",
            "left_elbow": "elbow",
            "left_hand": "hand",
        },
        end_effectors=["hand"],
    )


class TestSkeletonConfig:
    def test_post_init_defaults(self) -> None:
        s = _simple_skel()
        assert s.joint_axes is not None
        assert s.joint_axes.shape == (4, 3)
        assert s.joint_limits is not None
        assert s.joint_limits.shape == (4, 2)
        assert s.n_joints == 4

    def test_validation_parents(self) -> None:
        with pytest.raises(ValueError, match="parent_indices"):
            SkeletonConfig(
                name="x",
                joint_names=["a", "b"],
                parent_indices=[-1],
                joint_offsets=np.zeros((2, 3)),
            )

    def test_validation_offsets(self) -> None:
        with pytest.raises(ValueError, match="joint_offsets"):
            SkeletonConfig(
                name="x",
                joint_names=["a", "b"],
                parent_indices=[-1, 0],
                joint_offsets=np.zeros((1, 3)),
            )

    def test_get_joint_index(self) -> None:
        s = _simple_skel()
        assert s.get_joint_index("elbow") == 2
        with pytest.raises(ValueError, match="not found"):
            s.get_joint_index("nope")

    def test_get_semantic_joint(self) -> None:
        s = _simple_skel()
        assert s.get_semantic_joint("left_shoulder") == "shoulder"
        assert s.get_semantic_joint("missing") is None

    def test_kinematic_chain(self) -> None:
        s = _simple_skel()
        chain = s.get_kinematic_chain("hand")
        assert chain == ["root", "shoulder", "elbow", "hand"]

    def test_create_humanoid(self) -> None:
        h = SkeletonConfig.create_humanoid()
        assert h.n_joints == 22
        assert "head" in h.end_effectors
        assert h.get_joint_index("pelvis") == 0


class TestMotionRetargeter:
    def test_init_and_mapping(self) -> None:
        src = _simple_skel("src")
        tgt = _simple_skel("tgt")
        r = MotionRetargeter(src, tgt)
        mapping = r.get_joint_mapping()
        assert mapping["shoulder"] == "shoulder"
        assert mapping["root"] == "root"

    def test_retarget_direct(self) -> None:
        src = _simple_skel("src")
        tgt = _simple_skel("tgt")
        r = MotionRetargeter(src, tgt)
        motion = np.zeros((3, 4))
        motion[:, 1] = 0.5
        out = r.retarget(motion, method="direct")
        assert out.shape == (3, 4)
        np.testing.assert_allclose(out[:, 1], 0.5)

    def test_retarget_direct_clipped_by_limits(self) -> None:
        src = _simple_skel("src")
        tgt = _simple_skel("tgt")
        # narrow limits on the target
        tgt.joint_limits = np.tile(np.array([-0.2, 0.2]), (4, 1))
        r = MotionRetargeter(src, tgt)
        motion = np.full((1, 4), 5.0)
        out = r.retarget(motion, method="direct")
        assert np.all(out <= 0.2)

    def test_retarget_optimization_runs(self) -> None:
        src = _simple_skel("src")
        tgt = _simple_skel("tgt")
        r = MotionRetargeter(src, tgt)
        motion = np.zeros((2, 4))
        out = r.retarget(motion, method="optimization")
        assert out.shape == (2, 4)

    def test_retarget_ik_runs(self) -> None:
        src = _simple_skel("src")
        tgt = _simple_skel("tgt")
        r = MotionRetargeter(src, tgt)
        out = r.retarget(np.zeros((1, 4)), method="ik")
        assert out.shape == (1, 4)

    def test_retarget_unknown_method(self) -> None:
        r = MotionRetargeter(_simple_skel(), _simple_skel())
        with pytest.raises(ValueError, match="Unknown"):
            r.retarget(np.zeros((1, 4)), method="bogus")

    def test_retarget_from_mocap_default_mapping(self) -> None:
        # Use humanoid skeleton to align with marker name inference table.
        h = SkeletonConfig.create_humanoid()
        r = MotionRetargeter(h, h)
        marker_names = ["LSHO", "RSHO", "LELB"]
        marker_pos = np.zeros((2, 3, 3))
        out = r.retarget_from_mocap(marker_pos, marker_names)
        assert out.shape == (2, h.n_joints)

    def test_retarget_from_mocap_explicit_mapping(self) -> None:
        h = SkeletonConfig.create_humanoid()
        r = MotionRetargeter(h, h)
        marker_names = ["m1"]
        mapping = {"m1": "left_shoulder"}
        marker_pos = np.ones((1, 1, 3)) * 0.05
        out = r.retarget_from_mocap(marker_pos, marker_names, mapping)
        assert out.shape == (1, h.n_joints)

    def test_visualize_mapping(self) -> None:
        r = MotionRetargeter(_simple_skel("a"), _simple_skel("b"))
        text = r.visualize_mapping()
        assert "Motion Retargeting" in text
        assert "Mapped joints" in text
