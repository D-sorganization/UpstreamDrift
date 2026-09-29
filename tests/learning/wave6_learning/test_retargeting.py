"""Wave 6 coverage: src.learning.retargeting.retargeter."""

from __future__ import annotations

import numpy as np
import pytest

from src.learning.retargeting.retargeter import (
    MotionRetargeter,
    SkeletonConfig,
)

pytestmark = pytest.mark.unit


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


def _scaled_skel(name: str) -> SkeletonConfig:
    return SkeletonConfig(
        name=name,
        joint_names=["root", "shoulder", "elbow", "hand"],
        parent_indices=[-1, 0, 1, 2],
        joint_offsets=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.02, 0.0],
                [0.25, 0.0, 0.0],
                [0.15, 0.01, 0.0],
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


class _CountingSkeleton(SkeletonConfig):
    def __post_init__(self) -> None:
        self.kinematic_chain_calls = 0
        self.joint_index_calls = 0
        super().__post_init__()

    def get_joint_index(self, name: str) -> int:
        self.joint_index_calls += 1
        return super().get_joint_index(name)

    def get_kinematic_chain(self, end_joint: str) -> list[str]:
        self.kinematic_chain_calls += 1
        return super().get_kinematic_chain(end_joint)

    def reset_counts(self) -> None:
        self.kinematic_chain_calls = 0
        self.joint_index_calls = 0


class _MarkerNames(list[str]):
    def __init__(self, names: list[str]) -> None:
        super().__init__(names)
        self.index_calls = 0

    def index(self, value: str, *args: object) -> int:
        self.index_calls += 1
        return super().index(value, *args)


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

    def test_retarget_optimization_characterization_output(self) -> None:
        src = _simple_skel("src")
        tgt = _scaled_skel("tgt")
        r = MotionRetargeter(src, tgt)
        motion = np.array(
            [
                [0.1, 0.2, -0.15, 0.05],
                [0.0, -0.25, 0.2, -0.1],
            ],
            dtype=float,
        )

        out = r.retarget(motion, method="optimization")

        # Golden values updated in the fix for #7980: forward kinematics now
        # rotates about each joint's own `joint_axes` entry and applies a
        # joint's rotation to its *descendants* (previously a leaf joint's
        # angle moved its own position, and every axis was assumed to be z).
        np.testing.assert_allclose(
            out,
            np.array(
                [
                    [
                        0.08340713355732823,
                        0.18683814659599052,
                        -0.1537751445489205,
                        0.05,
                    ],
                    [-0.011346501679795895, -0.2563994, 0.1938440, -0.1],
                ]
            ),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_fk_reuses_cached_end_effector_index_chains(self) -> None:
        skeleton = _CountingSkeleton(
            name="counting",
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
        r = MotionRetargeter(skeleton, skeleton)
        skeleton.reset_counts()

        for _ in range(3):
            out = r._compute_end_effector_positions(np.zeros(4), skeleton)

        assert set(out) == {"hand"}
        assert skeleton.kinematic_chain_calls == 0
        assert skeleton.joint_index_calls == 0

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

    def test_retarget_from_mocap_precomputes_marker_indices(self) -> None:
        h = SkeletonConfig.create_humanoid()
        r = MotionRetargeter(h, h)
        marker_names = _MarkerNames(["m1", "m2"])
        mapping = {"m1": "left_shoulder", "missing": "left_elbow"}
        marker_pos = np.ones((4, 2, 3)) * 0.05

        out = r.retarget_from_mocap(marker_pos, marker_names, mapping)

        assert out.shape == (4, h.n_joints)
        assert marker_names.index_calls == 0

    def test_retarget_from_mocap_rejects_bad_marker_shape(self) -> None:
        h = SkeletonConfig.create_humanoid()
        r = MotionRetargeter(h, h)

        with pytest.raises(ValueError, match="marker_positions.shape"):
            r.retarget_from_mocap(np.zeros((2, 1, 3)), ["m1", "m2"], {"m1": "head"})

    def test_compute_end_effector_positions_rejects_non_finite_angles(self) -> None:
        r = MotionRetargeter(_simple_skel("src"), _simple_skel("tgt"))

        with pytest.raises(ValueError, match="joint_angles must contain finite"):
            r._compute_end_effector_positions(
                np.array([0.0, np.nan, 0.0, 0.0]), r.target
            )

    def test_visualize_mapping(self) -> None:
        r = MotionRetargeter(_simple_skel("a"), _simple_skel("b"))
        text = r.visualize_mapping()
        assert "Motion Retargeting" in text
        assert "Mapped joints" in text
