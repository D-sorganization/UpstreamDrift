"""Branch joints must remain contiguous in the Pinocchio model."""

import pytest

from src.engines.physics_engines.pinocchio.python.native_model import depth_first_joints


def test_interleaved_native_branches_become_contiguous() -> None:
    joints = [
        {"parent": "world", "child": "base"},
        {"parent": "base", "child": "left"},
        {"parent": "base", "child": "right"},
        {"parent": "left", "child": "left_hand"},
        {"parent": "right", "child": "right_hand"},
    ]
    assert [joint["child"] for joint in depth_first_joints(joints)] == [
        "base",
        "left",
        "left_hand",
        "right",
        "right_hand",
    ]


def test_invalid_tree_is_rejected_before_building_native_model() -> None:
    with pytest.raises(ValueError, match="tree"):
        depth_first_joints([{"parent": "missing", "child": "orphan"}])
    with pytest.raises(ValueError, match="tree"):
        depth_first_joints(
            [
                {"parent": "world", "child": "a"},
                {"parent": "a", "child": "a"},
            ]
        )
