"""Tests for src/tools/model_explorer/_chain_model.py — KinematicTree, ChainNode."""

from __future__ import annotations

import pytest

from src.tools.model_explorer._chain_model import ChainNode, KinematicTree
from tests.tools.model_explorer._fixtures import BRANCH_URDF, EE_URDF, SIMPLE_URDF


class TestChainNode:
    def test_leaf_node(self) -> None:
        node = ChainNode(name="foo")
        assert node.is_leaf()
        assert node.get_chain_to_root() == [node]
        assert node.get_all_descendants() == []

    def test_chain_to_root(self) -> None:
        root = ChainNode(name="root")
        child = ChainNode(name="child", parent=root)
        root.children.append(child)
        grand = ChainNode(name="grand", parent=child)
        child.children.append(grand)

        chain = grand.get_chain_to_root()
        assert [n.name for n in chain] == ["root", "child", "grand"]

    def test_get_all_descendants(self) -> None:
        root = ChainNode(name="root")
        a = ChainNode(name="a", parent=root)
        b = ChainNode(name="b", parent=root)
        c = ChainNode(name="c", parent=a)
        root.children.extend([a, b])
        a.children.append(c)

        names = {n.name for n in root.get_all_descendants()}
        assert names == {"a", "b", "c"}

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("left_hand", True),
            ("gripper_link", True),
            ("tool_tip", True),
            ("end_effector", True),
            ("finger_1", True),
            ("head_pan", True),
            ("foot_l", True),
            ("palm_r", True),
            ("base_link", False),
            ("torso", False),
        ],
    )
    def test_is_end_effector(self, name: str, expected: bool) -> None:
        node = ChainNode(name=name)
        # leaf required
        assert node.is_leaf()
        assert node.is_end_effector() is expected

    def test_is_end_effector_non_leaf_returns_false(self) -> None:
        parent = ChainNode(name="hand")
        child = ChainNode(name="finger", parent=parent)
        parent.children.append(child)
        # parent matches hint but has child -> False
        assert parent.is_end_effector() is False


class TestKinematicTree:
    def test_build_simple(self) -> None:
        t = KinematicTree()
        t.build_from_urdf(SIMPLE_URDF)
        assert t.root is not None
        assert t.root.name == "base"
        assert "arm" in t.nodes
        assert "hand" in t.nodes
        assert t.nodes["hand"].depth == 2
        assert t.nodes["arm"].depth == 1
        assert t.nodes["base"].depth == 0

    def test_build_invalid_xml_returns_silently(self) -> None:
        t = KinematicTree()
        t.build_from_urdf("<not really xml")
        assert t.root is None
        assert t.nodes == {}

    def test_empty_string_raises(self) -> None:
        t = KinematicTree()
        with pytest.raises((ValueError, AssertionError)):
            t.build_from_urdf("   ")

    def test_get_chain_linear(self) -> None:
        t = KinematicTree()
        t.build_from_urdf(SIMPLE_URDF)
        chain = t.get_chain("base", "hand")
        assert [n.name for n in chain] == ["base", "arm", "hand"]

    def test_get_chain_across_branches(self) -> None:
        t = KinematicTree()
        t.build_from_urdf(BRANCH_URDF)
        chain = t.get_chain("left_hand", "right_gripper")
        names = [n.name for n in chain]
        # both leaves connect through root
        assert names[0] == "left_hand"
        assert names[-1] == "right_gripper"
        assert "root" in names

    def test_get_chain_missing_link_returns_empty(self) -> None:
        t = KinematicTree()
        t.build_from_urdf(SIMPLE_URDF)
        assert t.get_chain("base", "nope") == []
        assert t.get_chain("nope", "hand") == []

    def test_get_all_chains(self) -> None:
        t = KinematicTree()
        t.build_from_urdf(BRANCH_URDF)
        chains = t.get_all_chains()
        # two leaves -> two root-to-leaf chains
        assert len(chains) == 2
        leaf_names = {c[-1].name for c in chains}
        assert leaf_names == {"left_hand", "right_gripper"}

    def test_get_end_effectors_and_branch_points(self) -> None:
        t = KinematicTree()
        t.build_from_urdf(BRANCH_URDF)
        leaves = {n.name for n in t.get_end_effectors()}
        assert leaves == {"left_hand", "right_gripper"}

        branches = {n.name for n in t.get_branch_points()}
        assert "root" in branches  # root has 2 children

    def test_joint_with_missing_parent_or_child_skipped(self) -> None:
        urdf = """<?xml version="1.0"?>
        <robot name="x">
            <link name="a"/>
            <link name="b"/>
            <joint name="bad" type="fixed">
                <parent link="a"/>
                <!-- missing child -->
            </joint>
        </robot>"""
        t = KinematicTree()
        t.build_from_urdf(urdf)
        # Both links are roots (no edges); first encountered becomes self.root
        assert t.root is not None
        assert t.nodes["a"].children == []
        assert t.nodes["b"].children == []
