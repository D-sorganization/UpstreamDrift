"""Rigid-frame composition and loop-consistency tests."""

import numpy as np
import pytest

from src.shared.python.motion_matching.native_assembly import RigidFrameGraph


def test_composes_forward_and_inverse_transforms() -> None:
    graph = RigidFrameGraph()
    offset = np.eye(4)
    offset[0, 3] = 2
    graph.connect("a", "b", offset)
    graph.connect("b", "c", offset)
    poses = graph.component("c")
    assert poses["a"][0, 3] == -4
    assert poses["b"][0, 3] == -2


def test_rejects_inconsistent_rigid_cycles_and_invalid_rotations() -> None:
    graph = RigidFrameGraph()
    graph.connect("a", "b", np.eye(4))
    graph.connect("b", "c", np.eye(4))
    offset = np.eye(4)
    offset[2, 3] = 0.01
    graph.connect("a", "c", offset)
    with pytest.raises(ValueError, match="Inconsistent rigid cycle"):
        graph.component("a")
    offset[0, 0] = 2
    with pytest.raises(ValueError, match="rotation"):
        graph.connect("d", "e", offset)
