"""DEFAULT_GOLFER_URDF is the Drake canonical URDF, with no dead branch (#13454)."""

from __future__ import annotations

from src.engines.physics_engines.drake.python.motion_matching import simulate
from src.engines.physics_engines.drake.python.motion_matching.humanoid_urdf import (
    CANONICAL_URDF,
)


def test_default_golfer_urdf_is_drake_canonical() -> None:
    assert simulate.DEFAULT_GOLFER_URDF == CANONICAL_URDF


def test_no_dead_pinocchio_default_branch() -> None:
    assert not hasattr(simulate, "_CANONICAL_PINOCCHIO_URDF")
