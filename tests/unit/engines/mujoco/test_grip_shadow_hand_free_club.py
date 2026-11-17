"""MuJoCo free-club scene with Shadow hand visuals (#15122)."""

from __future__ import annotations

import pytest

from src.engines.physics_engines.mujoco.python.grip_shadow_hand_free_club import (
    build_free_club_shadow_hands_mjcf,
)
from src.shared.python.grip_spec import GripSpec

pytestmark = pytest.mark.unit

mujoco = pytest.importorskip("mujoco")


def test_free_club_shadow_hands_mjcf_compiles() -> None:
    xml, mesh_dir, geom_names = build_free_club_shadow_hands_mjcf(GripSpec.legacy())
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.ngeom >= 3
    assert set(geom_names) == {"lead_hand_vis", "trail_hand_vis"}
    assert mesh_dir.is_dir()
    for gname in geom_names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        assert gid >= 0
        assert model.geom_contype[gid] == 0
        assert model.geom_conaffinity[gid] == 0
