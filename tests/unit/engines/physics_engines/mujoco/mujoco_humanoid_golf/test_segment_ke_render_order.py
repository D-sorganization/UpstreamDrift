"""Stored-KE scene colours must be applied before axial force colours (#15396)."""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.unit


def test_render_once_applies_stored_ke_before_axial_force() -> None:
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf import (
        sim_rendering_mixin as mixin_mod,
    )

    source = inspect.getsource(mixin_mod.SimRenderingMixin._render_once)
    ke_pos = source.index("_apply_stored_ke_scene_colors")
    force_pos = source.index("_apply_axial_scene_colors")
    assert ke_pos < force_pos

    mesh_pos = source.index("_update_meshcat_frame_overlays")
    mesh_source = inspect.getsource(mixin_mod.SimRenderingMixin._update_meshcat_frame_overlays)
    ke_mesh = mesh_source.index("update_stored_ke_colors")
    force_mesh = mesh_source.index("update_force_colors")
    assert ke_mesh < force_mesh