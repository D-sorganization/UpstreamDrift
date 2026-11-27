"""Render pipeline applies net-power shading before axial force (#15399)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_sim_rendering_mixin_orders_net_power_before_force():
    root = Path(__file__).resolve().parents[6]
    source = (
        root
        / "src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/sim_rendering_mixin.py"
    ).read_text(encoding="utf-8")
    render_body = source.split("def _render_once", 1)[1].split("\n    def ", 1)[0]
    net_idx = render_body.find("_apply_net_power_scene_colors")
    force_idx = render_body.find("_apply_axial_scene_colors")
    iface_once = render_body.count("_current_interface_power_frame()")
    assert net_idx >= 0 and force_idx >= 0
    assert net_idx < force_idx
    assert iface_once == 1
