"""MeshCat KE shading must not be cleared by a disabled net-power pass (#15399)."""

from __future__ import annotations

from dataclasses import replace

import pytest

pytestmark = pytest.mark.unit


def test_meshcat_ke_colours_persist_when_net_power_overlay_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meshcat = pytest.importorskip("meshcat")
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.meshcat_adapter import (
        MuJoCoMeshcatAdapter,
    )
    from src.shared.python.body_part_viz.segment_net_power import (
        default_net_power_color_scale,
    )
    from src.shared.python.body_part_viz.scalar_shading import ScalarColorScale

    class Window:
        web_url = "http://127.0.0.1/unused-test-transport"

        def __init__(self) -> None:
            self.commands: list[dict[str, object]] = []

        def send(self, command: object) -> None:
            if isinstance(command, dict):
                self.commands.append(command)

    window = Window()
    visualizer = meshcat.Visualizer(window=window)
    monkeypatch.setattr(meshcat, "Visualizer", lambda: visualizer)
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="rod">'
        '<geom name="shape" type="capsule" fromto="0 0 0 0 0 -1" size=".03" '
        'rgba=".2 .4 .6 .8"/></body></worldbody></mujoco>'
    )
    adapter = MuJoCoMeshcatAdapter(model)
    assert adapter._stored_ke_colors is not adapter._net_power_colors

    ke_scale = ScalarColorScale(
        enabled=True,
        scale_kind="sequential",
        units="J",
        colormap_id="cividis",
        low_limit=0.0,
        high_limit=100.0,
    )
    window.commands.clear()
    adapter.update_stored_ke_colors({"rod": 50.0}, ke_scale)
    assert window.commands, "stored KE should publish a colour"
    ke_command = window.commands[-1]
    assert ke_command.get("type") == "set_property"
    ke_rgb = list(ke_command["value"])[:3]  # type: ignore[index]

    net_off = replace(default_net_power_color_scale(), enabled=False)
    adapter.update_net_power_colors(None, net_off)
    assert window.commands[-1]["value"][:3] == pytest.approx(ke_rgb)
