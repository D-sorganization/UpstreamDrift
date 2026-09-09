"""Shared MeshCat transport seam preserves every bound object's base RGBA."""

import pytest

from src.shared.python.body_part_viz import AxialLoadFrame, ForceColorScale
from src.shared.python.body_part_viz.meshcat_force_colors import MeshcatForceColors

pytestmark = pytest.mark.unit


def test_session_rejects_stale_loads_and_clears_bindings_on_model_replacement():
    from src.shared.python.body_part_viz.meshcat_force_colors import (
        MeshcatForceColorSession,
    )

    calls = []
    adapter = MeshcatForceColors(
        lambda *args: calls.append(args), {"x": {"/a/<object>": (0, 1, 0, 0.5)}}
    )
    session = MeshcatForceColorSession()
    model = object()
    session.update(model, 0)
    session.bind(adapter, model)
    session.set_frame(AxialLoadFrame(0, {"x": 1000}, "Known section"))
    assert calls == []
    session.set_axial_color_scale(ForceColorScale(enabled=True))
    assert calls[-1][2] == [0, 0, 1, 0.5]
    session.update(model, 1)
    assert calls[-1][2] == [0, 1, 0, 0.5]
    session.set_frame(AxialLoadFrame(1, {"x": -1000}, "Known section"))
    assert calls[-1][2] == [1, 0, 0, 0.5]
    session.set_axial_color_scale(ForceColorScale())
    assert calls[-1][2] == [0, 1, 0, 0.5]
    session.set_axial_color_scale(ForceColorScale(enabled=True))
    session.update(object(), 1)
    count = len(calls)
    session.set_frame(AxialLoadFrame(1, {"x": 1000}, "Old model data"))
    assert len(calls) == count
    with pytest.raises(ValueError):
        session.bind(adapter, model)
    with pytest.raises(ValueError):
        session.update(model, float("nan"))


def test_shared_transport_handles_multiple_geometries_and_restoration():
    calls = []
    bindings = {
        "link": {
            "/visual/a/<object>": (0.1, 0.2, 0.3, 0.4),
            "/visual/b/<object>": (0.5, 0.6, 0.7, 0.8),
        }
    }
    display = MeshcatForceColors(
        lambda path, prop, value: calls.append((path, prop, value)), bindings
    )
    enabled = ForceColorScale(enabled=True, tension_limit_n=1, compression_limit_n=1)
    display.apply(AxialLoadFrame(0, {"link": 10}, "Known section"), enabled)
    assert calls[-2:] == [
        ("/visual/a/<object>", "color", [0, 0, 1, 0.4]),
        ("/visual/b/<object>", "color", [0, 0, 1, 0.8]),
    ]
    display.apply(AxialLoadFrame(1, {"link": -10}, "Known section"), enabled)
    assert calls[-1][2] == [1, 0, 0, 0.8]
    display.apply(None, enabled)
    assert calls[-1][2] == [0.5, 0.6, 0.7, 0.8]
    display.apply(AxialLoadFrame(2, {"link": 10}, "Known section"), enabled)
    display.apply(None, ForceColorScale())
    assert calls[-1][2] == [0.5, 0.6, 0.7, 0.8]


def test_off_does_not_touch_an_unmodified_scene():
    calls = []
    display = MeshcatForceColors(
        lambda *args: calls.append(args), {"x": {"/a/<object>": (1, 1, 1, 1)}}
    )
    display.apply(None, ForceColorScale())
    assert calls == []
    with pytest.raises(ValueError):
        MeshcatForceColors(lambda *args: None, {"x": {"/a/<object>": (1, 1, 1, 2)}})


@pytest.mark.requires_pinocchio
@pytest.mark.requires_drake
def test_native_meshcat_host_settings_and_display_clock(monkeypatch):
    """Exercise real optional GUIs, their settings actions and display lifecycle."""
    pytest.importorskip("pinocchio.visualize")
    pytest.importorskip("pydrake.all")
    pytest.importorskip("meshcat")
    from PyQt6.QtWidgets import QApplication
    from src.engines.physics_engines.pinocchio.python.pinocchio_golf.gui import (
        PinocchioGUI,
    )
    from src.engines.physics_engines.drake.python.src.drake_gui_app import DrakeSimApp

    monkeypatch.setenv("MESHCAT_HOST", "localhost")
    app = QApplication.instance() or QApplication([])
    for cls in (PinocchioGUI, DrakeSimApp):
        window = cls()
        try:
            window.timer.stop()
            menus = [a.menu() for a in window.menuBar().actions() if a.menu()]
            actions = [
                a
                for menu in menus
                for a in menu.actions()
                if a.text() == "Segment Force Colors…"
            ]
            assert len(actions) == 1
            actions[0].trigger()
            app.processEvents()
            window.update_visualization()
            model = window.model if isinstance(window, PinocchioGUI) else window.plant
            time_s = (
                window.sim_time
                if isinstance(window, PinocchioGUI)
                else window.context.get_time()
            )
            calls = []
            adapter = MeshcatForceColors(
                lambda *args, target=calls: target.append(args),
                {"x": {"/test/<object>": (0, 1, 0, 1)}},
            )
            session = window.segment_force_colors
            session.bind(adapter, model)
            session.set_frame(AxialLoadFrame(time_s, {"x": 1000}, "Qualified fixture"))
            session.set_axial_color_scale(ForceColorScale(enabled=True))
            assert calls[-1][2] == [0, 0, 1, 1]
            session.set_axial_color_scale(ForceColorScale())
            assert calls[-1][2] == [0, 1, 0, 1]
        finally:
            window.close()
            app.processEvents()


def test_mujoco_meshcat_uses_native_commands_without_changing_geometry(monkeypatch):
    native = pytest.importorskip("meshcat")
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.meshcat_adapter import (
        MuJoCoMeshcatAdapter,
    )

    class Window:
        web_url = "http://127.0.0.1/unused-test-transport"

        def __init__(self):
            self.commands = []

        def send(self, command):
            self.commands.append(command.lower())

    window = Window()
    visualizer = native.Visualizer(window=window)
    monkeypatch.setattr(native, "Visualizer", lambda: visualizer)
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="rod"><geom name="shape" type="capsule" fromto="0 0 0 0 0 -1" size=".03" rgba=".2 .4 .6 .8"/></body></worldbody></mujoco>'
    )
    adapter = MuJoCoMeshcatAdapter(model)
    window.commands.clear()
    adapter.update_force_colors(
        AxialLoadFrame(0, {"rod": 2}, "Qualified test"),
        ForceColorScale(enabled=True, tension_limit_n=1),
    )
    assert len(window.commands) == 1
    command = window.commands[-1]
    assert command["type"] == "set_property"
    assert command["property"] == "color"
    assert command["path"].endswith("/visuals/shape/geometry/<object>")
    assert command["value"][:3] == [0, 0, 1]
    assert command["value"][3] == pytest.approx(0.8)
    adapter.update_force_colors(None, ForceColorScale())
    assert window.commands[-1]["value"][:3] == pytest.approx([0.2, 0.4, 0.6])
