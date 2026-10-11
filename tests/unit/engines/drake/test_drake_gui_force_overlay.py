"""Headless tests for the Drake GUI force/torque overlay (FTO-12, #11297)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.engines.physics_engines.drake.python.src.drake_force_overlay import (
    ForceOverlayController,
    drake_color_bindings,
    kinds_for_toggles,
)
from src.engines.physics_engines.drake.python.src.drake_gui_viz import (
    VisualizationMixin,
)
from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.body_part_viz.force_colors import ForceColorScale
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)

pytestmark = pytest.mark.unit


def _wrench(kind: WrenchKind, name: str, force: float = 50.0) -> OverlayWrench:
    return OverlayWrench(
        kind=kind,
        label=f"{kind.value}:{name}",
        body=name,
        point_m=(0.0, 0.0, 1.0),
        force_n=(0.0, 0.0, force),
        torque_nm=None if kind is WrenchKind.GRAVITY else (0.0, 5.0, 0.0),
        source="test",
    )


def _frame(t: float = 0.0) -> ForceTorqueFrame:
    return ForceTorqueFrame(
        time_s=t,
        engine="drake",
        wrenches=(
            _wrench(WrenchKind.JOINT_REACTION, "elbow"),
            _wrench(WrenchKind.CONTACT, "foot"),
            _wrench(WrenchKind.JOINT_ACTUATOR, "shoulder"),
            _wrench(WrenchKind.GRAVITY, "arm"),
        ),
        axial_loads=AxialLoadFrame(t, {"arm": 12.0}, "test"),
    )


class FakeSink:
    """Records MeshcatSink calls; satisfies the runtime-checkable protocol."""

    def __init__(self) -> None:
        self.cylinders: dict[str, Any] = {}
        self.deleted: list[str] = []

    def set_cylinder(self, path, length_m, radius_top_m, radius_bottom_m, rgba):
        self.cylinders[path] = rgba

    def set_transform(self, path, matrix4x4):
        pass

    def delete(self, path):
        self.deleted.append(path)


class RecordingSession:
    def __init__(self) -> None:
        self.frames: list[Any] = []
        self.scales: list[ForceColorScale] = []

    def set_frame(self, frame):
        self.frames.append(frame)

    def set_axial_color_scale(self, scale):
        self.scales.append(scale)


def _controller(frame_fn=None, session=None, sink=None):
    sink = sink or FakeSink()
    session = session or RecordingSession()
    calls: list[bool] = []

    def provider(include_gravity: bool):
        calls.append(include_gravity)
        return frame_fn() if frame_fn else _frame()

    ctl = ForceOverlayController(provider, sink, session)
    return ctl, sink, session, calls


def _labels(sink: FakeSink) -> set[str]:
    return {p.split("/")[2] for p in sink.cylinders}


def test_kinds_for_toggles_maps_checkboxes() -> None:
    assert kinds_for_toggles(forces=True, torques=False, gravity=False) == {
        WrenchKind.JOINT_REACTION,
        WrenchKind.CONTACT,
        WrenchKind.EXTERNAL,
    }
    assert kinds_for_toggles(forces=False, torques=True, gravity=False) == {
        WrenchKind.JOINT_ACTUATOR
    }
    assert kinds_for_toggles(forces=False, torques=False, gravity=True) == {
        WrenchKind.GRAVITY
    }
    assert kinds_for_toggles(forces=False, torques=False, gravity=False) == set()


def test_kinds_for_toggles_rejects_non_bool() -> None:
    with pytest.raises(TypeError):
        kinds_for_toggles(forces=1, torques=False, gravity=False)  # type: ignore[arg-type]


def test_forces_draw_reaction_and_contact_only() -> None:
    ctl, sink, _, calls = _controller()
    ctl.tick(forces=True, torques=False, gravity=False)
    assert _labels(sink) == {"joint_reaction:elbow", "contact:foot"}
    assert calls == [False]


def test_gravity_draws_only_gravity_and_requests_it() -> None:
    ctl, sink, _, calls = _controller()
    ctl.tick(forces=False, torques=False, gravity=True)
    assert _labels(sink) == {"gravity:arm"}
    assert calls == [True]


def test_torques_draw_actuator_arcs() -> None:
    ctl, sink, _, _ = _controller()
    ctl.tick(forces=False, torques=True, gravity=False)
    assert _labels(sink) == {"joint_actuator:shoulder"}


def test_unchecking_deletes_removed_paths() -> None:
    ctl, sink, _, _ = _controller()
    ctl.tick(forces=True, torques=False, gravity=False)
    assert sink.cylinders
    ctl.tick(forces=False, torques=False, gravity=False)
    assert "/force_overlay" in sink.deleted


def test_status_legend_lists_unavailable_channels() -> None:
    wrench = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:foot",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=None,
        torque_nm=(0.0, 1.0, 0.0),
        source="test",
    )
    frame = ForceTorqueFrame(time_s=0.0, engine="drake", wrenches=(wrench,))
    ctl, _, _, _ = _controller(lambda: frame)
    text = ctl.tick(forces=True, torques=False, gravity=False)
    assert "Unavailable: contact:foot" in text
    assert "Engine: drake" in text


def test_colors_receive_axial_frame_each_tick_when_enabled() -> None:
    ctl, _, session, _ = _controller()
    ctl.set_axial_color_scale(ForceColorScale(enabled=True))
    ctl.tick(forces=False, torques=False, gravity=False)
    ctl.tick(forces=False, torques=False, gravity=False)
    assert len(session.frames) == 2
    assert all(isinstance(f, AxialLoadFrame) for f in session.frames)
    assert session.scales[-1].enabled is True


def test_colors_not_fed_when_disabled() -> None:
    ctl, _, session, _ = _controller()
    ctl.set_axial_color_scale(ForceColorScale(enabled=False))
    ctl.tick(forces=True, torques=True, gravity=True)
    assert session.frames == []


def test_missing_frame_clears_overlay_and_colors() -> None:
    ctl, sink, session, _ = _controller(lambda: None)
    ctl.set_axial_color_scale(ForceColorScale(enabled=True))
    assert ctl.tick(forces=True, torques=False, gravity=False) == ""
    assert session.frames == [None]
    assert "/force_overlay" in sink.deleted


def test_color_bindings_map_illustration_geometry_to_leaf_paths() -> None:
    plant = SimpleNamespace(
        GetBodyFrameIdOrThrow=lambda idx: "frame1",
    )
    inspector = MagicMock()
    inspector.GetGeometries.return_value = ["g1"]
    inspector.GetName.side_effect = lambda x: {
        "frame1": "golfer::arm",
        "g1": "golfer::arm_visual",
    }[x]
    inspector.GetIllustrationProperties.return_value = None
    bindings = drake_color_bindings(
        plant, inspector, {1: "arm"}, base_rgba_of=lambda gid: (0.2, 0.3, 0.4, 1.0)
    )
    assert bindings == {
        "arm": {
            "visualizer/golfer/arm/golfer/arm_visual/<object>": (0.2, 0.3, 0.4, 1.0)
        }
    }


class _Box:
    def __init__(self, on: bool) -> None:
        self._on = on

    def isChecked(self) -> bool:
        return self._on


def test_mixin_update_uses_provider_and_never_gravity_generalized_forces() -> None:
    plant = MagicMock()
    host = SimpleNamespace(
        plant=plant,
        chk_show_forces=_Box(True),
        chk_show_torques=_Box(True),
        chk_show_gravity=_Box(False),
        _force_overlay=MagicMock(),
        _update_status=MagicMock(),
    )
    host._force_overlay.tick.return_value = "Engine: drake"
    VisualizationMixin._update_force_glyphs(host)  # type: ignore[arg-type]
    host._force_overlay.tick.assert_called_once_with(
        forces=True, torques=True, gravity=False
    )
    host._update_status.assert_called_once_with("Engine: drake")
    plant.CalcGravityGeneralizedForces.assert_not_called()
    assert not hasattr(VisualizationMixin, "_draw_torque_vectors")
    assert not hasattr(VisualizationMixin, "_draw_gravity_force_vectors")


def test_mixin_update_without_overlay_is_noop() -> None:
    host = SimpleNamespace(_force_overlay=None)
    VisualizationMixin._update_force_glyphs(host)  # type: ignore[arg-type]


_PENDULUM_URDF = """<?xml version="1.0"?>
<robot name="pend">
  <link name="base"/>
  <link name="rod">
    <inertial><origin xyz="0 0 -0.5"/><mass value="2"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0 0 -0.5"/>
      <geometry><cylinder length="1" radius="0.05"/></geometry>
      <material name="m"><color rgba="0.2 0.4 0.6 1"/></material></visual>
  </link>
  <joint name="hinge" type="revolute">
    <parent link="base"/><child link="rod"/><axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" effort="10" velocity="10"/>
  </joint>
</robot>
"""


def test_real_drake_plant_draws_arrows_and_binds_shading() -> None:
    """Offscreen smoke: real plant, real MeshCat, no browser, one tick."""
    pydrake = pytest.importorskip("pydrake")
    if type(pydrake).__module__ == "unittest.mock":
        pytest.skip(
            "pydrake is mocked by tests/unit/conftest.py; this smoke test needs "
            "the real Drake bindings"
        )
    from pydrake.geometry import Meshcat, MeshcatVisualizer, SceneGraph
    from pydrake.multibody.parsing import Parser
    from pydrake.systems.analysis import Simulator
    from pydrake.systems.framework import DiagramBuilder
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph

    from src.engines.physics_engines.drake.python.drake_force_torque import (
        DrakeForceTorqueSource,
    )
    from src.engines.physics_engines.drake.python.src.drake_meshcat_sink import (
        DrakeMeshcatSink,
    )
    from src.shared.python.body_part_viz.meshcat_force_colors import (
        MeshcatForceColors,
        MeshcatForceColorSession,
    )

    from src.engines.physics_engines.drake.python.src.drake_force_overlay import (
        illustration_base_rgba,
    )

    meshcat = Meshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
    Parser(plant).AddModelsFromString(_PENDULUM_URDF, "urdf")
    plant.Finalize()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    sim = Simulator(diagram)
    sim.Initialize()
    sim.AdvanceTo(0.01)
    ctx = plant.GetMyContextFromRoot(sim.get_mutable_context())

    source = DrakeForceTorqueSource(plant, diagram)
    session = MeshcatForceColorSession()
    session.update(plant, 0.0)
    inspector = next(
        s for s in diagram.GetSystems() if isinstance(s, SceneGraph)
    ).model_inspector()
    bindings = drake_color_bindings(
        plant,
        inspector,
        source.body_labels,
        base_rgba_of=illustration_base_rgba(inspector),
    )
    assert "rod" in bindings
    (path,) = bindings["rod"]
    assert meshcat.HasPath(path.removesuffix("/<object>"))
    session.bind(MeshcatForceColors(meshcat.SetProperty, bindings), plant)

    ctl = ForceOverlayController(
        lambda g: source.sample(ctx, include_gravity=g),
        DrakeMeshcatSink(meshcat),
        session,
    )
    ctl.set_axial_color_scale(ForceColorScale(enabled=True))
    session.update(plant, float(ctx.get_time()))
    text = ctl.tick(forces=True, torques=True, gravity=True)
    assert "Engine: drake" in text
    assert meshcat.HasPath("force_overlay")
