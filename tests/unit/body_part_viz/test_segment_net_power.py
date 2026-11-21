"""Segment net-power shading contracts (#15399)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.biomechanics.robertson_winter_interface_power import (
    InterfacePowerFrame,
    InterfacePowerJointSample,
    InterfacePowerSegment,
)
from src.shared.python.body_part_viz.force_colors import (
    DEFAULT_COMPRESSION_COLOR,
    DEFAULT_TENSION_COLOR,
    ForceColorScale,
)
from src.shared.python.body_part_viz.mujoco_force_colors import (
    apply_mujoco_scene_colors,
)
from src.shared.python.body_part_viz.mujoco_segment_scene_colors import (
    apply_mujoco_scalar_scene_colors,
)
from src.shared.python.body_part_viz.scalar_shading import ScalarColorScale
from src.shared.python.body_part_viz.segment_net_power import (
    INTERFACE_TRANSFER_LABEL,
    JOINT_COORDINATE_POWER_LABEL,
    SEGMENT_NET_POWER_LABEL,
    default_net_power_color_scale,
    interface_edge_label,
    net_power_frame_from_interface,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
LIMITS = ROOT / "scripts/config/segment_net_power_limits.json"


def test_diverging_palette_differs_from_tension_compression_and_zero_is_neutral():
    scale = default_net_power_color_scale(500.0)
    assert scale.low_color != DEFAULT_TENSION_COLOR
    assert scale.high_color != DEFAULT_COMPRESSION_COLOR
    assert DEFAULT_TENSION_COLOR != scale.low_color
    assert DEFAULT_COMPRESSION_COLOR != scale.high_color
    active = replace(scale, enabled=True)
    assert active.color(0.0, "#888888").lower() == scale.neutral_color


def test_string_labels_use_transfer_only_on_interface_edge():
    assert SEGMENT_NET_POWER_LABEL == "dE/dt (net power)"
    assert "transfer" not in SEGMENT_NET_POWER_LABEL.lower()
    assert "transfer" not in JOINT_COORDINATE_POWER_LABEL.lower()
    joint = InterfacePowerJointSample(
        joint="j1",
        joint_index=1,
        child_body="child",
        child_body_index=2,
        parent_body="parent",
        parent_body_index=1,
        power_on_child_w=120.0,
        power_on_parent_w=-120.0,
        force_power_on_child_w=100.0,
        moment_power_on_child_w=20.0,
        joint_coordinate_power_w=0.0,
    )
    label = interface_edge_label(joint)
    assert INTERFACE_TRANSFER_LABEL in label
    assert "transfer" in label.lower()


def test_limits_json_documents_swing_fixture_excluding_glitch_frames():
    payload = json.loads(LIMITS.read_text(encoding="utf-8"))
    assert payload["fixture"] == "tests/fixtures/club_face/swing_q_driver.npz"
    assert payload["exclude_frame_ranges"] == [[661, 664]]
    limit = float(payload["symmetric_limit_w"])
    assert limit > 0.0
    assert default_net_power_color_scale().high_limit == pytest.approx(limit)


def test_interface_frame_builds_presentation_with_residual_and_clipping():
    iface = InterfacePowerFrame(
        schema_version=1,
        time_s=0.1,
        engine="test",
        joints=(),
        segments=(
            InterfacePowerSegment(
                body="arm",
                body_index=1,
                mass_kg=2.0,
                total_interface_power_w=10.0,
                external_power_w=-4.0,
                actuator_power_w=0.0,
                mechanical_energy_rate_w=1200.0,
                balance_residual_w=0.01,
            ),
        ),
    )
    scale = default_net_power_color_scale(1000.0)
    enabled = replace(scale, enabled=True)
    presentation = net_power_frame_from_interface(iface, enabled)
    assert presentation.segment_frame.quantity_label == SEGMENT_NET_POWER_LABEL
    assert presentation.extras.gravity_power_w == pytest.approx(-4.0)
    assert "arm" in presentation.extras.clipped_segments


@pytest.mark.skipif(
    not pytest.importorskip("mujoco", reason="mujoco not installed"),
    reason="mujoco",
)
def test_force_shading_unchanged_when_net_power_applied_first():
    import mujoco

    from src.shared.python.body_part_viz import AxialLoadFrame

    xml = """<mujoco><worldbody><body name="rod">
    <geom type="capsule" size="0.05 0.2" rgba="1 1 0 0.4"/>
    </body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    scene = mujoco.MjvScene(model, maxgeom=32)
    mujoco.mj_forward(model, data)
    mujoco.mjv_updateScene(
        model,
        data,
        mujoco.MjvOption(),
        None,
        mujoco.MjvCamera(),
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )

    def force_only():
        scene2 = mujoco.MjvScene(model, maxgeom=32)
        mujoco.mjv_updateScene(
            model,
            data,
            mujoco.MjvOption(),
            None,
            mujoco.MjvCamera(),
            mujoco.mjtCatBit.mjCAT_ALL,
            scene2,
        )
        scale = ForceColorScale(
            enabled=True, tension_limit_n=50.0, compression_limit_n=50.0
        )
        apply_mujoco_scene_colors(
            model, scene2, AxialLoadFrame(0.0, {"rod": 40.0}, "t"), scale
        )
        return [float(v) for v in scene2.geoms[0].rgba]

    def stacked():
        scene3 = mujoco.MjvScene(model, maxgeom=32)
        mujoco.mjv_updateScene(
            model,
            data,
            mujoco.MjvOption(),
            None,
            mujoco.MjvCamera(),
            mujoco.mjtCatBit.mjCAT_ALL,
            scene3,
        )
        power = default_net_power_color_scale(1000.0)
        power_on = replace(power, enabled=True)
        apply_mujoco_scalar_scene_colors(model, scene3, {"rod": 800.0}, power_on)
        scale = ForceColorScale(
            enabled=True, tension_limit_n=50.0, compression_limit_n=50.0
        )
        apply_mujoco_scene_colors(
            model, scene3, AxialLoadFrame(0.0, {"rod": 40.0}, "t"), scale
        )
        return [float(v) for v in scene3.geoms[0].rgba]

    assert force_only() == pytest.approx(stacked())


@pytest.mark.skipif(
    not pytest.importorskip("mujoco", reason="mujoco not installed"),
    reason="mujoco",
)
def test_two_link_interface_balance_matches_segment_rate():
    from mujoco_humanoid_golf.interface_power import MujocoInterfacePowerAnalyzer

    xml = """
    <mujoco>
      <option gravity="0 0 -9.80665"/>
      <worldbody>
        <body name="l1" pos="0 0 1">
          <joint name="j1" type="hinge" axis="1 0 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.05" mass="2"/>
          <body name="l2" pos="0 0 -0.5">
            <joint name="j2" type="hinge" axis="1 0 0"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.05" mass="1.5"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    import mujoco

    model = mujoco.MjModel.from_xml_string(xml)
    qpos = np.array([0.4, -0.6])
    qvel = np.array([2.1, -1.3])
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    qacc = data.qacc.copy()
    frame = MujocoInterfacePowerAnalyzer(model).frame_at(qpos, qvel, qacc)
    for segment in frame.segments:
        accounted = (
            segment.total_interface_power_w
            + segment.external_power_w
            + segment.actuator_power_w
        )
        assert segment.mechanical_energy_rate_w == pytest.approx(
            accounted + segment.balance_residual_w, rel=0, abs=0.05
        )
