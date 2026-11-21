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
from src.shared.python.body_part_viz.mujoco_scalar_colors import (
    apply_mujoco_scalar_scene_colors,
)
from src.shared.python.body_part_viz.scalar_shading import ScalarColorScale
from src.shared.python.body_part_viz.segment_net_power import (
    INTERFACE_TRANSFER_LABEL,
    JOINT_COORDINATE_POWER_LABEL,
    NET_POWER_COLORMAP_ID,
    SEGMENT_NET_POWER_LABEL,
    default_net_power_color_scale,
    interface_edge_label,
    net_power_frame_from_interface,
)
from src.shared.python.body_part_viz.segment_net_power_limits import (
    DEFAULT_EXCLUDE_FRAME_RANGES,
    DEFAULT_IMPACT_CORE_FRAMES,
    DEFAULT_IMPACT_MARGIN_FRAMES,
    derive_symmetric_limit_w,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
LIMITS = ROOT / "scripts/config/segment_net_power_limits.json"


def test_diverging_palette_differs_from_tension_compression_and_zero_is_neutral():
    scale = default_net_power_color_scale(500.0)
    assert scale.low_color != DEFAULT_TENSION_COLOR
    assert scale.high_color != DEFAULT_COMPRESSION_COLOR
    assert scale.colormap_id == NET_POWER_COLORMAP_ID
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


def test_limits_json_documents_ball_impact_exclusion_and_regenerates_limit():
    payload = json.loads(LIMITS.read_text(encoding="utf-8"))
    assert payload["fixture"] == "tests/fixtures/club_face/swing_q_driver.npz"
    assert payload["impact_core_frames"] == [
        list(pair) for pair in DEFAULT_IMPACT_CORE_FRAMES
    ]
    assert payload["exclude_frame_ranges"] == [
        list(pair) for pair in DEFAULT_EXCLUDE_FRAME_RANGES
    ]
    assert DEFAULT_EXCLUDE_FRAME_RANGES == (
        (661 - DEFAULT_IMPACT_MARGIN_FRAMES, 664 + DEFAULT_IMPACT_MARGIN_FRAMES),
    )
    computed = default_net_power_color_scale().high_limit
    assert computed == pytest.approx(float(payload["symmetric_limit_w"]))
    assert computed > 0.0


@pytest.mark.skipif(
    not pytest.importorskip("mujoco", reason="mujoco not installed"),
    reason="mujoco",
)
def test_material_geom_uses_scene_colour_base_for_scalar_shading():
    import mujoco

    xml = """
    <mujoco>
      <asset>
        <material name="rod_mat" rgba="0.1 0.6 0.2 0.55"/>
      </asset>
      <worldbody>
        <body name="rod">
          <geom type="capsule" size="0.05 0.2" material="rod_mat"/>
        </body>
      </worldbody>
    </mujoco>
    """
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
    scene_base = tuple(float(v) for v in scene.geoms[0].rgba[:3])
    model_rgba = tuple(float(v) for v in model.geom_rgba[0][:3])
    assert scene_base != pytest.approx(model_rgba)

    replace_scale = default_net_power_color_scale(1000.0)
    shaded = replace(replace_scale, enabled=True, blend_mode="replace")
    tinted = replace(replace_scale, enabled=True, opacity=0.5, blend_mode="tint")

    scene_replace = mujoco.MjvScene(model, maxgeom=32)
    mujoco.mjv_updateScene(
        model,
        data,
        mujoco.MjvOption(),
        None,
        mujoco.MjvCamera(),
        mujoco.mjtCatBit.mjCAT_ALL,
        scene_replace,
    )
    apply_mujoco_scalar_scene_colors(model, scene_replace, {"rod": 400.0}, shaded)
    replace_rgb = tuple(float(v) for v in scene_replace.geoms[0].rgba[:3])
    assert replace_rgb != pytest.approx(scene_base)

    scene_tint = mujoco.MjvScene(model, maxgeom=32)
    mujoco.mjv_updateScene(
        model,
        data,
        mujoco.MjvOption(),
        None,
        mujoco.MjvCamera(),
        mujoco.mjtCatBit.mjCAT_ALL,
        scene_tint,
    )
    apply_mujoco_scalar_scene_colors(model, scene_tint, {"rod": 400.0}, tinted)
    tint_rgb = tuple(float(v) for v in scene_tint.geoms[0].rgba[:3])
    assert tint_rgb != pytest.approx(replace_rgb)


def test_interface_frame_builds_presentation_with_gravity_and_clipping():
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
                gravity_power_w=-12.0,
                external_power_w=1.0,
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
    assert presentation.extras.gravity_power_w == pytest.approx(-12.0)
    assert presentation.extras.applied_force_power_w == pytest.approx(1.0)
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
def test_two_link_energy_balance_against_independent_finite_difference():
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
    analyzer = MujocoInterfacePowerAnalyzer(model)
    frame = analyzer.frame_at(qpos, qvel, qacc)

    dt = 1e-4
    for segment in frame.segments:
        body_id = segment.body_index
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = qpos
        scratch.qvel[:] = qvel
        scratch.qacc[:] = qacc
        mujoco.mj_forward(model, scratch)
        e0 = analyzer._body_mechanical_energy(scratch, body_id, qvel)  # noqa: SLF001
        qpos1 = qpos + dt * qvel + 0.5 * (dt**2) * qacc
        qvel1 = qvel + dt * qacc
        scratch.qpos[:] = qpos1
        scratch.qvel[:] = qvel1
        mujoco.mj_forward(model, scratch)
        e1 = analyzer._body_mechanical_energy(scratch, body_id, qvel1)  # noqa: SLF001
        independent_rate = float((e1 - e0) / dt)
        accounted = (
            segment.total_interface_power_w
            + segment.external_power_w
            + segment.actuator_power_w
        )
        assert independent_rate == pytest.approx(
            segment.mechanical_energy_rate_w, abs=0.05
        )
        assert abs(independent_rate - accounted) <= 0.01
        assert abs(segment.balance_residual_w) <= 0.01


@pytest.mark.skipif(
    not pytest.importorskip("mujoco", reason="mujoco not installed"),
    reason="mujoco",
)
def test_gravity_power_nonzero_under_gravity():
    from mujoco_humanoid_golf.interface_power import MujocoInterfacePowerAnalyzer

    import mujoco

    xml = """
    <mujoco><option gravity="0 0 -10"/>
    <worldbody><body name="bob" pos="0 0 1">
      <joint type="slide" axis="0 0 1"/>
      <geom type="sphere" size="0.1" mass="2"/>
    </body></worldbody></mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    qpos = np.array([0.5])
    qvel = np.array([-0.3])
    qacc = np.zeros(1)
    frame = MujocoInterfacePowerAnalyzer(model).frame_at(qpos, qvel, qacc)
    assert abs(frame.segments[0].gravity_power_w) > 1.0
    assert frame.segments[0].external_power_w == pytest.approx(0.0)


def test_derive_limit_from_samples_is_stable():
    samples = [float(v) for v in range(1, 200)]
    derived = derive_symmetric_limit_w(
        samples, source="synthetic", sample_count=len(samples)
    )
    assert derived.symmetric_limit_w == pytest.approx(
        float(np.percentile(samples, 98.0))
    )
