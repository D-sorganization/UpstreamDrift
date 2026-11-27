"""Smoke test for COM-centered segment KE globe overlay (#15389)."""

from __future__ import annotations

import os

import mujoco
import numpy as np
import pytest

from mujoco_humanoid_golf.ke_globe_scene import (
    GrfPlateOutline,
    add_com_rgb_axes_to_scene,
    add_grf_plate_outlines_to_scene,
    add_ke_globes_to_scene,
)
from mujoco_humanoid_golf.segment_kinetic_energy_provider import (
    MujocoSegmentKineticEnergyProvider,
)
from src.shared.python.segment_kinetic_energy import SegmentKineticEnergySeries

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]


def _swing_arm_model() -> mujoco.MjModel:
    xml = """
    <mujoco>
      <option gravity="0 0 -9.80665"/>
      <worldbody>
        <geom name="floor" type="plane" size="2 2 0.1" rgba="0.3 0.3 0.3 1"/>
        <body name="upper" pos="0 0 1.2">
          <joint name="j1" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.04" mass="2"/>
          <body name="lower" pos="0.4 0 0">
            <joint name="j2" type="hinge" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0.35 0 0" size="0.035" mass="1.5"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


@pytest.mark.headless_safe
def test_ke_globe_overlay_renders_on_synthetic_swing() -> None:
    """Overlay draws globes, RGB axes, and optional plate outlines."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    model = _swing_arm_model()
    data = mujoco.MjData(model)
    provider = MujocoSegmentKineticEnergyProvider(model)

    times = np.linspace(0.0, 0.04, 5)
    qpos_traj = []
    qvel_traj = []
    for t in times:
        qpos = np.array([0.4 * np.sin(4.0 * t), 0.2 * np.cos(3.0 * t)])
        qvel = np.array([1.6 * np.cos(4.0 * t), -0.6 * np.sin(3.0 * t)])
        qpos_traj.append(qpos)
        qvel_traj.append(qvel)
    qpos_traj = np.asarray(qpos_traj)
    qvel_traj = np.asarray(qvel_traj)

    series = SegmentKineticEnergySeries.from_trajectory(
        provider, times, qpos_traj, qvel_traj
    )
    peaks = series.peak_totals()
    frame = series.frames[2]

    data.qpos[:] = qpos_traj[2]
    data.qvel[:] = qvel_traj[2]
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=240, width=320)
    renderer.update_scene(data)
    base_geoms = renderer.scene.ngeom

    receipt = add_ke_globes_to_scene(renderer.scene, frame, peaks)
    assert receipt.globes_drawn >= 1

    upper = next(seg for seg in frame.segments if seg.body == "upper")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "upper")
    rot = data.xmat[body_id].reshape(3, 3)
    axes = add_com_rgb_axes_to_scene(renderer.scene, upper.com_m, rot)
    assert axes == 3

    plate = GrfPlateOutline(
        corners_m=(
            (-0.4, -0.3, 0.0),
            (0.4, -0.3, 0.0),
            (0.4, 0.3, 0.0),
            (-0.4, 0.3, 0.0),
        )
    )
    plates = add_grf_plate_outlines_to_scene(renderer.scene, [plate])
    assert plates == 4

    assert renderer.scene.ngeom > base_geoms
    rgb = renderer.render()
    assert rgb.shape == (240, 320, 3)
    assert int(rgb.max()) > 10
