# ruff: noqa: E501
"""Four-bar linkage mechanism generators for MuJoCo.

Provides Grashof crank-rocker, double-crank, double-rocker, and
parallel/antiparallel four-bar linkage XML generation.

Extracted from __init__.py for SRP (#1485).
"""

import numpy as np

from src.shared.python.core.constants import GRAVITY_M_S2


def _four_bar_asset_xml() -> str:
    return """    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
                 rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
        <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0.2"/>
        <material name="crank_mat" rgba="0.9 0.1 0.1 1"/>
        <material name="coupler_mat" rgba="0.1 0.7 0.1 1"/>
        <material name="follower_mat" rgba="0.1 0.1 0.9 1"/>
        <material name="ground_mat" rgba="0.3 0.3 0.3 1"/>
    </asset>"""


def _four_bar_worldbody_xml(
    ground: float, crank: float, coupler: float, follower: float
) -> str:
    cx = crank * np.cos(np.pi / 4)
    cy = crank * np.sin(np.pi / 4)
    return f"""    <worldbody>
        <light directional="true" diffuse=".8 .8 .8" specular="0.3 0.3 0.3"
              pos="0 0 5" dir="0 0 -1"/>
        <geom type="plane" size="10 10 0.1" material="grid"/>

        <!-- Ground link visualization -->
        <geom type="cylinder" fromto="0 0 0.5  {ground} 0 0.5" size="0.03"
              material="ground_mat" contype="0" conaffinity="0"/>

        <!-- Fixed pivot points -->
        <geom name="pivot_crank" type="sphere" pos="0 0 0.5" size="0.08"
              rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
        <geom name="pivot_follower" type="sphere" pos="{ground} 0 0.5" size="0.08"
              rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>

        <!-- Crank link -->
        <body name="crank" pos="0 0 0.5">
            <joint name="crank_joint" type="hinge" axis="0 0 1" damping="0.1"/>
            <geom type="capsule"
                  fromto="0 0 0  {cx} {cy} 0"
                  size="0.05" material="crank_mat" mass="0.5"/>
            <geom name="crank_end" type="sphere"
                  pos="{cx} {cy} 0"
                  size="0.07" rgba="1 0.5 0 1" contype="0" conaffinity="0"/>

            <!-- Coupler link attached to crank end -->
            <body name="coupler" pos="{cx} {cy} 0">
                <joint name="coupler_crank_joint" type="hinge" axis="0 0 1" damping="0.05"/>
                <geom type="capsule" fromto="0 0 0  {coupler * 0.7} {coupler * 0.3} 0"
                      size="0.05" material="coupler_mat" mass="0.5"/>
                <geom name="coupler_point" type="sphere" pos="{coupler * 0.35} {coupler * 0.15} 0"
                      size="0.06" rgba="1 1 0 1" contype="0" conaffinity="0"/>
            </body>
        </body>

        <!-- Follower link -->
        <body name="follower" pos="{ground} 0 0.5">
            <joint name="follower_joint" type="hinge" axis="0 0 1" damping="0.1"/>
            <geom type="capsule" fromto="0 0 0  {-follower * 0.7} {follower * 0.3} 0"
                  size="0.05" material="follower_mat" mass="0.5"/>
            <geom name="follower_end" type="sphere" pos="{-follower * 0.7} {follower * 0.3} 0"
                  size="0.07" rgba="0.5 0 1 1" contype="0" conaffinity="0"/>
        </body>
    </worldbody>"""


def generate_four_bar_linkage_xml(
    link_lengths: list[float] | None = None,
    link_type: str = "grashof_crank_rocker",
) -> str:
    """Generate a four-bar linkage mechanism.

    Parameters
    ----------
    link_lengths : list of 4 floats, optional
        Lengths of [ground, crank, coupler, follower]. If None, uses defaults.
    link_type : str
        Type of four-bar: "grashof_crank_rocker", "grashof_double_crank",
        "grashof_double_rocker", "non_grashof", "parallel", "antiparallel"

    Returns
    -------
    str : MuJoCo XML string
    """
    configs = {
        "grashof_crank_rocker": [4.0, 1.0, 3.5, 3.0],
        "grashof_double_crank": [4.0, 2.0, 4.5, 3.5],
        "grashof_double_rocker": [4.0, 3.8, 2.5, 3.0],
        "non_grashof": [4.0, 4.5, 2.0, 3.0],
        "parallel": [4.0, 2.0, 4.0, 2.0],
        "antiparallel": [4.0, 2.0, 4.0, 2.0],
    }

    if link_lengths is None:
        link_lengths = configs.get(link_type, configs["grashof_crank_rocker"])

    ground, crank, coupler, follower = link_lengths

    asset_xml = _four_bar_asset_xml()
    worldbody_xml = _four_bar_worldbody_xml(ground, crank, coupler, follower)

    return f"""
<mujoco model="four_bar_linkage_{link_type}">
    <option timestep="0.002" gravity="0 0 -{GRAVITY_M_S2}"/>

    <visual>
        <global offwidth="1920" offheight="1080"/>
        <quality shadowsize="4096"/>
        <map stiffness="700" shadowscale="0.5" fogstart="10" fogend="15"/>
    </visual>

{asset_xml}

{worldbody_xml}

    <actuator>
        <motor name="crank_motor" joint="crank_joint" gear="20"
              ctrllimited="true" ctrlrange="-5 5"/>
    </actuator>
</mujoco>
"""
