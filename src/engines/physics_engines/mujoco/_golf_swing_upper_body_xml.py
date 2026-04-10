"""MJCF XML model definitions — extracted from golf_swing_models_xml.py.

Import via golf_swing_models_xml, not directly.
"""

from __future__ import annotations

from src.shared.python.core.constants import (
    DEFAULT_TIME_STEP,
    GOLF_BALL_MASS_KG,
    GOLF_BALL_RADIUS_M,
    GRAVITY_M_S2,
)

# Convert to float for use in f-strings
_BALL_MASS = float(GOLF_BALL_MASS_KG)
_BALL_RADIUS = float(GOLF_BALL_RADIUS_M)
_BALL_RADIUS_INNER = _BALL_RADIUS * 0.998
_TIME_STEP = float(DEFAULT_TIME_STEP)

UPPER_BODY_GOLF_SWING_XML = rf"""<mujoco model="golf_upper_body_swing">
  <option timestep="0.002" gravity="0 0 -{GRAVITY_M_S2}" integrator="RK4"/>

  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>

  <visual>
    <global offwidth="1024" offheight="1024"/>
    <map znear="0.01" zfar="50"/>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <!-- Define materials for better visualization -->
    <material name="torso_mat" rgba="0.7 0.5 0.4 1"/>
    <material name="arm_left_mat" rgba="0.6 0.4 0.3 1"/>
    <material name="arm_right_mat" rgba="0.6 0.4 0.3 1"/>
    <material name="club_mat" rgba="0.2 0.2 0.2 1"/>
    <material name="grip_mat" rgba="0.1 0.1 0.1 1"/>
    <material name="ground_mat" rgba="0.4 0.6 0.3 1"/>
  </asset>

  <worldbody>
    <!-- Ground plane -->
    <geom name="floor" type="plane" size="10 10 0.1" material="ground_mat"/>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>

    <!-- Camera views -->
    <camera name="side" pos="-4 -1.5 1.3" euler="0.1 0 0.3" mode="fixed"/>
    <camera name="front" pos="0 -4 1.3" euler="0.1 0 1.57" mode="fixed"/>
    <camera name="top" pos="0 0 5" euler="0 0 0" mode="fixed"/>

    <!-- Pelvis (fixed base at hip height) -->
    <body name="pelvis" pos="0 0 0.95">
      <geom name="pelvis_geom" type="box" size="0.15 0.08 0.08"
            rgba="0.6 0.5 0.4 1" mass="10"/>
      <geom name="pelvis_marker" type="sphere" size="0.03"
            pos="0 0 0" rgba="1 0 0 0.5"/>

      <!-- Torso connected via spine rotation joint -->
      <body name="torso" pos="0 0 0.1">
        <joint name="spine_rotation" type="hinge" axis="0 0 1"
               range="-1.57 1.57" damping="2.0"/>

        <!-- Torso (spine) segment -->
        <geom name="lower_torso" type="capsule" fromto="0 0 0 0 0 0.25"
              size="0.12" material="torso_mat" mass="15"/>
        <geom name="upper_torso" type="capsule" fromto="0 0 0.25 0 0 0.50"
              size="0.14" material="torso_mat" mass="15"/>

        <!-- Left shoulder -->
        <body name="left_shoulder" pos="-0.2 0 0.50" euler="0 0 0">
          <joint name="left_shoulder_swing" type="hinge" axis="0 1 0"
                 range="-2.0 2.8" damping="1.5"/>
          <joint name="left_shoulder_lift" type="hinge" axis="1 0 0"
                 range="-1.5 1.5" damping="1.5"/>

          <!-- Left upper arm -->
          <geom name="left_upper_arm" type="capsule"
                fromto="0 0 0 0.25 0 -0.05" size="0.035"
                material="arm_left_mat" mass="2.5"/>

          <!-- Left elbow -->
          <body name="left_elbow" pos="0.25 0 -0.05">
            <joint name="left_elbow" type="hinge" axis="0 1 0"
                   range="-2.4 0" damping="1.0"/>

            <!-- Left forearm -->
            <geom name="left_forearm" type="capsule"
                  fromto="0 0 0 0.25 0 0" size="0.03"
                  material="arm_left_mat" mass="1.5"/>

            <!-- Left hand/wrist -->
            <body name="left_hand" pos="0.25 0 0">
              <joint name="left_wrist" type="hinge" axis="0 1 0"
                     range="-1.57 1.57" damping="0.5"/>
              <geom name="left_hand_geom" type="box"
                    size="0.04 0.02 0.08"
                    rgba="0.9 0.7 0.6 1" mass="0.4"/>
            </body>
          </body>
        </body>

        <!-- Right shoulder -->
        <body name="right_shoulder" pos="0.2 0 0.50" euler="0 0 0">
          <joint name="right_shoulder_swing" type="hinge" axis="0 1 0"
                 range="-2.8 2.0" damping="1.5"/>
          <joint name="right_shoulder_lift" type="hinge" axis="1 0 0"
                 range="-1.5 1.5" damping="1.5"/>

          <!-- Right upper arm -->
          <geom name="right_upper_arm" type="capsule"
                fromto="0 0 0 0.25 0 -0.05" size="0.035"
                material="arm_right_mat" mass="2.5"/>

          <!-- Right elbow -->
          <body name="right_elbow" pos="0.25 0 -0.05">
            <joint name="right_elbow" type="hinge" axis="0 1 0"
                   range="-2.4 0" damping="1.0"/>

            <!-- Right forearm -->
            <geom name="right_forearm" type="capsule"
                  fromto="0 0 0 0.25 0 0" size="0.03"
                  material="arm_right_mat" mass="1.5"/>

            <!-- Right hand/wrist connected to club -->
            <body name="right_hand" pos="0.25 0 0">
              <joint name="right_wrist" type="hinge" axis="0 1 0"
                     range="-1.57 1.57" damping="0.5"/>
              <geom name="right_hand_geom" type="box"
                    size="0.04 0.02 0.08"
                    rgba="0.9 0.7 0.6 1" mass="0.4"/>

              <!-- Golf club attached to right hand
                   (left hand connects via equality constraint) -->
              <body name="club" pos="0 0 -0.08" euler="0 -0.3 0">
                <joint name="club_wrist" type="hinge" axis="0 1 0"
                       range="-1.0 1.0" damping="0.3"/>

                <!-- Club grip -->
                <geom name="club_grip" type="capsule"
                      fromto="0 0 0 0 0 -0.25" size="0.015"
                      material="grip_mat" mass="0.1"/>

                <!-- Club shaft -->
                <geom name="club_shaft" type="capsule"
                      fromto="0 0 -0.25 0 0 -1.05" size="0.012"
                      material="club_mat" mass="0.25"/>

                <!-- Club head (driver) -->
                <body name="clubhead" pos="0 0 -1.05">
                  <geom name="clubhead_geom" type="box"
                        size="0.055 0.04 0.03"
                        rgba="0.15 0.15 0.15 1" mass="0.2"/>
                  <!-- Club face indicator -->
                  <geom name="clubface" type="box"
                        size="0.056 0.041 0.005"
                        pos="0 0.041 0"
                        rgba="0.8 0.2 0.2 0.7"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Ball positioned at address -->
    <body name="ball" pos="0 0.1 0.02">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="{_BALL_RADIUS}"
            rgba="1 1 1 1" mass="{_BALL_MASS}"
            condim="3" friction="0.8 0.005 0.0001"/>
    </body>
  </worldbody>

  <!-- Equality constraints to connect left hand to club -->
  <equality>
    <weld body1="left_hand" body2="club"
          relpose="0 0 -0.16 1 0 0 0" active="true"/>
  </equality>

  <actuator>
    <!-- Torso -->
    <motor name="spine_rotation_motor" joint="spine_rotation" gear="100"
           ctrllimited="true" ctrlrange="-100 100"/>

    <!-- Left arm -->
    <motor name="left_shoulder_swing_motor" joint="left_shoulder_swing" gear="50"
           ctrllimited="true" ctrlrange="-80 80"/>
    <motor name="left_shoulder_lift_motor" joint="left_shoulder_lift" gear="50"
           ctrllimited="true" ctrlrange="-80 80"/>
    <motor name="left_elbow_motor" joint="left_elbow" gear="40"
           ctrllimited="true" ctrlrange="-60 60"/>
    <motor name="left_wrist_motor" joint="left_wrist" gear="20"
           ctrllimited="true" ctrlrange="-30 30"/>

    <!-- Right arm -->
    <motor name="right_shoulder_swing_motor" joint="right_shoulder_swing" gear="50"
           ctrllimited="true" ctrlrange="-80 80"/>
    <motor name="right_shoulder_lift_motor" joint="right_shoulder_lift" gear="50"
           ctrllimited="true" ctrlrange="-80 80"/>
    <motor name="right_elbow_motor" joint="right_elbow" gear="40"
           ctrllimited="true" ctrlrange="-60 60"/>
    <motor name="right_wrist_motor" joint="right_wrist" gear="20"
           ctrllimited="true" ctrlrange="-30 30"/>

    <!-- Club -->
    <motor name="club_wrist_motor" joint="club_wrist" gear="15"
           ctrllimited="true" ctrlrange="-20 20"/>
  </actuator>
</mujoco>
"""
