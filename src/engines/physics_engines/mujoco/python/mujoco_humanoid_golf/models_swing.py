"""MJCF golf swing models – upper body, full body, and advanced biomechanical.

These are biomechanically realistic golf swing models with anthropometric body
segments, two-handed grip, and multi-DOF joint hierarchies.
"""

from __future__ import annotations

from src.shared.python.core import constants
from src.shared.python.physics.physics_parameters import get_parameter_registry

GRAVITY_M_S2 = float(constants.GRAVITY_M_S2)
DEFAULT_TIME_STEP = float(constants.DEFAULT_TIME_STEP)

_registry = get_parameter_registry()
_ball_mass_param = _registry.get("BALL_MASS")
_ball_radius_param = _registry.get("BALL_RADIUS")

BALL_MASS = (
    float(_ball_mass_param.value)
    if _ball_mass_param
    else float(constants.GOLF_BALL_MASS_KG)
)
BALL_RADIUS = (
    float(_ball_radius_param.value)
    if _ball_radius_param
    else float(constants.GOLF_BALL_RADIUS_M)
)


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
    <body name="ball" pos="0 0.1 {BALL_RADIUS}">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="{BALL_RADIUS}"
            rgba="1 1 1 1" mass="{BALL_MASS}"
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


FULL_BODY_GOLF_SWING_XML = rf"""
<mujoco model="golf_full_body_swing">
  <option timestep="0.002" gravity="0 0 -{GRAVITY_M_S2}" integrator="RK4"/>

  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>

  <visual>
    <global offwidth="1024" offheight="1024"/>
    <map znear="0.01" zfar="50"/>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <material name="torso_mat" rgba="0.7 0.5 0.4 1"/>
    <material name="arm_mat" rgba="0.6 0.4 0.3 1"/>
    <material name="leg_mat" rgba="0.5 0.5 0.6 1"/>
    <material name="club_mat" rgba="0.2 0.2 0.2 1"/>
    <material name="grip_mat" rgba="0.1 0.1 0.1 1"/>
    <material name="ground_mat" rgba="0.4 0.6 0.3 1"/>
  </asset>

  <worldbody>
    <!-- Ground plane -->
    <geom name="floor" type="plane" size="10 10 0.1" material="ground_mat"/>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>

    <!-- Camera views -->
    <camera name="side" pos="-5 -2 1.5" euler="0.15 0 0.35" mode="fixed"/>
    <camera name="front" pos="0 -5 1.5" euler="0.15 0 1.57" mode="fixed"/>
    <camera name="top" pos="0 0 6" euler="0 0 0" mode="fixed"/>
    <camera name="follow" pos="-3 -1 2" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>

    <!-- Left foot (grounded) -->
    <body name="left_foot" pos="-0.15 0 0.05">
      <geom name="left_foot_geom" type="box" size="0.08 0.12 0.04"
            rgba="0.3 0.3 0.3 1" mass="1.0"/>

      <!-- Left ankle -->
      <body name="left_shin" pos="0 0 0.05">
        <joint name="left_ankle" type="hinge" axis="1 0 0"
               range="-0.8 0.8" damping="1.5"/>

        <!-- Left shin -->
        <geom name="left_shin_geom" type="capsule"
              fromto="0 0 0 0 0 0.43" size="0.04"
              material="leg_mat" mass="4.0"/>

        <!-- Left knee -->
        <body name="left_thigh" pos="0 0 0.43">
          <joint name="left_knee" type="hinge" axis="1 0 0"
                 range="-2.0 0" damping="2.0"/>

          <!-- Left thigh -->
          <geom name="left_thigh_geom" type="capsule"
                fromto="0 0 0 0 0 0.45" size="0.055"
                material="leg_mat" mass="8.0"/>

          <!-- Left hip connection point -->
          <site name="left_hip_site" pos="0 0 0.45"/>
        </body>
      </body>
    </body>

    <!-- Right foot (grounded) -->
    <body name="right_foot" pos="0.15 0 0.05">
      <geom name="right_foot_geom" type="box" size="0.08 0.12 0.04"
            rgba="0.3 0.3 0.3 1" mass="1.0"/>

      <!-- Right ankle -->
      <body name="right_shin" pos="0 0 0.05">
        <joint name="right_ankle" type="hinge" axis="1 0 0"
               range="-0.8 0.8" damping="1.5"/>

        <!-- Right shin -->
        <geom name="right_shin_geom" type="capsule"
              fromto="0 0 0 0 0 0.43" size="0.04"
              material="leg_mat" mass="4.0"/>

        <!-- Right knee -->
        <body name="right_thigh" pos="0 0 0.43">
          <joint name="right_knee" type="hinge" axis="1 0 0"
                 range="-2.0 0" damping="2.0"/>

          <!-- Right thigh -->
          <geom name="right_thigh_geom" type="capsule"
                fromto="0 0 0 0 0 0.45" size="0.055"
                material="leg_mat" mass="8.0"/>

          <!-- Right hip connection point -->
          <site name="right_hip_site" pos="0 0 0.45"/>
        </body>
      </body>
    </body>

    <!-- Pelvis (connected to both legs via equality constraints) -->
    <body name="pelvis" pos="0 0 0.93">
      <freejoint/>
      <geom name="pelvis_geom" type="box" size="0.16 0.09 0.09"
            rgba="0.6 0.5 0.4 1" mass="12"/>

      <!-- Torso connected via spine joints -->
      <body name="lower_torso" pos="0 0 0.1">
        <joint name="spine_bend" type="hinge" axis="1 0 0"
               range="-0.5 0.8" damping="3.0"/>

        <geom name="lower_torso_geom" type="capsule"
              fromto="0 0 0 0 0 0.25" size="0.12"
              material="torso_mat" mass="12"/>

        <body name="upper_torso" pos="0 0 0.25">
          <joint name="spine_rotation" type="hinge" axis="0 0 1"
                 range="-1.8 1.8" damping="3.0"/>

          <geom name="upper_torso_geom" type="capsule"
                fromto="0 0 0 0 0 0.25" size="0.14"
                material="torso_mat" mass="15"/>

          <!-- Head (for reference) -->
          <body name="head" pos="0 0 0.30">
            <geom name="head_geom" type="sphere" size="0.1"
                  rgba="0.9 0.7 0.6 1" mass="5"/>
          </body>

          <!-- Left shoulder -->
          <body name="left_shoulder" pos="-0.20 0 0.20">
            <joint name="left_shoulder_swing" type="hinge" axis="0 1 0"
                   range="-2.2 3.0" damping="1.5"/>
            <joint name="left_shoulder_lift" type="hinge" axis="1 0 0"
                   range="-1.5 1.5" damping="1.5"/>

            <geom name="left_upper_arm" type="capsule"
                  fromto="0 0 0 0.28 0 -0.06" size="0.035"
                  material="arm_mat" mass="2.5"/>

            <body name="left_elbow" pos="0.28 0 -0.06">
              <joint name="left_elbow" type="hinge" axis="0 1 0"
                     range="-2.5 0" damping="1.0"/>

              <geom name="left_forearm" type="capsule"
                    fromto="0 0 0 0.26 0 0" size="0.03"
                    material="arm_mat" mass="1.5"/>

              <body name="left_hand" pos="0.26 0 0">
                <joint name="left_wrist" type="hinge" axis="0 1 0"
                       range="-1.6 1.6" damping="0.5"/>
                <geom name="left_hand_geom" type="box"
                      size="0.04 0.02 0.09"
                      rgba="0.9 0.7 0.6 1" mass="0.4"/>
              </body>
            </body>
          </body>

          <!-- Right shoulder -->
          <body name="right_shoulder" pos="0.20 0 0.20">
            <joint name="right_shoulder_swing" type="hinge" axis="0 1 0"
                   range="-3.0 2.2" damping="1.5"/>
            <joint name="right_shoulder_lift" type="hinge" axis="1 0 0"
                   range="-1.5 1.5" damping="1.5"/>

            <geom name="right_upper_arm" type="capsule"
                  fromto="0 0 0 0.28 0 -0.06" size="0.035"
                  material="arm_mat" mass="2.5"/>

            <body name="right_elbow" pos="0.28 0 -0.06">
              <joint name="right_elbow" type="hinge" axis="0 1 0"
                     range="-2.5 0" damping="1.0"/>

              <geom name="right_forearm" type="capsule"
                    fromto="0 0 0 0.26 0 0" size="0.03"
                    material="arm_mat" mass="1.5"/>

              <body name="right_hand" pos="0.26 0 0">
                <joint name="right_wrist" type="hinge" axis="0 1 0"
                       range="-1.6 1.6" damping="0.5"/>
                <geom name="right_hand_geom" type="box"
                      size="0.04 0.02 0.09"
                      rgba="0.9 0.7 0.6 1" mass="0.4"/>

                <!-- Club attached to right hand -->
                <body name="club" pos="0 0 -0.09" euler="0 -0.35 0">
                  <joint name="club_wrist" type="hinge" axis="0 1 0"
                         range="-1.2 1.2" damping="0.3"/>

                  <geom name="club_grip" type="capsule"
                        fromto="0 0 0 0 0 -0.28" size="0.015"
                        material="grip_mat" mass="0.1"/>

                  <geom name="club_shaft" type="capsule"
                        fromto="0 0 -0.28 0 0 -1.08" size="0.012"
                        material="club_mat" mass="0.28"/>

                  <body name="clubhead" pos="0 0 -1.08">
                    <geom name="clubhead_geom" type="box"
                          size="0.06 0.04 0.032"
                          rgba="0.15 0.15 0.15 1" mass="0.2"/>
                    <geom name="clubface" type="box"
                          size="0.061 0.041 0.005"
                          pos="0 0.041 0"
                          rgba="0.9 0.2 0.2 0.7"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Golf ball -->
    <body name="ball" pos="0 0.15 {BALL_RADIUS}">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="{BALL_RADIUS}"
            rgba="1 1 1 1" mass="{BALL_MASS}"
            condim="3" friction="0.8 0.005 0.0001"/>
    </body>
  </worldbody>

  <!-- Connect pelvis to legs -->
  <equality>
    <connect body1="pelvis" body2="left_thigh" anchor="0 0 0"/>
    <connect body1="pelvis" body2="right_thigh" anchor="0 0 0"/>
    <weld body1="left_hand" body2="club" relpose="0 0 -0.18 1 0 0 0"/>
  </equality>

  <actuator>
    <!-- Legs -->
    <motor name="left_ankle_motor" joint="left_ankle" gear="30"
           ctrllimited="true" ctrlrange="-40 40"/>
    <motor name="left_knee_motor" joint="left_knee" gear="80"
           ctrllimited="true" ctrlrange="-120 120"/>
    <motor name="right_ankle_motor" joint="right_ankle" gear="30"
           ctrllimited="true" ctrlrange="-40 40"/>
    <motor name="right_knee_motor" joint="right_knee" gear="80"
           ctrllimited="true" ctrlrange="-120 120"/>

    <!-- Torso -->
    <motor name="spine_bend_motor" joint="spine_bend" gear="120"
           ctrllimited="true" ctrlrange="-100 100"/>
    <motor name="spine_rotation_motor" joint="spine_rotation" gear="120"
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
