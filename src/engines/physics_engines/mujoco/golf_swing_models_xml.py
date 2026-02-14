"""MJCF XML model definitions.

Split from head_models.py for maintainability.
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
    <body name="ball" pos="0 0.15 0.02135">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="{_BALL_RADIUS}"
            rgba="1 1 1 1" mass="{_BALL_MASS}"
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


ADVANCED_BIOMECHANICAL_GOLF_SWING_XML = rf"""
<mujoco model="advanced_biomechanical_golf_swing">
  <option timestep="0.001" gravity="0 0 -{GRAVITY_M_S2}"
          integrator="RK4" solver="Newton" iterations="50"/>

  <compiler angle="radian" coordinate="local"
            inertiafromgeom="false" balanceinertia="true"/>

  <visual>
    <global offwidth="1024" offheight="1024"/>
    <map znear="0.01" zfar="50"/>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3"/>
    <quality shadowsize="8192"/>
    <scale framelength="0.2" framewidth="0.005"/>
  </visual>

  <default>
    <geom condim="3" friction="0.9 0.005 0.0001"/>
    <joint damping="0.5" armature="0.01"/>
  </default>

  <asset>
    <material name="torso_mat" rgba="0.75 0.55 0.45 1"
              specular="0.3" shininess="0.2"/>
    <material name="arm_mat" rgba="0.65 0.45 0.35 1"
              specular="0.3" shininess="0.2"/>
    <material name="leg_mat" rgba="0.55 0.55 0.65 1"
              specular="0.3" shininess="0.2"/>
    <material name="scapula_mat" rgba="0.85 0.75 0.65 0.8"
              specular="0.4" shininess="0.3"/>
    <material name="club_grip_mat" rgba="0.1 0.1 0.1 1"
              specular="0.1" shininess="0.05"/>
    <material name="club_shaft_mat" rgba="0.3 0.3 0.35 1"
              specular="0.8" shininess="0.6"/>
    <material name="club_head_mat" rgba="0.15 0.15 0.18 1"
              specular="0.9" shininess="0.8"/>
    <material name="ground_mat" rgba="0.35 0.55 0.25 1"/>
  </asset>

  <worldbody>
    <!-- Ground -->
    <geom name="floor" type="plane" size="15 15 0.1" material="ground_mat"/>
    <light pos="3 -3 4" dir="-0.3 0.3 -1" directional="true" diffuse="0.8 0.8 0.8"/>
    <light pos="-3 -3 4" dir="0.3 0.3 -1" directional="true" diffuse="0.4 0.4 0.4"/>

    <!-- Cameras -->
    <camera name="side" pos="-5 -2 1.6" euler="0.15 0 0.35" mode="fixed"/>
    <camera name="front" pos="0 -6 1.6" euler="0.15 0 1.57" mode="fixed"/>
    <camera name="top" pos="0 0 8" euler="0 0 0" mode="fixed"/>
    <camera name="follow" pos="-4 -2 2.5" xyaxes="0 -1 0 0.5 0 1" mode="trackcom"/>
    <camera name="dtl" pos="-2 -2 1.2" xyaxes="0.7 -0.7 0 0.3 0.3 1" mode="fixed"/>

    <!-- Left Foot -->
    <body name="left_foot" pos="-0.12 0 0.04">
      <inertial pos="0.05 0 -0.01" mass="1.2"
                diaginertia="0.0012 0.0035 0.0039"/>
      <geom name="left_foot_geom" type="box" size="0.09 0.14 0.03"
            rgba="0.25 0.25 0.25 1"/>

      <!-- Left Ankle - 2 DOF -->
      <body name="left_talus" pos="0 -0.05 0.04">
        <joint name="left_ankle_plantar" type="hinge" axis="1 0 0"
               range="-0.7 0.9" damping="2.0" armature="0.02"/>
        <joint name="left_ankle_inversion" type="hinge" axis="0 1 0.2"
               range="-0.6 0.6" damping="1.5" armature="0.015"/>

        <!-- Shin -->
        <inertial pos="0 0 0.215" mass="3.8"
                  diaginertia="0.0504 0.0504 0.0048"/>
        <geom name="left_shin_geom" type="capsule" fromto="0 0 0 0 0 0.43"
              size="0.04" material="leg_mat"/>

        <!-- Left Knee - 1 DOF -->
        <body name="left_femur" pos="0 0 0.43">
          <joint name="left_knee" type="hinge" axis="1 0 0"
                 range="-2.3 0.1" damping="3.0" armature="0.03"/>

          <inertial pos="0 0 0.215" mass="9.800"
                    diaginertia="0.1268 0.1268 0.0145"/>
          <geom name="left_thigh_geom" type="capsule" fromto="0 0 0 0 0 0.43"
                size="0.06" material="leg_mat"/>

          <site name="left_hip_site" pos="0 0 0.43"/>
        </body>
      </body>
    </body>

    <!-- Right Foot -->
    <body name="right_foot" pos="0.12 0 0.04">
      <inertial pos="0.05 0 -0.01" mass="1.2"
                diaginertia="0.0012 0.0035 0.0039"/>
      <geom name="right_foot_geom" type="box" size="0.09 0.14 0.03"
            rgba="0.25 0.25 0.25 1"/>

      <!-- Right Ankle - 2 DOF -->
      <body name="right_talus" pos="0 -0.05 0.04">
        <joint name="right_ankle_plantar" type="hinge" axis="1 0 0"
               range="-0.7 0.9" damping="2.0" armature="0.02"/>
        <joint name="right_ankle_inversion" type="hinge" axis="0 1 0.2"
               range="-0.6 0.6" damping="1.5" armature="0.015"/>

        <!-- Shin -->
        <inertial pos="0 0 0.215" mass="3.8"
                  diaginertia="0.0504 0.0504 0.0048"/>
        <geom name="right_shin_geom" type="capsule" fromto="0 0 0 0 0 0.43"
              size="0.04" material="leg_mat"/>

        <!-- Right Knee - 1 DOF -->
        <body name="right_femur" pos="0 0 0.43">
          <joint name="right_knee" type="hinge" axis="1 0 0"
                 range="-2.3 0.1" damping="3.0" armature="0.03"/>

          <inertial pos="0 0 0.215" mass="9.800"
                    diaginertia="0.1268 0.1268 0.0145"/>
          <geom name="right_thigh_geom" type="capsule" fromto="0 0 0 0 0 0.43"
                size="0.06" material="leg_mat"/>

          <site name="right_hip_site" pos="0 0 0.43"/>
        </body>
      </body>
    </body>

    <!-- Pelvis -->
    <body name="pelvis" pos="0 0 0.90">
      <freejoint/>
      <inertial pos="0 0 0" mass="11.7"
                diaginertia="0.0956 0.0929 0.0614"/>
      <geom name="pelvis_geom" type="box" size="0.17 0.10 0.09"
            material="torso_mat"/>

      <!-- Lower Spine -->
      <body name="lumbar" pos="0 0 0.10">
        <!-- Spine - 3 DOF: lateral bend, sagittal bend, axial rotation -->
        <joint name="spine_lateral" type="hinge" axis="0 1 0"
               range="-0.5 0.5" damping="4.0" armature="0.04"/>
        <joint name="spine_sagittal" type="hinge" axis="1 0 0"
               range="-0.7 1.0" damping="4.5" armature="0.045"/>
        <joint name="spine_rotation" type="hinge" axis="0 0 1"
               range="-1.9 1.9" damping="3.5" armature="0.035"/>

        <inertial pos="0 0 0.125" mass="13.2"
                  diaginertia="0.1139 0.1139 0.0221"/>
        <geom name="lumbar_geom" type="capsule" fromto="0 0 0 0 0 0.15"
              size="0.12" material="torso_mat"/>

        <!-- Upper Thorax -->
        <body name="thorax" pos="0 0 0.15">
          <inertial pos="0 0 0.15" mass="17.5"
                    diaginertia="0.1996 0.1996 0.0384"/>
          <geom name="thorax_geom" type="capsule" fromto="0 0 0 0 0 0.30"
                size="0.14" material="torso_mat"/>

          <!-- Head and Neck -->
          <body name="neck" pos="0 0 0.30">
            <inertial pos="0 0 0.12" mass="5.6"
                      diaginertia="0.0268 0.0268 0.0102"/>
            <geom name="neck_geom" type="capsule" fromto="0 0 0 0 0 0.08"
                  size="0.045" material="torso_mat"/>
            <geom name="head_geom" type="sphere" pos="0 0 0.16" size="0.11"
                  rgba="0.9 0.7 0.6 1"/>
          </body>

          <!-- LEFT SCAPULA - 2 DOF (elevation/depression, protraction/retraction) -->
          <body name="left_scapula" pos="-0.08 0 0.25" euler="0 0 -0.2">
            <joint name="left_scapula_elevation" type="hinge" axis="0 1 0"
                   range="-0.5 0.8" damping="1.2" armature="0.012"/>
            <joint name="left_scapula_protraction" type="hinge" axis="0 0 1"
                   range="-0.7 0.7" damping="1.0" armature="0.01"/>

            <inertial pos="-0.06 0 0" mass="0.8"
                      diaginertia="0.0008 0.0012 0.0008"/>
            <geom name="left_scapula_geom" type="box" size="0.06 0.04 0.10"
                  material="scapula_mat"/>

            <!-- LEFT SHOULDER - 3 DOF
                 (flexion/extension, abduction/adduction, rotation) -->
            <body name="left_humerus" pos="-0.08 0 0" euler="0 0 0">
              <joint name="left_shoulder_flexion" type="hinge" axis="0 1 0"
                     range="-1.0 3.0" damping="2.5" armature="0.025"/>
              <joint name="left_shoulder_abduction" type="hinge" axis="1 0 0"
                     range="-0.5 2.8" damping="2.5" armature="0.025"/>
              <joint name="left_shoulder_rotation" type="hinge" axis="0 0 1"
                     range="-1.6 1.6" damping="2.0" armature="0.02"/>

              <inertial pos="0.16 0 0" mass="2.1"
                        diaginertia="0.0116 0.0116 0.0018"/>
              <geom name="left_humerus_geom" type="capsule" fromto="0 0 0 0.32 0 -0.02"
                    size="0.036" material="arm_mat"/>

              <!-- LEFT ELBOW - 1 DOF -->
              <body name="left_radius" pos="0.32 0 -0.02">
                <joint name="left_elbow" type="hinge" axis="0 1 0"
                       range="-2.6 0" damping="1.8" armature="0.018"/>

                <inertial pos="0.14 0 0" mass="1.3"
                          diaginertia="0.0064 0.0064 0.0009"/>
                <geom name="left_forearm_geom" type="capsule" fromto="0 0 0 0.28 0 0"
                      size="0.032" material="arm_mat"/>

                <!-- LEFT WRIST - 2 DOF (flexion/extension, radial/ulnar deviation) -->
                <body name="left_hand" pos="0.28 0 0">
                  <joint name="left_wrist_flexion" type="hinge" axis="0 1 0"
                         range="-1.4 1.4" damping="0.8" armature="0.008"/>
                  <joint name="left_wrist_deviation" type="hinge" axis="1 0 0"
                         range="-0.6 0.5" damping="0.6" armature="0.006"/>

                  <inertial pos="0.045 0 0" mass="0.45"
                            diaginertia="0.0008 0.0008 0.0002"/>
                  <geom name="left_hand_geom" type="box" size="0.045 0.025 0.10"
                        rgba="0.9 0.7 0.6 1"/>
                </body>
              </body>
            </body>
          </body>

          <!-- RIGHT SCAPULA - 2 DOF -->
          <body name="right_scapula" pos="0.08 0 0.25" euler="0 0 0.2">
            <joint name="right_scapula_elevation" type="hinge" axis="0 1 0"
                   range="-0.5 0.8" damping="1.2" armature="0.012"/>
            <joint name="right_scapula_protraction" type="hinge" axis="0 0 1"
                   range="-0.7 0.7" damping="1.0" armature="0.01"/>

            <inertial pos="0.06 0 0" mass="0.8"
                      diaginertia="0.0008 0.0012 0.0008"/>
            <geom name="right_scapula_geom" type="box" size="0.06 0.04 0.10"
                  material="scapula_mat"/>

            <!-- RIGHT SHOULDER - 3 DOF -->
            <body name="right_humerus" pos="0.08 0 0">
              <joint name="right_shoulder_flexion" type="hinge" axis="0 1 0"
                     range="-1.0 3.0" damping="2.5" armature="0.025"/>
              <joint name="right_shoulder_abduction" type="hinge" axis="1 0 0"
                     range="-2.8 0.5" damping="2.5" armature="0.025"/>
              <joint name="right_shoulder_rotation" type="hinge" axis="0 0 1"
                     range="-1.6 1.6" damping="2.0" armature="0.02"/>

              <inertial pos="0.16 0 0" mass="2.1"
                        diaginertia="0.0116 0.0116 0.0018"/>
              <geom name="right_humerus_geom" type="capsule" fromto="0 0 0 0.32 0 -0.02"
                    size="0.036" material="arm_mat"/>

              <!-- RIGHT ELBOW - 1 DOF -->
              <body name="right_radius" pos="0.32 0 -0.02">
                <joint name="right_elbow" type="hinge" axis="0 1 0"
                       range="-2.6 0" damping="1.8" armature="0.018"/>

                <inertial pos="0.14 0 0" mass="1.3"
                          diaginertia="0.0064 0.0064 0.0009"/>
                <geom name="right_forearm_geom" type="capsule" fromto="0 0 0 0.28 0 0"
                      size="0.032" material="arm_mat"/>

                <!-- RIGHT WRIST - 2 DOF -->
                <body name="right_hand" pos="0.28 0 0">
                  <joint name="right_wrist_flexion" type="hinge" axis="0 1 0"
                         range="-1.4 1.4" damping="0.8" armature="0.008"/>
                  <joint name="right_wrist_deviation" type="hinge" axis="1 0 0"
                         range="-0.5 0.6" damping="0.6" armature="0.006"/>

                  <inertial pos="0.045 0 0" mass="0.45"
                            diaginertia="0.0008 0.0008 0.0002"/>
                  <geom name="right_hand_geom" type="box" size="0.045 0.025 0.10"
                        rgba="0.9 0.7 0.6 1"/>

                  <!-- GOLF CLUB - FLEXIBLE SHAFT -->
                  <body name="club_grip" pos="0 0 -0.10" euler="0 -0.4 0">
                    <inertial pos="0 0 -0.14" mass="0.050"
                              diaginertia="0.00008 0.00008 0.000001"/>
                    <geom name="grip_geom" type="capsule"
                          fromto="0 0 0 0 0 -0.28"
                          size="0.0145" material="club_grip_mat"/>

                    <!-- Shaft Segment 1 (upper) -->
                    <body name="shaft_upper" pos="0 0 -0.28">
                      <joint name="shaft_flex_upper" type="hinge" axis="1 0 0"
                             range="-0.15 0.15" damping="0.4"
                             stiffness="150" armature="0.001"/>

                      <inertial pos="0 0 -0.13" mass="0.045"
                                diaginertia="0.00006 0.00006 0.000001"/>
                      <geom name="shaft_upper_geom" type="capsule"
                            fromto="0 0 0 0 0 -0.26"
                            size="0.0062" material="club_shaft_mat"/>

                      <!-- Shaft Segment 2 (middle) -->
                      <body name="shaft_middle" pos="0 0 -0.26">
                        <joint name="shaft_flex_middle" type="hinge" axis="1 0 0"
                               range="-0.2 0.2" damping="0.3"
                               stiffness="120" armature="0.001"/>

                        <inertial pos="0 0 -0.13" mass="0.055"
                                  diaginertia="0.00007 0.00007 0.000001"/>
                        <geom name="shaft_middle_geom" type="capsule"
                              fromto="0 0 0 0 0 -0.26"
                              size="0.0060" material="club_shaft_mat"/>

                        <!-- Shaft Segment 3 (lower - tip) -->
                        <body name="shaft_tip" pos="0 0 -0.26">
                          <joint name="shaft_flex_tip" type="hinge" axis="1 0 0"
                                 range="-0.25 0.25" damping="0.2"
                                 stiffness="100" armature="0.001"/>

                          <inertial pos="0 0 -0.12" mass="0.058"
                                    diaginertia="0.00008 0.00008 0.000001"/>
                          <geom name="shaft_tip_geom" type="capsule"
                                fromto="0 0 0 0 0 -0.24"
                                size="0.0058" material="club_shaft_mat"/>

                          <!-- Hosel and Club Head -->
                          <body name="hosel" pos="0 0 -0.24" euler="0 0.17 0">
                            <inertial pos="0 0.02 -0.01" mass="0.010"
                                      diaginertia="0.000005 0.000005 0.000002"/>
                            <geom name="hosel_geom" type="cylinder"
                                  fromto="0 0 0 0 0.035 -0.005"
                                  size="0.008" material="club_head_mat"/>

                            <!-- Driver Head -->
                            <body name="clubhead" pos="0 0.050 -0.010">
                              <inertial pos="0 0.025 0.002" mass="0.198"
                                        diaginertia="0.00035 0.00025 0.00018"/>

                              <!-- Club head body -->
                              <geom name="head_body" type="box" size="0.062 0.048 0.038"
                                    pos="0 0.025 0" material="club_head_mat"/>

                              <!-- Face -->
                              <geom name="face" type="box" size="0.063 0.003 0.039"
                                    pos="0 0.051 0" rgba="0.85 0.15 0.15 0.9"/>

                              <!-- Crown (top) -->
                              <geom name="crown" type="box" size="0.055 0.042 0.002"
                                    pos="0 0.025 0.040" rgba="0.12 0.12 0.15 1"/>

                              <!-- Sole (bottom) -->
                              <geom name="sole" type="box" size="0.056 0.043 0.003"
                                    pos="0 0.025 -0.041" rgba="0.18 0.18 0.20 1"/>

                              <!-- Alignment aid -->
                              <geom name="alignment" type="box" size="0.002 0.035 0.001"
                                    pos="0 0.025 0.043" rgba="1 1 1 0.95"/>
                            </body>
                          </body>
                        </body>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Golf Ball -->
    <body name="ball" pos="0 0.20 {_BALL_RADIUS}">
      <freejoint/>
      <inertial pos="0 0 0" mass="{_BALL_MASS}"
                diaginertia="0.000017 0.000017 0.000017"/>
      <geom name="ball_geom" type="sphere" size="{_BALL_RADIUS}"
            rgba="1 1 1 1" condim="3" friction="0.8 0.005 0.0001"/>
      <!-- Dimple visualization (cosmetic) -->
      <geom name="ball_detail" type="sphere" size="{_BALL_RADIUS_INNER}"
            rgba="0.95 0.95 0.95 0.3"/>
    </body>
  </worldbody>

  <!-- Connect pelvis to legs via equality constraints -->
  <equality>
    <connect body1="pelvis" body2="left_femur" anchor="-0.12 0 0"/>
    <connect body1="pelvis" body2="right_femur" anchor="0.12 0 0"/>
    <!-- Left hand grips club -->
    <weld body1="left_hand" body2="club_grip"
          relpose="0 0 -0.18 1 0 0 0" active="true"/>
  </equality>

  <actuator>
    <!-- Left Leg (3 DOF) -->
    <motor name="l_ankle_plantar" joint="left_ankle_plantar"
           gear="40" ctrllimited="true" ctrlrange="-50 50"/>
    <motor name="l_ankle_invert" joint="left_ankle_inversion"
           gear="30" ctrllimited="true" ctrlrange="-35 35"/>
    <motor name="l_knee" joint="left_knee"
           gear="120" ctrllimited="true" ctrlrange="-150 150"/>

    <!-- Right Leg (3 DOF) -->
    <motor name="r_ankle_plantar" joint="right_ankle_plantar"
           gear="40" ctrllimited="true" ctrlrange="-50 50"/>
    <motor name="r_ankle_invert" joint="right_ankle_inversion"
           gear="30" ctrllimited="true" ctrlrange="-35 35"/>
    <motor name="r_knee" joint="right_knee"
           gear="120" ctrllimited="true" ctrlrange="-150 150"/>

    <!-- Spine (3 DOF) -->
    <motor name="spine_lateral" joint="spine_lateral"
           gear="140" ctrllimited="true" ctrlrange="-120 120"/>
    <motor name="spine_sagittal" joint="spine_sagittal"
           gear="150" ctrllimited="true" ctrlrange="-130 130"/>
    <motor name="spine_rotation" joint="spine_rotation"
           gear="130" ctrllimited="true" ctrlrange="-110 110"/>

    <!-- Left Scapula (2 DOF) -->
    <motor name="l_scap_elev" joint="left_scapula_elevation"
           gear="35" ctrllimited="true" ctrlrange="-40 40"/>
    <motor name="l_scap_prot" joint="left_scapula_protraction"
           gear="30" ctrllimited="true" ctrlrange="-35 35"/>

    <!-- Left Shoulder (3 DOF) -->
    <motor name="l_shldr_flex" joint="left_shoulder_flexion"
           gear="70" ctrllimited="true" ctrlrange="-90 90"/>
    <motor name="l_shldr_abd" joint="left_shoulder_abduction"
           gear="70" ctrllimited="true" ctrlrange="-90 90"/>
    <motor name="l_shldr_rot" joint="left_shoulder_rotation"
           gear="50" ctrllimited="true" ctrlrange="-65 65"/>

    <!-- Left Elbow (1 DOF) -->
    <motor name="l_elbow" joint="left_elbow"
           gear="60" ctrllimited="true" ctrlrange="-75 75"/>

    <!-- Left Wrist (2 DOF) -->
    <motor name="l_wrist_flex" joint="left_wrist_flexion"
           gear="25" ctrllimited="true" ctrlrange="-35 35"/>
    <motor name="l_wrist_dev" joint="left_wrist_deviation"
           gear="20" ctrllimited="true" ctrlrange="-28 28"/>

    <!-- Right Scapula (2 DOF) -->
    <motor name="r_scap_elev" joint="right_scapula_elevation"
           gear="35" ctrllimited="true" ctrlrange="-40 40"/>
    <motor name="r_scap_prot" joint="right_scapula_protraction"
           gear="30" ctrllimited="true" ctrlrange="-35 35"/>

    <!-- Right Shoulder (3 DOF) -->
    <motor name="r_shldr_flex" joint="right_shoulder_flexion"
           gear="70" ctrllimited="true" ctrlrange="-90 90"/>
    <motor name="r_shldr_abd" joint="right_shoulder_abduction"
           gear="70" ctrllimited="true" ctrlrange="-90 90"/>
    <motor name="r_shldr_rot" joint="right_shoulder_rotation"
           gear="50" ctrllimited="true" ctrlrange="-65 65"/>

    <!-- Right Elbow (1 DOF) -->
    <motor name="r_elbow" joint="right_elbow"
           gear="60" ctrllimited="true" ctrlrange="-75 75"/>

    <!-- Right Wrist (2 DOF) -->
    <motor name="r_wrist_flex" joint="right_wrist_flexion"
           gear="25" ctrllimited="true" ctrlrange="-35 35"/>
    <motor name="r_wrist_dev" joint="right_wrist_deviation"
           gear="20" ctrllimited="true" ctrlrange="-28 28"/>

    <!-- Flexible Shaft (3 DOF) - typically controlled passively,
         but can be actuated for study -->
    <motor name="shaft_upper" joint="shaft_flex_upper"
           gear="2" ctrllimited="true" ctrlrange="-5 5"/>
    <motor name="shaft_middle" joint="shaft_flex_middle"
           gear="2" ctrllimited="true" ctrlrange="-5 5"/>
    <motor name="shaft_tip" joint="shaft_flex_tip"
           gear="2" ctrllimited="true" ctrlrange="-5 5"/>
  </actuator>
</mujoco>
"""


# ==============================================================================

# ==============================================================================
# MyoSuite Musculoskeletal Model Paths
# ==============================================================================
# These models use the MyoSuite musculoskeletal framework with realistic
# muscle-tendon units, biomechanical constraints, and physiological parameters.
# Models are loaded from external XML files in the myo_sim directory.

# Upper body model (19 DOF, 20 actuators): Torso + head + both arms
MYOUPPERBODY_PATH = "myo_sim/body/myoupperbody.xml"

# Full body model (52 DOF, 290 actuators): Complete musculoskeletal system
# Includes torso, head, arms, and legs with muscle-tendon units
MYOBODY_PATH = "myo_sim/body/myobody.xml"

# Simplified arm model (bilateral, 14 DOF): Both arms with simplified torso
MYOARM_SIMPLE_PATH = "myo_sim/arm/myoarm_simple.xml"


# GOLF CLUB CONFIGURATIONS
# ==============================================================================

# Golf club parameters (realistic values)
CLUB_CONFIGS: dict[str, dict[str, float | list[float]]] = {
    "driver": {
        "grip_length": 0.28,
        "grip_radius": 0.0145,
        "grip_mass": 0.050,
        "shaft_length": 1.10,  # Total shaft length
        "shaft_radius": 0.0062,
        "shaft_mass": 0.065,
        "head_mass": 0.198,
        "head_size": [0.062, 0.048, 0.038],
        "total_length": 1.16,
        "club_loft": 0.17,  # 10 degrees
        "flex_stiffness": [180, 150, 120],  # Upper, middle, lower
    },
    "iron_7": {
        "grip_length": 0.26,
        "grip_radius": 0.0140,
        "grip_mass": 0.048,
        "shaft_length": 0.94,
        "shaft_radius": 0.0058,
        "shaft_mass": 0.072,
        "head_mass": 0.253,
        "head_size": [0.038, 0.025, 0.045],
        "total_length": 0.95,
        "club_loft": 0.56,  # 32 degrees
        "flex_stiffness": [220, 200, 180],
    },
    "wedge": {
        "grip_length": 0.25,
        "grip_radius": 0.0138,
        "grip_mass": 0.045,
        "shaft_length": 0.89,
        "shaft_radius": 0.0056,
        "shaft_mass": 0.078,
        "head_mass": 0.288,
        "head_size": [0.032, 0.022, 0.048],
        "total_length": 0.90,
        "club_loft": 0.96,  # 55 degrees
        "flex_stiffness": [240, 220, 200],
    },
}
