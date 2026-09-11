"""Canonical 25-DOF Floating-Hip MuJoCo Humanoid Model.

Direct 1-to-1 physical analogue of Simscape Multibody `GolfSwing3D_Kinetic.slx`
and `shared/models/golf_humanoid_topology.yaml`.

Topology summary (25 total DOFs, 19 internal actuated joints):
- 6-DOF floating base at pelvis (`pelvis_floating`)
- Universal joint at spine (2 DOF: `spine_universal_1/2`)
- Revolute joint at torso (1 DOF: `spine_twist`)
- Universal joint at left scapula (2 DOF: `left_scapula_universal_1/2`)
- Gimbal joint at left shoulder (3 DOF: `left_shoulder_gimbal_1/2/3`)
- Revolute joint at left elbow (1 DOF: `left_elbow`)
- Universal joint at left wrist (2 DOF: `left_wrist_universal_1/2`)
- Universal joint at right scapula (2 DOF: `right_scapula_universal_1/2`)
- Gimbal joint at right shoulder (3 DOF: `right_shoulder_gimbal_1/2/3`)
- Revolute joint at right elbow (1 DOF: `right_elbow`)
- Universal joint at right wrist (2 DOF: `right_wrist_universal_1/2`)
- Closed dual-arm grip constraint: weld equality between right hand and club grip.
"""

from __future__ import annotations

from src.shared.python.core.constants import (
    DEFAULT_TIME_STEP,
    GOLF_BALL_MASS_KG,
    GOLF_BALL_RADIUS_M,
    GRAVITY_M_S2,
)

_GRAVITY = float(GRAVITY_M_S2)
_BALL_MASS = float(GOLF_BALL_MASS_KG)
_BALL_RADIUS = float(GOLF_BALL_RADIUS_M)
_TIME_STEP = float(DEFAULT_TIME_STEP)

CANONICAL_GOLF_HUMANOID_XML = rf"""
<mujoco model="canonical_golf_humanoid">
  <option timestep="{_TIME_STEP}" gravity="0 0 -{_GRAVITY}" integrator="RK4"/>

  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>

  <visual>
    <global offwidth="1024" offheight="1024"/>
    <map znear="0.01" zfar="50"/>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <material name="pelvis_mat" rgba="0.3 0.4 0.7 1.0"/>
    <material name="spine_mat" rgba="0.4 0.6 0.8 1.0"/>
    <material name="torso_mat" rgba="0.7 0.5 0.4 1.0"/>
    <material name="arm_mat" rgba="0.6 0.4 0.3 1.0"/>
    <material name="hand_mat" rgba="0.8 0.6 0.5 1.0"/>
    <material name="club_shaft_mat" rgba="0.3 0.3 0.3 1.0"/>
    <material name="club_head_mat" rgba="0.8 0.2 0.2 1.0"/>
    <material name="floor_mat" rgba="0.4 0.6 0.3 1.0"/>
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" material="floor_mat"/>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>

    <!-- Cameras -->
    <camera name="side" pos="-4 -2 1.5" euler="0.15 0 0.35" mode="fixed"/>
    <camera name="front" pos="0 -4 1.5" euler="0.15 0 1.57" mode="fixed"/>
    <camera name="top" pos="0 0 5" euler="0 0 0" mode="fixed"/>

    <!-- 25-DOF Floating Humanoid Skeleton -->
    <body name="pelvis" pos="0 0 1.0">
      <inertial pos="0 0 0" quat="0.5 0.5 0.5 0.5" mass="12.0" diaginertia="0.13 0.13 0.09"/>
      <joint name="pelvis_floating" type="free"/>
      <geom name="pelvis_geom" size="0.15 0.10 0.10" type="box" material="pelvis_mat"/>

      <!-- Spine Universal Joint (2 DOF) -->
      <body name="lower_spine_dummy_1" pos="0 0 0.10">
        <inertial pos="0 0 0" mass="0.001" diaginertia="1e-06 1e-06 1e-06"/>
        <joint name="spine_universal_1" pos="0 0 0" axis="1 0 0" range="-0.6 0.6" damping="0.5"/>
        <body name="lower_spine">
          <inertial pos="0 0 0" mass="7.5" diaginertia="0.06 0.06 0.05"/>
          <joint name="spine_universal_2" pos="0 0 0" axis="0 1 0" range="-0.6 0.6" damping="0.5"/>
          <geom name="lower_spine_geom" size="0.10 0.10 0.125" type="box" material="spine_mat"/>

          <!-- Torso Revolute Joint (1 DOF: axial twist) -->
          <body name="upper_spine" pos="0 0 0.25">
            <inertial pos="0 -0.10 0" quat="0.5 0.5 -0.5 0.5" mass="12.5" diaginertia="0.3125 0.3075 0.12"/>
            <joint name="spine_twist" pos="0 0 0" axis="0 0 1" range="-1.0 1.0" damping="0.5"/>
            <geom name="upper_spine_geom" size="0.10 0.10 0.125" type="box" material="torso_mat"/>
            <geom name="chest_geom" size="0.15 0.15 0.10" pos="0 0 0.25" type="box" material="torso_mat"/>

            <!-- Right Arm Kinematic Chain -->
            <body name="right_scapula_dummy_1" pos="0 -0.18 0.35">
              <inertial pos="0 0 0" mass="0.001" diaginertia="1e-06 1e-06 1e-06"/>
              <joint name="right_scapula_universal_1" pos="0 0 0" axis="1 0 0" range="-0.5 0.5" damping="0.3"/>
              <body name="right_scapula_rod">
                <inertial pos="0 0 0" mass="1.0" diaginertia="0.0014 0.0014 0.00045"/>
                <joint name="right_scapula_universal_2" pos="0 0 0" axis="0 1 0" range="-0.5 0.5" damping="0.3"/>
                <geom name="r_scap_geom" size="0.03 0.06" type="cylinder" material="arm_mat"/>

                <body name="right_upper_arm_dummy_1" pos="0 0 0.12">
                  <inertial pos="0 0 0" mass="0.001" diaginertia="1e-06 1e-06 1e-06"/>
                  <joint name="right_shoulder_gimbal_1" pos="0 0 0" axis="0 0 1" range="-3.14 3.14" damping="0.3"/>
                  <body name="right_upper_arm_dummy_2">
                    <inertial pos="0 0 0" mass="0.001" diaginertia="1e-06 1e-06 1e-06"/>
                    <joint name="right_shoulder_gimbal_2" pos="0 0 0" axis="0 1 0" range="-1.5 1.5" damping="0.3"/>
                    <body name="right_upper_arm">
                      <inertial pos="0 0 0" mass="2.0" diaginertia="0.018 0.018 0.0024"/>
                      <joint name="right_shoulder_gimbal_3" pos="0 0 0" axis="1 0 0" range="-1.5 1.5" damping="0.3"/>
                      <geom name="r_upper_arm_geom" size="0.04 0.15" type="cylinder" material="arm_mat"/>

                      <body name="right_forearm" pos="0 0 -0.30">
                        <inertial pos="0 0 0" mass="1.5" diaginertia="0.012 0.012 0.0014"/>
                        <joint name="right_elbow" pos="0 0 0" axis="0 1 0" range="-2.5 0.0" damping="0.3"/>
                        <geom name="r_forearm_geom" size="0.035 0.135" type="cylinder" material="arm_mat"/>

                        <body name="right_hand_dummy_1" pos="0 0 -0.27">
                          <inertial pos="0 0 0" mass="0.001" diaginertia="1e-06 1e-06 1e-06"/>
                          <joint name="right_wrist_universal_1" pos="0 0 0" axis="1 0 0" range="-1.0 1.0" damping="0.2"/>
                          <body name="right_hand">
                            <inertial pos="0 0 -0.044" mass="0.9" diaginertia="0.0398 0.0398 0.00025"/>
                            <joint name="right_wrist_universal_2" pos="0 0 0" axis="0 1 0" range="-1.0 1.0" damping="0.2"/>
                            <geom name="r_hand_geom" size="0.03 0.05" type="cylinder" material="hand_mat"/>
                            <site name="right_grip_site" pos="0 0 -0.05" size="0.01"/>
                          </body>
                        </body>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>

            <!-- Left Arm Kinematic Chain -->
            <body name="left_scapula_dummy_1" pos="0 0.18 0.35">
              <inertial pos="0 0 0" mass="0.001" diaginertia="1e-06 1e-06 1e-06"/>
              <joint name="left_scapula_universal_1" pos="0 0 0" axis="1 0 0" range="-0.5 0.5" damping="0.3"/>
              <body name="left_scapula_rod">
                <inertial pos="0 0 0" mass="1.0" diaginertia="0.0014 0.0014 0.00045"/>
                <joint name="left_scapula_universal_2" pos="0 0 0" axis="0 1 0" range="-0.5 0.5" damping="0.3"/>
                <geom name="l_scap_geom" size="0.03 0.06" type="cylinder" material="arm_mat"/>

                <body name="left_upper_arm_dummy_1" pos="0 0 0.12">
                  <inertial pos="0 0 0" mass="0.001" diaginertia="1e-06 1e-06 1e-06"/>
                  <joint name="left_shoulder_gimbal_1" pos="0 0 0" axis="0 0 1" range="-3.14 3.14" damping="0.3"/>
                  <body name="left_upper_arm_dummy_2">
                    <inertial pos="0 0 0" mass="0.001" diaginertia="1e-06 1e-06 1e-06"/>
                    <joint name="left_shoulder_gimbal_2" pos="0 0 0" axis="0 1 0" range="-1.5 1.5" damping="0.3"/>
                    <body name="left_upper_arm">
                      <inertial pos="0 0 0" mass="2.0" diaginertia="0.018 0.018 0.0024"/>
                      <joint name="left_shoulder_gimbal_3" pos="0 0 0" axis="1 0 0" range="-1.5 1.5" damping="0.3"/>
                      <geom name="l_upper_arm_geom" size="0.04 0.15" type="cylinder" material="arm_mat"/>

                      <body name="left_forearm" pos="0 0 -0.30">
                        <inertial pos="0 0 0" mass="1.5" diaginertia="0.012 0.012 0.0014"/>
                        <joint name="left_elbow" pos="0 0 0" axis="0 1 0" range="-2.5 0.0" damping="0.3"/>
                        <geom name="l_forearm_geom" size="0.035 0.135" type="cylinder" material="arm_mat"/>

                        <body name="left_hand_dummy_1" pos="0 0 -0.27">
                          <inertial pos="0 0 0" mass="0.001" diaginertia="1e-06 1e-06 1e-06"/>
                          <joint name="left_wrist_universal_1" pos="0 0 0" axis="1 0 0" range="-1.0 1.0" damping="0.2"/>
                          <body name="left_hand">
                            <inertial pos="0 0 0" mass="0.5" diaginertia="0.00057 0.00057 0.000225"/>
                            <joint name="left_wrist_universal_2" pos="0 0 0" axis="0 1 0" range="-1.0 1.0" damping="0.2"/>
                            <geom name="l_hand_geom" size="0.03 0.05" type="cylinder" material="hand_mat"/>

                            <!-- Club rigidly welded to Left Hand -->
                            <body name="club_grip" pos="0 0 -0.05">
                              <inertial pos="0 0 -0.45" mass="0.35" diaginertia="0.04 0.04 0.0005"/>
                              <geom name="shaft_geom" size="0.012 0.50" pos="0 0 -0.50" type="cylinder" material="club_shaft_mat"/>
                              <site name="club_grip_site" pos="0 0 0" size="0.01"/>

                              <!-- Clubhead -->
                              <body name="clubhead" pos="0 0 -1.0">
                                <inertial pos="0 0 0" mass="0.20" diaginertia="0.0005 0.0005 0.0005"/>
                                <geom name="head_geom" size="0.05 0.03 0.03" type="box" material="club_head_mat"/>
                                <site name="clubhead_site" pos="0 0 0" size="0.01"/>
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
    </body>
  </worldbody>

  <!-- Dual-Hand Closed Grip Equality Constraint -->
  <equality>
    <weld site1="right_grip_site" site2="club_grip_site"/>
  </equality>

  <!-- 19 Actuators strictly matching Simscape q_order -->
  <actuator>
    <motor name="spine_universal_1_motor" joint="spine_universal_1" ctrllimited="true" ctrlrange="-200 200"/>
    <motor name="spine_universal_2_motor" joint="spine_universal_2" ctrllimited="true" ctrlrange="-200 200"/>
    <motor name="spine_twist_motor" joint="spine_twist" ctrllimited="true" ctrlrange="-200 200"/>

    <motor name="left_scapula_universal_1_motor" joint="left_scapula_universal_1" ctrllimited="true" ctrlrange="-200 200"/>
    <motor name="left_scapula_universal_2_motor" joint="left_scapula_universal_2" ctrllimited="true" ctrlrange="-200 200"/>
    <motor name="left_shoulder_gimbal_1_motor" joint="left_shoulder_gimbal_1" ctrllimited="true" ctrlrange="-200 200"/>
    <motor name="left_shoulder_gimbal_2_motor" joint="left_shoulder_gimbal_2" ctrllimited="true" ctrlrange="-200 200"/>
    <motor name="left_shoulder_gimbal_3_motor" joint="left_shoulder_gimbal_3" ctrllimited="true" ctrlrange="-200 200"/>
    <motor name="left_elbow_motor" joint="left_elbow" ctrllimited="true" ctrlrange="-200 200"/>
    <motor name="left_wrist_universal_1_motor" joint="left_wrist_universal_1" ctrllimited="true" ctrlrange="-200 200"/>
    <motor name="left_wrist_universal_2_motor" joint="left_wrist_universal_2" ctrllimited="true" ctrlrange="-200 200"/>

    <motor name="right_scapula_universal_1_motor" joint="right_scapula_universal_1" ctrllimited="true" ctrlrange="-200 200"/>
    <motor name="right_scapula_universal_2_motor" joint="right_scapula_universal_2" ctrllimited="true" ctrlrange="-200 200"/>
    <motor name="right_shoulder_gimbal_1_motor" joint="right_shoulder_gimbal_1" ctrllimited="true" ctrlrange="-200 200"/>
    <motor name="right_shoulder_gimbal_2_motor" joint="right_shoulder_gimbal_2" ctrllimited="true" ctrlrange="-200 200"/>
    <motor name="right_shoulder_gimbal_3_motor" joint="right_shoulder_gimbal_3" ctrllimited="true" ctrlrange="-200 200"/>
    <motor name="right_elbow_motor" joint="right_elbow" ctrllimited="true" ctrlrange="-200 200"/>
    <motor name="right_wrist_universal_1_motor" joint="right_wrist_universal_1" ctrllimited="true" ctrlrange="-200 200"/>
    <motor name="right_wrist_universal_2_motor" joint="right_wrist_universal_2" ctrllimited="true" ctrlrange="-200 200"/>
  </actuator>
</mujoco>
"""
