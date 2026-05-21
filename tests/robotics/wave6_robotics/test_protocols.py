"""Tests for src.robotics.core.protocols runtime checks and helpers."""

from __future__ import annotations

import numpy as np

from src.robotics.core import protocols


class _MinimalRC:
    def get_state(self):
        return np.zeros(3), np.zeros(3)

    def set_state(self, q, v):
        return None

    def compute_mass_matrix(self):
        return np.eye(3)

    def compute_bias_forces(self):
        return np.zeros(3)

    def compute_gravity_forces(self):
        return np.zeros(3)

    def compute_jacobian(self, body_name):
        return None

    def get_time(self):
        return 0.0


class _Humanoid(_MinimalRC):
    def get_com_position(self):
        return np.zeros(3)

    def get_com_velocity(self):
        return np.zeros(3)

    def get_total_mass(self):
        return 70.0

    def compute_centroidal_momentum(self):
        return np.zeros(6)

    def compute_centroidal_momentum_matrix(self):
        return np.zeros((6, 3))

    def get_foot_position(self, foot):
        return np.zeros(3)

    def get_foot_velocity(self, foot):
        return np.zeros(3)

    def get_foot_jacobian(self, foot):
        return np.zeros((6, 3))


class _Manip(_MinimalRC):
    def get_end_effector_pose(self, ee_name):
        return np.zeros(7)

    def get_end_effector_velocity(self, ee_name):
        return np.zeros(6)

    def get_end_effector_jacobian(self, ee_name):
        return np.zeros((6, 3))

    def get_gripper_state(self, gripper_name):
        return {"position": 0.0, "velocity": 0.0, "force": 0.0, "is_grasping": False}

    def set_gripper_command(self, gripper_name, command):
        return None


def test_is_robotics_capable_true() -> None:
    assert protocols.is_robotics_capable(_MinimalRC())


def test_is_robotics_capable_false() -> None:
    assert not protocols.is_robotics_capable(object())


def test_is_humanoid_capable_true() -> None:
    assert protocols.is_humanoid_capable(_Humanoid())


def test_is_humanoid_capable_false_for_plain_rc() -> None:
    assert not protocols.is_humanoid_capable(_MinimalRC())


def test_is_manipulation_capable_true() -> None:
    assert protocols.is_manipulation_capable(_Manip())


def test_is_manipulation_capable_false() -> None:
    assert not protocols.is_manipulation_capable(_MinimalRC())


def test_contact_capable_runtime_check() -> None:
    class _C:
        def get_contact_count(self):
            return 0

        def get_contact_info(self, idx):
            return {}

        def get_contact_jacobian(self, idx):
            return None

    assert isinstance(_C(), protocols.ContactCapable)


def test_dynamics_computable_runtime_check() -> None:
    class _D:
        def compute_inverse_dynamics(self, q, v, a):
            return np.zeros(3)

        def compute_forward_dynamics(self, q, v, tau):
            return np.zeros(3)

        def compute_aba(self, q, v, tau):
            return np.zeros(3)

    assert isinstance(_D(), protocols.DynamicsComputable)


def test_simulatable_runtime_check() -> None:
    class _S:
        def step(self, dt=None):
            return None

        def reset(self):
            return None

        def forward(self):
            return None

    assert isinstance(_S(), protocols.Simulatable)
