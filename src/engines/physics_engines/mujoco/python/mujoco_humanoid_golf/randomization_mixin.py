"""Mixin exposing domain-randomization hooks on :class:`MuJoCoPhysicsEngine`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.domain_randomization_api import (
    read_actuator_gains,
    read_friction_coefficients,
    read_gravity,
    read_joint_damping,
    read_link_masses,
    read_motor_strength,
    write_actuator_gains,
    write_friction_coefficients,
    write_gravity,
    write_joint_damping,
    write_link_masses,
    write_motor_strength,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MuJoCoDomainRandomizationMixin:
    """Optional parameter getters/setters consumed by ``DomainRandomizer``.

    The host class must provide ``model`` and ``require_initialized`` (as
    :class:`BasePhysicsEngine` does); they are declared here for type checking
    only so the mixin never shadows the host's implementation at runtime.
    """

    if TYPE_CHECKING:
        model: Any

        def require_initialized(self, operation: str = "this operation") -> None:
            """Raise if the host engine has no loaded model."""

    def get_link_masses(self) -> NDArray[np.floating]:
        self.require_initialized("get_link_masses")
        return read_link_masses(self.model)

    def set_link_masses(self, masses: NDArray[np.floating]) -> None:
        self.require_initialized("set_link_masses")
        write_link_masses(self.model, masses)

    def get_joint_damping(self) -> NDArray[np.floating]:
        self.require_initialized("get_joint_damping")
        return read_joint_damping(self.model)

    def set_joint_damping(self, damping: NDArray[np.floating]) -> None:
        self.require_initialized("set_joint_damping")
        write_joint_damping(self.model, damping)

    def get_friction_coefficients(self) -> NDArray[np.floating]:
        self.require_initialized("get_friction_coefficients")
        return read_friction_coefficients(self.model)

    def set_friction_coefficients(self, friction: NDArray[np.floating]) -> None:
        self.require_initialized("set_friction_coefficients")
        write_friction_coefficients(self.model, friction)

    def get_motor_strength(self) -> NDArray[np.floating]:
        self.require_initialized("get_motor_strength")
        return read_motor_strength(self.model)

    def set_motor_strength(self, strength: NDArray[np.floating]) -> None:
        self.require_initialized("set_motor_strength")
        write_motor_strength(self.model, strength)

    def get_actuator_gains(self) -> NDArray[np.floating]:
        self.require_initialized("get_actuator_gains")
        return read_actuator_gains(self.model)

    def set_actuator_gains(self, gains: NDArray[np.floating]) -> None:
        self.require_initialized("set_actuator_gains")
        write_actuator_gains(self.model, gains)

    def get_gravity(self) -> NDArray[np.floating]:
        self.require_initialized("get_gravity")
        return read_gravity(self.model)

    def set_gravity(self, gravity: NDArray[np.floating]) -> None:
        self.require_initialized("set_gravity")
        write_gravity(self.model, gravity)
