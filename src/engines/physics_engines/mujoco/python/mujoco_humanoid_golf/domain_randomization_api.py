"""MuJoCo accessors for :mod:`src.learning.sim2real.domain_randomization`."""

from __future__ import annotations

import mujoco
import numpy as np
from numpy.typing import NDArray


def _require_model(model: mujoco.MjModel | None) -> mujoco.MjModel:
    if model is None:
        raise ValueError("MuJoCo model is not loaded")
    return model


def read_link_masses(model: mujoco.MjModel) -> NDArray[np.floating]:
    """Return per-body masses (world body excluded)."""
    model = _require_model(model)
    if model.nbody <= 1:
        return np.zeros(0, dtype=float)
    return np.asarray(model.body_mass[1:], dtype=float).copy()


def write_link_masses(model: mujoco.MjModel, masses: NDArray[np.floating]) -> None:
    model = _require_model(model)
    values = np.asarray(masses, dtype=float)
    expected = model.nbody - 1
    if values.shape != (expected,):
        raise ValueError(f"masses must have shape ({expected},), got {values.shape}")
    model.body_mass[1:] = values


def read_joint_damping(model: mujoco.MjModel) -> NDArray[np.floating]:
    model = _require_model(model)
    return np.asarray(model.dof_damping, dtype=float).copy()


def write_joint_damping(model: mujoco.MjModel, damping: NDArray[np.floating]) -> None:
    model = _require_model(model)
    values = np.asarray(damping, dtype=float)
    if values.shape != model.dof_damping.shape:
        raise ValueError(
            f"damping must match model.dof_damping shape {model.dof_damping.shape}"
        )
    model.dof_damping[:] = values


def read_friction_coefficients(model: mujoco.MjModel) -> NDArray[np.floating]:
    model = _require_model(model)
    return np.asarray(model.geom_friction[:, 0], dtype=float).copy()


def write_friction_coefficients(
    model: mujoco.MjModel, friction: NDArray[np.floating]
) -> None:
    model = _require_model(model)
    values = np.asarray(friction, dtype=float)
    if values.shape != (model.ngeom,):
        raise ValueError(
            f"friction must have shape ({model.ngeom},), got {values.shape}"
        )
    model.geom_friction[:, 0] = values


def read_motor_strength(model: mujoco.MjModel) -> NDArray[np.floating]:
    model = _require_model(model)
    return np.asarray(model.actuator_gear[:, 0], dtype=float).copy()


def write_motor_strength(model: mujoco.MjModel, strength: NDArray[np.floating]) -> None:
    model = _require_model(model)
    values = np.asarray(strength, dtype=float)
    if values.shape != (model.nu,):
        raise ValueError(f"strength must have shape ({model.nu},), got {values.shape}")
    model.actuator_gear[:, 0] = values


def read_actuator_gains(model: mujoco.MjModel) -> NDArray[np.floating]:
    model = _require_model(model)
    return np.asarray(model.actuator_gainprm[:, 0], dtype=float).copy()


def write_actuator_gains(model: mujoco.MjModel, gains: NDArray[np.floating]) -> None:
    model = _require_model(model)
    values = np.asarray(gains, dtype=float)
    if values.shape != (model.nu,):
        raise ValueError(f"gains must have shape ({model.nu},), got {values.shape}")
    model.actuator_gainprm[:, 0] = values


def read_gravity(model: mujoco.MjModel) -> NDArray[np.floating]:
    model = _require_model(model)
    return np.asarray(model.opt.gravity, dtype=float).copy()


def write_gravity(model: mujoco.MjModel, gravity: NDArray[np.floating]) -> None:
    model = _require_model(model)
    values = np.asarray(gravity, dtype=float)
    if values.shape != (3,):
        raise ValueError(f"gravity must have shape (3,), got {values.shape}")
    model.opt.gravity[:] = values
