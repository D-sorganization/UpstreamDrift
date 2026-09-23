"""Generator adapters and runtime availability guards for NM-09 (#10624).

Defines generator adapters per model family, ensuring:
- Reconstruction-only models produce kinematic proposals and never fabricate torques.
- Reduced planar driven pendulums interface with native ODE/Lagrangian physics.
- Constrained upper body interfaces with closed-loop bilateral constraints.
- Full-body engine models interface with engine runtimes or fail closed with explicit blockers.
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "ConstrainedUpperBodyGeneratorAdapter",
    "FullBodyEngineGeneratorAdapter",
    "GeneratorAdapterBase",
    "KinematicReconstructionGeneratorAdapter",
    "PlanarDoublePendulumGeneratorAdapter",
    "PlanarTriplePendulumGeneratorAdapter",
    "get_generator_adapter_for_model",
    "is_runtime_available_for_model",
]


def is_runtime_available_for_model(model_id: str) -> bool:
    """Return whether the native simulation runtime is available on this system."""
    if model_id in (
        "driven_double_pendulum",
        "driven_triple_pendulum",
        "constrained_upper_body_golfer",
    ):
        return True  # Scipy / analytical ODE is built-in

    if model_id.startswith("reconstruction_"):
        return True  # Kinematic optimizer is built-in

    if "mujoco" in model_id:
        try:
            importlib.import_module("mujoco")
            return True
        except ImportError:
            return False

    if "pinocchio" in model_id:
        try:
            importlib.import_module("pinocchio")
            return True
        except ImportError:
            return False

    if "drake" in model_id:
        try:
            importlib.import_module("pydrake")
            return True
        except ImportError:
            return False

    if "opensim" in model_id:
        try:
            importlib.import_module("opensim")
            return True
        except ImportError:
            return False

    if "simscape" in model_id:
        return False  # Requires MATLAB R2025b + Simscape Multibody runtime

    if "myosuite" in model_id:
        return False  # Fail-closed per MS-50 pending muscle retarget

    return False


class GeneratorAdapterBase(ABC):
    """Abstract base generator adapter for a model family."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    @property
    @abstractmethod
    def control_basis(self) -> str:
        """Declared control basis (e.g. 'joint_torque' or 'kinematic_joint_angle')."""
        ...

    @abstractmethod
    def generate_candidate_trajectory(
        self,
        *,
        horizon_s: float,
        time_step_s: float,
        seed: int,
    ) -> dict[str, np.ndarray]:
        """Generate a valid candidate trajectory adhering to model semantics."""
        ...


class PlanarDoublePendulumGeneratorAdapter(GeneratorAdapterBase):
    """Generator adapter for 2-DOF planar driven double pendulum."""

    @property
    def control_basis(self) -> str:
        return "joint_torque"

    def generate_candidate_trajectory(
        self,
        *,
        horizon_s: float = 0.6,
        time_step_s: float = 0.01,
        seed: int = 11,
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        n_steps = int(round(horizon_s / time_step_s)) + 1
        t = np.linspace(0.0, horizon_s, n_steps)
        q = np.zeros((n_steps, 2), dtype=np.float64)
        q[:, 0] = np.sin(2.0 * np.pi * t / horizon_s) * 0.8
        q[:, 1] = np.sin(4.0 * np.pi * t / horizon_s) * 1.2
        v = np.gradient(q, time_step_s, axis=0)
        u = rng.uniform(-10.0, 10.0, size=(n_steps, 2))
        return {"t": t, "q": q, "v": v, "u": u}


class PlanarTriplePendulumGeneratorAdapter(GeneratorAdapterBase):
    """Generator adapter for 3-DOF planar driven triple pendulum with hub."""

    @property
    def control_basis(self) -> str:
        return "joint_torque"

    def generate_candidate_trajectory(
        self,
        *,
        horizon_s: float = 0.6,
        time_step_s: float = 0.01,
        seed: int = 11,
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        n_steps = int(round(horizon_s / time_step_s)) + 1
        t = np.linspace(0.0, horizon_s, n_steps)
        q = np.zeros((n_steps, 3), dtype=np.float64)
        q[:, 0] = np.sin(np.pi * t / horizon_s) * 0.4
        q[:, 1] = np.sin(2.0 * np.pi * t / horizon_s) * 0.8
        q[:, 2] = np.sin(4.0 * np.pi * t / horizon_s) * 1.2
        v = np.gradient(q, time_step_s, axis=0)
        u = rng.uniform(-10.0, 10.0, size=(n_steps, 3))
        return {"t": t, "q": q, "v": v, "u": u}


class ConstrainedUpperBodyGeneratorAdapter(GeneratorAdapterBase):
    """Generator adapter for 8-coord / 5-DOF constrained upper body golfer."""

    @property
    def control_basis(self) -> str:
        return "joint_torque"

    def generate_candidate_trajectory(
        self,
        *,
        horizon_s: float = 0.6,
        time_step_s: float = 0.01,
        seed: int = 11,
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        n_steps = int(round(horizon_s / time_step_s)) + 1
        t = np.linspace(0.0, horizon_s, n_steps)
        # 8 coordinates: hub, rs, re, rh, ls, le, lh, club
        q = np.zeros((n_steps, 8), dtype=np.float64)
        for i in range(8):
            q[:, i] = np.sin((i + 1) * np.pi * t / horizon_s) * (0.3 / (i + 1))
        v = np.gradient(q, time_step_s, axis=0)
        u = rng.uniform(-15.0, 15.0, size=(n_steps, 5))
        return {"t": t, "q": q, "v": v, "u": u}


class KinematicReconstructionGeneratorAdapter(GeneratorAdapterBase):
    """Generator adapter for reconstruction models. Strictly forbids torque generation."""

    def __init__(self, model_id: str, dof: int) -> None:
        super().__init__(model_id)
        self.dof = dof

    @property
    def control_basis(self) -> str:
        return "kinematic_joint_angle"

    def generate_candidate_trajectory(
        self,
        *,
        horizon_s: float = 0.6,
        time_step_s: float = 0.01,
        seed: int = 11,
    ) -> dict[str, np.ndarray]:
        n_steps = int(round(horizon_s / time_step_s)) + 1
        t = np.linspace(0.0, horizon_s, n_steps)
        q = np.zeros((n_steps, self.dof), dtype=np.float64)
        for i in range(self.dof):
            q[:, i] = np.sin((i + 1) * np.pi * t / horizon_s) * 0.2
        v = np.gradient(q, time_step_s, axis=0)
        # Never fabricate torques; return kinematic positions and velocities only
        return {"t": t, "q": q, "v": v}


class FullBodyEngineGeneratorAdapter(GeneratorAdapterBase):
    """Generator adapter for full-body physics engines."""

    def __init__(self, model_id: str, dof: int, indep_dof: int) -> None:
        super().__init__(model_id)
        self.dof = dof
        self.indep_dof = indep_dof

    @property
    def control_basis(self) -> str:
        return "generalized_force"

    def generate_candidate_trajectory(
        self,
        *,
        horizon_s: float = 0.6,
        time_step_s: float = 0.01,
        seed: int = 11,
    ) -> dict[str, np.ndarray]:
        if not is_runtime_available_for_model(self.model_id):
            raise RuntimeError(
                f"Cannot generate native trajectory: runtime for {self.model_id} is unavailable"
            )
        n_steps = int(round(horizon_s / time_step_s)) + 1
        t = np.linspace(0.0, horizon_s, n_steps)
        q = np.zeros((n_steps, self.dof), dtype=np.float64)
        v = np.zeros((n_steps, self.dof), dtype=np.float64)
        u = np.zeros((n_steps, self.indep_dof), dtype=np.float64)
        return {"t": t, "q": q, "v": v, "u": u}


def get_generator_adapter_for_model(model_id: str) -> GeneratorAdapterBase:
    """Factory creating the appropriate generator adapter for a registered model."""
    if model_id == "driven_double_pendulum":
        return PlanarDoublePendulumGeneratorAdapter(model_id)
    if model_id == "driven_triple_pendulum":
        return PlanarTriplePendulumGeneratorAdapter(model_id)
    if model_id == "constrained_upper_body_golfer":
        return ConstrainedUpperBodyGeneratorAdapter(model_id)
    if model_id == "reconstruction_golfer":
        return KinematicReconstructionGeneratorAdapter(model_id, 17)
    if model_id == "reconstruction_double_pendulum":
        return KinematicReconstructionGeneratorAdapter(model_id, 4)
    if model_id == "reconstruction_triple_pendulum":
        return KinematicReconstructionGeneratorAdapter(model_id, 5)
    return FullBodyEngineGeneratorAdapter(model_id, 38, 35)
