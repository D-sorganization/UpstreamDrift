"""JaxSim backend adapter for kinematics and dynamics terms."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from src.shared.python.core.contracts import StateError
from src.shared.python.engine_core.base_physics_engine import BasePhysicsEngine
from src.shared.python.engine_core.capabilities import (
    CapabilityLevel,
    EngineCapabilities,
)
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

BASE_POSITION_SIZE = 3
BASE_QUATERNION_SIZE = 4
BASE_VELOCITY_SIZE = 6
DEFAULT_TIME_STEP = 0.001


class JaxSimBackend(BasePhysicsEngine):
    """Optional JaxSim adapter implementing core query and dynamics protocols.

    The adapter exposes suite-canonical state vectors:
    ``q = [base_position(3), base_quaternion(wxyz), joint_positions...]`` and
    ``v = [base_angular_velocity(3), base_linear_velocity(3), joint_velocities...]``.
    JaxSim calls are executed in inertial velocity representation so the output
    aligns with the engine-core convention introduced for #6652.
    """

    def __init__(
        self,
        *,
        api_module: Any | None = None,
        time_step: float = DEFAULT_TIME_STEP,
    ) -> None:
        super().__init__()
        if time_step <= 0.0:
            raise ValueError("time_step must be positive")
        self.model: Any | None = None
        self.data: Any | None = None
        self._api_module = api_module
        self.time_step = float(time_step)
        self.time = 0.0
        self.tau = np.array([], dtype=np.float64)

    @staticmethod
    def is_available() -> bool:
        """Return whether the optional ``jaxsim.api`` package imports."""
        try:
            importlib.import_module("jaxsim.api")
        except ImportError:
            return False
        return True

    @property
    def engine_type(self) -> str:
        """Return the engine registry identifier."""
        return "jaxsim"

    @property
    def model_name(self) -> str:
        """Return the loaded JaxSim model name."""
        if self.model is not None and hasattr(self.model, "name"):
            return str(self.model.name)
        return self.model_name_str

    @property
    def is_initialized(self) -> bool:
        """Check if a JaxSim model/data pair has been loaded."""
        return self.model is not None and self.data is not None

    def get_capabilities(self) -> EngineCapabilities:
        """Report the JaxSim capability profile from the optional adapter."""
        return EngineCapabilities(
            engine_name="JaxSim",
            mass_matrix=CapabilityLevel.FULL,
            jacobian=CapabilityLevel.FULL,
            contact_forces=CapabilityLevel.PARTIAL,
            inverse_dynamics=CapabilityLevel.FULL,
            drift_acceleration=CapabilityLevel.PARTIAL,
            parameter_gradients=CapabilityLevel.FULL,
            state_control_gradients=CapabilityLevel.FULL,
            forward_sim=CapabilityLevel.FULL,
            contact_step=CapabilityLevel.PARTIAL,
            trajectory_opt=CapabilityLevel.NONE,
            extra={
                "spatial_jacobian_order": "angular_linear",
                "velocity_representation": "inertial",
            },
        )

    def _load_from_path_impl(self, path: str) -> None:
        js = self._require_api()
        model_path = Path(path)
        extension = model_path.suffix.lower()
        if extension not in {".sdf", ".urdf"}:
            raise ValueError("JaxSimBackend supports .sdf and .urdf model files")
        self.model = js.model.JaxSimModel.build_from_model_description(
            model_path,
            is_urdf=extension == ".urdf",
            time_step=self.time_step,
        )
        self._initialize_data(js)

    def _load_from_string_impl(self, content: str, extension: str | None) -> None:
        js = self._require_api()
        normalized_extension = (extension or "sdf").lower().lstrip(".")
        if normalized_extension not in {"sdf", "urdf"}:
            raise ValueError("JaxSimBackend supports sdf or urdf content")
        self.model = js.model.JaxSimModel.build_from_model_description(
            content,
            is_urdf=normalized_extension == "urdf",
            time_step=self.time_step,
        )
        self._initialize_data(js)

    def reset(self) -> None:
        """Reset model data and generalized forces."""
        self._require_initialized("reset")
        self._initialize_data(self._require_api())

    def step(self, dt: float | None = None) -> None:
        """Advance JaxSim one step using the current joint force references."""
        self._require_initialized("step")
        if dt is not None and dt <= 0.0:
            raise ValueError("dt must be positive")
        js = self._require_api()
        self.data = js.model.step(
            self.model,
            self.data,
            dt=dt,
            joint_force_references=self._joint_torques(),
        )
        self.time += self.time_step if dt is None else float(dt)

    def forward(self) -> None:
        """Compute kinematics by forcing JaxSim to materialize transforms."""
        self._require_initialized("forward")
        js = self._require_api()
        js.model.forward_kinematics(self.model, self.data)

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return suite-canonical generalized position and velocity vectors."""
        self._require_initialized("get_state")
        data = self._require_data()
        q = np.concatenate(
            [
                self._array(data.base_position),
                self._array(data.base_quaternion),
                self._array(data.joint_positions),
            ]
        )
        v = np.concatenate(
            [
                self._array(data.base_angular_velocity),
                self._array(data.base_linear_velocity),
                self._array(data.joint_velocities),
            ]
        )
        return q, v

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Set state from suite-canonical ``q`` and ``v`` vectors."""
        self._require_initialized("set_state")
        q_arr = self._as_vector(q, "q")
        v_arr = self._as_vector(v, "v")
        dofs = self._dofs()
        if q_arr.shape != (BASE_POSITION_SIZE + BASE_QUATERNION_SIZE + dofs,):
            raise ValueError(
                "q must have shape "
                f"({BASE_POSITION_SIZE + BASE_QUATERNION_SIZE + dofs},)"
            )
        if v_arr.shape != (BASE_VELOCITY_SIZE + dofs,):
            raise ValueError(f"v must have shape ({BASE_VELOCITY_SIZE + dofs},)")
        self.data = self._require_data().replace(
            self.model,
            base_position=q_arr[:3],
            base_quaternion=q_arr[3:7],
            joint_positions=q_arr[7:],
            base_angular_velocity=v_arr[:3],
            base_linear_velocity=v_arr[3:6],
            joint_velocities=v_arr[6:],
            validate=True,
        )

    def set_control(self, u: np.ndarray) -> None:
        """Store generalized force references for the next step."""
        self._require_initialized("set_control")
        u_arr = self._as_vector(u, "u")
        if u_arr.shape == (BASE_VELOCITY_SIZE + self._dofs(),):
            u_arr = u_arr[BASE_VELOCITY_SIZE:]
        if u_arr.shape != (self._dofs(),):
            raise ValueError(f"u must have shape ({self._dofs()},)")
        self.tau = u_arr.copy()

    def get_time(self) -> float:
        """Return simulation time in seconds."""
        return self.time

    def compute_mass_matrix(self) -> np.ndarray:
        """Compute the free-floating mass matrix."""
        self._require_initialized("compute_mass_matrix")
        matrix = self._array(
            self._require_api().model.free_floating_mass_matrix(self.model, self.data)
        )
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("JaxSim mass matrix must be square")
        return matrix

    def compute_bias_forces(self) -> np.ndarray:
        """Compute free-floating bias forces."""
        self._require_initialized("compute_bias_forces")
        return self._array(
            self._require_api().model.free_floating_bias_forces(self.model, self.data)
        )

    def compute_gravity_forces(self) -> np.ndarray:
        """Compute free-floating gravity forces."""
        self._require_initialized("compute_gravity_forces")
        return self._array(
            self._require_api().model.free_floating_gravity_forces(
                self.model, self.data
            )
        )

    def compute_coriolis_matrix(self) -> np.ndarray:
        """Compute the free-floating Coriolis matrix."""
        self._require_initialized("compute_coriolis_matrix")
        return self._array(
            self._require_api().model.free_floating_coriolis_matrix(
                self.model,
                self.data,
            )
        )

    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        """Compute generalized forces for canonical accelerations."""
        self._require_initialized("compute_inverse_dynamics")
        qacc_arr = self._as_generalized_velocity(qacc, "qacc")
        base_force, joint_forces = self._require_api().model.inverse_dynamics(
            self.model,
            self.data,
            base_acceleration=qacc_arr[:BASE_VELOCITY_SIZE],
            joint_accelerations=qacc_arr[BASE_VELOCITY_SIZE:],
        )
        return np.concatenate([self._array(base_force), self._array(joint_forces)])

    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        """Compute a link spatial Jacobian in angular-linear row order."""
        self._require_initialized("compute_jacobian")
        if not body_name or not body_name.strip():
            raise ValueError("body_name must be non-empty")
        js = self._require_api()
        try:
            link_index = js.link.name_to_idx(self.model, link_name=body_name)
        except (KeyError, ValueError):
            return None
        jacobians = self._array(
            js.model.generalized_free_floating_jacobian(
                self.model,
                self.data,
                output_vel_repr=self._inertial_velocity_representation(js),
            )
        )
        spatial = jacobians[int(link_index)]
        return {
            "angular": spatial[:3, :],
            "linear": spatial[3:, :],
            "spatial": spatial,
        }

    def compute_frame_transforms(self) -> np.ndarray:
        """Return stacked homogeneous transforms for JaxSim links."""
        self._require_initialized("compute_frame_transforms")
        return self._array(
            self._require_api().model.forward_kinematics(self.model, self.data)
        )

    def compute_drift_acceleration(self) -> np.ndarray:
        """Compute passive acceleration with zero joint torques."""
        self._require_initialized("compute_drift_acceleration")
        return self._system_acceleration(np.zeros(self._dofs(), dtype=np.float64))

    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        """Compute acceleration induced by the supplied joint torques."""
        self._require_initialized("compute_control_acceleration")
        torque = self._as_vector(tau, "tau")
        if torque.shape == (BASE_VELOCITY_SIZE + self._dofs(),):
            torque = torque[BASE_VELOCITY_SIZE:]
        if torque.shape != (self._dofs(),):
            raise ValueError(f"tau must have shape ({self._dofs()},)")
        return self._system_acceleration(torque)

    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute the zero-torque counterfactual acceleration."""
        self.set_state(q, v)
        return self.compute_drift_acceleration()

    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        """Compute the zero-velocity counterfactual acceleration."""
        self._require_initialized("compute_zvcf")
        q_arr = self._as_vector(q, "q")
        self.set_state(q_arr, np.zeros(BASE_VELOCITY_SIZE + self._dofs()))
        return self.compute_drift_acceleration()

    def _initialize_data(self, js: ModuleType | Any) -> None:
        self.data = js.data.JaxSimModelData.build(
            self.model,
            velocity_representation=self._inertial_velocity_representation(js),
        )
        self.tau = np.zeros(self._dofs(), dtype=np.float64)
        self.time = 0.0

    def _system_acceleration(self, joint_torques: np.ndarray) -> np.ndarray:
        js = self._require_api()
        if not hasattr(js, "ode") or not hasattr(js.ode, "system_acceleration"):
            raise NotImplementedError(
                "JaxSim ode.system_acceleration is required for acceleration terms "
                "(tracked: #6653)"
            )
        base_acceleration, joint_acceleration, _ = js.ode.system_acceleration(
            self.model,
            self.data,
            joint_torques=joint_torques,
        )
        return np.concatenate(
            [self._array(base_acceleration), self._array(joint_acceleration)]
        )

    def _require_api(self) -> Any:
        if self._api_module is not None:
            return self._api_module
        try:
            self._api_module = importlib.import_module("jaxsim.api")
        except ImportError as exc:
            raise ImportError(
                "JaxSim optional dependency is not installed. "
                'Install with: python -m pip install "upstream-drift[jaxsim]"'
            ) from exc
        return self._api_module

    def _require_initialized(self, operation: str) -> None:
        if not self.is_initialized:
            raise StateError(
                f"Cannot perform '{operation}' - JaxSimBackend is not initialized.",
                current_state="uninitialized",
                required_state="initialized",
                operation=operation,
            )

    def _require_data(self) -> Any:
        if self.data is None:
            raise StateError(
                "Cannot access JaxSim data - backend is not initialized.",
                current_state="uninitialized",
                required_state="initialized",
                operation="data access",
            )
        return self.data

    def _joint_torques(self) -> np.ndarray:
        if self.tau.shape == (self._dofs(),):
            return self.tau
        return np.zeros(self._dofs(), dtype=np.float64)

    def _dofs(self) -> int:
        if self.model is None:
            return 0
        return int(self.model.dofs())

    def _as_generalized_velocity(self, value: np.ndarray, name: str) -> np.ndarray:
        array = self._as_vector(value, name)
        expected = BASE_VELOCITY_SIZE + self._dofs()
        if array.shape != (expected,):
            raise ValueError(f"{name} must have shape ({expected},)")
        return array

    @staticmethod
    def _as_vector(value: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional vector")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array

    @staticmethod
    def _array(value: Any) -> np.ndarray:
        return np.asarray(value, dtype=np.float64)

    @staticmethod
    def _inertial_velocity_representation(js: Any) -> Any:
        vel_repr = getattr(getattr(js, "common", None), "VelRepr", None)
        return getattr(vel_repr, "Inertial", "inertial")
