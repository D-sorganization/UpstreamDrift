"""JaxSim adapter for engine-core dynamics protocols."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.engine_core.capabilities import (
    CapabilityLevel,
    EngineCapabilities,
)
from src.shared.python.engine_core.velocity_conventions import (
    CANONICAL_FLOATING_BASE_CONVENTION,
    CANONICAL_VELOCITY_REPRESENTATION,
    FloatingBaseConvention,
    VelocityRepresentation,
)


@dataclass(frozen=True)
class _JaxSimApis:
    model: Any
    data: Any
    frame: Any
    common: Any


class JaxSimBackend:
    """JaxSim-backed implementation of load/query/dynamics protocols.

    The adapter keeps all JaxSim imports lazy. Core installs can import this
    module without the optional ``upstream-drift[jaxsim]`` extra, while loader
    code raises an actionable error when a live JaxSim backend is requested
    without the dependency installed.
    """

    def __init__(
        self,
        *,
        apis: _JaxSimApis | None = None,
        convention: FloatingBaseConvention = CANONICAL_FLOATING_BASE_CONVENTION,
    ) -> None:
        if convention.velocity_representation != CANONICAL_VELOCITY_REPRESENTATION:
            raise ValueError(
                "JaxSimBackend normalizes to the suite canonical convention"
            )
        self._apis = apis
        self._convention = convention
        self._model: Any | None = None
        self._data: Any | None = None
        self._model_name = ""
        self._control: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        self._time = 0.0

    @property
    def model_name(self) -> str:
        """Return the loaded model name, or an empty string before load."""

        return self._model_name

    @property
    def convention(self) -> FloatingBaseConvention:
        """Return the suite-normalized floating-base convention."""

        return self._convention

    def load_from_path(self, path: str) -> None:
        """Load a URDF/SDF/MJCF model from disk through JaxSim."""

        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(path)
        self._load_model_description(model_path.read_text(encoding="utf-8"), model_path)

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        """Load model description text through JaxSim."""

        if not content.strip():
            raise ValueError("content must be non-empty")
        suffix = f".{extension.lstrip('.')}" if extension else ".sdf"
        self._load_model_description(content, Path(f"inline_model{suffix}"))

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return generalized position and canonical generalized velocity."""

        self._require_loaded()
        base_position = _vector_attr(self._data, "base_position", 3)
        base_quaternion = _vector_attr(self._data, "base_quaternion", 4)
        joint_positions = _array_attr(self._data, "joint_positions")
        q = np.concatenate([base_position, base_quaternion, joint_positions])

        base_angular = _vector_attr(self._data, "base_angular_velocity", 3)
        base_linear = _vector_attr(self._data, "base_linear_velocity", 3)
        joint_velocities = _array_attr(self._data, "joint_velocities")
        v = np.concatenate([base_angular, base_linear, joint_velocities])
        return q, v

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Set generalized position and canonical generalized velocity."""

        self._require_loaded()
        q = np.asarray(q, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        expected_q = 7 + self._dofs()
        expected_v = 6 + self._dofs()
        if q.shape != (expected_q,):
            raise ValueError(f"q must have shape ({expected_q},)")
        if v.shape != (expected_v,):
            raise ValueError(f"v must have shape ({expected_v},)")
        self._data = self._build_data(
            base_position=q[:3],
            base_quaternion=q[3:7],
            joint_positions=q[7:],
            base_angular_velocity=v[:3],
            base_linear_velocity=v[3:6],
            joint_velocities=v[6:],
        )

    def set_control(self, u: np.ndarray) -> None:
        """Store joint generalized forces for the next forward/step call."""

        self._require_loaded()
        control: NDArray[np.float64] = np.asarray(u, dtype=np.float64)
        if control.shape != (self._dofs(),):
            raise ValueError(f"u must have shape ({self._dofs()},)")
        self._control = control.copy()

    def get_time(self) -> float:
        """Return simulation time in seconds."""

        return self._time

    def reset(self) -> None:
        """Reset loaded model data to the zero state."""

        model, apis = self._model_and_apis()
        self._data = apis.data.JaxSimModelData.zero(
            model,
            velocity_representation=self._jaxsim_vel_repr(),
        )
        self._control = np.zeros(self._dofs(), dtype=np.float64)
        self._time = 0.0

    def forward(self) -> None:
        """Refresh JaxSim forward-kinematics caches without advancing time."""

        model, data, apis = self._loaded()
        apis.model.forward_kinematics(model, data)

    def step(self, dt: float | None = None) -> None:
        """Advance the JaxSim model by one step."""

        model, data, apis = self._loaded()
        if dt is not None and dt <= 0.0:
            raise ValueError("dt must be positive")
        kwargs: dict[str, Any] = {"joint_force_references": self._control}
        if dt is not None:
            kwargs["dt"] = dt
        self._data = apis.model.step(model, data, **kwargs)
        self._time += float(dt if dt is not None else getattr(model, "time_step", 0.0))

    def compute_mass_matrix(self) -> np.ndarray:
        """Compute the canonical free-floating inertia matrix M(q)."""

        matrix = self._call_model_array("free_floating_mass_matrix")
        _assert_spd(matrix, "mass matrix")
        return matrix

    def compute_bias_forces(self) -> np.ndarray:
        """Compute canonical free-floating bias forces h(q, v)."""

        return self._call_model_vector("free_floating_bias_forces")

    def compute_gravity_forces(self) -> np.ndarray:
        """Compute canonical free-floating gravity forces g(q)."""

        return self._call_model_vector("free_floating_gravity_forces")

    def compute_coriolis_matrix(self) -> np.ndarray:
        """Compute the canonical free-floating Coriolis matrix."""

        return self._call_model_array("free_floating_coriolis_matrix")

    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        """Compute generalized forces for the requested acceleration."""

        _model, _data, apis = self._loaded()
        qacc = np.asarray(qacc, dtype=np.float64)
        expected_v = 6 + self._dofs()
        if qacc.shape != (expected_v,):
            raise ValueError(f"qacc must have shape ({expected_v},)")
        base_force, joint_forces = self._with_canonical_data(
            apis.model.inverse_dynamics,
            joint_accelerations=qacc[6:],
            base_acceleration=qacc[:6],
        )
        return np.concatenate(
            [
                np.asarray(base_force, dtype=np.float64).reshape(6),
                np.asarray(joint_forces, dtype=np.float64).reshape(self._dofs()),
            ]
        )

    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        """Compute a canonical spatial Jacobian for a link or frame."""

        _model, _data, apis = self._loaded()
        if not body_name.strip():
            raise ValueError("body_name must be non-empty")
        frame_index = self._frame_index(body_name)
        if frame_index is None:
            return None
        jacobians = self._with_canonical_data(
            apis.model.generalized_free_floating_jacobian,
            output_vel_repr=self._jaxsim_vel_repr(),
        )
        spatial = np.asarray(jacobians[frame_index], dtype=np.float64)
        return {
            "spatial": spatial,
            "angular": spatial[:3, :],
            "linear": spatial[3:, :],
        }

    def compute_drift_acceleration(self) -> np.ndarray:
        """Compute passive acceleration from bias forces with zero control."""

        mass = self.compute_mass_matrix()
        bias = self.compute_bias_forces()
        return np.linalg.solve(mass, -bias)

    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        """Compute acceleration attributable to applied generalized forces."""

        tau = np.asarray(tau, dtype=np.float64)
        expected_v = 6 + self._dofs()
        if tau.shape != (expected_v,):
            raise ValueError(f"tau must have shape ({expected_v},)")
        return np.linalg.solve(self.compute_mass_matrix(), tau)

    def get_capabilities(self) -> EngineCapabilities:
        """Declare JaxSim support through the engine capability taxonomy."""

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
            trajectory_opt=CapabilityLevel.PARTIAL,
            dataset_export=CapabilityLevel.PARTIAL,
            model_positioning=CapabilityLevel.PARTIAL,
        )

    def _load_model_description(self, content: str, source_path: Path) -> None:
        self._apis = self._apis or _load_jaxsim_apis()
        self._model = self._apis.model.JaxSimModel.build_from_model_description(
            model_description=content,
            model_name=source_path.stem,
        )
        self._model_name = str(getattr(self._model, "model_name", source_path.stem))
        self.reset()

    def _require_loaded(self) -> None:
        if self._model is None or self._data is None or self._apis is None:
            raise RuntimeError("JaxSimBackend has no loaded model")

    def _loaded(self) -> tuple[Any, Any, _JaxSimApis]:
        self._require_loaded()
        assert self._model is not None
        assert self._data is not None
        assert self._apis is not None
        return self._model, self._data, self._apis

    def _model_and_apis(self) -> tuple[Any, _JaxSimApis]:
        if self._model is None or self._apis is None:
            raise RuntimeError("JaxSimBackend has no loaded model")
        return self._model, self._apis

    def _dofs(self) -> int:
        model, data, _apis = self._loaded()
        if hasattr(model, "dofs"):
            return int(model.dofs)
        return int(_array_attr(data, "joint_positions").shape[0])

    def _build_data(self, **kwargs: Any) -> Any:
        model, _data, apis = self._loaded()
        return apis.data.JaxSimModelData.build(
            model,
            velocity_representation=self._jaxsim_vel_repr(),
            **kwargs,
        )

    def _jaxsim_vel_repr(self) -> Any:
        if self._apis is None:
            raise RuntimeError("JaxSimBackend has no loaded model")
        if self._convention.velocity_representation == VelocityRepresentation.INERTIAL:
            return self._apis.common.VelRepr.Inertial
        if (
            self._convention.velocity_representation
            == VelocityRepresentation.BODY_FIXED
        ):
            return self._apis.common.VelRepr.Body
        return self._apis.common.VelRepr.Mixed

    def _with_canonical_data(self, func: Any, **kwargs: Any) -> Any:
        model, data, _apis = self._loaded()
        target_repr = self._jaxsim_vel_repr()
        if getattr(data, "velocity_representation", target_repr) == target_repr:
            return func(model, data, **kwargs)
        with data.switch_velocity_representation(target_repr) as switched_data:
            return func(model, switched_data, **kwargs)

    def _call_model_array(self, name: str) -> np.ndarray:
        _model, _data, apis = self._loaded()
        value = self._with_canonical_data(getattr(apis.model, name))
        return np.asarray(value, dtype=np.float64)

    def _call_model_vector(self, name: str) -> np.ndarray:
        vector = self._call_model_array(name).reshape(-1)
        expected_v = 6 + self._dofs()
        if vector.shape != (expected_v,):
            raise ValueError(
                f"{name} returned shape {vector.shape}, expected ({expected_v},)"
            )
        return vector

    def _frame_index(self, body_name: str) -> int | None:
        self._require_loaded()
        for api_name in ("frame", "link"):
            api = getattr(self._apis, api_name, None)
            if api is None or not hasattr(api, "name_to_idx"):
                continue
            try:
                return int(api.name_to_idx(self._model, frame_name=body_name))
            except TypeError:
                try:
                    return int(api.name_to_idx(self._model, link_name=body_name))
                except (KeyError, ValueError, RuntimeError, TypeError):
                    continue
            except (KeyError, ValueError, RuntimeError):
                continue
        return None


def _load_jaxsim_apis() -> _JaxSimApis:
    try:
        js = importlib.import_module("jaxsim.api")
    except ImportError as exc:
        raise ImportError(
            "JaxSim is not installed. Install with `pip install upstream-drift[jaxsim]`."
        ) from exc
    return _JaxSimApis(
        model=js.model,
        data=js.data,
        frame=js.frame,
        common=js.common,
    )


def _array_attr(obj: Any, name: str) -> np.ndarray:
    return np.asarray(getattr(obj, name), dtype=np.float64).reshape(-1)


def _vector_attr(obj: Any, name: str, size: int) -> np.ndarray:
    vector = _array_attr(obj, name)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return vector


def _assert_spd(matrix: np.ndarray, name: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.allclose(matrix, matrix.T, atol=1e-9):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if np.any(eigenvalues <= 0.0):
        raise ValueError(f"{name} must be positive definite")


def make_mock_jaxsim_apis(model_api: Any, data_api: Any, frame_api: Any) -> _JaxSimApis:
    """Create injectable JaxSim API seams for unit tests."""

    return _JaxSimApis(
        model=model_api,
        data=data_api,
        frame=frame_api,
        common=SimpleNamespace(
            VelRepr=SimpleNamespace(Inertial="inertial", Body="body", Mixed="mixed")
        ),
    )
