"""Unit contracts for the optional JaxSim backend adapter."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.engines.physics_engines.jaxsim.python.jaxsim_backend import JaxSimBackend
from src.engines.loaders import LOADER_MAP
from src.shared.python.engine_core.capabilities import CapabilityLevel
from src.shared.python.engine_core.engine_availability import get_engine_status
from src.shared.python.engine_core.engine_probes import JaxSimProbe, ProbeStatus
from src.shared.python.engine_core.engine_registry import EngineType


class _FakeJaxSimData:
    def __init__(
        self,
        *,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        base_position: np.ndarray,
        base_quaternion: np.ndarray,
        base_linear_velocity: np.ndarray,
        base_angular_velocity: np.ndarray,
        velocity_representation: str = "inertial",
    ) -> None:
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities
        self.base_position = base_position
        self.base_quaternion = base_quaternion
        self.base_linear_velocity = base_linear_velocity
        self.base_angular_velocity = base_angular_velocity
        self.velocity_representation = velocity_representation

    @property
    def base_transform(self) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, 3] = self.base_position
        return transform

    def replace(self, model, **kwargs):  # noqa: ANN001
        values = {
            "joint_positions": self.joint_positions,
            "joint_velocities": self.joint_velocities,
            "base_position": self.base_position,
            "base_quaternion": self.base_quaternion,
            "base_linear_velocity": self.base_linear_velocity,
            "base_angular_velocity": self.base_angular_velocity,
        }
        values.update(
            {k: v for k, v in kwargs.items() if v is not None and k != "validate"}
        )
        return _FakeJaxSimData(**values, velocity_representation="inertial")


class _FakeJaxSimModel:
    def __init__(self, dofs: int = 2) -> None:
        self._dofs = dofs
        self.name = "fake_jaxsim_model"
        self.loaded_from: object | None = None
        self.is_urdf = False

    def dofs(self) -> int:
        return self._dofs

    def number_of_links(self) -> int:
        return 2


class _FakeDataFactory:
    @staticmethod
    def build(model, **kwargs):  # noqa: ANN001
        dofs = model.dofs()
        return _FakeJaxSimData(
            joint_positions=np.zeros(dofs),
            joint_velocities=np.zeros(dofs),
            base_position=np.zeros(3),
            base_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            base_linear_velocity=np.zeros(3),
            base_angular_velocity=np.zeros(3),
            velocity_representation=kwargs.get("velocity_representation", "inertial"),
        )


class _FakeModelApi:
    def __init__(self, model: _FakeJaxSimModel) -> None:
        self._model = model
        self.step_joint_forces: np.ndarray | None = None
        self.JaxSimModel = self

    def build_from_model_description(self, model_description, **kwargs):  # noqa: ANN001
        self._model.loaded_from = model_description
        self._model.is_urdf = kwargs["is_urdf"]
        return self._model

    def free_floating_mass_matrix(self, model, data):  # noqa: ANN001
        return np.diag(np.arange(1.0, 9.0))

    def free_floating_bias_forces(self, model, data):  # noqa: ANN001
        return np.arange(8.0)

    def free_floating_gravity_forces(self, model, data):  # noqa: ANN001
        return np.ones(8)

    def free_floating_coriolis_matrix(self, model, data):  # noqa: ANN001
        return np.eye(8) * 2.0

    def inverse_dynamics(self, model, data, **kwargs):  # noqa: ANN001
        return kwargs["base_acceleration"], kwargs["joint_accelerations"]

    def generalized_free_floating_jacobian(self, model, data, **kwargs):  # noqa: ANN001
        return np.arange(2 * 6 * 8, dtype=float).reshape(2, 6, 8)

    def forward_kinematics(self, model, data):  # noqa: ANN001
        return np.repeat(np.eye(4)[None, :, :], 2, axis=0)

    def step(self, model, data, **kwargs):  # noqa: ANN001
        self.step_joint_forces = kwargs.get("joint_force_references")
        return data


class _FakeLinkApi:
    @staticmethod
    def name_to_idx(model, *, link_name: str):  # noqa: ANN001
        if link_name == "base":
            return 0
        if link_name == "club":
            return 1
        raise ValueError(link_name)


def _fake_js() -> SimpleNamespace:
    model = _FakeJaxSimModel()
    return SimpleNamespace(
        common=SimpleNamespace(VelRepr=SimpleNamespace(Inertial="inertial")),
        data=SimpleNamespace(JaxSimModelData=_FakeDataFactory),
        link=_FakeLinkApi(),
        model=_FakeModelApi(model),
        ode=SimpleNamespace(
            system_acceleration=lambda model, data, joint_torques=None: (
                np.arange(6.0),
                np.arange(model.dofs(), dtype=float),
                {},
            )
        ),
    )


def test_jaxsim_backend_declares_capability_taxonomy() -> None:
    backend = JaxSimBackend(api_module=_fake_js())

    caps = backend.get_capabilities()

    assert caps.engine_name == "JaxSim"
    assert caps.mass_matrix == CapabilityLevel.FULL
    assert caps.jacobian == CapabilityLevel.FULL
    assert caps.inverse_dynamics == CapabilityLevel.FULL
    assert caps.parameter_gradients == CapabilityLevel.FULL
    assert caps.state_control_gradients == CapabilityLevel.FULL
    assert caps.extra["velocity_representation"] == "inertial"


def test_jaxsim_backend_loads_sdf_and_exposes_canonical_state(tmp_path: Path) -> None:
    model_file = tmp_path / "robot.sdf"
    model_file.write_text("<sdf version='1.10' />", encoding="utf-8")
    backend = JaxSimBackend(api_module=_fake_js())

    backend.load_from_path(str(model_file))
    q, v = backend.get_state()

    assert backend.model_name == "fake_jaxsim_model"
    assert backend.model is not None
    assert backend.model.loaded_from == model_file
    assert backend.model.is_urdf is False
    assert q.shape == (9,)
    assert v.shape == (8,)


def test_jaxsim_backend_sets_state_using_canonical_angular_linear_velocity() -> None:
    backend = JaxSimBackend(api_module=_fake_js())
    backend.load_from_string("<sdf />", extension="sdf")
    q = np.array([1.0, 2.0, 3.0, 0.5, 0.5, 0.5, 0.5, 0.1, 0.2])
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.7, 0.8])

    backend.set_state(q, v)
    observed_q, observed_v = backend.get_state()

    np.testing.assert_allclose(observed_q, q)
    np.testing.assert_allclose(observed_v, v)
    assert backend.data is not None
    np.testing.assert_allclose(backend.data.base_angular_velocity, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(backend.data.base_linear_velocity, [4.0, 5.0, 6.0])


def test_jaxsim_backend_delegates_dynamics_and_jacobian_terms() -> None:
    backend = JaxSimBackend(api_module=_fake_js())
    backend.load_from_string("<sdf />", extension="sdf")

    mass = backend.compute_mass_matrix()
    bias = backend.compute_bias_forces()
    gravity = backend.compute_gravity_forces()
    coriolis = backend.compute_coriolis_matrix()
    jacobian = backend.compute_jacobian("club")
    tau = backend.compute_inverse_dynamics(np.arange(8.0))

    assert mass.shape == (8, 8)
    assert np.allclose(mass, mass.T)
    assert bias.shape == (8,)
    assert gravity.shape == (8,)
    assert coriolis.shape == (8, 8)
    assert jacobian is not None
    assert jacobian["spatial"].shape == (6, 8)
    np.testing.assert_allclose(jacobian["angular"], jacobian["spatial"][:3])
    np.testing.assert_allclose(jacobian["linear"], jacobian["spatial"][3:])
    np.testing.assert_allclose(tau, np.arange(8.0))


def test_jaxsim_backend_registered_in_loader_map() -> None:
    assert EngineType.JAXSIM in LOADER_MAP


def test_jaxsim_probe_reports_missing_optional_dependency() -> None:
    result = JaxSimProbe(Path.cwd()).probe()

    assert result.engine_name == "JaxSim"
    assert result.status in {ProbeStatus.AVAILABLE, ProbeStatus.NOT_INSTALLED}
    if result.status == ProbeStatus.NOT_INSTALLED:
        assert "jaxsim" in result.missing_dependencies


def test_jaxsim_availability_name_is_supported() -> None:
    assert get_engine_status("jaxsim").value in {"available", "not_installed", "broken"}


def test_unavailable_jaxsim_backend_raises_clear_import_error() -> None:
    backend = JaxSimBackend(api_module=None)
    with pytest.raises(
        ImportError, match="JaxSim optional dependency is not installed"
    ):
        backend.load_from_string("<sdf />", extension="sdf")
