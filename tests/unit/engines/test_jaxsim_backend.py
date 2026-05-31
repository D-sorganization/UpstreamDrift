"""JaxSim backend adapter contract tests."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from src.engines.physics_engines.jaxsim import JaxSimBackend
from src.engines.physics_engines.jaxsim.jaxsim_backend import make_mock_jaxsim_apis
from src.shared.python.engine_core.engine_registry import EngineType
from src.shared.python.engine_core.engine_availability import get_engine_status
from src.shared.python.engine_core.sub_protocols import (
    DynamicsComputable,
    Loadable,
    Queryable,
)
from src.shared.python.engine_core.velocity_conventions import (
    CANONICAL_VELOCITY_REPRESENTATION,
)


class _FakeModel:
    model_name = "canonical_jaxsim"
    dofs = 1
    time_step = 0.01


class _FakeData:
    def __init__(
        self,
        *,
        base_position: np.ndarray | None = None,
        base_quaternion: np.ndarray | None = None,
        joint_positions: np.ndarray | None = None,
        base_angular_velocity: np.ndarray | None = None,
        base_linear_velocity: np.ndarray | None = None,
        joint_velocities: np.ndarray | None = None,
        velocity_representation: str = "inertial",
    ) -> None:
        self.base_position = np.zeros(3) if base_position is None else base_position
        self.base_quaternion = (
            np.array([1.0, 0.0, 0.0, 0.0])
            if base_quaternion is None
            else base_quaternion
        )
        self.joint_positions = (
            np.zeros(1) if joint_positions is None else joint_positions
        )
        self.base_angular_velocity = (
            np.zeros(3) if base_angular_velocity is None else base_angular_velocity
        )
        self.base_linear_velocity = (
            np.zeros(3) if base_linear_velocity is None else base_linear_velocity
        )
        self.joint_velocities = (
            np.zeros(1) if joint_velocities is None else joint_velocities
        )
        self.velocity_representation = velocity_representation

    @contextmanager
    def switch_velocity_representation(self, velocity_representation: str):
        previous = self.velocity_representation
        self.velocity_representation = velocity_representation
        try:
            yield self
        finally:
            self.velocity_representation = previous


class _FakeDataApi:
    class JaxSimModelData:
        @staticmethod
        def zero(
            model: _FakeModel, velocity_representation: str = "mixed"
        ) -> _FakeData:
            return _FakeData(velocity_representation=velocity_representation)

        @staticmethod
        def build(
            model: _FakeModel,
            *,
            velocity_representation: str = "mixed",
            **kwargs: np.ndarray,
        ) -> _FakeData:
            return _FakeData(velocity_representation=velocity_representation, **kwargs)


class _FakeModelApi:
    class JaxSimModel:
        @staticmethod
        def build_from_model_description(
            *, model_description: str, model_name: str
        ) -> _FakeModel:
            assert model_description.strip()
            return _FakeModel()

    @staticmethod
    def free_floating_mass_matrix(model: _FakeModel, data: _FakeData) -> np.ndarray:
        return np.diag([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    @staticmethod
    def free_floating_bias_forces(model: _FakeModel, data: _FakeData) -> np.ndarray:
        return np.arange(7, dtype=np.float64)

    @staticmethod
    def free_floating_gravity_forces(model: _FakeModel, data: _FakeData) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, -9.81, 1.0])

    @staticmethod
    def free_floating_coriolis_matrix(model: _FakeModel, data: _FakeData) -> np.ndarray:
        return np.eye(7) * 0.5

    @staticmethod
    def inverse_dynamics(
        model: _FakeModel,
        data: _FakeData,
        *,
        joint_accelerations: np.ndarray,
        base_acceleration: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return base_acceleration + 1.0, joint_accelerations + 2.0

    @staticmethod
    def generalized_free_floating_jacobian(
        model: _FakeModel,
        data: _FakeData,
        *,
        output_vel_repr: str,
    ) -> np.ndarray:
        assert output_vel_repr == "inertial"
        first = np.arange(42, dtype=np.float64).reshape(6, 7)
        second = first + 100.0
        return np.stack([first, second])

    @staticmethod
    def forward_kinematics(model: _FakeModel, data: _FakeData) -> None:
        return None

    @staticmethod
    def step(
        model: _FakeModel,
        data: _FakeData,
        *,
        joint_force_references: np.ndarray,
        dt: float | None = None,
    ) -> _FakeData:
        return data


class _FakeFrameApi:
    @staticmethod
    def name_to_idx(model: _FakeModel, *, frame_name: str) -> int:
        if frame_name != "club":
            raise ValueError(frame_name)
        return 1


def _backend() -> JaxSimBackend:
    backend = JaxSimBackend(
        apis=make_mock_jaxsim_apis(
            model_api=_FakeModelApi,
            data_api=_FakeDataApi,
            frame_api=_FakeFrameApi,
        )
    )
    backend.load_from_string("<sdf version='1.10'/>", extension="sdf")
    return backend


def test_jaxsim_backend_satisfies_core_sub_protocols() -> None:
    backend = _backend()

    assert isinstance(backend, Loadable)
    assert isinstance(backend, Queryable)
    assert isinstance(backend, DynamicsComputable)
    assert (
        backend.convention.velocity_representation is CANONICAL_VELOCITY_REPRESENTATION
    )


def test_jaxsim_backend_state_round_trips_in_canonical_order() -> None:
    backend = _backend()
    q = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.25])
    v = np.array([0.1, 0.2, 0.3, 1.0, 2.0, 3.0, -0.5])

    backend.set_state(q, v)

    returned_q, returned_v = backend.get_state()
    np.testing.assert_allclose(returned_q, q)
    np.testing.assert_allclose(returned_v, v)


def test_jaxsim_backend_dynamics_terms_have_expected_shapes_and_invariants() -> None:
    backend = _backend()

    mass = backend.compute_mass_matrix()
    bias = backend.compute_bias_forces()
    gravity = backend.compute_gravity_forces()
    coriolis = backend.compute_coriolis_matrix()

    assert mass.shape == (7, 7)
    np.testing.assert_allclose(mass, mass.T)
    assert np.all(np.linalg.eigvalsh(mass) > 0.0)
    assert bias.shape == (7,)
    assert gravity.shape == (7,)
    assert coriolis.shape == (7, 7)


def test_jaxsim_backend_inverse_dynamics_and_acceleration_decomposition() -> None:
    backend = _backend()
    qacc = np.arange(7, dtype=np.float64)

    tau = backend.compute_inverse_dynamics(qacc)
    drift = backend.compute_drift_acceleration()
    control = backend.compute_control_acceleration(np.ones(7))

    np.testing.assert_allclose(tau[:6], qacc[:6] + 1.0)
    np.testing.assert_allclose(tau[6:], qacc[6:] + 2.0)
    assert drift.shape == (7,)
    assert control.shape == (7,)


def test_jaxsim_backend_jacobian_uses_spatial_order() -> None:
    backend = _backend()

    jacobian = backend.compute_jacobian("club")

    assert jacobian is not None
    assert jacobian["spatial"].shape == (6, 7)
    np.testing.assert_allclose(jacobian["angular"], jacobian["spatial"][:3])
    np.testing.assert_allclose(jacobian["linear"], jacobian["spatial"][3:])
    assert backend.compute_jacobian("missing") is None


def test_jaxsim_backend_declares_capabilities_and_loader_registration() -> None:
    backend = _backend()

    capabilities = backend.get_capabilities()

    assert capabilities.engine_name == "JaxSim"
    assert capabilities.has_parameter_gradients
    assert capabilities.has_state_control_gradients
    pytest.importorskip("pandas")
    from src.engines.loaders import LOADER_MAP, load_jaxsim_engine

    assert EngineType.JAXSIM in LOADER_MAP
    assert LOADER_MAP[EngineType.JAXSIM] is load_jaxsim_engine
    assert get_engine_status("jaxsim").value in {"available", "not_installed", "broken"}


def test_mock_api_factory_uses_inertial_velocity_repr_names() -> None:
    apis = make_mock_jaxsim_apis(
        model_api=_FakeModelApi,
        data_api=_FakeDataApi,
        frame_api=_FakeFrameApi,
    )

    assert apis.common.VelRepr.Inertial == "inertial"
    assert isinstance(apis.common, SimpleNamespace)
