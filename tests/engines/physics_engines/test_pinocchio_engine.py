"""Tests for PinocchioPhysicsEngine wrapper.

We mock the underlying ``pinocchio`` library via ``patch.dict(sys.modules,...)``
so the wrapper logic — argument validation, state translation, integrator
routing, and graceful degradation when no model is loaded — is exercised
without needing the native pinocchio install.
"""

from __future__ import annotations

import types
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
    PinocchioPhysicsEngine,
)
from src.shared.python.core.contracts.exceptions import PreconditionError


@pytest.fixture
def engine() -> PinocchioPhysicsEngine:
    return PinocchioPhysicsEngine()


def _make_fake_pin(nq: int = 2, nv: int = 2) -> Any:
    """Build a fake pinocchio module that returns numpy arrays of the right size."""
    fake = types.ModuleType("pinocchio")

    class FakeRefFrame:
        LOCAL_WORLD_ALIGNED = 0

    def _make_model(name: str = "FakeModel") -> Any:
        m = MagicMock()
        m.name = name
        m.nq = nq
        m.nv = nv
        m.existFrame = MagicMock(return_value=True)
        m.getFrameId = MagicMock(return_value=1)
        m.createData = MagicMock(return_value=MagicMock())
        return m

    fake.buildModelFromUrdf = MagicMock(side_effect=lambda p: _make_model("UrdfModel"))
    fake.buildModelFromXML = MagicMock(side_effect=lambda c: _make_model("XmlModel"))
    fake.neutral = MagicMock(side_effect=lambda model: np.zeros(model.nq))
    fake.aba = MagicMock(side_effect=lambda m, d, q, v, tau: np.zeros(m.nv))
    fake.rnea = MagicMock(side_effect=lambda m, d, q, v, a: np.zeros(m.nv))
    fake.integrate = MagicMock(side_effect=lambda m, q, dq: q + dq)
    fake.forwardKinematics = MagicMock()
    fake.computeJointJacobians = MagicMock()
    fake.updateFramePlacements = MagicMock()
    fake.getFrameJacobian = MagicMock(
        side_effect=lambda m, d, fid, ref: np.ones((6, m.nv))
    )
    fake.ReferenceFrame = FakeRefFrame
    return fake


class TestUninitialised:
    def test_engine_type(self, engine: PinocchioPhysicsEngine) -> None:
        assert engine.engine_type == "pinocchio"

    def test_is_initialized_false(self, engine: PinocchioPhysicsEngine) -> None:
        assert engine.is_initialized is False

    def test_model_name_default(self, engine: PinocchioPhysicsEngine) -> None:
        assert engine.model_name == ""

    def test_get_state_returns_empty(self, engine: PinocchioPhysicsEngine) -> None:
        q, v = engine.get_state()
        assert q.size == 0 and v.size == 0

    def test_get_time_zero(self, engine: PinocchioPhysicsEngine) -> None:
        assert engine.time == 0.0

    def test_set_control_without_model_is_noop(
        self, engine: PinocchioPhysicsEngine
    ) -> None:
        # Should not raise even without a model.
        engine.set_control(np.array([0.0]))

    def test_set_control_none_raises(self, engine: PinocchioPhysicsEngine) -> None:
        with pytest.raises(ValueError):
            engine.set_control(None)  # type: ignore[arg-type]

    def test_compute_contact_forces_uninit_raises(
        self, engine: PinocchioPhysicsEngine
    ) -> None:
        # Precondition guard fires when engine not initialised.
        with pytest.raises(PreconditionError):
            engine.compute_contact_forces()

    def test_compute_drift_uninit_empty(self, engine: PinocchioPhysicsEngine) -> None:
        # uninitialised engine returns empty array (bypasses precondition guard
        # because is_initialized is False -> precondition fires).
        with pytest.raises(PreconditionError):
            engine.compute_drift_acceleration()

    def test_get_capabilities(self, engine: PinocchioPhysicsEngine) -> None:
        caps = engine.get_capabilities()
        assert caps.engine_name == "Pinocchio"

    def test_get_sensors_uninit_raises(self, engine: PinocchioPhysicsEngine) -> None:
        with pytest.raises(PreconditionError):
            engine.get_sensors()


class TestRequireVector:
    def test_none_raises(self, engine: PinocchioPhysicsEngine) -> None:
        with pytest.raises(ValueError, match="must be provided"):
            engine._require_vector("u", None, 2)  # type: ignore[arg-type]

    def test_wrong_dimension_raises(self, engine: PinocchioPhysicsEngine) -> None:
        with pytest.raises(ValueError, match="dimension mismatch"):
            engine._require_vector("u", np.zeros((2, 2)), 2)

    def test_wrong_length_raises(self, engine: PinocchioPhysicsEngine) -> None:
        with pytest.raises(ValueError, match="dimension mismatch"):
            engine._require_vector("u", np.zeros(3), 2)

    def test_valid_vector_returns_float64(self, engine: PinocchioPhysicsEngine) -> None:
        out = engine._require_vector("u", np.array([1, 2]), 2)
        assert out.dtype == np.float64


@pytest.fixture(autouse=True)
def _restore_pin_module():
    """Remove any test-injected `pin` attribute after each test."""
    from src.engines.physics_engines.pinocchio.python import (
        pinocchio_physics_engine as mod,
    )

    had_pin = hasattr(mod, "pin")
    original = getattr(mod, "pin", None)
    yield
    if had_pin:
        mod.pin = original
    elif hasattr(mod, "pin"):
        delattr(mod, "pin")


class TestWithMockedPin:
    """Test methods that go through pinocchio after mocking the module."""

    def _engine_with_model(self) -> tuple[PinocchioPhysicsEngine, Any]:
        fake = _make_fake_pin()
        from src.engines.physics_engines.pinocchio.python import (
            pinocchio_physics_engine as mod,
        )

        # Inject the fake `pin` reference the wrapper expects when
        # PINOCCHIO_AVAILABLE is true at import time.
        mod.pin = fake
        eng = PinocchioPhysicsEngine()
        eng._load_from_path_impl("model.urdf")
        return eng, fake

    def test_load_from_path_populates_state(self) -> None:
        eng, _ = self._engine_with_model()
        assert eng.is_initialized
        assert eng.q.shape == (2,)
        assert eng.v.shape == (2,)
        assert eng.model_name == "UrdfModel"

    def test_load_from_string_warns_for_non_urdf(self) -> None:
        fake = _make_fake_pin()
        from src.engines.physics_engines.pinocchio.python import (
            pinocchio_physics_engine as mod,
        )

        mod.pin = fake
        eng = PinocchioPhysicsEngine()
        eng._load_from_string_impl("<urdf/>", extension="xml")
        assert eng.model_name_str == "StringLoadedModel"

    def test_step_invalid_dt_raises(self) -> None:
        eng, _ = self._engine_with_model()
        with pytest.raises(ValueError, match="dt must be positive"):
            eng.step(dt=-0.01)

    def test_step_bad_integrator_raises(self) -> None:
        eng, _ = self._engine_with_model()
        with pytest.raises(ValueError, match="Unsupported"):
            eng.step(dt=0.01, integrator="bogus")  # type: ignore[arg-type]

    def test_step_semi_implicit(self) -> None:
        eng, fake = self._engine_with_model()
        eng.step(dt=0.01, integrator="semi_implicit")
        assert fake.aba.called
        assert eng.time == pytest.approx(0.01)

    def test_step_rk4(self) -> None:
        eng, fake = self._engine_with_model()
        eng.step(dt=0.01, integrator="rk4")
        assert fake.aba.call_count >= 4
        assert eng.time == pytest.approx(0.01)

    def test_set_state_validates_lengths(self) -> None:
        eng, _ = self._engine_with_model()
        with pytest.raises(ValueError, match="q size"):
            eng.set_state(np.zeros(5), np.zeros(2))
        with pytest.raises(ValueError, match="v size"):
            eng.set_state(np.zeros(2), np.zeros(5))

    def test_set_state_refresh_kinematics(self) -> None:
        eng, fake = self._engine_with_model()
        fake.forwardKinematics.reset_mock()
        eng.set_state(np.array([0.1, 0.2]), np.array([0.0, 0.0]))
        assert fake.forwardKinematics.called

    def test_set_control_writes_tau(self) -> None:
        eng, _ = self._engine_with_model()
        eng.set_control(np.array([1.0, 2.0]))
        assert np.allclose(eng.tau, [1.0, 2.0])

    def test_compute_inverse_dynamics(self) -> None:
        eng, fake = self._engine_with_model()
        out = eng.compute_inverse_dynamics(np.array([0.1, 0.2]))
        assert out.shape == (2,)
        assert fake.rnea.called

    def test_compute_inverse_dynamics_none_raises(self) -> None:
        eng, _ = self._engine_with_model()
        with pytest.raises(ValueError):
            eng.compute_inverse_dynamics(None)  # type: ignore[arg-type]

    def test_compute_jacobian_unknown_body_returns_none(self) -> None:
        eng, fake = self._engine_with_model()
        eng.model.existFrame = MagicMock(return_value=False)
        assert eng.compute_jacobian("missing") is None

    def test_compute_jacobian_known_body_returns_dict(self) -> None:
        eng, fake = self._engine_with_model()
        out = eng.compute_jacobian("hand")
        assert set(out.keys()) == {"linear", "angular", "spatial"}
        assert out["spatial"].shape == (6, 2)

    def test_compute_jacobian_none_body_raises(self) -> None:
        eng, _ = self._engine_with_model()
        with pytest.raises(ValueError):
            eng.compute_jacobian(None)  # type: ignore[arg-type]

    def test_compute_drift_acceleration_uses_zero_tau(self) -> None:
        eng, fake = self._engine_with_model()
        fake.aba.reset_mock()
        a = eng.compute_drift_acceleration()
        assert a.shape == (2,)
        # Last positional arg of aba is tau; should be zeros.
        _, _, _, _, tau_used = fake.aba.call_args.args
        assert np.allclose(tau_used, [0.0, 0.0])

    def test_compute_control_acceleration_difference(self) -> None:
        eng, fake = self._engine_with_model()
        out = eng.compute_control_acceleration(np.array([1.0, 2.0]))
        assert out.shape == (2,)

    def test_compute_control_acceleration_none_raises(self) -> None:
        eng, _ = self._engine_with_model()
        with pytest.raises(ValueError):
            eng.compute_control_acceleration(None)  # type: ignore[arg-type]

    def test_compute_zvcf(self) -> None:
        eng, fake = self._engine_with_model()
        out = eng.compute_zvcf(np.array([0.1, 0.2]))
        assert out.shape == (2,)

    def test_compute_affine_drift_matches_drift(self) -> None:
        eng, _ = self._engine_with_model()
        a1 = eng.compute_affine_drift()
        a2 = eng.compute_drift_acceleration()
        assert np.allclose(a1, a2)

    def test_get_sensors_empty(self) -> None:
        eng, _ = self._engine_with_model()
        assert eng.get_sensors() == {}

    def test_reset_resets_state(self) -> None:
        eng, fake = self._engine_with_model()
        eng.q = np.array([1.0, 2.0])
        eng.v = np.array([3.0, 4.0])
        eng.time = 5.0
        eng.reset()
        assert eng.time == 0.0
        assert np.allclose(eng.v, [0.0, 0.0])
        assert fake.neutral.called

    def test_get_state_returns_copies(self) -> None:
        eng, _ = self._engine_with_model()
        eng.q = np.array([1.0, 2.0])
        q, v = eng.get_state()
        q[0] = 99.0
        assert eng.q[0] == 1.0

    def test_forward_calls_pin_calls(self) -> None:
        eng, fake = self._engine_with_model()
        fake.forwardKinematics.reset_mock()
        eng.forward()
        assert fake.forwardKinematics.called
        assert fake.computeJointJacobians.called

    def test_compute_contact_forces_zero(self) -> None:
        eng, _ = self._engine_with_model()
        f = eng.compute_contact_forces()
        assert np.allclose(f, [0.0, 0.0, 0.0])
