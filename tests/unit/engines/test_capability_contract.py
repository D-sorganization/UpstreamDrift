"""Contract tests for the Capability enum and engine capabilities() method.

Issue #3052 — every method listed in an engine's capabilities() set must not
raise NotImplementedError when called with valid inputs.  Conversely, methods
that ARE NOT in the capability set may legitimately raise NotImplementedError.

Design by Contract:
    Invariant: if Capability.X in engine.capabilities(), then the corresponding
    method must be callable without raising NotImplementedError.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.engine_core.capabilities import Capability

# ---------------------------------------------------------------------------
# Tests for the Capability enum itself
# ---------------------------------------------------------------------------


class TestCapabilityEnum:
    """Verify the Capability enum members exist and are distinct."""

    def test_forward_dynamics_exists(self) -> None:
        assert Capability.FORWARD_DYNAMICS is not None

    def test_inverse_dynamics_exists(self) -> None:
        assert Capability.INVERSE_DYNAMICS is not None

    def test_contact_forces_exists(self) -> None:
        assert Capability.CONTACT_FORCES is not None

    def test_energy_computation_exists(self) -> None:
        assert Capability.ENERGY_COMPUTATION is not None

    def test_mass_matrix_exists(self) -> None:
        assert Capability.MASS_MATRIX is not None

    def test_jacobian_exists(self) -> None:
        assert Capability.JACOBIAN is not None

    def test_drift_control_exists(self) -> None:
        assert Capability.DRIFT_CONTROL is not None

    def test_counterfactual_exists(self) -> None:
        assert Capability.COUNTERFACTUAL is not None

    def test_all_members_distinct(self) -> None:
        members = list(Capability)
        assert len(members) == len(set(members))

    def test_capability_is_enum(self) -> None:
        from enum import Enum

        assert issubclass(Capability, Enum)


# ---------------------------------------------------------------------------
# Tests for BasePhysicsEngine.capabilities()
# ---------------------------------------------------------------------------


class TestBasePhysicsEngineCapabilities:
    """BasePhysicsEngine.capabilities() must return an empty frozenset by default."""

    def _make_concrete_base(self):  # type: ignore[return]
        """Create a minimal concrete BasePhysicsEngine for testing."""
        try:
            from src.shared.python.engine_core.base_physics_engine import (
                BasePhysicsEngine,
            )
        except ImportError as exc:
            pytest.skip(f"BasePhysicsEngine not importable: {exc}")

        class _MinimalEngine(BasePhysicsEngine):
            @property
            def engine_type(self) -> str:
                return "minimal"

            def _load_from_path_impl(self, path: str) -> None:
                pass

            def _load_from_string_impl(self, content: str, extension) -> None:
                pass

            def reset(self) -> None:
                pass

            def step(self, dt=None) -> None:
                pass

            def forward(self) -> None:
                pass

            def get_state(self):
                return np.zeros(1), np.zeros(1)

            def set_state(self, q, v) -> None:
                pass

            def set_control(self, u) -> None:
                pass

            def compute_mass_matrix(self):
                return np.eye(1)

            def compute_bias_forces(self):
                return np.zeros(1)

            def compute_gravity_forces(self):
                return np.zeros(1)

            def compute_inverse_dynamics(self, qacc):
                return np.zeros(1)

            def compute_jacobian(self, body_name):
                return None

            def compute_drift_acceleration(self):
                return np.zeros(1)

            def compute_control_acceleration(self, tau):
                return np.zeros(1)

            def compute_ztcf(self, q, v):
                return np.zeros(1)

            def compute_zvcf(self, q):
                return np.zeros(1)

        return _MinimalEngine()

    def test_base_capabilities_returns_frozenset(self) -> None:
        engine = self._make_concrete_base()
        assert isinstance(engine.capabilities(), frozenset)

    def test_base_capabilities_is_empty(self) -> None:
        engine = self._make_concrete_base()
        assert engine.capabilities() == frozenset()

    def test_capabilities_method_exists(self) -> None:
        engine = self._make_concrete_base()
        assert callable(engine.capabilities)


# ---------------------------------------------------------------------------
# Tests for PinocchioPhysicsEngine capabilities
# ---------------------------------------------------------------------------


class TestPinocchioCapabilityContract:
    """Pinocchio engine capability declarations must match its implementations."""

    def _make_pinocchio_engine(self):  # type: ignore[return]
        try:
            from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
                PinocchioPhysicsEngine,
            )
        except ImportError as exc:
            pytest.skip(f"Pinocchio not available: {exc}")
        return PinocchioPhysicsEngine()

    def test_capabilities_returns_frozenset(self) -> None:
        engine = self._make_pinocchio_engine()
        assert isinstance(engine.capabilities(), frozenset)

    def test_forward_dynamics_declared(self) -> None:
        engine = self._make_pinocchio_engine()
        assert Capability.FORWARD_DYNAMICS in engine.capabilities()

    def test_mass_matrix_declared(self) -> None:
        engine = self._make_pinocchio_engine()
        assert Capability.MASS_MATRIX in engine.capabilities()

    def test_inverse_dynamics_declared(self) -> None:
        engine = self._make_pinocchio_engine()
        assert Capability.INVERSE_DYNAMICS in engine.capabilities()

    def test_jacobian_declared(self) -> None:
        engine = self._make_pinocchio_engine()
        assert Capability.JACOBIAN in engine.capabilities()

    def test_drift_control_declared(self) -> None:
        engine = self._make_pinocchio_engine()
        assert Capability.DRIFT_CONTROL in engine.capabilities()

    def test_counterfactual_declared(self) -> None:
        engine = self._make_pinocchio_engine()
        assert Capability.COUNTERFACTUAL in engine.capabilities()

    def test_contact_forces_not_declared(self) -> None:
        """CONTACT_FORCES must NOT appear in Pinocchio capabilities.

        This is the core fix for issue #3052: callers must be able to
        determine at capability-check time that contact forces are
        unsupported, rather than discovering it via a runtime crash.
        """
        engine = self._make_pinocchio_engine()
        assert Capability.CONTACT_FORCES not in engine.capabilities()

    def test_compute_contact_forces_raises_not_implemented(self) -> None:
        """compute_contact_forces must raise NotImplementedError (not return zeros).

        Returning zeros is a silent lie; raising NotImplementedError with a
        helpful message forces callers to check capabilities() first.
        """
        from unittest.mock import MagicMock

        from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
            PinocchioPhysicsEngine,
        )

        engine = PinocchioPhysicsEngine.__new__(PinocchioPhysicsEngine)
        engine.time = 0.0
        engine.tau = np.zeros(6)
        engine.a = np.zeros(6)
        engine.q = np.zeros(7)
        engine.v = np.zeros(6)
        mock_model = MagicMock()
        mock_model.nq = 7
        mock_model.nv = 6
        engine.model = mock_model
        engine.data = MagicMock()
        engine._is_initialized = True
        engine.model_name_str = "test"
        engine.allowed_dirs = []

        with pytest.raises(NotImplementedError, match="capabilities"):
            engine.compute_contact_forces()

    def test_compute_contact_forces_error_mentions_capabilities(self) -> None:
        """The NotImplementedError message must guide callers to capabilities()."""
        from unittest.mock import MagicMock

        from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
            PinocchioPhysicsEngine,
        )

        engine = PinocchioPhysicsEngine.__new__(PinocchioPhysicsEngine)
        engine.time = 0.0
        engine.tau = np.zeros(6)
        engine.a = np.zeros(6)
        engine.q = np.zeros(7)
        engine.v = np.zeros(6)
        engine.model = MagicMock()
        engine.data = MagicMock()
        engine._is_initialized = True
        engine.model_name_str = "test"
        engine.allowed_dirs = []

        with pytest.raises(NotImplementedError) as exc_info:
            engine.compute_contact_forces()
        assert "capabilities" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Tests for PendulumPhysicsEngine capabilities
# ---------------------------------------------------------------------------


class TestPendulumCapabilityContract:
    """Pendulum engine capability declarations must match its implementations."""

    def _make_pendulum_engine(self):  # type: ignore[return]
        try:
            from src.engines.physics_engines.pendulum.python.pendulum_physics_engine import (
                PendulumPhysicsEngine,
            )
        except ImportError as exc:
            pytest.skip(f"PendulumPhysicsEngine not available: {exc}")
        return PendulumPhysicsEngine()

    def test_capabilities_returns_frozenset(self) -> None:
        engine = self._make_pendulum_engine()
        assert isinstance(engine.capabilities(), frozenset)

    def test_forward_dynamics_declared(self) -> None:
        engine = self._make_pendulum_engine()
        assert Capability.FORWARD_DYNAMICS in engine.capabilities()

    def test_mass_matrix_declared(self) -> None:
        engine = self._make_pendulum_engine()
        assert Capability.MASS_MATRIX in engine.capabilities()

    def test_inverse_dynamics_declared(self) -> None:
        engine = self._make_pendulum_engine()
        assert Capability.INVERSE_DYNAMICS in engine.capabilities()

    def test_drift_control_declared(self) -> None:
        engine = self._make_pendulum_engine()
        assert Capability.DRIFT_CONTROL in engine.capabilities()

    def test_counterfactual_declared(self) -> None:
        engine = self._make_pendulum_engine()
        assert Capability.COUNTERFACTUAL in engine.capabilities()

    def test_contact_forces_not_declared(self) -> None:
        engine = self._make_pendulum_engine()
        assert Capability.CONTACT_FORCES not in engine.capabilities()

    def test_jacobian_not_declared(self) -> None:
        """Pendulum Jacobian returns None; it should not be declared as supported."""
        engine = self._make_pendulum_engine()
        assert Capability.JACOBIAN not in engine.capabilities()

    def test_declared_mass_matrix_does_not_raise(self) -> None:
        """MASS_MATRIX is declared; compute_mass_matrix() must not raise NotImplementedError."""
        engine = self._make_pendulum_engine()
        assert Capability.MASS_MATRIX in engine.capabilities()
        result = engine.compute_mass_matrix()
        assert result is not None

    def test_declared_inverse_dynamics_does_not_raise(self) -> None:
        """INVERSE_DYNAMICS is declared; compute_inverse_dynamics() must work."""
        engine = self._make_pendulum_engine()
        assert Capability.INVERSE_DYNAMICS in engine.capabilities()
        result = engine.compute_inverse_dynamics(np.zeros(2))
        assert result is not None

    def test_declared_drift_control_does_not_raise(self) -> None:
        """DRIFT_CONTROL is declared; compute_drift_acceleration() must work."""
        engine = self._make_pendulum_engine()
        assert Capability.DRIFT_CONTROL in engine.capabilities()
        result = engine.compute_drift_acceleration()
        assert result is not None


# ---------------------------------------------------------------------------
# General contract: no engine raises NotImplementedError for declared caps
# ---------------------------------------------------------------------------


class TestEngineCapabilityContractGeneral:
    """Engines must not raise NotImplementedError for any declared capability."""

    _CAPABILITY_METHODS: dict[Capability, str] = {
        Capability.MASS_MATRIX: "compute_mass_matrix",
        Capability.INVERSE_DYNAMICS: "compute_inverse_dynamics",
        Capability.DRIFT_CONTROL: "compute_drift_acceleration",
    }

    def _check_engine(self, engine, qacc: np.ndarray) -> None:
        """Assert that every declared capability method is callable."""
        caps = engine.capabilities()

        if Capability.MASS_MATRIX in caps:
            result = engine.compute_mass_matrix()
            assert not isinstance(
                result, type(NotImplemented)
            ), "compute_mass_matrix raised NotImplementedError for declared capability"

        if Capability.INVERSE_DYNAMICS in caps:
            result = engine.compute_inverse_dynamics(qacc)
            assert result is not None

        if Capability.DRIFT_CONTROL in caps:
            result = engine.compute_drift_acceleration()
            assert result is not None

    def test_pendulum_contract(self) -> None:
        try:
            from src.engines.physics_engines.pendulum.python.pendulum_physics_engine import (
                PendulumPhysicsEngine,
            )
        except ImportError as exc:
            pytest.skip(f"Pendulum not available: {exc}")

        engine = PendulumPhysicsEngine()
        self._check_engine(engine, np.zeros(2))
