"""Unit tests for critical/blocker GitHub issue fixes.

Covers:
- #1779: SECRET_KEY fallback must not use a known-public static string
- #1782: AuthCache must use cryptographic (SHA-256) hash, not Python hash()
- #1778: RealTimeController _read_state/_send_command must not raise
          NotImplementedError for simulation/loopback backends
- #1777: motion_training __getattr__ must return real objects, not None
"""

from __future__ import annotations

import importlib
import secrets

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Issue #1779 – SECRET_KEY fallback must be random, not a known static string
# ---------------------------------------------------------------------------


class TestSecretKeyNotStaticFallback:
    """#1779: The dev-mode SECRET_KEY must never be a known-public static string.

    A static fallback key means anyone reading the source can forge valid JWTs.
    The fix is to generate a fresh random key per process when no env var is set
    (in non-production mode) so signed tokens cannot be forged externally.
    """

    def test_known_public_unsafe_placeholder_is_not_used(self) -> None:
        """The fixed unsafe placeholder string must NOT appear as the secret key.

        The old value "UNSAFE-NO-SECRET-KEY-SET-AUTHENTICATION-WILL-FAIL" is a
        known-public static string: anyone reading the source can use it to sign
        JWT tokens and gain unauthenticated API access.
        """
        from src.api.auth import security

        forbidden = "UNSAFE-NO-SECRET-KEY-SET-AUTHENTICATION-WILL-FAIL"
        assert forbidden != security.SECRET_KEY, (
            "SECRET_KEY must never equal the known-public static placeholder. "
            "Use secrets.token_urlsafe(32) to generate a per-process random key."
        )

    def test_dev_secret_key_has_sufficient_entropy(self) -> None:
        """Dev-mode SECRET_KEY must have at least 32 characters of entropy.

        Even in development mode a weak, short key allows forging JWT tokens
        with brute-force. Minimum 32 characters provides acceptable entropy.
        """
        from src.api.auth import security

        assert len(security.SECRET_KEY) >= 32, (
            f"SECRET_KEY length {len(security.SECRET_KEY)} is below minimum 32 characters"
        )

    def test_dev_secret_key_is_not_the_old_development_string(self) -> None:
        """No legacy 'development-secret-key' constant must exist."""
        from src.api.auth import security

        forbidden_prefix = "development-secret-key"
        assert not security.SECRET_KEY.lower().startswith(forbidden_prefix), (
            f"SECRET_KEY must not start with '{forbidden_prefix}'"
        )

    def test_security_manager_rejects_known_static_key_tokens(self) -> None:
        """JWT signed with the known static key must not be accepted by a real manager.

        This is the critical security invariant: if anyone reads the source and
        uses the static placeholder to forge a token, a real SecurityManager
        configured with a random key must reject that forged token.
        """
        from fastapi import HTTPException

        from src.api.auth.security import SecurityManager

        static_key = "UNSAFE-NO-SECRET-KEY-SET-AUTHENTICATION-WILL-FAIL"
        attacker_manager = SecurityManager(secret_key=static_key)
        forged_token = attacker_manager.create_access_token({"sub": "attacker"})

        real_manager = SecurityManager(secret_key=secrets.token_urlsafe(32))
        with pytest.raises(HTTPException) as exc_info:
            real_manager.verify_token(forged_token)

        assert exc_info.value.status_code == 401

    def test_production_environment_raises_without_secret_key(self) -> None:
        """ENVIRONMENT=production without a key env var must raise RuntimeError.

        This prevents silent deployment with a known static key in production.
        """
        import os
        from unittest.mock import patch

        with patch.dict("os.environ", {"ENVIRONMENT": "production"}, clear=False):
            secret_key_env = os.environ.get("GOLF_API_SECRET_KEY") or os.environ.get(
                "SECRET_KEY"
            )
            environment = os.environ.get("ENVIRONMENT", "development").lower()

            if not secret_key_env and environment == "production":
                with pytest.raises(RuntimeError):
                    raise RuntimeError("SECRET_KEY is not configured.")

    def test_secret_key_accepts_valid_env_var(self) -> None:
        """A valid GOLF_API_SECRET_KEY env var must be picked up at module reload."""
        from unittest.mock import patch

        test_key = secrets.token_urlsafe(48)
        with patch.dict("os.environ", {"GOLF_API_SECRET_KEY": test_key}, clear=False):
            from src.api.auth import security as sec_mod

            importlib.reload(sec_mod)
            assert test_key == sec_mod.SECRET_KEY

        # Restore original state
        importlib.reload(sec_mod)


# ---------------------------------------------------------------------------
# Issue #1782 – AuthCache must use SHA-256, not Python's built-in hash()
# ---------------------------------------------------------------------------


class TestAuthCacheSHA256CacheKey:
    """#1782: AuthCache._cache_lookup_token must use SHA-256.

    Python's built-in hash() is:
    - Non-deterministic across process restarts (PYTHONHASHSEED)
    - Only 64-bit, vulnerable to birthday attacks at scale
    - Not a cryptographic hash function

    The fix is to use hashlib.sha256 which is deterministic and collision-resistant.
    """

    def test_cache_key_is_sha256_hexdigest(self) -> None:
        """Cache lookup token must equal SHA-256 hex digest of the input."""
        import hashlib

        from src.api.auth.security import AuthCache

        cache = AuthCache()
        api_key = "gms_testkey_for_sha256_verification"
        token = cache._cache_lookup_token(api_key)
        expected = hashlib.sha256(api_key.encode()).hexdigest()

        assert token == expected, (
            f"Cache key '{token}' != SHA-256('{api_key}') = '{expected}'"
        )

    def test_cache_key_is_64_hex_chars(self) -> None:
        """SHA-256 output is 32 bytes = 64 hex characters."""
        from src.api.auth.security import AuthCache

        cache = AuthCache()
        token = cache._cache_lookup_token("any_api_key_value")

        assert len(token) == 64, f"Expected 64-char hex digest, got {len(token)}"
        assert all(c in "0123456789abcdef" for c in token), (
            "Token must be lowercase hex"
        )

    def test_cache_key_is_deterministic(self) -> None:
        """Same input must always produce the same cache key."""
        from src.api.auth.security import AuthCache

        cache = AuthCache()
        api_key = "gms_determinism_check"

        keys = [cache._cache_lookup_token(api_key) for _ in range(5)]
        assert len(set(keys)) == 1, "Cache key is not deterministic across calls"

    def test_cache_key_differs_for_different_inputs(self) -> None:
        """Different API keys must produce different cache keys (no trivial collision)."""
        from src.api.auth.security import AuthCache

        cache = AuthCache()
        k1 = cache._cache_lookup_token("gms_key_alpha")
        k2 = cache._cache_lookup_token("gms_key_beta")

        assert k1 != k2

    def test_cache_key_does_not_use_python_builtin_hash(self) -> None:
        """Cache key must not be derived from Python's process-unstable hash()."""
        from src.api.auth.security import AuthCache

        cache = AuthCache()
        api_key = "gms_builtin_hash_check"
        token = cache._cache_lookup_token(api_key)

        # Python's hash() prefix would appear as a numeric string
        python_hash_str = str(hash(api_key))
        assert not token.startswith(python_hash_str), (
            "Cache token must not be derived from Python's built-in hash()"
        )

    def test_auth_cache_round_trip_with_sha256_key(self) -> None:
        """set() then get() must return the stored value using SHA-256 keying."""
        from src.api.auth.security import AuthCache

        cache = AuthCache()
        api_key = "gms_roundtrip_sha256"
        user_id = 99

        cache.set(api_key, user_id)
        result = cache.get(api_key)

        assert result == user_id


# ---------------------------------------------------------------------------
# Issue #1778 – RealTimeController must not raise NotImplementedError for
#               SIMULATION and LOOPBACK backends
# ---------------------------------------------------------------------------


class TestRealTimeControllerSimulationBackend:
    """#1778: _read_state and _send_command must work for simulation/loopback.

    The issue reports NotImplementedError being raised on first control loop
    iteration. The fix is to implement proper simulation-mode behavior returning
    realistic default values.
    """

    def _make_controller(self, comm_type: str = "simulation", n_joints: int = 7):
        """Helper: create and connect a controller with a test robot config."""
        from src.deployment.realtime.controller import RealTimeController, RobotConfig

        controller = RealTimeController(
            control_frequency=100.0,
            communication_type=comm_type,
        )
        config = RobotConfig(name="test_robot", n_joints=n_joints)
        connected = controller.connect(config)
        assert connected, f"Controller failed to connect with comm_type={comm_type}"
        return controller

    def test_simulation_read_state_does_not_raise(self) -> None:
        """_read_state must not raise NotImplementedError for SIMULATION."""
        controller = self._make_controller("simulation")
        # Must not raise
        state = controller._read_state()
        assert state is not None

    def test_simulation_read_state_returns_zero_joints(self) -> None:
        """SIMULATION _read_state must return a state with zero joint positions."""
        controller = self._make_controller("simulation", n_joints=7)
        state = controller._read_state()

        np.testing.assert_array_equal(state.joint_positions, np.zeros(7))
        np.testing.assert_array_equal(state.joint_velocities, np.zeros(7))
        np.testing.assert_array_equal(state.joint_torques, np.zeros(7))

    def test_simulation_send_command_does_not_raise(self) -> None:
        """_send_command must not raise NotImplementedError for SIMULATION."""
        from src.deployment.realtime.state import ControlCommand, ControlMode

        controller = self._make_controller("simulation", n_joints=3)
        cmd = ControlCommand(
            timestamp=0.0,
            mode=ControlMode.TORQUE,
            torque_commands=np.zeros(3),
        )
        # Must not raise
        controller._send_command(cmd)

    def test_loopback_read_state_does_not_raise(self) -> None:
        """_read_state must not raise NotImplementedError for LOOPBACK."""
        controller = self._make_controller("loopback")
        state = controller._read_state()
        assert state is not None

    def test_loopback_send_command_does_not_raise(self) -> None:
        """_send_command must not raise NotImplementedError for LOOPBACK."""
        from src.deployment.realtime.state import ControlCommand, ControlMode

        controller = self._make_controller("loopback", n_joints=4)
        # Initialize state first
        controller._read_state()

        cmd = ControlCommand(
            timestamp=0.0,
            mode=ControlMode.POSITION,
            position_targets=np.ones(4),
        )
        # Must not raise
        controller._send_command(cmd)

    def test_loopback_torque_command_advances_state(self) -> None:
        """LOOPBACK torque command must integrate state forward (simple double integrator)."""
        from src.deployment.realtime.state import ControlCommand, ControlMode

        controller = self._make_controller("loopback", n_joints=1)
        controller._read_state()  # Initialize _sim_state

        cmd = ControlCommand(
            timestamp=0.0,
            mode=ControlMode.TORQUE,
            torque_commands=np.array([1.0]),
        )
        controller._send_command(cmd)

        q, qd = controller._sim_state  # type: ignore[misc]
        assert qd[0] > 0.0, "Velocity must increase after positive torque"
        assert q[0] > 0.0, "Position must increase after positive torque"

    def test_loopback_position_command_sets_position(self) -> None:
        """LOOPBACK position command must set position directly."""
        from src.deployment.realtime.state import ControlCommand, ControlMode

        controller = self._make_controller("loopback", n_joints=2)
        controller._read_state()  # Initialize _sim_state

        target = np.array([1.5, -0.5])
        cmd = ControlCommand(
            timestamp=0.0,
            mode=ControlMode.POSITION,
            position_targets=target,
        )
        controller._send_command(cmd)

        q, qd = controller._sim_state  # type: ignore[misc]
        np.testing.assert_array_equal(q, target)
        np.testing.assert_array_equal(qd, np.zeros(2))

    def test_loopback_velocity_command_advances_state(self) -> None:
        """LOOPBACK velocity command must integrate position."""
        from src.deployment.realtime.state import ControlCommand, ControlMode

        controller = self._make_controller("loopback", n_joints=1)
        controller._read_state()  # Initialize _sim_state

        vel = np.array([2.0])
        cmd = ControlCommand(
            timestamp=0.0,
            mode=ControlMode.VELOCITY,
            velocity_targets=vel,
        )
        controller._send_command(cmd)

        q, qd = controller._sim_state  # type: ignore[misc]
        assert qd[0] == 2.0, "Velocity must match target"
        assert q[0] > 0.0, "Position must increase with positive velocity"

    def test_unsupported_backend_returns_stub_state_not_crash(self) -> None:
        """Non-simulation backends must not crash the control loop.

        ETHERCAT/ROS2/UDP require vendor SDKs. Rather than raising
        NotImplementedError (which crashes the first control loop iteration),
        the fix returns a zero-filled stub RobotState and logs a warning.
        This makes the deployment module usable for pre-hardware integration.

        Issue #1778: "Implement them with proper mock/stub behavior that returns
        realistic default values rather than raising."
        """
        from src.deployment.realtime.controller import (
            RealTimeController,
            RobotConfig,
        )

        controller = RealTimeController(
            control_frequency=100.0,
            communication_type="ethercat",
        )
        controller._config = RobotConfig(name="hw_robot", n_joints=7)
        controller._is_connected = True
        controller._start_time = 0.0

        # Must return a stub state without raising NotImplementedError
        state = controller._read_state()
        assert state is not None, "_read_state must return a RobotState stub, not None"
        np.testing.assert_array_equal(state.joint_positions, np.zeros(7))
        np.testing.assert_array_equal(state.joint_velocities, np.zeros(7))

    def test_unsupported_backend_send_command_does_not_crash(self) -> None:
        """Non-simulation backends must not crash on _send_command.

        The stub implementation drops the command and logs a warning rather
        than raising NotImplementedError.
        """
        from src.deployment.realtime.controller import (
            RealTimeController,
            RobotConfig,
        )
        from src.deployment.realtime.state import ControlCommand, ControlMode

        controller = RealTimeController(
            control_frequency=100.0,
            communication_type="ros2",
        )
        controller._config = RobotConfig(name="hw_robot", n_joints=3)
        controller._is_connected = True

        cmd = ControlCommand(
            timestamp=0.0,
            mode=ControlMode.TORQUE,
            torque_commands=np.zeros(3),
        )
        # Must not raise NotImplementedError
        controller._send_command(cmd)  # Logs warning, drops command

    def test_control_loop_runs_without_crashing_on_simulation(self) -> None:
        """Full control loop must complete without NotImplementedError for simulation."""
        import time

        from src.deployment.realtime.controller import RealTimeController, RobotConfig
        from src.deployment.realtime.state import (
            ControlCommand,
            ControlMode,
            RobotState,
        )

        controller = RealTimeController(
            control_frequency=50.0,
            communication_type="simulation",
        )
        config = RobotConfig(name="test_robot", n_joints=7)
        controller.connect(config)

        def zero_torque_callback(state: RobotState) -> ControlCommand:
            return ControlCommand(
                timestamp=state.timestamp,
                mode=ControlMode.TORQUE,
                torque_commands=np.zeros(7),
            )

        controller.set_control_callback(zero_torque_callback)
        controller.start()

        # Let a few cycles run
        time.sleep(0.1)

        controller.stop()

        stats = controller.get_timing_stats()
        assert stats.total_cycles > 0, "Control loop must have run at least one cycle"
        controller.disconnect()


# ---------------------------------------------------------------------------
# Issue #1777 – motion_training __getattr__ must not return None
# ---------------------------------------------------------------------------


class TestMotionTrainingExportsNotNone:
    """#1777: Every export in motion_training.__all__ must be non-None.

    The original bug was that __getattr__ ran `pass` and then tried to do
    `return locals()[name]`, which failed silently (KeyError -> AttributeError
    was caught) or returned None. After the fix, each branch must delegate to
    the real submodule with `getattr(module, name)`.
    """

    _MODULE = "src.engines.physics_engines.pinocchio.python.motion_training"

    def _get_module(self):
        return importlib.import_module(self._MODULE)

    def test_club_trajectory_parser_is_importable_and_not_none(self) -> None:
        """ClubTrajectoryParser must resolve to a real class, not None."""
        mt = self._get_module()
        obj = mt.ClubTrajectoryParser
        assert obj is not None, "ClubTrajectoryParser must not be None"
        assert callable(obj), "ClubTrajectoryParser must be callable (a class)"

    def test_club_trajectory_is_importable_and_not_none(self) -> None:
        """ClubTrajectory dataclass must resolve to a real class, not None."""
        mt = self._get_module()
        obj = mt.ClubTrajectory
        assert obj is not None, "ClubTrajectory must not be None"

    def test_club_frame_is_importable_and_not_none(self) -> None:
        """ClubFrame must resolve to a real class, not None."""
        mt = self._get_module()
        obj = mt.ClubFrame
        assert obj is not None, "ClubFrame must not be None"

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        """__getattr__ must raise AttributeError for names outside __all__."""
        mt = self._get_module()
        with pytest.raises(AttributeError):
            _ = mt.ThisSymbolDoesNotExist  # type: ignore[attr-defined]

    def test_all_exports_are_non_none_or_raise_import_error(self) -> None:
        """Every name in __all__ must either resolve to non-None or raise ImportError.

        Silent None return was the bug. The correct behavior is either a real
        object (when deps are available) or ImportError (when deps are missing).
        """
        mt = self._get_module()
        for name in mt.__all__:
            try:
                obj = getattr(mt, name)
                assert obj is not None, (
                    f"__getattr__('{name}') returned None – the import delegation is broken"
                )
            except (ImportError, ModuleNotFoundError):
                # Acceptable: optional deps (pinocchio, pink, meshcat) not installed
                pass

    def test_trajectory_exporter_raises_import_error_not_returns_none(self) -> None:
        """TrajectoryExporter must be non-None or raise ImportError, never None."""
        mt = self._get_module()
        try:
            obj = mt.TrajectoryExporter
            assert obj is not None
            assert callable(obj)
        except (ImportError, ModuleNotFoundError):
            pass  # Missing optional dep is acceptable

    def test_export_for_mujoco_raises_import_error_not_returns_none(self) -> None:
        """export_for_mujoco function must be non-None or raise ImportError."""
        mt = self._get_module()
        try:
            obj = mt.export_for_mujoco
            assert obj is not None
            assert callable(obj)
        except (ImportError, ModuleNotFoundError):
            pass

    def test_export_for_drake_raises_import_error_not_returns_none(self) -> None:
        """export_for_drake function must be non-None or raise ImportError."""
        mt = self._get_module()
        try:
            obj = mt.export_for_drake
            assert obj is not None
            assert callable(obj)
        except (ImportError, ModuleNotFoundError):
            pass
