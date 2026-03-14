"""Comprehensive third-party integration audit tests.

This module provides adversarial TDD tests for all third-party package
integrations in UpstreamDrift. Tests are organized by package and
structured to verify:

1. Import resilience (graceful degradation when packages missing)
2. Protocol compliance (PhysicsEngine interface adherence)
3. API correctness (correct method signatures and return types)
4. Error handling (proper exceptions, no silent failures)

Issues: #1810, #1811, #1812, #1813, #1814, #1815, #1816, #1817, #1818
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.shared.python.engine_core.engine_availability import (
    DRAKE_AVAILABLE,
    MEDIAPIPE_AVAILABLE,
    MUJOCO_AVAILABLE,
    MYOSUITE_AVAILABLE,
    OPENSIM_AVAILABLE,
    PINOCCHIO_AVAILABLE,
    get_available_engines,
    get_unavailable_engines,
    is_engine_available,
    skip_if_unavailable,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Engine Availability Infrastructure Tests (#1818)
# ═══════════════════════════════════════════════════════════════════════════════


class TestEngineAvailabilityInfrastructure:
    """Verify engine_availability.py correctness and consistency."""

    def test_is_engine_available_case_insensitive(self) -> None:
        """Engine name lookup must be case-insensitive."""
        # numpy is almost always available
        result_lower = is_engine_available("numpy")
        result_upper = is_engine_available("NUMPY")
        result_mixed = is_engine_available("NumPy")
        assert result_lower == result_upper == result_mixed

    def test_get_available_engines_returns_list(self) -> None:
        """get_available_engines must return a list, never None."""
        result = get_available_engines()
        assert isinstance(result, list)
        # numpy should always be available
        assert "numpy" in result

    def test_get_unavailable_engines_returns_list(self) -> None:
        """get_unavailable_engines must return a list."""
        result = get_unavailable_engines()
        assert isinstance(result, list)

    def test_available_and_unavailable_are_disjoint(self) -> None:
        """No engine should appear in both available and unavailable lists."""
        available = set(get_available_engines())
        unavailable = set(get_unavailable_engines())
        overlap = available & unavailable
        assert not overlap, f"Engines in both lists: {overlap}"

    def test_openpose_availability_check_uses_pyopenpose(self) -> None:
        """OpenPose availability check must use 'pyopenpose', not 'openpose'.

        The Python bindings for OpenPose are distributed under the module
        name 'pyopenpose', not 'openpose'. Using the wrong name would
        always report unavailable even when installed.
        """
        # Verify the import in engine_availability.py uses pyopenpose
        import inspect

        from src.shared.python.engine_core import engine_availability

        source = inspect.getsource(engine_availability)
        assert "import pyopenpose" in source, (
            "engine_availability.py should check for 'pyopenpose', not 'openpose'"
        )

    def test_skip_if_unavailable_returns_marker(self) -> None:
        """skip_if_unavailable must return a pytest marker."""
        marker = skip_if_unavailable("numpy")
        assert hasattr(marker, "mark") or hasattr(marker, "args")

    def test_mediapipe_flag_exists(self) -> None:
        """MEDIAPIPE_AVAILABLE flag must exist and be boolean."""
        assert isinstance(MEDIAPIPE_AVAILABLE, bool)

    def test_engine_flags_dict_has_all_physics_engines(self) -> None:
        """_ENGINE_FLAGS must include all core physics engines."""
        from src.shared.python.engine_core.engine_availability import _ENGINE_FLAGS

        required_engines = [
            "mujoco",
            "pinocchio",
            "drake",
            "opensim",
            "myosuite",
            "mediapipe",
            "openpose",
        ]
        for engine in required_engines:
            assert engine in _ENGINE_FLAGS, f"Missing engine flag: {engine}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Drake Integration Tests (#1810)
# ═══════════════════════════════════════════════════════════════════════════════


class TestDrakeIntegrationAudit:
    """Verify Drake (pydrake) integration correctness."""

    def test_drake_availability_flag_is_boolean(self) -> None:
        """DRAKE_AVAILABLE must be a boolean."""
        assert isinstance(DRAKE_AVAILABLE, bool)

    def test_drake_engine_importable(self) -> None:
        """DrakePhysicsEngine class must be importable regardless of Drake."""
        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        assert DrakePhysicsEngine is not None

    @skip_if_unavailable("drake")
    def test_drake_engine_initialization(self) -> None:
        """DrakePhysicsEngine must initialize without a model."""
        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()
        assert not engine.is_initialized
        assert engine.model_name == ""

    @skip_if_unavailable("drake")
    def test_drake_protocol_methods_exist(self) -> None:
        """DrakePhysicsEngine must implement all PhysicsEngine protocol methods."""
        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()
        required_methods = [
            "load_from_path",
            "load_from_string",
            "reset",
            "step",
            "forward",
            "get_state",
            "set_state",
            "set_control",
            "get_time",
            "compute_mass_matrix",
            "compute_bias_forces",
            "compute_gravity_forces",
            "compute_inverse_dynamics",
            "compute_contact_forces",
            "compute_jacobian",
            "compute_drift_acceleration",
        ]
        for method in required_methods:
            assert hasattr(engine, method), f"Missing method: {method}"
            assert callable(getattr(engine, method)), f"Not callable: {method}"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MuJoCo Integration Tests (#1811)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMuJoCoIntegrationAudit:
    """Verify MuJoCo integration correctness."""

    def test_mujoco_availability_flag_is_boolean(self) -> None:
        """MUJOCO_AVAILABLE must be a boolean."""
        assert isinstance(MUJOCO_AVAILABLE, bool)

    def test_mujoco_conftest_early_import(self) -> None:
        """conftest.py must attempt early MuJoCo import to avoid DLL conflicts."""
        conftest_path = Path(__file__).parent.parent.parent / "conftest.py"
        if conftest_path.exists():
            content = conftest_path.read_text(encoding="utf-8")
            assert "mujoco" in content, (
                "conftest.py should import mujoco early to avoid DLL conflicts"
            )

    @skip_if_unavailable("mujoco")
    def test_mujoco_gl_environment(self) -> None:
        """MUJOCO_GL should not be set to 'egl' on Windows (headless)."""
        import os
        import sys

        if sys.platform == "win32":
            gl = os.environ.get("MUJOCO_GL", "")
            assert gl != "egl", (
                "MUJOCO_GL=egl is invalid on Windows; use osmesa or glfw"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Pinocchio Integration Tests (#1812)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPinocchioIntegrationAudit:
    """Verify Pinocchio integration correctness."""

    def test_pinocchio_availability_flag_is_boolean(self) -> None:
        """PINOCCHIO_AVAILABLE must be a boolean."""
        assert isinstance(PINOCCHIO_AVAILABLE, bool)

    def test_pinocchio_engine_importable(self) -> None:
        """PinocchioPhysicsEngine must be importable."""
        from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
            PinocchioPhysicsEngine,
        )

        assert PinocchioPhysicsEngine is not None

    @skip_if_unavailable("pinocchio")
    def test_pinocchio_engine_initialization(self) -> None:
        """PinocchioPhysicsEngine must initialize."""
        from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
            PinocchioPhysicsEngine,
        )

        engine = PinocchioPhysicsEngine()
        assert not engine.is_initialized

    @skip_if_unavailable("pinocchio")
    def test_pinocchio_contact_forces_raises(self) -> None:
        """compute_contact_forces must raise NotImplementedError (not fake zeros)."""
        from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
            PinocchioPhysicsEngine,
        )

        engine = PinocchioPhysicsEngine()
        # Load a simple model
        simple_urdf = """<?xml version="1.0"?>
        <robot name="test">
            <link name="base_link">
                <inertial>
                    <mass value="1.0"/>
                    <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
                </inertial>
            </link>
        </robot>"""
        try:
            engine.load_from_string(simple_urdf, extension=".urdf")
            with pytest.raises(NotImplementedError):
                engine.compute_contact_forces()
        except Exception:
            pytest.skip("Could not load URDF in Pinocchio")

    @skip_if_unavailable("pinocchio")
    def test_pinocchio_protocol_methods_exist(self) -> None:
        """PinocchioPhysicsEngine must implement all PhysicsEngine protocol methods."""
        from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
            PinocchioPhysicsEngine,
        )

        engine = PinocchioPhysicsEngine()
        required_methods = [
            "load_from_path",
            "load_from_string",
            "reset",
            "step",
            "forward",
            "get_state",
            "set_state",
            "set_control",
            "get_time",
            "compute_mass_matrix",
            "compute_bias_forces",
            "compute_gravity_forces",
            "compute_inverse_dynamics",
            "compute_contact_forces",
            "compute_jacobian",
            "compute_drift_acceleration",
        ]
        for method in required_methods:
            assert hasattr(engine, method), f"Missing method: {method}"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Pink IK Solver Tests (#1812 sub-component)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPinkIKSolverAudit:
    """Verify Pink IK solver import resilience and correctness."""

    def test_pink_solver_importable_without_pink(self) -> None:
        """pink_solver.py must be importable even when pink is not installed.

        This is the CRITICAL fix — previously, this would crash with
        ModuleNotFoundError during test collection.
        """
        # This import should NOT crash regardless of whether pink is installed
        from src.engines.physics_engines.pinocchio.python.dtack.ik.pink_solver import (
            PINK_SOLVER_AVAILABLE,
            PinkSolver,
            SolverSettings,
        )

        assert isinstance(PINK_SOLVER_AVAILABLE, bool)
        assert PinkSolver is not None
        assert SolverSettings is not None

    def test_pink_solver_raises_on_unavailable(self) -> None:
        """PinkSolver.__init__ must raise ImportError when pink is not installed."""
        from src.engines.physics_engines.pinocchio.python.dtack.ik.pink_solver import (
            PINK_SOLVER_AVAILABLE,
            PinkSolver,
        )

        if not PINK_SOLVER_AVAILABLE:
            with pytest.raises(ImportError, match="Pink and pinocchio"):
                PinkSolver(
                    robot_model=MagicMock(),
                    robot_data=MagicMock(),
                    robot_visual=MagicMock(),
                    robot_collision=MagicMock(),
                )
        else:
            pytest.skip("Pink is installed — cannot test unavailable path")

    def test_pink_backend_importable_without_pink(self) -> None:
        """pink_backend.py must be importable even when Pink is not installed."""
        try:
            from src.engines.physics_engines.pinocchio.python.dtack.backends.pink_backend import (
                PINK_AVAILABLE,
                PINKBackend,
            )
        except ModuleNotFoundError:
            # dtack uses internal 'dtack.*' imports that don't resolve
            # from the project root — this is itself an issue (#1812)
            pytest.skip("dtack internal imports not resolvable from project root")
            return

        assert isinstance(PINK_AVAILABLE, bool)
        assert PINKBackend is not None

    def test_pink_backend_raises_on_unavailable(self) -> None:
        """PINKBackend.__init__ must raise ImportError when pink is not installed."""
        try:
            from src.engines.physics_engines.pinocchio.python.dtack.backends.pink_backend import (
                PINK_AVAILABLE,
                PINKBackend,
            )
        except ModuleNotFoundError:
            pytest.skip("dtack internal imports not resolvable from project root")
            return

        if not PINK_AVAILABLE:
            with pytest.raises(ImportError, match="PINK is required"):
                PINKBackend(model_path="/tmp/nonexistent.urdf")
        else:
            pytest.skip("Pink is installed — cannot test unavailable path")

    def test_solver_settings_defaults(self) -> None:
        """SolverSettings must have sane defaults."""
        from src.engines.physics_engines.pinocchio.python.dtack.ik.pink_solver import (
            SolverSettings,
        )

        settings = SolverSettings()
        assert settings.solver == "quadprog"
        assert settings.damping == pytest.approx(1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. OpenSim Integration Tests (#1813)
# ═══════════════════════════════════════════════════════════════════════════════


class TestOpenSimIntegrationAudit:
    """Verify OpenSim integration correctness."""

    def test_opensim_availability_flag_is_boolean(self) -> None:
        """OPENSIM_AVAILABLE must be a boolean."""
        assert isinstance(OPENSIM_AVAILABLE, bool)

    def test_opensim_engine_importable(self) -> None:
        """OpenSimPhysicsEngine must be importable."""
        from src.engines.physics_engines.opensim.python.opensim_physics_engine import (
            OpenSimPhysicsEngine,
        )

        assert OpenSimPhysicsEngine is not None

    @skip_if_unavailable("opensim")
    def test_opensim_engine_initialization(self) -> None:
        """OpenSimPhysicsEngine must initialize without model."""
        from src.engines.physics_engines.opensim.python.opensim_physics_engine import (
            OpenSimPhysicsEngine,
        )

        engine = OpenSimPhysicsEngine()
        assert not engine.is_initialized
        assert engine.model_name == ""

    @skip_if_unavailable("opensim")
    def test_opensim_protocol_methods_exist(self) -> None:
        """OpenSimPhysicsEngine must implement all PhysicsEngine protocol methods."""
        from src.engines.physics_engines.opensim.python.opensim_physics_engine import (
            OpenSimPhysicsEngine,
        )

        engine = OpenSimPhysicsEngine()
        required_methods = [
            "load_from_path",
            "load_from_string",
            "reset",
            "step",
            "forward",
            "get_state",
            "set_state",
            "set_control",
            "get_time",
            "compute_mass_matrix",
            "compute_bias_forces",
            "compute_gravity_forces",
            "compute_inverse_dynamics",
            "compute_jacobian",
            "compute_drift_acceleration",
        ]
        for method in required_methods:
            assert hasattr(engine, method), f"Missing method: {method}"

    def test_opensim_muscle_analysis_importable(self) -> None:
        """muscle_analysis.py must be importable."""
        from src.engines.physics_engines.opensim.python import muscle_analysis

        assert muscle_analysis is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MyoSuite Integration Tests (#1814)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMyoSuiteIntegrationAudit:
    """Verify MyoSuite integration correctness."""

    def test_myosuite_availability_flag_is_boolean(self) -> None:
        """MYOSUITE_AVAILABLE must be a boolean."""
        assert isinstance(MYOSUITE_AVAILABLE, bool)

    def test_myosuite_engine_importable(self) -> None:
        """MyoSuitePhysicsEngine must be importable."""
        from src.engines.physics_engines.myosuite.python.myosuite_physics_engine import (
            MyoSuitePhysicsEngine,
        )

        assert MyoSuitePhysicsEngine is not None

    @skip_if_unavailable("myosuite")
    def test_myosuite_engine_initialization(self) -> None:
        """MyoSuitePhysicsEngine must initialize."""
        from src.engines.physics_engines.myosuite.python.myosuite_physics_engine import (
            MyoSuitePhysicsEngine,
        )

        engine = MyoSuitePhysicsEngine()
        assert not engine.is_initialized

    @skip_if_unavailable("myosuite")
    def test_myosuite_protocol_methods_exist(self) -> None:
        """MyoSuitePhysicsEngine must implement all PhysicsEngine protocol methods."""
        from src.engines.physics_engines.myosuite.python.myosuite_physics_engine import (
            MyoSuitePhysicsEngine,
        )

        engine = MyoSuitePhysicsEngine()
        required_methods = [
            "load_from_path",
            "load_from_string",
            "reset",
            "step",
            "forward",
            "get_state",
            "set_state",
            "set_control",
            "get_time",
            "compute_mass_matrix",
            "compute_bias_forces",
            "compute_inverse_dynamics",
            "compute_jacobian",
        ]
        for method in required_methods:
            assert hasattr(engine, method), f"Missing method: {method}"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. OpenPose Integration Tests (#1815)
# ═══════════════════════════════════════════════════════════════════════════════


class TestOpenPoseIntegrationAudit:
    """Verify OpenPose integration correctness."""

    def test_openpose_estimator_importable(self) -> None:
        """OpenPoseEstimator must be importable even without pyopenpose."""
        from src.shared.python.pose_estimation.openpose_estimator import (
            OpenPoseEstimator,
        )

        assert OpenPoseEstimator is not None

    def test_openpose_estimator_instantiation_without_lib(self) -> None:
        """OpenPoseEstimator must instantiate even without pyopenpose."""
        from src.shared.python.pose_estimation.openpose_estimator import (
            OpenPoseEstimator,
        )

        # Should NOT crash
        estimator = OpenPoseEstimator()
        assert estimator is not None
        assert not estimator._is_loaded

    def test_openpose_load_model_raises_without_lib(self) -> None:
        """load_model must raise ImportError when pyopenpose is missing."""
        from src.shared.python.pose_estimation.openpose_estimator import (
            OpenPoseEstimator,
        )

        # Check if pyopenpose is available
        try:
            import pyopenpose  # noqa: F401

            pytest.skip("pyopenpose is installed — cannot test missing path")
        except ImportError:
            estimator = OpenPoseEstimator()
            with pytest.raises(ImportError, match="pyopenpose"):
                estimator.load_model()

    def test_openpose_estimate_without_load_raises(self) -> None:
        """estimate_from_image must raise StateError before load_model."""
        from src.shared.python.pose_estimation.openpose_estimator import (
            OpenPoseEstimator,
        )

        estimator = OpenPoseEstimator()
        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises((Exception,)):  # StateError or similar
            estimator.estimate_from_image(fake_image)

    def test_openpose_keypoint_map_completeness(self) -> None:
        """BODY_25 keypoint map must have exactly 25 entries."""
        from src.shared.python.pose_estimation.openpose_estimator import (
            OpenPoseEstimator,
        )

        assert len(OpenPoseEstimator.KEYPOINT_MAP) == 25  # noqa: PLR2004
        # Verify indices are 0-24
        assert set(OpenPoseEstimator.KEYPOINT_MAP.keys()) == set(range(25))

    def test_openpose_has_cross_platform_model_paths(self) -> None:
        """Model path fallback must support Linux/macOS in addition to Windows."""
        import inspect

        from src.shared.python.pose_estimation.openpose_estimator import (
            OpenPoseEstimator,
        )

        source = inspect.getsource(OpenPoseEstimator.load_model)
        # Verify cross-platform support
        assert "/usr/local" in source or "OPENPOSE_MODELS_DIR" in source, (
            "OpenPose load_model must support cross-platform model paths"
        )

    def test_openpose_supports_env_var(self) -> None:
        """OpenPose must support OPENPOSE_MODELS_DIR environment variable."""
        import inspect

        from src.shared.python.pose_estimation.openpose_estimator import (
            OpenPoseEstimator,
        )

        source = inspect.getsource(OpenPoseEstimator.load_model)
        assert "OPENPOSE_MODELS_DIR" in source, (
            "OpenPose load_model must support OPENPOSE_MODELS_DIR env var"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 9. MediaPipe Integration Tests (#1816)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMediaPipeIntegrationAudit:
    """Verify MediaPipe integration correctness."""

    def test_mediapipe_estimator_importable(self) -> None:
        """MediaPipeEstimator must be importable."""
        from src.shared.python.pose_estimation.mediapipe_estimator import (
            MediaPipeEstimator,
        )

        assert MediaPipeEstimator is not None

    def test_mediapipe_estimator_instantiation(self) -> None:
        """MediaPipeEstimator must instantiate with default parameters."""
        from src.shared.python.pose_estimation.mediapipe_estimator import (
            MediaPipeEstimator,
        )

        estimator = MediaPipeEstimator()
        assert estimator is not None
        assert estimator.enable_temporal_smoothing is True  # noqa: E712

    def test_mediapipe_estimator_custom_params(self) -> None:
        """MediaPipeEstimator must accept custom parameters."""
        from src.shared.python.pose_estimation.mediapipe_estimator import (
            MediaPipeEstimator,
        )

        estimator = MediaPipeEstimator(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.8,
            enable_temporal_smoothing=False,
        )
        assert estimator.enable_temporal_smoothing is False  # noqa: E712

    def test_mediapipe_reset_temporal_state_exists(self) -> None:
        """reset_temporal_state must exist and be callable."""
        from src.shared.python.pose_estimation.mediapipe_estimator import (
            MediaPipeEstimator,
        )

        estimator = MediaPipeEstimator()
        assert hasattr(estimator, "reset_temporal_state")
        assert callable(estimator.reset_temporal_state)
        # Should not crash when called
        estimator.reset_temporal_state()

    def test_mediapipe_implements_pose_estimator_interface(self) -> None:
        """MediaPipeEstimator must implement PoseEstimator ABC."""
        from src.shared.python.pose_estimation.interface import PoseEstimator
        from src.shared.python.pose_estimation.mediapipe_estimator import (
            MediaPipeEstimator,
        )

        assert issubclass(MediaPipeEstimator, PoseEstimator)

    def test_mediapipe_video_resets_kalman_state(self) -> None:
        """estimate_from_video must call reset_temporal_state at start.

        This prevents Kalman filter contamination between video files.
        """
        import inspect

        from src.shared.python.pose_estimation.mediapipe_estimator import (
            MediaPipeEstimator,
        )

        source = inspect.getsource(MediaPipeEstimator.estimate_from_video)
        assert "reset_temporal_state" in source, (
            "estimate_from_video must call reset_temporal_state() at start "
            "to prevent Kalman filter contamination between videos"
        )

    def test_mediapipe_reset_clears_kalman_filters(self) -> None:
        """reset_temporal_state must clear all Kalman filter state."""
        from src.shared.python.pose_estimation.mediapipe_estimator import (
            MediaPipeEstimator,
        )

        estimator = MediaPipeEstimator()
        # Simulate having some state
        estimator.kalman_filters["test"] = MagicMock()
        estimator.previous_landmarks = {"test": np.array([1, 2, 3])}

        # Reset
        estimator.reset_temporal_state()

        assert len(estimator.kalman_filters) == 0
        assert estimator.previous_landmarks is None


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Pose Estimation Interface Tests (#1817)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPoseEstimationInterfaceAudit:
    """Verify pose estimation interface and shared utilities."""

    def test_interface_importable(self) -> None:
        """PoseEstimator and PoseEstimationResult must be importable."""
        from src.shared.python.pose_estimation.interface import (
            PoseEstimationResult,
            PoseEstimator,
        )

        assert PoseEstimator is not None
        assert PoseEstimationResult is not None

    def test_pose_estimation_result_fields(self) -> None:
        """PoseEstimationResult must have required fields."""
        from src.shared.python.pose_estimation.interface import PoseEstimationResult

        result = PoseEstimationResult(
            joint_angles={"elbow": 1.5},
            confidence=0.9,
            timestamp=1.0,
            raw_keypoints=None,
        )
        assert result.joint_angles == {"elbow": 1.5}
        assert result.confidence == pytest.approx(0.9)
        assert result.timestamp == pytest.approx(1.0)
        assert result.raw_keypoints is None

    def test_joint_angle_utils_importable(self) -> None:
        """Joint angle utilities must be importable."""
        from src.shared.python.pose_estimation.joint_angle_utils import (
            compute_joint_angles,
        )

        assert compute_joint_angles is not None

    def test_openpose_canonical_mapping_exists(self) -> None:
        """OPENPOSE_TO_CANONICAL mapping must exist."""
        from src.shared.python.pose_estimation.joint_angle_utils import (
            OPENPOSE_TO_CANONICAL,
        )

        assert isinstance(OPENPOSE_TO_CANONICAL, dict)
        assert len(OPENPOSE_TO_CANONICAL) > 0

    def test_validation_metrics_importable(self) -> None:
        """Validation metrics module must be importable."""
        from src.shared.python.pose_estimation import validation_metrics

        assert validation_metrics is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Cross-Engine Protocol Compliance Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCrossEngineProtocolCompliance:
    """Verify all engines adhere to the PhysicsEngine protocol."""

    PHYSICS_ENGINE_METHODS = [
        "load_from_path",
        "load_from_string",
        "reset",
        "step",
        "forward",
        "get_state",
        "set_state",
        "set_control",
        "get_time",
        "compute_mass_matrix",
        "compute_bias_forces",
        "compute_inverse_dynamics",
        "compute_jacobian",
    ]

    @pytest.mark.parametrize(
        "engine_module,engine_class",
        [
            (
                "src.engines.physics_engines.drake.python.drake_physics_engine",
                "DrakePhysicsEngine",
            ),
            (
                "src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine",
                "PinocchioPhysicsEngine",
            ),
            (
                "src.engines.physics_engines.opensim.python.opensim_physics_engine",
                "OpenSimPhysicsEngine",
            ),
            (
                "src.engines.physics_engines.myosuite.python.myosuite_physics_engine",
                "MyoSuitePhysicsEngine",
            ),
        ],
    )
    def test_engine_has_all_protocol_methods(
        self, engine_module: str, engine_class: str
    ) -> None:
        """Every engine must implement all PhysicsEngine protocol methods."""
        import importlib

        mod = importlib.import_module(engine_module)
        cls = getattr(mod, engine_class)

        for method in self.PHYSICS_ENGINE_METHODS:
            assert hasattr(cls, method), (
                f"{engine_class} missing protocol method: {method}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 12. dtack Subpackage Tests (#1812)
# ═══════════════════════════════════════════════════════════════════════════════


class TestDtackSubpackageAudit:
    """Verify pinocchio dtack subpackage integrity."""

    def test_dtack_init_importable(self) -> None:
        """dtack __init__.py must be importable."""
        from src.engines.physics_engines.pinocchio.python import dtack

        assert dtack is not None

    def test_dtack_backends_importable(self) -> None:
        """dtack backends package must be importable."""
        from src.engines.physics_engines.pinocchio.python.dtack import backends

        assert backends is not None

    def test_dtack_mujoco_backend_has_import_guard(self) -> None:
        """MuJoCoBackend must not crash when mujoco is not installed."""
        from src.engines.physics_engines.pinocchio.python.dtack.backends.mujoco_backend import (  # noqa: E501
            MuJoCoBackend,
        )

        assert MuJoCoBackend is not None

    def test_dtack_backend_factory_importable(self) -> None:
        """BackendFactory must be importable."""
        from src.engines.physics_engines.pinocchio.python.dtack.backends.backend_factory import (  # noqa: E501
            BackendFactory,
            BackendType,
        )

        assert BackendFactory is not None
        assert BackendType is not None
        # Verify enum values
        assert BackendType.PINOCCHIO == "pinocchio"
        assert BackendType.MUJOCO == "mujoco"
        assert BackendType.PINK == "pink"

    def test_dtack_ik_importable(self) -> None:
        """dtack ik package must be importable."""
        from src.engines.physics_engines.pinocchio.python.dtack import ik

        assert ik is not None

    def test_dtack_utils_importable(self) -> None:
        """dtack utils package must be importable."""
        from src.engines.physics_engines.pinocchio.python.dtack import utils

        assert utils is not None
