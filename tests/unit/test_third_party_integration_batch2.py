"""Additional third-party integration tests — Batch 2.

Tests for remaining issues:
- #1810: Drake engine specific verification
- #1813: OpenSim engine specific verification
- #1814: MyoSuite gymnasium compatibility
- #1817: Video Pose Pipeline end-to-end
- #1818: Engine availability additional checks

These tests supplement test_third_party_integration_audit.py.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.engine_core.engine_availability import (
    CV2_AVAILABLE,
    DRAKE_AVAILABLE,
    MEDIAPIPE_AVAILABLE,
    MUJOCO_AVAILABLE,
    MYOSUITE_AVAILABLE,
    OPENSIM_AVAILABLE,
    skip_if_unavailable,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Drake Specific Tests (#1810)
# ═══════════════════════════════════════════════════════════════════════════════


class TestDrakeSpecificVerification:
    """Deep verification of Drake integration correctness."""

    @skip_if_unavailable("drake")
    def test_drake_engine_load_from_string_sdf(self) -> None:
        """DrakePhysicsEngine must accept SDF string."""
        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()
        # load_from_string should exist and be callable
        assert callable(engine.load_from_string)

    def test_drake_engine_importable_without_pydrake(self) -> None:
        """drake_physics_engine module must be importable even without pydrake."""
        from src.engines.physics_engines.drake.python import drake_physics_engine

        assert drake_physics_engine is not None

    @skip_if_unavailable("drake")
    def test_drake_engine_get_state_returns_tuple(self) -> None:
        """get_state must return a (qpos, qvel) tuple."""
        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()
        state = engine.get_state()
        assert isinstance(state, tuple)
        assert len(state) == 2  # noqa: PLR2004

    @skip_if_unavailable("drake")
    def test_drake_engine_get_time_returns_float(self) -> None:
        """get_time must return a float."""
        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()
        time_val = engine.get_time()
        assert isinstance(time_val, float)
        assert time_val >= 0.0

    def test_drake_engine_has_contact_forces(self) -> None:
        """Drake must implement compute_contact_forces."""
        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        assert hasattr(DrakePhysicsEngine, "compute_contact_forces")
        assert callable(DrakePhysicsEngine.compute_contact_forces)

    def test_drake_engine_has_drift_acceleration(self) -> None:
        """Drake must implement compute_drift_acceleration."""
        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        assert hasattr(DrakePhysicsEngine, "compute_drift_acceleration")
        assert callable(DrakePhysicsEngine.compute_drift_acceleration)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. OpenSim Specific Tests (#1813)
# ═══════════════════════════════════════════════════════════════════════════════


class TestOpenSimSpecificVerification:
    """Deep verification of OpenSim integration correctness."""

    def test_opensim_muscle_analysis_module(self) -> None:
        """OpenSim muscle_analysis module must be importable."""
        from src.engines.physics_engines.opensim.python import muscle_analysis

        assert muscle_analysis is not None

    def test_opensim_engine_has_load_from_string(self) -> None:
        """OpenSimPhysicsEngine must implement load_from_string."""
        from src.engines.physics_engines.opensim.python.opensim_physics_engine import (  # noqa: E501
            OpenSimPhysicsEngine,
        )

        engine = OpenSimPhysicsEngine()
        assert callable(engine.load_from_string)

    @skip_if_unavailable("opensim")
    def test_opensim_engine_get_state_returns_tuple(self) -> None:
        """get_state must return a (qpos, qvel) tuple with correct types."""
        from src.engines.physics_engines.opensim.python.opensim_physics_engine import (  # noqa: E501
            OpenSimPhysicsEngine,
        )

        engine = OpenSimPhysicsEngine()
        state = engine.get_state()
        assert isinstance(state, tuple)
        assert len(state) == 2  # noqa: PLR2004

    @skip_if_unavailable("opensim")
    def test_opensim_engine_get_time_returns_float(self) -> None:
        """get_time must return a float."""
        from src.engines.physics_engines.opensim.python.opensim_physics_engine import (  # noqa: E501
            OpenSimPhysicsEngine,
        )

        engine = OpenSimPhysicsEngine()
        time_val = engine.get_time()
        assert isinstance(time_val, float)

    def test_opensim_engine_has_drift_acceleration(self) -> None:
        """OpenSim must implement compute_drift_acceleration."""
        from src.engines.physics_engines.opensim.python.opensim_physics_engine import (  # noqa: E501
            OpenSimPhysicsEngine,
        )

        assert hasattr(OpenSimPhysicsEngine, "compute_drift_acceleration")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MyoSuite Gymnasium Compatibility Tests (#1814)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMyoSuiteGymnasiumCompatibility:
    """Verify MyoSuite uses gymnasium with gym fallback."""

    def test_myosuite_prefers_gymnasium(self) -> None:
        """MyoSuite engine must prefer gymnasium over legacy gym."""
        source_code = inspect.getsource(
            __import__(
                "src.engines.physics_engines.myosuite.python.myosuite_physics_engine",
                fromlist=["MyoSuitePhysicsEngine"],
            )
        )
        assert "import gymnasium" in source_code, (
            "MyoSuite engine should prefer 'import gymnasium' over legacy 'import gym'"
        )

    def test_myosuite_has_gym_fallback(self) -> None:
        """MyoSuite must fall back to legacy gym if gymnasium is missing."""
        source_code = inspect.getsource(
            __import__(
                "src.engines.physics_engines.myosuite.python.myosuite_physics_engine",
                fromlist=["MyoSuitePhysicsEngine"],
            )
        )
        assert "import gym" in source_code, (
            "MyoSuite engine must have a fallback to legacy 'import gym'"
        )

    def test_myosuite_engine_load_from_string_raises(self) -> None:
        """load_from_string must raise (Gym envs don't support string loading)."""
        from src.engines.physics_engines.myosuite.python.myosuite_physics_engine import (  # noqa: E501
            MyoSuitePhysicsEngine,
        )

        engine = MyoSuitePhysicsEngine()
        with pytest.raises(RuntimeError, match="does not support"):
            engine.load_from_string("<xml/>")

    def test_myosuite_engine_set_control_stores_action(self) -> None:
        """set_control must store the action for the next step."""
        from src.engines.physics_engines.myosuite.python.myosuite_physics_engine import (  # noqa: E501
            MyoSuitePhysicsEngine,
        )

        engine = MyoSuitePhysicsEngine()
        u = np.array([0.5, 0.3, 0.1])
        engine.set_control(u)
        assert hasattr(engine, "_last_action")
        np.testing.assert_array_equal(engine._last_action, u)

    def test_myosuite_engine_model_name_default(self) -> None:
        """model_name must return a default when no model loaded."""
        from src.engines.physics_engines.myosuite.python.myosuite_physics_engine import (  # noqa: E501
            MyoSuitePhysicsEngine,
        )

        engine = MyoSuitePhysicsEngine()
        assert engine.model_name == "MyoSuite_NoModel"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Video Pose Pipeline Tests (#1817)
# ═══════════════════════════════════════════════════════════════════════════════


class TestVideoPosePipelineAudit:
    """Verify video pose pipeline integration."""

    pytestmark = pytest.mark.skipif(
        not CV2_AVAILABLE,
        reason="cv2 not installed",
    )

    def test_pipeline_importable(self) -> None:
        """VideoPosePipeline must be importable."""
        from src.shared.python.gui_pkg.video_pose_pipeline import (
            VideoPosePipeline,
        )

        assert VideoPosePipeline is not None

    def test_config_importable(self) -> None:
        """VideoProcessingConfig must be importable."""
        from src.shared.python.gui_pkg.video_pose_pipeline import (
            VideoProcessingConfig,
        )

        assert VideoProcessingConfig is not None

    def test_result_importable(self) -> None:
        """VideoProcessingResult must be importable."""
        from src.shared.python.gui_pkg.video_pose_pipeline import (
            VideoProcessingResult,
        )

        assert VideoProcessingResult is not None

    def test_config_defaults(self) -> None:
        """VideoProcessingConfig must have sane defaults."""
        from src.shared.python.gui_pkg.video_pose_pipeline import (
            VideoProcessingConfig,
        )

        config = VideoProcessingConfig()
        assert config.estimator_type == "mediapipe"
        assert config.min_confidence == pytest.approx(0.5)
        assert config.enable_temporal_smoothing is True
        assert config.output_format == "json"

    def test_pipeline_initialization(self) -> None:
        """VideoPosePipeline must initialize with default config."""
        from src.shared.python.gui_pkg.video_pose_pipeline import (
            VideoPosePipeline,
        )

        if not MEDIAPIPE_AVAILABLE:
            # Pipeline calls _load_estimator in __init__ which requires MediaPipe
            with pytest.raises(ImportError):
                VideoPosePipeline()
            return

        pipeline = VideoPosePipeline()
        assert pipeline is not None
        assert pipeline.config is not None

    def test_pipeline_custom_config(self) -> None:
        """VideoProcessingConfig must accept custom parameters."""
        from src.shared.python.gui_pkg.video_pose_pipeline import (
            VideoProcessingConfig,
        )

        config = VideoProcessingConfig(
            estimator_type="openpose",
            min_confidence=0.7,
            enable_temporal_smoothing=False,
            output_format="csv",
        )
        assert config.estimator_type == "openpose"
        assert config.min_confidence == pytest.approx(0.7)
        assert config.output_format == "csv"
        assert config.enable_temporal_smoothing is False

    @skip_if_unavailable("mediapipe")
    def test_pipeline_process_video_rejects_missing_file(self) -> None:
        """process_video must raise for missing video file."""
        from src.shared.python.gui_pkg.video_pose_pipeline import (
            VideoPosePipeline,
        )

        pipeline = VideoPosePipeline()
        with pytest.raises((FileNotFoundError, Exception)):
            pipeline.process_video(Path("/tmp/nonexistent_video.mp4"))

    @skip_if_unavailable("mediapipe")
    def test_pipeline_process_batch_rejects_empty_list(self) -> None:
        """process_batch must handle empty video list gracefully."""
        from src.shared.python.gui_pkg.video_pose_pipeline import (
            VideoPosePipeline,
        )

        pipeline = VideoPosePipeline()
        results = pipeline.process_batch([], Path("/tmp"))
        assert isinstance(results, list)
        assert len(results) == 0

    @skip_if_unavailable("mediapipe")
    def test_pipeline_has_fit_to_model(self) -> None:
        """VideoPosePipeline must have fit_to_model method."""
        from src.shared.python.gui_pkg.video_pose_pipeline import (
            VideoPosePipeline,
        )

        pipeline = VideoPosePipeline()
        assert hasattr(pipeline, "fit_to_model")
        assert callable(pipeline.fit_to_model)

    @skip_if_unavailable("mediapipe")
    def test_pipeline_filter_quality_callable(self) -> None:
        """_filter_by_quality must be callable."""
        from src.shared.python.gui_pkg.video_pose_pipeline import (
            VideoPosePipeline,
        )

        pipeline = VideoPosePipeline()
        assert hasattr(pipeline, "_filter_by_quality")
        assert callable(pipeline._filter_by_quality)

    @skip_if_unavailable("mediapipe")
    def test_pipeline_filter_quality_empty_input(self) -> None:
        """_filter_by_quality must handle empty list."""
        from src.shared.python.gui_pkg.video_pose_pipeline import (
            VideoPosePipeline,
        )

        pipeline = VideoPosePipeline()
        result = pipeline._filter_by_quality([])
        assert isinstance(result, list)
        assert len(result) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Engine Availability Deep Tests (#1818)
# ═══════════════════════════════════════════════════════════════════════════════


class TestEngineAvailabilityDeep:
    """Deep verification of engine availability infrastructure."""

    def test_all_engine_flags_are_boolean(self) -> None:
        """Every flag in _ENGINE_FLAGS must be a boolean."""
        from src.shared.python.engine_core.engine_availability import (
            _ENGINE_FLAGS,
        )

        for name, flag in _ENGINE_FLAGS.items():
            assert isinstance(flag, bool), (
                f"_ENGINE_FLAGS['{name}'] is {type(flag).__name__}, expected bool"
            )

    def test_engine_availability_consistent_with_import(self) -> None:
        """DRAKE_AVAILABLE must be consistent with pydrake.all import."""
        try:
            import pydrake.all  # noqa: F401

            assert DRAKE_AVAILABLE
        except Exception as e:  # noqa: BLE001
            assert not DRAKE_AVAILABLE

    def test_mujoco_availability_consistent_with_import(self) -> None:
        """MUJOCO_AVAILABLE must be consistent with mujoco import."""
        try:
            import mujoco  # noqa: F401

            assert MUJOCO_AVAILABLE
        except ImportError:
            assert not MUJOCO_AVAILABLE

    def test_mediapipe_availability_consistent_with_import(self) -> None:
        """MEDIAPIPE_AVAILABLE must be consistent with mediapipe import."""
        try:
            import mediapipe  # noqa: F401

            assert MEDIAPIPE_AVAILABLE
        except ImportError:
            assert not MEDIAPIPE_AVAILABLE

    def test_opensim_availability_consistent_with_import(self) -> None:
        """OPENSIM_AVAILABLE must be consistent with opensim import."""
        try:
            import opensim  # noqa: F401

            assert OPENSIM_AVAILABLE
        except ImportError:
            assert not OPENSIM_AVAILABLE

    def test_myosuite_availability_consistent_with_import(self) -> None:
        """MYOSUITE_AVAILABLE must be consistent with myosuite import."""
        try:
            import myosuite  # noqa: F401

            assert MYOSUITE_AVAILABLE
        except ImportError:
            assert not MYOSUITE_AVAILABLE
