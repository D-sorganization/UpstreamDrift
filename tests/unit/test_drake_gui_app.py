"""Unit tests for Drake GUI App and its mixin decomposition.

Tests the DrakeInducedAccelerationAnalyzer, DrakeRecorder,
and the individual mixin modules (UI, Sim, Viz, Analysis).
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Check PyQt6 availability without importing engine_availability
# (which triggers a torch import that may fail on some platforms)
try:
    import PyQt6  # noqa: F401

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

skip_if_no_pyqt6 = pytest.mark.skipif(not HAS_PYQT6, reason="PyQt6 not installed")

# Drake engine module paths that may get imported and must be cleaned up
_DRAKE_ENGINE_MODULES = [
    "src.engines.physics_engines.drake",
    "src.engines.physics_engines.drake.python",
    "src.engines.physics_engines.drake.python.src",
    "src.engines.physics_engines.drake.python.src.drake_gui_app",
    "src.engines.physics_engines.drake.python.src.drake_gui_ui",
    "src.engines.physics_engines.drake.python.src.drake_gui_sim",
    "src.engines.physics_engines.drake.python.src.drake_gui_viz",
    "src.engines.physics_engines.drake.python.src.drake_gui_analysis",
    "src.engines.physics_engines.drake.python.src.drake_analysis",
]


@pytest.fixture(autouse=True, scope="function")
def _mock_pydrake():
    """Provide mock pydrake modules only during test execution.

    Also cleanup drake engine modules to prevent pollution of test_drake_wrapper.py.
    When drake_gui_app is imported, it brings in the parent package
    src.engines.physics_engines.drake.python into sys.modules. This causes
    test_drake_wrapper.py to fail when it tries to patch
    src.engines.physics_engines.drake.python.drake_physics_engine, because the
    parent package exists but drake_physics_engine was never imported.
    """
    # Save existing drake modules so we can restore them
    saved_modules = {}
    for module_name in _DRAKE_ENGINE_MODULES:
        if module_name in sys.modules:
            saved_modules[module_name] = sys.modules[module_name]

    # Create fresh mocks for each test session to prevent pollution
    pydrake_mocks = {
        "pydrake": MagicMock(),
        "pydrake.all": MagicMock(),
        "pydrake.multibody": MagicMock(),
        "pydrake.multibody.plant": MagicMock(),
        "pydrake.multibody.tree": MagicMock(),
        # Mock torch and cv2 to prevent DLL loading errors on Windows
        # (drake pkg __init__ → logger_utils → reproducibility → engine_availability → torch/cv2)
        "torch": MagicMock(),
        "cv2": MagicMock(),
        "cv2.dnn": MagicMock(),
        "cv2.typing": MagicMock(),
    }
    with patch.dict("sys.modules", pydrake_mocks):
        yield

    # Clean up drake engine modules to prevent pollution
    for module_name in _DRAKE_ENGINE_MODULES:
        if module_name in sys.modules:
            del sys.modules[module_name]

    # Restore saved modules
    for module_name, module in saved_modules.items():
        sys.modules[module_name] = module


# ==================================================================
# DrakeInducedAccelerationAnalyzer Tests
# ==================================================================


@skip_if_no_pyqt6
class TestDrakeInducedAccelerationAnalyzer:
    """Tests for the DrakeInducedAccelerationAnalyzer class."""

    def test_compute_specific_control_none_plant(self) -> None:
        """Test compute_specific_control when plant is None returns empty."""
        from src.engines.physics_engines.drake.python.src.drake_analysis import (
            DrakeInducedAccelerationAnalyzer,
        )

        analyzer = DrakeInducedAccelerationAnalyzer(None)
        result = analyzer.compute_specific_control(MagicMock(), np.array([1.0]))
        assert len(result) == 0

    def test_compute_specific_control_identity_mass(self) -> None:
        """Test compute_specific_control with identity mass matrix."""
        from src.engines.physics_engines.drake.python.src.drake_analysis import (
            DrakeInducedAccelerationAnalyzer,
        )

        plant = MagicMock()
        context = MagicMock()
        analyzer = DrakeInducedAccelerationAnalyzer(plant)

        # M = I => a = tau
        plant.CalcMassMatrix.return_value = np.eye(2)
        tau = np.array([1.0, 2.0])
        result = analyzer.compute_specific_control(context, tau)

        np.testing.assert_array_almost_equal(result, np.array([1.0, 2.0]))
        plant.CalcMassMatrix.assert_called_with(context)

    def test_compute_specific_control_scaled_mass(self) -> None:
        """Test compute_specific_control with a scaled mass matrix."""
        from src.engines.physics_engines.drake.python.src.drake_analysis import (
            DrakeInducedAccelerationAnalyzer,
        )

        plant = MagicMock()
        context = MagicMock()
        analyzer = DrakeInducedAccelerationAnalyzer(plant)

        # M = 2*I => a = tau/2
        plant.CalcMassMatrix.return_value = 2.0 * np.eye(3)
        tau = np.array([4.0, 6.0, 8.0])
        result = analyzer.compute_specific_control(context, tau)

        np.testing.assert_array_almost_equal(result, np.array([2.0, 3.0, 4.0]))

    def test_compute_components_none_plant(self) -> None:
        """Test compute_components returns empty arrays for None plant."""
        from src.engines.physics_engines.drake.python.src.drake_analysis import (
            DrakeInducedAccelerationAnalyzer,
        )

        analyzer = DrakeInducedAccelerationAnalyzer(None)
        result = analyzer.compute_components(MagicMock())
        assert "gravity" in result
        assert "velocity" in result
        assert "total" in result
        assert len(result["gravity"]) == 0
        assert len(result["velocity"]) == 0
        assert len(result["total"]) == 0

    def test_compute_components_valid_plant(self) -> None:
        """Test compute_components with valid plant and identity mass."""
        from src.engines.physics_engines.drake.python.src.drake_analysis import (
            DrakeInducedAccelerationAnalyzer,
        )

        plant = MagicMock()
        context = MagicMock()
        analyzer = DrakeInducedAccelerationAnalyzer(plant)

        plant.CalcMassMatrix.return_value = np.eye(2)
        plant.CalcGravityGeneralizedForces.return_value = np.array([0.0, -9.81])
        plant.CalcBiasTerm.return_value = np.array([0.1, 0.2])

        result = analyzer.compute_components(context)

        # gravity accel = M^-1 @ tau_g = [0, -9.81]
        np.testing.assert_array_almost_equal(result["gravity"], np.array([0.0, -9.81]))
        # velocity accel = M^-1 @ (-(bias + tau_g))
        expected_v = -(np.array([0.1, 0.2]) + np.array([0.0, -9.81]))
        np.testing.assert_array_almost_equal(result["velocity"], expected_v)
        # total = gravity + velocity
        np.testing.assert_array_almost_equal(
            result["total"], result["gravity"] + result["velocity"]
        )

    def test_compute_counterfactuals_none_plant(self) -> None:
        """Test compute_counterfactuals returns empty dict for None plant."""
        from src.engines.physics_engines.drake.python.src.drake_analysis import (
            DrakeInducedAccelerationAnalyzer,
        )

        analyzer = DrakeInducedAccelerationAnalyzer(None)
        result = analyzer.compute_counterfactuals(MagicMock())
        assert result == {}

    def test_compute_counterfactuals_valid_plant(self) -> None:
        """Test compute_counterfactuals returns ZTCF and ZVCF data."""
        from src.engines.physics_engines.drake.python.src.drake_analysis import (
            DrakeInducedAccelerationAnalyzer,
        )

        plant = MagicMock()
        context = MagicMock()
        analyzer = DrakeInducedAccelerationAnalyzer(plant)

        plant.CalcMassMatrix.return_value = np.eye(2)
        plant.CalcBiasTerm.return_value = np.array([0.5, 1.0])
        plant.CalcGravityGeneralizedForces.return_value = np.array([0.0, -9.81])

        result = analyzer.compute_counterfactuals(context)

        assert "ztcf_accel" in result
        assert "zvcf_torque" in result

        # ztcf_accel = M^-1 @ (-bias)
        np.testing.assert_array_almost_equal(
            result["ztcf_accel"], np.array([-0.5, -1.0])
        )
        # zvcf_torque = -tau_g
        np.testing.assert_array_almost_equal(
            result["zvcf_torque"], np.array([0.0, 9.81])
        )

    def test_compute_specific_control_singular_mass_uses_pinv(self) -> None:
        """Test pinv fallback for singular mass matrix."""
        from src.engines.physics_engines.drake.python.src.drake_analysis import (
            DrakeInducedAccelerationAnalyzer,
        )

        plant = MagicMock()
        context = MagicMock()
        analyzer = DrakeInducedAccelerationAnalyzer(plant)

        # Singular matrix - rank 1
        plant.CalcMassMatrix.return_value = np.array([[1.0, 0.0], [0.0, 0.0]])
        tau = np.array([2.0, 0.0])
        result = analyzer.compute_specific_control(context, tau)

        # Should use pinv and not crash
        assert result.shape == (2,)
        np.testing.assert_almost_equal(result[0], 2.0)


# ==================================================================
# DrakeRecorder Tests
# ==================================================================


@skip_if_no_pyqt6
class TestDrakeRecorder:
    """Tests for the DrakeRecorder class."""

    def test_init_state(self) -> None:
        """Test recorder starts in a clean state."""
        from src.engines.physics_engines.drake.python.src.drake_analysis import (
            DrakeRecorder,
        )

        rec = DrakeRecorder()
        assert rec.times == []
        assert rec.q_history == []
        assert rec.v_history == []
        assert rec.is_recording is False

    def test_start_stop(self) -> None:
        """Test start/stop recording toggles state."""
        from src.engines.physics_engines.drake.python.src.drake_analysis import (
            DrakeRecorder,
        )

        rec = DrakeRecorder()
        rec.start()
        assert rec.is_recording is True
        rec.stop()
        assert rec.is_recording is False

    def test_record_when_not_recording(self) -> None:
        """Test that record() is a no-op when not recording."""
        from src.engines.physics_engines.drake.python.src.drake_analysis import (
            DrakeRecorder,
        )

        rec = DrakeRecorder()
        rec.record(0.0, np.array([1.0]), np.array([2.0]))
        assert len(rec.times) == 0

    def test_record_when_recording(self) -> None:
        """Test that record() captures data when recording."""
        from src.engines.physics_engines.drake.python.src.drake_analysis import (
            DrakeRecorder,
        )

        rec = DrakeRecorder()
        rec.start()
        rec.record(0.0, np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        rec.record(0.001, np.array([1.1, 2.1]), np.array([3.1, 4.1]))

        assert len(rec.times) == 2
        assert len(rec.q_history) == 2
        np.testing.assert_array_almost_equal(rec.q_history[0], [1.0, 2.0])
        np.testing.assert_array_almost_equal(rec.v_history[1], [3.1, 4.1])

    def test_record_with_optional_data(self) -> None:
        """Test record() with club position and COM data."""
        from src.engines.physics_engines.drake.python.src.drake_analysis import (
            DrakeRecorder,
        )

        rec = DrakeRecorder()
        rec.start()
        rec.record(
            0.0,
            np.array([1.0]),
            np.array([2.0]),
            club_pos=np.array([0.1, 0.2, 0.3]),
            com_pos=np.array([0.4, 0.5, 0.6]),
            angular_momentum=np.array([0.7, 0.8, 0.9]),
        )

        np.testing.assert_array_almost_equal(
            rec.club_head_pos_history[0], [0.1, 0.2, 0.3]
        )
        np.testing.assert_array_almost_equal(
            rec.com_position_history[0], [0.4, 0.5, 0.6]
        )
        np.testing.assert_array_almost_equal(
            rec.angular_momentum_history[0], [0.7, 0.8, 0.9]
        )

    def test_record_without_optional_data_uses_zeros(self) -> None:
        """Test record() fills zeros when optional data is None."""
        from src.engines.physics_engines.drake.python.src.drake_analysis import (
            DrakeRecorder,
        )

        rec = DrakeRecorder()
        rec.start()
        rec.record(0.0, np.array([1.0]), np.array([2.0]))

        np.testing.assert_array_almost_equal(rec.club_head_pos_history[0], [0, 0, 0])
        np.testing.assert_array_almost_equal(rec.com_position_history[0], [0, 0, 0])
        np.testing.assert_array_almost_equal(rec.angular_momentum_history[0], [0, 0, 0])

    def test_reset_clears_data(self) -> None:
        """Test reset() clears all recorded data."""
        from src.engines.physics_engines.drake.python.src.drake_analysis import (
            DrakeRecorder,
        )

        rec = DrakeRecorder()
        rec.start()
        rec.record(0.0, np.array([1.0]), np.array([2.0]))
        assert len(rec.times) == 1

        rec.reset()
        assert len(rec.times) == 0
        assert len(rec.q_history) == 0
        assert rec.is_recording is False

    def test_get_time_series_joint_positions(self) -> None:
        """Test get_time_series returns joint position data."""
        from src.engines.physics_engines.drake.python.src.drake_analysis import (
            DrakeRecorder,
        )

        rec = DrakeRecorder()
        rec.start()
        rec.record(0.0, np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        rec.record(0.001, np.array([1.1, 2.1]), np.array([3.1, 4.1]))

        times, data = rec.get_time_series("joint_positions")
        assert len(times) == 2
        np.testing.assert_array_almost_equal(times, [0.0, 0.001])
        assert data.shape == (2, 2)

    def test_get_time_series_unknown_field(self) -> None:
        """Test get_time_series returns empty for unknown field."""
        from src.engines.physics_engines.drake.python.src.drake_analysis import (
            DrakeRecorder,
        )

        rec = DrakeRecorder()
        rec.start()
        rec.record(0.0, np.array([1.0]), np.array([2.0]))
        times, data = rec.get_time_series("nonexistent_field")
        assert len(data) == 0

    def test_export_to_dict(self) -> None:
        """Test export_to_dict returns complete recorded data."""
        from src.engines.physics_engines.drake.python.src.drake_analysis import (
            DrakeRecorder,
        )

        rec = DrakeRecorder()
        rec.start()
        rec.record(0.0, np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        rec.record(0.001, np.array([1.1, 2.1]), np.array([3.1, 4.1]))

        data = rec.export_to_dict()
        assert "times" in data
        assert "joint_positions" in data
        assert "joint_velocities" in data
        assert data["joint_positions"].shape == (2, 2)

    def test_set_analysis_config(self) -> None:
        """Test set_analysis_config stores configuration."""
        from src.engines.physics_engines.drake.python.src.drake_analysis import (
            DrakeRecorder,
        )

        rec = DrakeRecorder()
        config = {"live_analysis": True, "window_size": 50}
        rec.set_analysis_config(config)
        assert rec.analysis_config == config

    def test_get_induced_acceleration_series_missing(self) -> None:
        """Test get_induced_acceleration_series returns empty when no data."""
        from src.engines.physics_engines.drake.python.src.drake_analysis import (
            DrakeRecorder,
        )

        rec = DrakeRecorder()
        times, data = rec.get_induced_acceleration_series("gravity")
        assert len(times) == 0
        assert len(data) == 0

    def test_get_counterfactual_series_missing(self) -> None:
        """Test get_counterfactual_series returns empty when no data."""
        from src.engines.physics_engines.drake.python.src.drake_analysis import (
            DrakeRecorder,
        )

        rec = DrakeRecorder()
        times, data = rec.get_counterfactual_series("ztcf_accel")
        assert len(times) == 0
        assert len(data) == 0

    def test_start_resets_existing_data(self) -> None:
        """Test that start() resets any previously recorded data."""
        from src.engines.physics_engines.drake.python.src.drake_analysis import (
            DrakeRecorder,
        )

        rec = DrakeRecorder()
        rec.start()
        rec.record(0.0, np.array([1.0]), np.array([2.0]))
        assert len(rec.times) == 1

        # Starting again should reset
        rec.start()
        assert len(rec.times) == 0
        assert rec.is_recording is True
