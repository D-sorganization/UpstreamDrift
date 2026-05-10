"""Tests for drake_visualization_mixin.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.engines.physics_engines.drake.python.src.drake_visualization_mixin import (
    DrakeVisualizationMixin,
    HAS_QT,
)

if not HAS_QT:
    pytest.skip("Skipping PyQt tests since PyQt6 is not available", allow_module_level=True)


class DummyVizGUI(DrakeVisualizationMixin):
    """Dummy class to test DrakeVisualizationMixin."""

    def __init__(self):
        """Initialize mock components."""
        self.meshcat = MagicMock()
        self.plant = MagicMock()
        self.context = MagicMock()
        self.eval_context = MagicMock()
        self.visualizer = MagicMock()
        
        self.chk_mobility = MagicMock()
        self.chk_force_ellip = MagicMock()
        self.chk_induced_vec = MagicMock()
        self.chk_cf_vec = MagicMock()
        self.chk_show_forces = MagicMock()
        self.chk_show_torques = MagicMock()
        
        self.combo_induced_source = MagicMock()
        self.combo_cf_type = MagicMock()
        
        self.lbl_cond = MagicMock()
        self.lbl_rank = MagicMock()
        
        self.manip_analyzer = MagicMock()
        self.manip_checkboxes = {}
        self.recorder = MagicMock()


@pytest.fixture
def dummy_viz() -> DummyVizGUI:
    """Fixture providing a DummyVizGUI instance."""
    return DummyVizGUI()


class TestDrakeVisualizationMixin:
    """Tests for DrakeVisualizationMixin methods."""

    def test_on_visualization_changed(self, dummy_viz: DummyVizGUI) -> None:
        """Test toggling visualization triggers update."""
        dummy_viz._update_visualization = MagicMock()
        dummy_viz._on_visualization_changed()
        dummy_viz._update_visualization.assert_called_once()

    def test_update_visualization(self, dummy_viz: DummyVizGUI) -> None:
        """Test _update_visualization calls sub-methods correctly."""
        dummy_viz._draw_ellipsoids = MagicMock()
        dummy_viz._update_ellipsoids = MagicMock()
        dummy_viz._update_vectors = MagicMock()
        
        # Suppose checkboxes are off
        dummy_viz.chk_mobility.isChecked.return_value = False
        dummy_viz.chk_force_ellip.isChecked.return_value = False
        dummy_viz.chk_induced_vec.isChecked.return_value = False
        dummy_viz.chk_cf_vec.isChecked.return_value = False
        dummy_viz.chk_show_forces.isChecked.return_value = False
        dummy_viz.chk_show_torques.isChecked.return_value = False
        
        dummy_viz._update_visualization()
        
        dummy_viz.visualizer.update_frame_transforms.assert_called_once_with(dummy_viz.context)
        dummy_viz.visualizer.update_com_transforms.assert_called_once_with(dummy_viz.context)
        
        dummy_viz._draw_ellipsoids.assert_called_once()
        dummy_viz._update_ellipsoids.assert_called_once()
        dummy_viz._update_vectors.assert_called_once()
        
        # Since checkboxes are off, it should have deleted overlays
        dummy_viz.meshcat.Delete.assert_any_call("overlays/ellipsoids")
        dummy_viz.meshcat.Delete.assert_any_call("overlays/vectors")

    def test_cleanup_disabled_vector_categories(self, dummy_viz: DummyVizGUI) -> None:
        """Test that disabled categories are deleted."""
        dummy_viz.chk_show_torques.isChecked.return_value = False
        dummy_viz.chk_show_forces.isChecked.return_value = True
        dummy_viz.chk_induced_vec.isChecked.return_value = False
        dummy_viz.chk_cf_vec.isChecked.return_value = True
        
        dummy_viz._cleanup_disabled_vector_categories()
        
        dummy_viz.meshcat.Delete.assert_any_call("overlays/vectors/torques")
        dummy_viz.meshcat.Delete.assert_any_call("overlays/vectors/induced")
        # Should not have called delete on forces or cf

    def test_resolve_induced_accels_named_source(self, dummy_viz: DummyVizGUI) -> None:
        """Test resolving induced accelerations for a named source."""
        analyzer = MagicMock()
        dummy_viz.plant.num_velocities.return_value = 2
        
        # Test 'total' source
        analyzer.compute_components.return_value = {"total": np.array([1.0, 2.0])}
        res = dummy_viz._resolve_induced_accels(analyzer, "total")
        np.testing.assert_array_equal(res, np.array([1.0, 2.0]))
        
        # Test joint name source
        dummy_viz.plant.HasJointNamed.return_value = True
        joint_mock = MagicMock()
        joint_mock.num_velocities.return_value = 1
        joint_mock.velocity_start.return_value = 1
        dummy_viz.plant.GetJointByName.return_value = joint_mock
        
        analyzer.compute_specific_control.side_effect = lambda ctx, tau: tau * 3.0
        
        res = dummy_viz._resolve_induced_accels(analyzer, "j1")
        np.testing.assert_array_equal(res, np.array([0.0, 3.0]))

    def test_draw_accel_vectors(self, dummy_viz: DummyVizGUI) -> None:
        """Test drawing vectors."""
        dummy_viz.plant.num_joints.return_value = 1
        
        joint = MagicMock()
        joint.num_velocities.return_value = 1
        joint.velocity_start.return_value = 0
        joint.name.return_value = "j1"
        joint.revolute_axis.return_value = np.array([0, 0, 1])
        dummy_viz.plant.get_joint.return_value = joint
        
        pose_mock = MagicMock()
        pose_mock.translation.return_value = np.array([0, 0, 0])
        pose_mock.rotation().multiply.return_value = np.array([0, 0, 1])
        dummy_viz.plant.EvalBodyPoseInWorld.return_value = pose_mock
        
        color = MagicMock()
        values = np.array([2.0])
        
        dummy_viz._draw_accel_vectors(values, "test", color, scale=1.0)
        
        # Should have drawn line segment from [0,0,0] to [0,0,2]
        dummy_viz.meshcat.SetLineSegments.assert_called_once()
        path = dummy_viz.meshcat.SetLineSegments.call_args[0][0]
        assert "test/j1" in path

    @patch("src.engines.physics_engines.drake.python.src.drake_visualization_mixin.HAS_MATPLOTLIB", False)
    def test_show_induced_acceleration_plot_no_matplotlib(self, dummy_viz: DummyVizGUI) -> None:
        """Test plotting aborts if matplotlib is not available."""
        with patch("src.engines.physics_engines.drake.python.src.drake_visualization_mixin.QtWidgets.QMessageBox.warning") as mock_warn:
            dummy_viz._show_induced_acceleration_plot()
            mock_warn.assert_called_once_with(dummy_viz, "Error", "Matplotlib not found.")

    @patch("src.engines.physics_engines.drake.python.src.drake_visualization_mixin.HAS_MATPLOTLIB", False)
    def test_show_counterfactuals_plot_no_matplotlib(self, dummy_viz: DummyVizGUI) -> None:
        """Test plotting aborts if matplotlib is not available."""
        with patch("src.engines.physics_engines.drake.python.src.drake_visualization_mixin.QtWidgets.QMessageBox.warning") as mock_warn:
            dummy_viz._show_counterfactuals_plot()
            mock_warn.assert_called_once_with(dummy_viz, "Error", "Matplotlib not found.")
