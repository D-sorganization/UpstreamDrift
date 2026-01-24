"""Tests for dashboard enhancements."""

import unittest

import numpy as np
import pytest

from src.shared.python.engine_availability import PYQT6_AVAILABLE, skip_if_unavailable

if PYQT6_AVAILABLE:
    from PyQt6 import QtWidgets
else:
    QtWidgets = None  # type: ignore[misc, assignment]

pytestmark = skip_if_unavailable("pyqt6")

if PYQT6_AVAILABLE:
    from src.shared.python.dashboard.recorder import GenericPhysicsRecorder
    from src.shared.python.dashboard.widgets import LivePlotWidget
    from src.shared.python.dashboard.window import UnifiedDashboardWindow
    from src.shared.python.interfaces import PhysicsEngine

    # Mock Physics Engine - only defined when PyQt6 is available
    class MockPhysicsEngine(PhysicsEngine):
        def __init__(self):
            self._time = 0.0
            self._q = np.zeros(10)
            self._v = np.zeros(10)

        @property
        def model_name(self) -> str:
            return "MockModel"

        def load_from_path(self, path: str) -> None:
            pass

        def load_from_string(self, content: str, extension: str | None = None) -> None:
            pass

        def reset(self) -> None:
            self._time = 0.0
            self._q = np.zeros(10)
            self._v = np.zeros(10)

        def step(self, dt: float | None = None) -> None:
            self._time += dt or 0.01
            self._q += 0.01
            self._v += 0.01

        def forward(self) -> None:
            pass

        def get_state(self):
            return self._q, self._v

        def set_state(self, q, v) -> None:
            self._q = q
            self._v = v

        def set_control(self, u) -> None:
            pass

        def get_time(self) -> float:
            return self._time

        def compute_mass_matrix(self):
            return np.eye(10)

        def compute_bias_forces(self):
            return np.zeros(10)

        def compute_gravity_forces(self):
            return np.zeros(10)

        def compute_inverse_dynamics(self, qacc):
            return np.zeros(10)

        def compute_jacobian(self, body_name):
            return None

        def compute_drift_acceleration(self):
            return np.zeros(10)

        def compute_control_acceleration(self, tau):
            return np.zeros(10)

        def compute_ztcf(self, q, v):
            return np.zeros(10)

        def compute_zvcf(self, q):
            return np.zeros(10)

    class TestDashboardEnhancements(unittest.TestCase):
        app: "QtWidgets.QApplication | None" = None

        @classmethod
        def setUpClass(cls):
            # Create a QApplication instance for widgets
            if not QtWidgets.QApplication.instance():
                cls.app = QtWidgets.QApplication([])
            else:
                # Cast to QApplication since we know we created QApplication above
                existing = QtWidgets.QApplication.instance()
                cls.app = (
                    existing if isinstance(existing, QtWidgets.QApplication) else None
                )

        def setUp(self):
            self.engine = MockPhysicsEngine()
            self.recorder = GenericPhysicsRecorder(self.engine)
            self.recorder.start()

            # Populate recorder with some dummy data
            for _ in range(10):
                self.engine.step()
                self.recorder.record_step()

            # Manually populate induced acceleration for testing
            self.recorder.data["induced_accelerations"][0] = np.random.rand(100, 10)

        def test_live_plot_widget_modes(self):
            """Test LivePlotWidget new modes."""
            widget = LivePlotWidget(self.recorder)

            # Test default mode (All Dimensions)
            widget.update_plot()
            # Should have 10 lines
            self.assertEqual(len(widget.line_objects), 10)

            # Test Single Dimension mode
            widget.mode_combo.setCurrentText("Single Dimension")
            widget.dim_spin.setValue(2)
            widget.update_plot()
            # Should have 1 line (Dim 2)
            self.assertEqual(len(widget.line_objects), 1)
            self.assertEqual(widget.line_objects[0].get_label(), "Dim 2")

            # Test Norm mode
            widget.mode_combo.setCurrentText("Norm")
            widget.update_plot()
            # Should have 1 line (Norm)
            self.assertEqual(len(widget.line_objects), 1)
            self.assertEqual(widget.line_objects[0].get_label(), "Norm")

        def test_live_plot_ground_forces(self):
            """Test plotting Ground Forces."""
            widget = LivePlotWidget(self.recorder)

            # Select Ground Forces
            widget.set_plot_metric("Ground Forces")
            widget.update_plot()

            # Ground forces in recorder is initialized to shape (max_samples, 3)
            # So we expect 3 lines
            self.assertEqual(len(widget.line_objects), 3)

        def test_unified_window_static_plots(self):
            """Test new static plot options in UnifiedDashboardWindow."""
            window = UnifiedDashboardWindow(self.engine)

            # We just want to ensure these don't raise exceptions when called
            # with empty or dummy data.

            new_options = [
                "Joint Power Curves",
                "Impulse Accumulation",
                "Phase Diagram (Joint 0)",
                "Stability Diagram (CoM vs CoP)",
                "CoP Trajectory",
                "GRF Butterfly Diagram",
                "Club Head Trajectory (3D)",
                "Summary Dashboard",
            ]

            for option in new_options:
                with self.subTest(option=option):
                    idx = window.plot_type_combo.findText(option)
                    if idx >= 0:
                        window.plot_type_combo.setCurrentIndex(idx)
                        try:
                            window.refresh_static_plot()
                        except Exception as e:
                            self.fail(f"Plotting '{option}' raised exception: {e}")
                    else:
                        self.fail(f"Option '{option}' not found in combo box")


if __name__ == "__main__":
    unittest.main()
