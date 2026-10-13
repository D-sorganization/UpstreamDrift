"""Regression test for issue #11934: Drake embed widget offscreen abort.

Verifies:
- Signal blocking on QComboBox during kinematic controls rebuild.
- Presence of chk_show_gravity in UISetupMixin.
- Safe lifecycle handling via _drake_ui_initialized flag.
- Safe handling of empty or whitespace sources in induced acceleration resolution.
- Resilient force glyphs update without AttributeError.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.engines.physics_engines.drake.python.src.drake_gui_ui import (
    HAS_QT,
    UISetupMixin,
)
from src.engines.physics_engines.drake.python.src.drake_gui_viz import (
    VisualizationMixin,
)

pytestmark = pytest.mark.unit

if not HAS_QT:
    pytest.skip(
        "PyQt6 not available; skipping Qt regression tests",
        allow_module_level=True,
    )

from PyQt6 import QtWidgets  # noqa: E402


class DummyDrakeUI(UISetupMixin, VisualizationMixin, QtWidgets.QMainWindow):
    """Minimal test harness combining UI and visualization mixins."""

    def __init__(self) -> None:
        super().__init__()
        self.available_models = [{"name": "Default Model", "path": None}]
        self.operating_mode = "dynamic"
        self.is_running = False
        self.plant = MagicMock()
        self.context = MagicMock()
        self.eval_context = MagicMock()
        self.meshcat = MagicMock()
        self.sliders: dict[int, QtWidgets.QSlider] = {}
        self.spinboxes: dict[int, QtWidgets.QDoubleSpinBox] = {}
        self.recorder = MagicMock()
        self.recorder.get_time_series.return_value = ([], [])
        self.kinematic_layout = QtWidgets.QVBoxLayout()
        self._drake_ui_initialized = False

    def get_joint_names(self) -> list[str]:
        return ["joint_1", "joint_2"]

    def _update_status(self, msg: str) -> None:
        pass

    def _on_model_changed(self, idx: int) -> None:
        pass

    def _on_mode_changed(self, mode: str) -> None:
        pass

    def _toggle_run(self) -> None:
        pass

    def _reset_state(self) -> None:
        pass

    def _reset_simulation(self) -> None:
        pass

    def _on_slider_change(self, val: int, spin: Any, idx: int) -> None:
        pass

    def _on_spin_change(self, val: float, slider: Any, idx: int) -> None:
        pass

    def _toggle_recording(self) -> None:
        pass

    def _reset_recording(self) -> None:
        pass

    def _export_data(self) -> None:
        pass

    def _show_induced_acceleration_plot(self) -> None:
        pass

    def _show_counterfactuals_plot(self) -> None:
        pass

    def _show_swing_plane_analysis(self) -> None:
        pass

    def _show_advanced_plots(self) -> None:
        pass

    def _show_overlay_dialog(self) -> None:
        pass


def test_build_kinematic_controls_blocks_signals_during_clear(
    qapp,
) -> None:  # noqa: ANN001
    """Signal emissions on combo_induced_source must be blocked during clear/repopulate."""
    ui = DummyDrakeUI()
    central = QtWidgets.QWidget()
    ui.setCentralWidget(central)
    layout = QtWidgets.QVBoxLayout(central)

    ui._setup_visualization_panel(layout)

    # Wire a spy to currentIndexChanged
    spy = MagicMock()
    ui.combo_induced_source.currentIndexChanged.connect(spy)

    # Configure mock plant with 1-DOF joints
    mock_joint1 = MagicMock()
    mock_joint1.name.return_value = "joint_1"
    mock_joint1.num_velocities.return_value = 1
    mock_joint1.num_positions.return_value = 1
    mock_joint1.index.return_value = 0
    mock_joint1.position_lower_limits.return_value = [-1.0]
    mock_joint1.position_upper_limits.return_value = [1.0]

    ui.plant.num_joints.return_value = 1
    ui.plant.get_joint.return_value = mock_joint1

    # Call _build_kinematic_controls
    ui._build_kinematic_controls()

    # The spy should NOT be called with -1 from clear() because signals were blocked
    for call_args in spy.call_args_list:
        assert call_args[0][0] != -1, "Signal fired with index -1 during clear()"


def test_setup_ui_creates_chk_show_gravity_and_sets_initialized(
    qapp,
) -> None:  # noqa: ANN001
    """_setup_ui must create chk_show_gravity and mark _drake_ui_initialized True."""
    ui = DummyDrakeUI()
    assert ui._drake_ui_initialized is False

    ui._setup_ui()

    assert hasattr(ui, "chk_show_gravity")
    assert isinstance(ui.chk_show_gravity, QtWidgets.QCheckBox)
    assert ui._drake_ui_initialized is True


def test_visualization_mixin_guards_uninitialized() -> None:
    """_on_visualization_changed and _update_visualization must no-op when uninitialized."""
    ui = DummyDrakeUI()
    ui._drake_ui_initialized = False

    # Mock _update_visualization to verify _on_visualization_changed guards it
    ui._update_visualization = MagicMock()
    ui._on_visualization_changed()
    ui._update_visualization.assert_not_called()


def test_update_force_glyphs_safe_without_checkboxes() -> None:
    """_update_force_glyphs must safely proceed if checkboxes are missing."""
    ui = DummyDrakeUI()
    ui._force_overlay = MagicMock()

    # Should not raise AttributeError even if checkboxes are not on ui
    ui._update_force_glyphs()
    ui._force_overlay.tick.assert_called_once_with(
        forces=False,
        torques=False,
        gravity=False,
    )


def test_resolve_induced_accels_safe_on_empty_source() -> None:
    """_resolve_induced_accels must return zeros and not query plant with empty source."""
    ui = DummyDrakeUI()
    ui.plant.num_velocities.return_value = 4
    analyzer = MagicMock()

    res_empty = ui._resolve_induced_accels(analyzer, "")
    assert isinstance(res_empty, np.ndarray)
    assert len(res_empty) == 4
    assert np.all(res_empty == 0.0)
    ui.plant.HasJointNamed.assert_not_called()

    res_spaces = ui._resolve_induced_accels(analyzer, "   ")
    assert np.all(res_spaces == 0.0)
    ui.plant.HasJointNamed.assert_not_called()
