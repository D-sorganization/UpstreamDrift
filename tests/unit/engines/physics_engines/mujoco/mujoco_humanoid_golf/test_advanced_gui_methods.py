"""Unit tests for advanced_gui_methods.py."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.advanced_gui_methods import (
    AdvancedGuiMethodsMixin,
)


class MockGuiClass(AdvancedGuiMethodsMixin):
    def __init__(self):
        self.model_configs = [{"name": "full_body"}, {"name": "other"}]
        self.model_combo = MagicMock()
        self.sim_widget = MagicMock()
        self.sim_widget.model = MagicMock()


def test_load_launch_config(tmp_path):
    """Test loading configuration from a JSON file."""
    config_file = tmp_path / "simulation_config.json"
    config_data = {"colors": {"shirt": [1, 0, 0, 1]}}
    config_file.write_text(json.dumps(config_data))

    gui = MockGuiClass()
    gui._apply_config_colors = MagicMock()

    with patch(
        "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.advanced_gui_methods.Path.cwd",
        return_value=tmp_path,
    ):
        gui._load_launch_config()

    gui.model_combo.setCurrentIndex.assert_called_once_with(0)
    gui._apply_config_colors.assert_called_once_with({"shirt": [1, 0, 0, 1]})


def test_apply_config_colors():
    """Test applying colors based on geometry names."""
    import mujoco

    gui = MockGuiClass()
    gui.sim_widget.get_num_geoms.return_value = 2

    # Mock mujoco's mj_id2name to return specific names
    def mock_id2name(model, type, i):
        return ["torso_geom", "thigh_geom"][i]

    with patch("mujoco.mj_id2name", side_effect=mock_id2name):
        colors = {"shirt": [1, 0, 0, 1], "pants": [0, 1, 0, 1]}
        gui._apply_config_colors(colors)

        gui.sim_widget.set_geom_rgba.assert_any_call(0, [1, 0, 0, 1])
        gui.sim_widget.set_geom_rgba.assert_any_call(1, [0, 1, 0, 1])
        gui.sim_widget._render_once.assert_called_once()


def test_on_ellipsoid_visualization_changed():
    """Test toggling ellipsoid visualization."""
    gui = MockGuiClass()
    gui.show_mobility_ellipsoid_cb = MagicMock()
    gui.show_mobility_ellipsoid_cb.isChecked.return_value = True
    gui.show_force_ellipsoid_cb = MagicMock()
    gui.show_force_ellipsoid_cb.isChecked.return_value = False

    gui.on_ellipsoid_visualization_changed(1)

    gui.sim_widget.set_ellipsoid_visualization.assert_called_once_with(True, False)


def test_prepare_analysis_data():
    """Test preparation of analysis data arrays and metrics."""
    gui = MockGuiClass()

    mock_recorder = MagicMock()
    mock_recorder.get_time_series.side_effect = (
        lambda key: ([1], [2]) if key != "club_head_speed" else ([1], [100.0])
    )

    mock_analyzer_cls = MagicMock()
    mock_analyzer_instance = mock_analyzer_cls.return_value
    mock_analyzer_instance.generate_comprehensive_report.return_value = {
        "club_head_speed": {"peak_value": 100.0},
        "energy_efficiency": 80.0,
        "tempo": {"ratio": 3.5},
    }

    mock_plotter_cls = MagicMock()

    analyzer, report, plotter, metrics = gui._prepare_analysis_data(
        mock_recorder, np, mock_analyzer_cls, mock_plotter_cls
    )

    assert report["club_head_speed"]["peak_value"] == 100.0
    assert metrics["Speed"] == 1.0  # min(100.0 / 50.0, 1.0)
    assert metrics["Efficiency"] == 0.8
    assert metrics["Tempo"] == 0.75  # 1.0 - abs(3.5 - 3.0)/2.0


def test_detect_pelvis_torso_indices():
    """Test detecting pelvis and torso joint indices."""
    gui = MockGuiClass()

    def mock_name2id(model, type, name):
        if name == "pelvis":
            return 1
        if name == "spine_rotation":
            return 2
        return -1

    gui.sim_widget.model.jnt_dofadr = [0, 5, 10]

    with patch("mujoco.mj_name2id", side_effect=mock_name2id):
        p_idx, t_idx = gui._detect_pelvis_torso_indices()

        assert p_idx == 5
        assert t_idx == 10
