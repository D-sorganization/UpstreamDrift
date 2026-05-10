"""Unit tests for advanced_export.py."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.advanced_export import (
    _matlab_analyze_script,
    _matlab_animate_script,
    _matlab_plot_script,
    create_matlab_script,
    export_recording_all_formats,
)


def test_matlab_scripts_content():
    """Verify MATLAB scripts contain correct commands."""
    mat_file = "test_data.mat"

    plot_script = _matlab_plot_script(mat_file)
    assert mat_file in plot_script
    assert "plot(t, rad2deg(data.joint_positions));" in plot_script

    analyze_script = _matlab_analyze_script(mat_file)
    assert mat_file in analyze_script
    assert "Peak club head speed" in analyze_script

    animate_script = _matlab_animate_script(mat_file)
    assert mat_file in animate_script
    assert "plot(positions(i, :)" in animate_script


def test_create_matlab_script(tmp_path):
    """Test generating MATLAB scripts to disk."""
    out_file = tmp_path / "plot_script.m"
    mat_file = "/fake/path/to/data.mat"

    create_matlab_script(str(out_file), mat_file, script_type="plot")

    assert out_file.exists()
    content = out_file.read_text()
    assert "data.mat" in content  # Path(mat_file).name


@patch(
    "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.advanced_export._shared_export_all"
)
def test_export_recording_all_formats_fallback(mock_shared_export):
    """Test fallback to shared export when telemetry is unavailable."""
    # Hide telemetry module to force fallback
    with patch.dict(
        "sys.modules",
        {
            "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.telemetry": None
        },
    ):
        data = {"times": [1, 2, 3]}
        mock_shared_export.return_value = {"json": True}

        result = export_recording_all_formats("base", data)
        assert result == {"json": True}
        mock_shared_export.assert_called_once_with("base", data, None)


def test_export_recording_all_formats_with_telemetry():
    """Test using local telemetry and exporting various formats."""
    # Mock telemetry
    mock_telemetry_json = MagicMock(return_value=True)
    mock_telemetry_csv = MagicMock(return_value=True)
    mock_matlab = MagicMock(return_value=True)
    mock_hdf5 = MagicMock(return_value=True)
    mock_c3d = MagicMock(return_value=True)

    import numpy as np

    data = {"times": np.array([0, 1]), "joint_positions": np.array([[0, 0], [1, 1]])}

    with patch.dict("sys.modules"):
        import sys

        # Create a fake telemetry module
        import types

        fake_telemetry = types.ModuleType("telemetry")
        fake_telemetry.export_telemetry_json = mock_telemetry_json
        fake_telemetry.export_telemetry_csv = mock_telemetry_csv

        # Inject it so it's importable from relative import
        sys.modules[
            "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.telemetry"
        ] = fake_telemetry

        with (
            patch(
                "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.advanced_export.export_to_matlab",
                mock_matlab,
            ),
            patch(
                "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.advanced_export.export_to_hdf5",
                mock_hdf5,
            ),
            patch(
                "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.advanced_export.export_to_c3d",
                mock_c3d,
            ),
        ):
            result = export_recording_all_formats(
                "base", data, formats=["json", "csv", "mat", "hdf5", "c3d", "unknown"]
            )

            assert result["json"] is True
            assert result["csv"] is True
            assert result["mat"] is True
            assert result["hdf5"] is True
            assert result["c3d"] is True
            assert result["unknown"] is False

            mock_telemetry_json.assert_called_once()
            mock_telemetry_csv.assert_called_once()
            mock_matlab.assert_called_once()
            mock_hdf5.assert_called_once()
            mock_c3d.assert_called_once()
