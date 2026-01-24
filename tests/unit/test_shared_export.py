import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import paths configured at test runner level via pyproject.toml/conftest.py
from src.shared.python.export import (
    export_recording_all_formats,
    export_to_hdf5,
    export_to_matlab,
    get_available_export_formats,
)


@pytest.fixture
def sample_data() -> dict[str, Any]:
    """Create comprehensive sample data for export testing."""
    return {
        "times": np.array([0.0, 0.1, 0.2]),
        "position": np.array([1.0, 2.0, 3.0]),
        "velocity": np.array([0.5, 1.0, 1.5]),
        "matrix_2d": np.array([[1, 2], [3, 4], [5, 6]]),
        "scalar_int": 42,
        "scalar_float": 3.14,
        "string_val": "test_string",
        "nested_dict": {
            "sub_array": np.array([10, 20]),
            "sub_scalar": 100,
        },
        "induced_accelerations": {
            0: np.array([0.1, 0.2, 0.3]),
            "gravity": np.array([9.8, 9.8, 9.8]),
        },
    }


class TestSharedExport:
    """Tests for shared/python/export.py module."""

    def test_get_available_export_formats(self) -> None:
        """Test listing available formats."""
        formats = get_available_export_formats()
        assert "json" in formats
        assert "csv" in formats
        assert "mat" in formats
        assert "hdf5" in formats
        assert formats["json"]["extension"] == ".json"

    def test_export_to_matlab_success(
        self, tmp_path: Path, sample_data: dict[str, Any]
    ) -> None:
        """Test successful export to MATLAB .mat file."""
        output_path = str(tmp_path / "test.mat")

        # Mock scipy.io.savemat
        with patch("src.shared.python.export.savemat") as mock_savemat:
            with patch("src.shared.python.export.SCIPY_AVAILABLE", True):
                success = export_to_matlab(output_path, sample_data)
                assert success is True
                mock_savemat.assert_called_once()

                # Verify arguments
                args, kwargs = mock_savemat.call_args
                assert args[0] == output_path
                data_dict = args[1]

                # Check data transformation
                assert "times" in data_dict
                assert "scalar_int" in data_dict
                # Check nested flattening
                assert "nested_dict_sub_array" in data_dict
                assert "nested_dict_sub_scalar" in data_dict

    def test_export_to_matlab_missing_dependency(
        self, tmp_path: Path, sample_data: dict[str, Any]
    ) -> None:
        """Test behavior when scipy is missing."""
        with patch("src.shared.python.export.SCIPY_AVAILABLE", False):
            success = export_to_matlab(str(tmp_path / "test.mat"), sample_data)
            assert success is False

    def test_export_to_matlab_exception(
        self, tmp_path: Path, sample_data: dict[str, Any]
    ) -> None:
        """Test exception handling during export."""
        with patch(
            "src.shared.python.export.savemat", side_effect=Exception("Disk full")
        ):
            with patch("src.shared.python.export.SCIPY_AVAILABLE", True):
                success = export_to_matlab(str(tmp_path / "test.mat"), sample_data)
                assert success is False

    def test_export_to_hdf5_success(
        self, tmp_path: Path, sample_data: dict[str, Any]
    ) -> None:
        """Test successful export to HDF5 file."""
        output_path = str(tmp_path / "test.h5")

        # Mock h5py via sys.modules since it's imported at module level
        mock_h5py = MagicMock()
        mock_file = MagicMock()
        mock_h5py.File.return_value.__enter__.return_value = mock_file

        # Need to patch in sys.modules first so the attribute exists on the module
        with patch.dict(sys.modules, {"h5py": mock_h5py}):
            # Reload module to pick up the mocked h5py
            import src.shared.python.export as export_module

            export_module.h5py = mock_h5py
            export_module.H5PY_AVAILABLE = True
            try:
                success = export_to_hdf5(output_path, sample_data)
                assert success is True

                # Verify group creation
                mock_file.create_group.assert_any_call("timeseries")
                mock_file.create_group.assert_any_call("metadata")
            finally:
                # Cleanup
                delattr(export_module, "h5py")

    def test_export_to_hdf5_missing_dependency(
        self, tmp_path: Path, sample_data: dict[str, Any]
    ) -> None:
        """Test behavior when h5py is missing."""
        with patch("src.shared.python.export.H5PY_AVAILABLE", False):
            success = export_to_hdf5(str(tmp_path / "test.h5"), sample_data)
            assert success is False

    def test_export_to_hdf5_exception(
        self, tmp_path: Path, sample_data: dict[str, Any]
    ) -> None:
        """Test exception handling during HDF5 export."""
        mock_h5py = MagicMock()
        mock_h5py.File.side_effect = Exception("File locked")

        # Need to patch in sys.modules first so the attribute exists on the module
        with patch.dict(sys.modules, {"h5py": mock_h5py}):
            import src.shared.python.export as export_module

            export_module.h5py = mock_h5py
            export_module.H5PY_AVAILABLE = True
            try:
                success = export_to_hdf5(str(tmp_path / "test.h5"), sample_data)
                assert success is False
            finally:
                delattr(export_module, "h5py")

    def test_export_recording_all_formats_json(
        self, tmp_path: Path, sample_data: dict[str, Any]
    ) -> None:
        """Test JSON export via generic function."""
        base_path = str(tmp_path / "recording")

        results = export_recording_all_formats(base_path, sample_data, formats=["json"])

        assert results["json"] is True
        json_path = tmp_path / "recording.json"
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)
            assert data["scalar_int"] == 42
            assert len(data["times"]) == 3
            # Check nested dict handling
            assert data["nested_dict"]["sub_scalar"] == 100

    def test_export_recording_all_formats_csv(
        self, tmp_path: Path, sample_data: dict[str, Any]
    ) -> None:
        """Test CSV export via generic function."""
        base_path = str(tmp_path / "recording")

        results = export_recording_all_formats(base_path, sample_data, formats=["csv"])

        assert results["csv"] is True
        csv_path = tmp_path / "recording.csv"
        assert csv_path.exists()

        import pandas as pd

        df = pd.read_csv(csv_path)

        assert "time" in df.columns
        assert "position" in df.columns
        assert "velocity" in df.columns
        # Check matrix expansion
        assert "matrix_2d_0" in df.columns
        assert "matrix_2d_1" in df.columns
        # Check nested dict expansion
        assert "induced_accelerations_source_0" in df.columns
        assert "induced_accelerations_gravity" in df.columns

    def test_export_recording_all_formats_partial_failure(
        self, tmp_path: Path, sample_data: dict[str, Any]
    ) -> None:
        """Test that failure in one format doesn't stop others."""
        base_path = str(tmp_path / "recording")

        # Mock export_to_matlab to fail
        with patch("src.shared.python.export.export_to_matlab", return_value=False):
            # Mock json dump to succeed (real file IO)
            results = export_recording_all_formats(
                base_path, sample_data, formats=["json", "mat"]
            )

            assert results["json"] is True
            assert results["mat"] is False
