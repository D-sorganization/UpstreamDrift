"""Tests for C3D export security, versioning, and telemetry features."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Setup import path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_PATH = os.path.join(
    PROJECT_ROOT, "src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src"
)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from c3d_reader import SCHEMA_VERSION, C3DDataReader  # noqa: E402


class TestC3DExportFeatures:
    """Tests for the enhanced export functionality."""

    @pytest.fixture
    def mock_reader(self):
        """Create a reader with a mocked get_metadata method."""
        reader = C3DDataReader("test_capture.c3d")
        # Mock underlying data to allow export to proceed (it calls self.get_metadata().units)
        mock_meta = MagicMock()
        mock_meta.units = "mm"
        reader.get_metadata = MagicMock(return_value=mock_meta)
        return reader

    @pytest.fixture
    def sample_dataframe(self):
        """Create a dummy DataFrame for export."""
        return pd.DataFrame(
            {
                "frame": [1, 2],
                "marker": ["A", "A"],
                "x": [10, 11],
                "y": [20, 21],
                "z": [30, 31],
                "residual": [0.0, 0.0],
            }
        )

    @pytest.fixture
    def mock_project_root(self, tmp_path):
        """Make tmp_path appear as the project root."""
        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_cwd.return_value = Path(tmp_path).resolve()
            yield mock_cwd

    def test_security_prevents_directory_traversal(
        self, mock_reader, sample_dataframe, tmp_path
    ):
        """Ensure attempts to write outside the project root are blocked."""
        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_root = Path(tmp_path) / "project_root"
            mock_root.mkdir()
            mock_cwd.return_value = mock_root.resolve()

            # Create a path clearly outside the root
            outside_path = tmp_path / "outside.csv"

            with pytest.raises(ValueError, match="Security: Refusing to output to"):
                mock_reader._export_dataframe(
                    sample_dataframe, str(outside_path), file_format="csv"
                )

    def test_security_allows_project_root_files(
        self, mock_reader, sample_dataframe, mock_project_root, tmp_path
    ):
        """Ensure writing within the project root is allowed."""
        # Safe path (inside tmp_path which is mocked as root)
        safe_path = tmp_path / "safe_export.csv"

        # Should not raise
        result = mock_reader._export_dataframe(
            sample_dataframe, str(safe_path), file_format="csv"
        )
        assert result == safe_path

    def test_csv_metadata_sidecar_creation(
        self, mock_reader, sample_dataframe, mock_project_root, tmp_path
    ):
        """Verify _meta.json sidecar is created for CSV exports."""
        output_path = tmp_path / "export.csv"

        mock_reader._export_dataframe(
            sample_dataframe, str(output_path), file_format="csv"
        )

        # Check main file
        assert output_path.exists()

        # Check sidecar
        sidecar_path = tmp_path / "export_meta.json"
        assert sidecar_path.exists()

        with open(sidecar_path) as f:
            meta = json.load(f)

        assert meta["schema_version"] == SCHEMA_VERSION
        assert meta["source_file"] == "test_capture.c3d"
        assert meta["row_count"] == 2
        assert meta["units"] == "mm"
        assert "created_at_utc" in meta

    def test_json_envelope_structure(
        self, mock_reader, sample_dataframe, mock_project_root, tmp_path
    ):
        """Verify JSON export uses the envelope pattern."""
        output_path = tmp_path / "export.json"

        mock_reader._export_dataframe(
            sample_dataframe, str(output_path), file_format="json"
        )

        with open(output_path) as f:
            data = json.load(f)

        assert "metadata" in data
        assert "data" in data
        assert data["metadata"]["schema_version"] == SCHEMA_VERSION
        assert len(data["data"]) == 2

    def test_npz_metadata_embedding(
        self, mock_reader, sample_dataframe, mock_project_root, tmp_path
    ):
        """Verify NPZ export includes metadata in the archive."""
        output_path = tmp_path / "export.npz"

        mock_reader._export_dataframe(
            sample_dataframe, str(output_path), file_format="npz"
        )

        with np.load(output_path) as archive:
            assert "_metadata" in archive
            meta = json.loads(str(archive["_metadata"]))
            assert meta["schema_version"] == SCHEMA_VERSION

    def test_telemetry_logging(
        self, mock_reader, sample_dataframe, mock_project_root, tmp_path
    ):
        """Verify execution time is logged."""
        with patch("c3d_reader.log_execution_time") as mock_log_ctx:
            # Setup context manager mock
            mock_ctx_instance = MagicMock()
            mock_log_ctx.return_value = mock_ctx_instance
            mock_ctx_instance.__enter__.return_value = None

            output_path = tmp_path / "telemetry_test.csv"
            mock_reader._export_dataframe(
                sample_dataframe, str(output_path), file_format="csv"
            )

            mock_log_ctx.assert_called_once()
            args, _ = mock_log_ctx.call_args
            assert "export_csv" in args[0]

    def test_csv_injection_sanitization(self, mock_reader, mock_project_root, tmp_path):
        """Verify dangerous characters are escaped in CSV."""
        dangerous_df = pd.DataFrame(
            {
                "col1": ["=SUM(1,1)", "@EVIL", "+DATA", "-MINUS", "SAFE"],
                "col2": [1, 2, 3, 4, 5],
            }
        )

        output_path = tmp_path / "sanitized.csv"
        mock_reader._export_dataframe(dangerous_df, str(output_path), file_format="csv")

        # Read back purely as text so we don't eval
        content = output_path.read_text()

        # Check expected escaping
        assert "'=SUM(1,1)" in content
        assert "'@EVIL" in content
        assert "'+DATA" in content
        assert "'-MINUS" in content
        assert "SAFE" in content  # Unchanged
