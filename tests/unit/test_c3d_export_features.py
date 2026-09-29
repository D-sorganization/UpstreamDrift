"""Tests for C3D export security, versioning, and telemetry features.

Previously this file imported a top-level ``c3d_reader`` module that does not
exist in any supported configuration, so a conftest allowlist rule dropped all
seven tests from collection without a skip entry (#8006). The real code lives at
``sidekick.lab.bio``, and the export routine moved from a private
``C3DDataReader._export_dataframe`` method to the module-level
``_c3d_io.export_dataframe`` function; the tests are repointed accordingly.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from sidekick.lab.bio._c3d_io import export_dataframe
from sidekick.lab.bio.c3d_reader import SCHEMA_VERSION

pytestmark = pytest.mark.unit

SOURCE_FILE = "test_capture.c3d"
UNITS = "mm"


class TestC3DExportFeatures:
    """Tests for the enhanced export functionality."""

    @staticmethod
    def _export(dataframe: pd.DataFrame, path: Path | str, file_format: str) -> Path:
        return export_dataframe(
            dataframe,
            path,
            file_format=file_format,
            source_file_name=SOURCE_FILE,
            units=UNITS,
        )

    @pytest.fixture
    def sample_dataframe(self) -> pd.DataFrame:
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
    def mock_project_root(self, tmp_path) -> Generator[Any, None, None]:
        """Make tmp_path appear as the project root."""
        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_cwd.return_value = Path(tmp_path).resolve()
            yield mock_cwd

    def test_security_prevents_directory_traversal(
        self, sample_dataframe, tmp_path
    ) -> None:
        """Ensure attempts to write outside the project root are blocked.

        NOTE: ``_c3d_io.validate_export_path`` walks the call stack looking for
        this exact function name to decide whether to enforce the check, so the
        name is load-bearing. That coupling is a production smell tracked
        separately; do not rename this test without fixing it.
        """
        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_root = Path(tmp_path) / "project_root"
            mock_root.mkdir()
            mock_cwd.return_value = mock_root.resolve()

            # Create a path clearly outside the root
            outside_path = tmp_path / "outside.csv"

            with pytest.raises(ValueError, match="Security: Refusing to output to"):
                self._export(sample_dataframe, str(outside_path), "csv")

    def test_security_allows_project_root_files(
        self, sample_dataframe, mock_project_root, tmp_path
    ) -> None:
        """Ensure writing within the project root is allowed."""
        safe_path = tmp_path / "safe_export.csv"

        result = self._export(sample_dataframe, str(safe_path), "csv")
        assert result == safe_path.resolve()

    def test_csv_metadata_sidecar_creation(
        self, sample_dataframe, mock_project_root, tmp_path
    ) -> None:
        """Verify _meta.json sidecar is created for CSV exports."""
        output_path = tmp_path / "export.csv"

        self._export(sample_dataframe, str(output_path), "csv")

        assert output_path.exists()

        sidecar_path = tmp_path / "export_meta.json"
        assert sidecar_path.exists()

        meta = json.loads(sidecar_path.read_text())

        assert meta["schema_version"] == SCHEMA_VERSION
        assert meta["source_file"] == SOURCE_FILE
        assert meta["row_count"] == 2
        assert meta["units"] == UNITS
        assert "created_at_utc" in meta

    def test_json_envelope_structure(
        self, sample_dataframe, mock_project_root, tmp_path
    ) -> None:
        """Verify JSON export uses the envelope pattern."""
        output_path = tmp_path / "export.json"

        self._export(sample_dataframe, str(output_path), "json")

        data = json.loads(output_path.read_text())

        assert "metadata" in data
        assert "data" in data
        assert data["metadata"]["schema_version"] == SCHEMA_VERSION
        assert len(data["data"]) == 2

    def test_npz_metadata_embedding(
        self, sample_dataframe, mock_project_root, tmp_path
    ) -> None:
        """Verify NPZ export includes metadata in the archive."""
        output_path = tmp_path / "export.npz"

        self._export(sample_dataframe, str(output_path), "npz")

        with np.load(output_path, allow_pickle=False) as archive:
            assert "_metadata" in archive
            meta = json.loads(str(archive["_metadata"]))
            assert meta["schema_version"] == SCHEMA_VERSION

    def test_telemetry_logging(
        self, sample_dataframe, mock_project_root, tmp_path
    ) -> None:
        """Verify execution time is logged."""
        with patch("sidekick.lab.bio._c3d_io.log_execution_time") as mock_log_ctx:
            mock_ctx_instance = MagicMock()
            mock_log_ctx.return_value = mock_ctx_instance
            mock_ctx_instance.__enter__.return_value = None

            output_path = tmp_path / "telemetry_test.csv"
            self._export(sample_dataframe, str(output_path), "csv")

            mock_log_ctx.assert_called_once()
            args, _ = mock_log_ctx.call_args
            assert "export_csv" in args[0]

    def test_csv_injection_sanitization(self, mock_project_root, tmp_path) -> None:
        """Verify dangerous characters are escaped in CSV."""
        dangerous_df = pd.DataFrame(
            {
                "col1": ["=SUM(1,1)", "@EVIL", "+DATA", "-MINUS", "SAFE"],
                "col2": [1, 2, 3, 4, 5],
            }
        )

        output_path = tmp_path / "sanitized.csv"
        self._export(dangerous_df, str(output_path), "csv")

        # Read back purely as text so we don't eval
        content = output_path.read_text()

        assert "'=SUM(1,1)" in content
        assert "'@EVIL" in content
        assert "'+DATA" in content
        assert "'-MINUS" in content
        assert "SAFE" in content  # Unchanged
