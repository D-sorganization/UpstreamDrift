"""Unit tests for shared/python/provenance.py."""

import unittest
from datetime import datetime
from unittest.mock import mock_open, patch

import numpy as np

from src.shared.python.provenance import (
    ProvenanceInfo,
    add_provenance_header_file,
    add_provenance_to_csv,
)


class TestProvenance(unittest.TestCase):
    """Test suite for provenance tracking."""

    @patch("src.shared.python.provenance.subprocess.check_output")
    @patch("src.shared.python.provenance.datetime")
    @patch("src.shared.python.provenance.hashlib")
    def test_capture_provenance(self, mock_hashlib, mock_datetime, mock_subprocess):
        """Test capturing provenance with mocked environment."""
        # Mock datetime
        mock_now = datetime(2025, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = mock_now
        # Need to handle timezone.utc access if it's used in the code
        mock_datetime.timezone.utc = unittest.mock.MagicMock()

        # Mock git
        mock_subprocess.side_effect = [
            "abc1234",  # git sha
            "feature-branch",  # git branch
            "",  # git status (clean)
        ]

        # Mock hashlib
        mock_sha256 = mock_hashlib.sha256.return_value
        mock_sha256.hexdigest.return_value = "hash123"

        # Mock file existence
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=b"model data")),
        ):
            provenance = ProvenanceInfo.capture(
                model_path="test_model.xml",
                parameters={"param1": 10},
            )

        self.assertEqual(provenance.git_commit_sha, "abc1234")
        self.assertEqual(provenance.git_branch, "feature-branch")
        self.assertFalse(provenance.git_is_dirty)
        self.assertEqual(provenance.model_file_hash, "hash123")
        self.assertEqual(provenance.parameters["param1"], 10)
        self.assertEqual(provenance.numpy_version, np.__version__)

    def test_to_header_lines(self):
        """Test header line generation."""
        provenance = ProvenanceInfo(
            timestamp_utc="2025-01-01T12:00:00Z",
            timestamp_local="2025-01-01T12:00:00",
            git_commit_sha="abc1234",
            git_branch="main",
            model_file_path="model.xml",
            model_file_hash="hash123",
            parameters={"dt": 0.01},
            python_version="3.9.0",
            numpy_version="1.21.0",
            mujoco_version="2.3.0",
        )

        lines = provenance.to_header_lines()

        self.assertTrue(any("golf-modeling-suite" in line for line in lines))
        self.assertTrue(any("abc1234" in line for line in lines))
        self.assertTrue(any("model.xml" in line for line in lines))
        self.assertTrue(any("dt: 0.01" in line for line in lines))
        self.assertTrue(any("NumPy: 1.21.0" in line for line in lines))

    def test_add_provenance_header_file(self):
        """Test writing provenance to file object."""
        provenance = ProvenanceInfo(
            timestamp_utc="2025-01-01T12:00:00Z",
            timestamp_local="2025-01-01T12:00:00",
        )

        m = mock_open()
        with patch("builtins.open", m):
            with open("test.csv", "w") as f:
                add_provenance_header_file(f, provenance)

        m().write.assert_called()
        # Verify call args contain provenance info
        calls = [args[0] for args, _ in m().write.call_args_list]
        self.assertTrue(any("2025-01-01T12:00:00Z" in call for call in calls))

    @patch("src.shared.python.provenance.ProvenanceInfo.capture")
    def test_add_provenance_to_csv(self, mock_capture):
        """Test prepending provenance to CSV file."""
        mock_capture.return_value = ProvenanceInfo(
            timestamp_utc="2025-01-01T12:00:00Z",
            timestamp_local="2025-01-01T12:00:00",
        )

        # Mock reading original content
        with patch("pathlib.Path.read_text", return_value="col1,col2\n1,2"):
            # Mock writing new content
            m = mock_open()
            with patch("builtins.open", m):
                add_provenance_to_csv("results.csv")

        # Verify calls
        # 1. Header lines
        # 2. Blank line
        # 3. Original content
        handle = m()
        self.assertTrue(handle.write.called)
        # Check that original content was written last (or at least written)
        handle.write.assert_any_call("col1,col2\n1,2")


if __name__ == "__main__":
    unittest.main()
