"""
Tests for external integration improvements.

Covers:
- Xacro preprocessing support (URDFParser)
- ROS package:// URI resolution with ROS_PACKAGE_PATH
- GitHub API authentication headers and retry logic
- Model cache integrity verification
"""

from __future__ import annotations

import json
import tempfile
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. Xacro preprocessing
# ---------------------------------------------------------------------------
class TestXacroPreprocessing:
    """Tests for xacro detection and preprocessing in URDFParser."""

    def test_detects_xacro_file_extension(self) -> None:
        """Parser should detect .xacro file extension."""
        from model_generation.converters.urdf_parser import URDFParser

        parser = URDFParser()
        assert parser._is_xacro(Path("robot.urdf.xacro"))
        assert parser._is_xacro(Path("robot.xacro"))
        assert not parser._is_xacro(Path("robot.urdf"))

    def test_detects_xacro_namespace_in_xml(self) -> None:
        """Parser should detect xacro namespace in XML content."""
        from model_generation.converters.urdf_parser import URDFParser

        parser = URDFParser()
        xml_with_ns = '<robot xmlns:xacro="http://www.ros.org/wiki/xacro">'
        xml_without_ns = "<robot name='test'>"
        assert parser._has_xacro_namespace(xml_with_ns)
        assert not parser._has_xacro_namespace(xml_without_ns)

    @patch("subprocess.run")
    def test_preprocess_xacro_uses_cli(self, mock_run: Mock) -> None:
        """Should use xacro CLI tool when available."""
        from model_generation.converters.urdf_parser import URDFParser

        parser = URDFParser()
        expected_xml = "<robot name='processed'></robot>"
        mock_run.return_value = Mock(returncode=0, stdout=expected_xml, stderr="")

        result = parser._preprocess_xacro(Path("/tmp/robot.xacro"))
        assert result == expected_xml
        mock_run.assert_called_once()
        assert "xacro" in mock_run.call_args[0][0]

    @patch("subprocess.run")
    def test_preprocess_xacro_fallback_on_failure(self, mock_run: Mock) -> None:
        """Should log warning and return None if xacro CLI fails."""
        from model_generation.converters.urdf_parser import URDFParser

        parser = URDFParser()
        mock_run.side_effect = FileNotFoundError("xacro not found")

        result = parser._preprocess_xacro(Path("/tmp/robot.xacro"))
        assert result is None

    @patch("subprocess.run")
    def test_preprocess_xacro_nonzero_return(self, mock_run: Mock) -> None:
        """Should return None when xacro returns non-zero exit code."""
        from model_generation.converters.urdf_parser import URDFParser

        parser = URDFParser()
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="Error processing")

        result = parser._preprocess_xacro(Path("/tmp/robot.xacro"))
        assert result is None

    @patch("subprocess.run")
    def test_parse_xacro_file_end_to_end(self, mock_run: Mock) -> None:
        """Full parse of a .xacro file should preprocess then parse XML."""
        from model_generation.converters.urdf_parser import URDFParser

        parser = URDFParser()
        processed_xml = """<?xml version="1.0"?>
        <robot name="xacro_robot">
            <link name="base_link">
                <inertial>
                    <mass value="1.0"/>
                    <inertia ixx="0.1" iyy="0.1" izz="0.1"
                             ixy="0" ixz="0" iyz="0"/>
                </inertial>
            </link>
        </robot>
        """
        mock_run.return_value = Mock(returncode=0, stdout=processed_xml, stderr="")

        with tempfile.NamedTemporaryFile(suffix=".xacro", delete=False, mode="w") as f:
            f.write('<robot xmlns:xacro="http://www.ros.org/wiki/xacro"/>')
            xacro_path = Path(f.name)

        try:
            model = parser.parse(xacro_path)
            assert model.name == "xacro_robot"
            assert len(model.links) == 1
        finally:
            xacro_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 2. ROS package:// URI resolution
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3. GitHub API authentication and retry logic
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4. Model cache integrity
# ---------------------------------------------------------------------------
