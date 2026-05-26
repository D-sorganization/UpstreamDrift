"""
Tests for external integration improvements.

Covers:
- Xacro preprocessing support (URDFParser)
- ROS package:// URI resolution with ROS_PACKAGE_PATH
- GitHub API authentication headers and retry logic
- Model cache integrity verification
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# 1. Xacro preprocessing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2. ROS package:// URI resolution
# ---------------------------------------------------------------------------
class TestROSPackageResolution:
    """Tests for enhanced package:// URI resolution."""

    def test_resolve_via_ros_package_path(self) -> None:
        """Should resolve package:// using ROS_PACKAGE_PATH."""
        from model_generation.converters.urdf_parser import URDFParser

        parser = URDFParser()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake ROS package structure
            pkg_dir = Path(tmpdir) / "my_robot" / "meshes"
            pkg_dir.mkdir(parents=True)
            mesh_file = pkg_dir / "body.stl"
            mesh_file.touch()

            with patch.dict("os.environ", {"ROS_PACKAGE_PATH": tmpdir}):
                result = parser._resolve_mesh_path(
                    "package://my_robot/meshes/body.stl",
                    Path("/some/urdf/dir/robot.urdf"),
                )
            assert result is not None
            assert result.resolve() == mesh_file.resolve()

    def test_resolve_via_multiple_ros_paths(self) -> None:
        """Should search all paths in ROS_PACKAGE_PATH (colon-separated)."""
        from model_generation.converters.urdf_parser import URDFParser

        parser = URDFParser()

        with (
            tempfile.TemporaryDirectory() as tmpdir1,
            tempfile.TemporaryDirectory() as tmpdir2,
        ):
            # Mesh exists only in second path
            pkg_dir = Path(tmpdir2) / "my_robot" / "meshes"
            pkg_dir.mkdir(parents=True)
            mesh_file = pkg_dir / "body.stl"
            mesh_file.touch()

            ros_path = f"{tmpdir1}:{tmpdir2}"
            with patch.dict("os.environ", {"ROS_PACKAGE_PATH": ros_path}):
                result = parser._resolve_mesh_path(
                    "package://my_robot/meshes/body.stl",
                    Path("/some/urdf/dir/robot.urdf"),
                )
            assert result is not None
            assert result.resolve() == mesh_file.resolve()

    def test_resolve_catkin_workspace(self) -> None:
        """Should search catkin workspace src/ directories."""
        from model_generation.converters.urdf_parser import URDFParser

        parser = URDFParser()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create catkin workspace structure
            ws_src = Path(tmpdir) / "catkin_ws" / "src" / "my_robot" / "meshes"
            ws_src.mkdir(parents=True)
            mesh_file = ws_src / "body.stl"
            mesh_file.touch()

            catkin_path = str(Path(tmpdir) / "catkin_ws")
            with patch.dict(
                "os.environ",
                {
                    "ROS_PACKAGE_PATH": "",
                    "CMAKE_PREFIX_PATH": catkin_path,
                },
            ):
                result = parser._resolve_mesh_path(
                    "package://my_robot/meshes/body.stl",
                    Path("/some/urdf/dir/robot.urdf"),
                )
            assert result is not None
            assert result.resolve() == mesh_file.resolve()

    def test_resolve_logs_failure(self) -> None:
        """Should log warning when resolution fails."""
        from model_generation.converters.urdf_parser import URDFParser

        parser = URDFParser()

        with (
            patch.dict("os.environ", {"ROS_PACKAGE_PATH": ""}, clear=False),
            patch("model_generation.converters.urdf_parser.logger") as mock_logger,
        ):
            result = parser._resolve_mesh_path(
                "package://nonexistent_pkg/mesh.stl",
                Path("/some/urdf/dir/robot.urdf"),
            )
            assert result is None
            mock_logger.warning.assert_called()

    def test_resolve_still_works_for_relative_paths(self) -> None:
        """Existing relative path resolution should still work."""
        from model_generation.converters.urdf_parser import URDFParser

        parser = URDFParser()

        with tempfile.TemporaryDirectory() as tmpdir:
            mesh_file = Path(tmpdir) / "mesh.stl"
            mesh_file.touch()
            urdf_file = Path(tmpdir) / "robot.urdf"

            result = parser._resolve_mesh_path("mesh.stl", urdf_file)
            assert result is not None
            assert result == mesh_file


# ---------------------------------------------------------------------------
# 3. GitHub API authentication and retry logic
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4. Model cache integrity
# ---------------------------------------------------------------------------
