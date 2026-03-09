"""
Tests for external integration improvements.

Covers:
- Xacro preprocessing support (URDFParser)
- ROS package:// URI resolution with ROS_PACKAGE_PATH
- GitHub API authentication headers and retry logic
- Model cache integrity verification
"""

from __future__ import annotations

import hashlib
import http.client
import json
import tempfile
import time
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, mock_open, patch

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
        mock_run.return_value = Mock(
            returncode=0, stdout=expected_xml, stderr=""
        )

        result = parser._preprocess_xacro(Path("/tmp/robot.xacro"))
        assert result == expected_xml
        mock_run.assert_called_once()
        assert "xacro" in mock_run.call_args[0][0]

    @patch("subprocess.run")
    def test_preprocess_xacro_fallback_on_failure(
        self, mock_run: Mock
    ) -> None:
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
        mock_run.return_value = Mock(
            returncode=1, stdout="", stderr="Error processing"
        )

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
        mock_run.return_value = Mock(
            returncode=0, stdout=processed_xml, stderr=""
        )

        with tempfile.NamedTemporaryFile(
            suffix=".xacro", delete=False, mode="w"
        ) as f:
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

            with patch.dict(
                "os.environ", {"ROS_PACKAGE_PATH": tmpdir}
            ):
                result = parser._resolve_mesh_path(
                    "package://my_robot/meshes/body.stl",
                    Path("/some/urdf/dir/robot.urdf"),
                )
            assert result is not None
            assert result == mesh_file

    def test_resolve_via_multiple_ros_paths(self) -> None:
        """Should search all paths in ROS_PACKAGE_PATH (colon-separated)."""
        from model_generation.converters.urdf_parser import URDFParser

        parser = URDFParser()

        with tempfile.TemporaryDirectory() as tmpdir1, \
             tempfile.TemporaryDirectory() as tmpdir2:
            # Mesh exists only in second path
            pkg_dir = Path(tmpdir2) / "my_robot" / "meshes"
            pkg_dir.mkdir(parents=True)
            mesh_file = pkg_dir / "body.stl"
            mesh_file.touch()

            ros_path = f"{tmpdir1}:{tmpdir2}"
            with patch.dict(
                "os.environ", {"ROS_PACKAGE_PATH": ros_path}
            ):
                result = parser._resolve_mesh_path(
                    "package://my_robot/meshes/body.stl",
                    Path("/some/urdf/dir/robot.urdf"),
                )
            assert result is not None
            assert result == mesh_file

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
            assert result == mesh_file

    def test_resolve_logs_failure(self) -> None:
        """Should log warning when resolution fails."""
        from model_generation.converters.urdf_parser import URDFParser

        parser = URDFParser()

        with patch.dict(
            "os.environ", {"ROS_PACKAGE_PATH": ""}, clear=False
        ), patch(
            "model_generation.converters.urdf_parser.logger"
        ) as mock_logger:
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
class TestGitHubAPIAuth:
    """Tests for GitHub API authentication and resilience."""

    def test_auth_header_from_env(self) -> None:
        """Should add Authorization header when GITHUB_TOKEN is set."""
        from model_generation.library.repository import GitHubRepository

        repo = GitHubRepository(owner="test", repo="models")
        with patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_testtoken123"}):
            req = repo._build_api_request(
                "https://api.github.com/repos/test/models/contents/"
            )
        assert req.get_header("Authorization") == "token ghp_testtoken123"
        assert req.get_header("Accept") == "application/vnd.github.v3+json"

    def test_no_auth_header_without_token(self) -> None:
        """Should not include Authorization header without GITHUB_TOKEN."""
        from model_generation.library.repository import GitHubRepository

        repo = GitHubRepository(owner="test", repo="models")
        with patch.dict("os.environ", {}, clear=True):
            req = repo._build_api_request(
                "https://api.github.com/repos/test/models/contents/"
            )
        assert req.get_header("Authorization") is None
        # Accept header should always be present
        assert req.get_header("Accept") == "application/vnd.github.v3+json"

    @patch("urllib.request.urlopen")
    def test_retry_on_transient_failure(self, mock_urlopen: Mock) -> None:
        """Should retry on HTTP 5xx errors with exponential backoff."""
        from model_generation.library.repository import GitHubRepository

        repo = GitHubRepository(owner="test", repo="models")

        # First two calls fail with 503, third succeeds
        error_response = urllib.error.HTTPError(
            url="https://api.github.com/test",
            code=503,
            msg="Service Unavailable",
            hdrs=MagicMock(),  # type: ignore[arg-type]
            fp=None,
        )
        success_response = MagicMock()
        success_response.read.return_value = b"[]"
        success_response.headers = MagicMock()
        success_response.headers.get.return_value = None
        success_response.__enter__ = Mock(return_value=success_response)
        success_response.__exit__ = Mock(return_value=False)

        mock_urlopen.side_effect = [
            error_response,
            error_response,
            success_response,
        ]

        with patch("time.sleep"):  # Don't actually sleep
            result = repo._api_request_with_retry(
                "https://api.github.com/test"
            )
        assert result == []
        assert mock_urlopen.call_count == 3

    @patch("urllib.request.urlopen")
    def test_retry_exhausted_raises(self, mock_urlopen: Mock) -> None:
        """Should raise after all retries exhausted."""
        from model_generation.library.repository import GitHubRepository

        repo = GitHubRepository(owner="test", repo="models")

        error = urllib.error.HTTPError(
            url="https://api.github.com/test",
            code=500,
            msg="Internal Server Error",
            hdrs=MagicMock(),  # type: ignore[arg-type]
            fp=None,
        )
        mock_urlopen.side_effect = error

        with patch("time.sleep"), pytest.raises(urllib.error.HTTPError):
            repo._api_request_with_retry("https://api.github.com/test")
        # 1 initial + 3 retries = 4 total attempts
        assert mock_urlopen.call_count == 4

    @patch("urllib.request.urlopen")
    def test_no_retry_on_4xx(self, mock_urlopen: Mock) -> None:
        """Should NOT retry on client errors (4xx)."""
        from model_generation.library.repository import GitHubRepository

        repo = GitHubRepository(owner="test", repo="models")

        error = urllib.error.HTTPError(
            url="https://api.github.com/test",
            code=404,
            msg="Not Found",
            hdrs=MagicMock(),  # type: ignore[arg-type]
            fp=None,
        )
        mock_urlopen.side_effect = error

        with pytest.raises(urllib.error.HTTPError):
            repo._api_request_with_retry("https://api.github.com/test")
        assert mock_urlopen.call_count == 1

    @patch("urllib.request.urlopen")
    def test_timeout_handling(self, mock_urlopen: Mock) -> None:
        """Should handle request timeouts as retryable errors."""
        from model_generation.library.repository import GitHubRepository

        repo = GitHubRepository(owner="test", repo="models")

        success_response = MagicMock()
        success_response.read.return_value = b"[]"
        success_response.headers = MagicMock()
        success_response.headers.get.return_value = None
        success_response.__enter__ = Mock(return_value=success_response)
        success_response.__exit__ = Mock(return_value=False)

        mock_urlopen.side_effect = [
            TimeoutError("Connection timed out"),
            success_response,
        ]

        with patch("time.sleep"):
            result = repo._api_request_with_retry(
                "https://api.github.com/test"
            )
        assert result == []
        assert mock_urlopen.call_count == 2

    @patch("urllib.request.urlopen")
    def test_pagination_follows_link_header(
        self, mock_urlopen: Mock
    ) -> None:
        """Should follow Link headers for pagination."""
        from model_generation.library.repository import GitHubRepository

        repo = GitHubRepository(owner="test", repo="models")

        # First page with Link header pointing to page 2
        page1_response = MagicMock()
        page1_response.read.return_value = json.dumps(
            [{"name": "a.urdf", "type": "file", "path": "a.urdf"}]
        ).encode()
        page1_response.headers = MagicMock()
        page1_response.headers.get.return_value = (
            '<https://api.github.com/test?page=2>; rel="next"'
        )
        page1_response.__enter__ = Mock(return_value=page1_response)
        page1_response.__exit__ = Mock(return_value=False)

        # Second page with no Link header (last page)
        page2_response = MagicMock()
        page2_response.read.return_value = json.dumps(
            [{"name": "b.urdf", "type": "file", "path": "b.urdf"}]
        ).encode()
        page2_response.headers = MagicMock()
        page2_response.headers.get.return_value = None
        page2_response.__enter__ = Mock(return_value=page2_response)
        page2_response.__exit__ = Mock(return_value=False)

        mock_urlopen.side_effect = [page1_response, page2_response]

        with patch("time.sleep"):
            result = repo._api_request_with_retry(
                "https://api.github.com/test", paginate=True
            )
        assert len(result) == 2
        assert mock_urlopen.call_count == 2


# ---------------------------------------------------------------------------
# 4. Model cache integrity
# ---------------------------------------------------------------------------
class TestCacheIntegrity:
    """Tests for cache integrity improvements."""

    def test_checksum_computed_by_default(self) -> None:
        """Checksum should always be computed (not optional)."""
        from model_generation.library.cache import CacheConfig, ModelCache

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))
            cache = ModelCache(config=config)

            # Create a test file
            test_file = Path(tmpdir) / "test_model.urdf"
            test_file.write_text("<robot name='test'/>")

            # Put without explicitly requesting checksum
            entry = cache.put("test_model", test_file, source_url="http://example.com")

            # Checksum must always be present
            assert entry.checksum is not None
            assert len(entry.checksum) == 64  # SHA-256 hex length

    def test_version_metadata_in_cache_entries(self) -> None:
        """Cache entries should include version metadata."""
        from model_generation.library.cache import CacheConfig, ModelCache

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))
            cache = ModelCache(config=config)

            test_file = Path(tmpdir) / "test_model.urdf"
            test_file.write_text("<robot name='test'/>")

            entry = cache.put("test_model", test_file)
            assert entry.version is not None
            assert isinstance(entry.version, str)

    def test_version_metadata_serialized(self) -> None:
        """Version should survive serialization round-trip."""
        from model_generation.library.cache import CacheEntry

        entry = CacheEntry(
            model_id="test",
            source_url="http://example.com",
            local_path=Path("/tmp/test"),
            checksum="abc123",
            version="1.2.3",
        )
        data = entry.to_dict()
        assert "version" in data
        assert data["version"] == "1.2.3"

        restored = CacheEntry.from_dict(data)
        assert restored.version == "1.2.3"

    def test_cache_validates_on_retrieval(self) -> None:
        """Cache.get() should validate checksum and reject corrupted files."""
        from model_generation.library.cache import CacheConfig, ModelCache

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))
            cache = ModelCache(config=config)

            test_file = Path(tmpdir) / "test_model.urdf"
            test_file.write_text("<robot name='test'/>")

            entry = cache.put("test_model", test_file)
            original_checksum = entry.checksum
            assert original_checksum is not None

            # Corrupt the file
            test_file.write_text("<robot name='CORRUPTED'/>")

            # Retrieval should detect corruption
            retrieved = cache.get("test_model")
            assert retrieved is None

    def test_cache_returns_valid_entry(self) -> None:
        """Cache.get() should return entry when checksum matches."""
        from model_generation.library.cache import CacheConfig, ModelCache

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))
            cache = ModelCache(config=config)

            test_file = Path(tmpdir) / "test_model.urdf"
            test_file.write_text("<robot name='test'/>")

            cache.put("test_model", test_file)

            # File unchanged - should return entry
            retrieved = cache.get("test_model")
            assert retrieved is not None
            assert retrieved.model_id == "test_model"

    def test_cache_index_includes_version(self) -> None:
        """The cache index JSON should include version field."""
        from model_generation.library.cache import CacheConfig, ModelCache

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))
            cache = ModelCache(config=config)

            test_file = Path(tmpdir) / "test_model.urdf"
            test_file.write_text("<robot name='test'/>")

            cache.put("test_model", test_file)

            # Read the saved index
            index_path = Path(tmpdir) / "cache_index.json"
            index_data = json.loads(index_path.read_text())

            entries = index_data["entries"]
            assert len(entries) == 1
            assert "version" in entries[0]
            assert "checksum" in entries[0]
            assert entries[0]["checksum"] is not None
