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


# ---------------------------------------------------------------------------
# 2. ROS package:// URI resolution
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3. GitHub API authentication and retry logic
# ---------------------------------------------------------------------------


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
