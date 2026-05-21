"""Tests for model_generation.library.cache."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from model_generation.library.cache import (
    CacheConfig,
    CacheEntry,
    ModelCache,
)


def _cfg(tmp_path: Path, **kw) -> CacheConfig:
    return CacheConfig(cache_dir=tmp_path / "cache", **kw)


class TestCacheEntry:
    def test_to_from_dict_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "f.urdf"
        path.write_text("x")
        e = CacheEntry(
            model_id="m",
            source_url="https://x/y",
            local_path=path,
            checksum="abc",
            size_bytes=1,
            version="v1",
        )
        d = e.to_dict()
        e2 = CacheEntry.from_dict(d)
        assert e2.model_id == "m"
        assert e2.local_path == path
        assert e2.checksum == "abc"
        assert e2.version == "v1"

    def test_from_dict_defaults_filled(self) -> None:
        e = CacheEntry.from_dict({"model_id": "m", "local_path": "/tmp/x"})
        assert e.is_complete is True
        assert e.size_bytes == 0
        assert e.cached_at > 0


class TestModelCache:
    def test_creates_cache_dir(self, tmp_path: Path) -> None:
        cache = ModelCache(_cfg(tmp_path))
        assert cache.config.cache_dir.exists()
        assert len(cache) == 0

    def test_put_and_get_file(self, tmp_path: Path) -> None:
        cache = ModelCache(_cfg(tmp_path))
        f = tmp_path / "model.urdf"
        f.write_bytes(b"hello world")
        entry = cache.put("m1", f, source_url="https://x/m.urdf", version="v1")
        assert entry.checksum is not None
        assert entry.size_bytes == len(b"hello world")
        assert entry.version == "v1"

        got = cache.get("m1")
        assert got is not None
        assert got.model_id == "m1"
        assert "m1" in cache
        assert cache.contains("m1") is True

    def test_get_missing_returns_none(self, tmp_path: Path) -> None:
        cache = ModelCache(_cfg(tmp_path))
        assert cache.get("nope") is None
        assert "nope" not in cache

    def test_get_with_corrupted_file_returns_none(self, tmp_path: Path) -> None:
        cache = ModelCache(_cfg(tmp_path))
        f = tmp_path / "model.urdf"
        f.write_bytes(b"hello")
        cache.put("m1", f)
        # Corrupt the file
        f.write_bytes(b"tampered")
        assert cache.get("m1") is None

    def test_verify_no_checksum_returns_true(self, tmp_path: Path) -> None:
        cache = ModelCache(_cfg(tmp_path))
        f = tmp_path / "x.urdf"
        f.write_bytes(b"data")
        cache.put("m1", f)
        # Strip checksum to simulate legacy entry
        cache._entries["m1"].checksum = None
        assert cache.verify("m1") is True

    def test_verify_corrupted(self, tmp_path: Path) -> None:
        cache = ModelCache(_cfg(tmp_path))
        f = tmp_path / "x.urdf"
        f.write_bytes(b"data")
        cache.put("m1", f)
        f.write_bytes(b"different")
        assert cache.verify("m1") is False

    def test_verify_missing(self, tmp_path: Path) -> None:
        cache = ModelCache(_cfg(tmp_path))
        assert cache.verify("nope") is False

    def test_remove_with_file_deletion(self, tmp_path: Path) -> None:
        cache = ModelCache(_cfg(tmp_path))
        # Put file into a subdir of the cache dir so cleanup of parent applies
        sub = cache.config.cache_dir / "sub"
        sub.mkdir()
        f = sub / "a.urdf"
        f.write_bytes(b"x")
        cache.put("m1", f)
        assert cache.remove("m1", delete_files=True) is True
        assert not f.exists()
        assert "m1" not in cache

    def test_remove_without_file_deletion(self, tmp_path: Path) -> None:
        cache = ModelCache(_cfg(tmp_path))
        f = tmp_path / "a.urdf"
        f.write_bytes(b"x")
        cache.put("m1", f)
        assert cache.remove("m1", delete_files=False) is True
        assert f.exists()

    def test_remove_missing_returns_false(self, tmp_path: Path) -> None:
        cache = ModelCache(_cfg(tmp_path))
        assert cache.remove("missing") is False

    def test_get_cache_path_sanitizes_separators(self, tmp_path: Path) -> None:
        cache = ModelCache(_cfg(tmp_path))
        p = cache.get_cache_path("owner/name")
        assert "/" not in p.name
        assert "\\" not in p.name

    @pytest.mark.parametrize("bad", ["..", "../etc", "a/../b", "%2e%2e", "x%2Fy"])
    def test_get_cache_path_blocks_traversal(self, tmp_path: Path, bad: str) -> None:
        cache = ModelCache(_cfg(tmp_path))
        with pytest.raises(ValueError, match="Path traversal blocked"):
            cache.get_cache_path(bad)

    def test_statistics(self, tmp_path: Path) -> None:
        cache = ModelCache(_cfg(tmp_path, max_size_mb=1))
        f = tmp_path / "x.urdf"
        f.write_bytes(b"x" * 1024)
        cache.put("m1", f)
        stats = cache.get_statistics()
        assert stats["entry_count"] == 1
        assert stats["total_size_bytes"] >= 1024
        assert "cache_dir" in stats
        assert stats["max_size_mb"] == 1

    def test_index_persisted_and_reloaded(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        cache = ModelCache(cfg)
        f = tmp_path / "x.urdf"
        f.write_bytes(b"hello")
        cache.put("m1", f)

        # New instance with same cache_dir should load the entry
        cache2 = ModelCache(cfg)
        assert "m1" in cache2
        assert len(cache2) == 1

    def test_index_skips_missing_files_on_reload(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        cache = ModelCache(cfg)
        f = tmp_path / "x.urdf"
        f.write_bytes(b"hello")
        cache.put("m1", f)
        f.unlink()

        cache2 = ModelCache(cfg)
        assert "m1" not in cache2

    def test_clear(self, tmp_path: Path) -> None:
        cache = ModelCache(_cfg(tmp_path))
        f = tmp_path / "x.urdf"
        f.write_bytes(b"y")
        cache.put("m1", f)
        cache.clear()
        assert len(cache) == 0

    def test_get_size_directory(self, tmp_path: Path) -> None:
        cache = ModelCache(_cfg(tmp_path))
        d = tmp_path / "dir"
        d.mkdir()
        (d / "a").write_bytes(b"12345")
        (d / "b").write_bytes(b"67")
        entry = cache.put("m1", d)
        # Directory entries get no checksum but size reflects content
        assert entry.checksum is None
        assert entry.size_bytes == 7

    def test_get_size_nonexistent_path(self, tmp_path: Path) -> None:
        cache = ModelCache(_cfg(tmp_path))
        assert cache._get_size(tmp_path / "nonexistent") == 0

    def test_none_model_id_raises(self, tmp_path: Path) -> None:
        cache = ModelCache(_cfg(tmp_path))
        with pytest.raises(ValueError):
            cache.get(None)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            cache.contains(None)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            cache.verify(None)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            cache.remove(None)  # type: ignore[arg-type]

    def test_cleanup_respects_min_age(self, tmp_path: Path) -> None:
        cfg = _cfg(
            tmp_path,
            max_size_mb=1,
            cleanup_threshold=0.0,  # always trigger
            min_age_days=365,  # never old enough
        )
        cache = ModelCache(cfg)
        f = tmp_path / "x.urdf"
        f.write_bytes(b"x" * 2048)
        cache.put("m1", f)
        # Cleanup should not delete because cached_at is recent
        assert "m1" in cache

    def test_cleanup_removes_old(self, tmp_path: Path) -> None:
        cfg = _cfg(
            tmp_path,
            max_size_mb=1,
            cleanup_threshold=0.0,
            min_age_days=0,
        )
        cache = ModelCache(cfg)
        f = tmp_path / "x.urdf"
        f.write_bytes(b"x" * (2 * 1024 * 1024))  # > max
        cache.put("m1", f)
        # Force cached_at to be old
        cache._entries["m1"].cached_at = time.time() - 86400 * 30
        cache._maybe_cleanup()
        assert "m1" not in cache

    def test_load_corrupt_index(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        (cfg.cache_dir / ModelCache.INDEX_FILE).write_text("not json")
        # Should not raise; OSError/PermissionError chain is wide enough
        cache = ModelCache(cfg)
        # Index may be empty because parsing failed
        assert len(cache) == 0

    def test_save_index_failure_swallowed(self, tmp_path: Path, monkeypatch) -> None:
        cache = ModelCache(_cfg(tmp_path))

        def bad_dumps(*a, **k):
            raise ValueError("boom")

        monkeypatch.setattr("model_generation.library.cache.json.dumps", bad_dumps)
        f = tmp_path / "x.urdf"
        f.write_bytes(b"x")
        # put will trigger _save_index — should not raise
        cache.put("m1", f)

    def test_get_updates_last_accessed(self, tmp_path: Path) -> None:
        cache = ModelCache(_cfg(tmp_path))
        f = tmp_path / "x.urdf"
        f.write_bytes(b"x")
        cache.put("m1", f)
        cache._entries["m1"].last_accessed = 0.0
        got = cache.get("m1")
        assert got is not None
        assert cache._entries["m1"].last_accessed > 0.0
