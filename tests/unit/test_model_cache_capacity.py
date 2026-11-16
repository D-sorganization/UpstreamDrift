"""Cache entry pressure must consider additions while protecting recent files."""

from pathlib import Path

import pytest
from src.shared.python.model_generation.library.cache import (
    CacheConfig,
    CacheEntry,
    ModelCache,
)

pytestmark = pytest.mark.unit


def _make_cache(tmp_path: Path, max_entries: int, min_age_days: int = 7) -> ModelCache:
    return ModelCache(
        CacheConfig(
            cache_dir=tmp_path,
            max_size_mb=1,
            max_entries=max_entries,
            min_age_days=min_age_days,
        )
    )


def _add_model(cache: ModelCache, model_id: str) -> CacheEntry:
    path = cache.get_cache_path(model_id + ".urdf")
    path.write_text(f"<robot name='{model_id}'/>", encoding="utf-8")
    return cache.put(model_id, path)


@pytest.mark.parametrize("min_age_days", [0, 7])
def test_put_enforces_count_pressure_and_persists_remaining_entry(
    tmp_path: Path, min_age_days: int
) -> None:
    """An eligible old file is evicted on the insertion that exceeds count."""
    cache = _make_cache(tmp_path, 1, min_age_days)
    old = _add_model(cache, "old")
    old.cached_at = 0.0
    old.last_accessed = 0.0
    new = _add_model(cache, "new")

    assert cache.get_total_size() < 1024
    assert len(cache) == 1
    assert cache.get("old") is None
    assert not old.local_path.exists()
    assert cache.get("new") is new
    assert new.local_path.exists()
    restored = ModelCache(cache.config)
    assert len(restored) == 1
    loaded = restored.get("new")
    assert loaded is not None and loaded.local_path == new.local_path


def test_count_cleanup_retains_most_recently_accessed_eligible_model(
    tmp_path: Path,
) -> None:
    """Count-only pressure uses LRU order even below the byte-size target."""
    cache = _make_cache(tmp_path, 10)
    for index in range(3):
        entry = _add_model(cache, f"old{index}")
        entry.cached_at = 0.0
        entry.last_accessed = float(index)
    cache.config.max_entries = 2
    _add_model(cache, "new")

    assert len(cache) == 2
    assert cache.get("old0") is None
    assert cache.get("old1") is None
    assert cache.get("old2") is not None
    assert cache.get("new") is not None


def test_count_pressure_preserves_minimum_age_protection(tmp_path: Path) -> None:
    """Fresh files remain protected even when the soft count limit is exceeded."""
    cache = _make_cache(tmp_path, 1)
    first = _add_model(cache, "first")
    second = _add_model(cache, "second")
    assert len(cache) == 2
    assert first.local_path.exists() and second.local_path.exists()
    assert cache.get("first") is not None and cache.get("second") is not None
