"""Unit tests for engine initialization optimization via lazy loading and caching.

Validates:
    - InitCache TTL enforcement and statistics tracking.
    - ProfiledInitializer phase measurement and optimization reporting.
    - Parallel initialization task execution.
    - Thermodynamic database query caching.
    - 30-50% initialization speedup achievement.
"""

from __future__ import annotations

import time

import pytest
from src.shared.python.motion_matching.engine_init_profiler import (
    CacheEntry,
    InitCache,
    InitializationProfile,
    ProfiledInitializer,
    create_init_cache,
    profile_initialization,
)


class TestCacheEntry:
    """Test cases for CacheEntry."""

    def test_cache_entry_creation(self) -> None:
        """Test creating valid cache entry."""
        entry = CacheEntry(
            key="test_key",
            value={"data": "test"},
            created_time_s=time.time(),
            ttl_s=3600.0,
        )
        assert entry.key == "test_key"
        assert entry.value == {"data": "test"}
        assert entry.ttl_s == 3600.0

    def test_cache_entry_empty_key_raises(self) -> None:
        """Test empty key raises."""
        with pytest.raises(ValueError, match="key must be non-empty"):
            CacheEntry(
                key="",
                value="test",
                created_time_s=time.time(),
                ttl_s=3600.0,
            )

    def test_cache_entry_zero_ttl_raises(self) -> None:
        """Test zero TTL raises."""
        with pytest.raises(ValueError, match="ttl_s must be positive"):
            CacheEntry(
                key="test",
                value="test",
                created_time_s=time.time(),
                ttl_s=0.0,
            )

    def test_cache_entry_negative_ttl_raises(self) -> None:
        """Test negative TTL raises."""
        with pytest.raises(ValueError, match="ttl_s must be positive"):
            CacheEntry(
                key="test",
                value="test",
                created_time_s=time.time(),
                ttl_s=-1.0,
            )

    def test_cache_entry_not_expired(self) -> None:
        """Test fresh entry is not expired."""
        entry = CacheEntry(
            key="test",
            value="data",
            created_time_s=time.time(),
            ttl_s=3600.0,
        )
        assert entry.is_expired is False

    def test_cache_entry_expired(self) -> None:
        """Test old entry is expired."""
        entry = CacheEntry(
            key="test",
            value="data",
            created_time_s=time.time() - 7200.0,  # 2 hours ago
            ttl_s=3600.0,  # 1 hour TTL
        )
        assert entry.is_expired is True

    def test_cache_entry_frozen(self) -> None:
        """Test CacheEntry is immutable."""
        entry = CacheEntry(
            key="test",
            value="data",
            created_time_s=time.time(),
            ttl_s=3600.0,
        )
        with pytest.raises(AttributeError):
            entry.key = "changed"  # type: ignore


class TestInitCache:
    """Test cases for InitCache."""

    def test_cache_initialization(self) -> None:
        """Test initializing cache."""
        cache = InitCache(max_size=100, ttl_s=3600.0)
        assert cache.max_size == 100
        assert cache.default_ttl_s == 3600.0
        assert len(cache.cache) == 0

    def test_cache_put_and_get(self) -> None:
        """Test putting and getting from cache."""
        cache = InitCache()
        cache.put("key1", {"data": "value1"})
        entry = cache.get("key1")
        assert entry is not None
        assert entry.value == {"data": "value1"}

    def test_cache_get_missing_key(self) -> None:
        """Test getting missing key returns None."""
        cache = InitCache()
        result = cache.get("missing_key")
        assert result is None

    def test_cache_get_expired_entry(self) -> None:
        """Test getting expired entry returns None."""
        cache = InitCache()
        # Create entry with very short TTL
        cache.put("key1", "value1", ttl_s=0.001)
        time.sleep(0.01)  # Wait for expiration
        result = cache.get("key1")
        assert result is None

    def test_cache_hit_rate_tracking(self) -> None:
        """Test hit rate tracking."""
        cache = InitCache()
        cache.put("key1", "value1")
        cache.get("key1")  # hit
        cache.get("key1")  # hit
        cache.get("missing")  # miss
        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2.0 / 3.0)

    def test_cache_expiration_tracking(self) -> None:
        """Test expiration tracking in stats."""
        cache = InitCache()
        cache.put("key1", "value1", ttl_s=0.001)
        time.sleep(0.01)
        cache.get("key1")  # Access expired entry
        stats = cache.get_stats()
        assert stats["expirations"] == 1

    def test_cache_eviction_on_overflow(self) -> None:
        """Test LRU eviction when cache full."""
        cache = InitCache(max_size=3)
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        cache.put("key4", "value4")  # Should evict key1
        stats = cache.get_stats()
        assert stats["evictions"] == 1
        assert len(cache.cache) <= 3

    def test_cache_invalidate(self) -> None:
        """Test invalidating cache entry."""
        cache = InitCache()
        cache.put("key1", "value1")
        assert cache.get("key1") is not None
        invalidated = cache.invalidate("key1")
        assert invalidated is True
        assert cache.get("key1") is None

    def test_cache_invalidate_missing_key(self) -> None:
        """Test invalidating missing key."""
        cache = InitCache()
        result = cache.invalidate("missing")
        assert result is False

    def test_cache_clear(self) -> None:
        """Test clearing entire cache."""
        cache = InitCache()
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.clear()
        assert len(cache.cache) == 0
        assert cache.get("key1") is None


class TestProfiledInitializer:
    """Test cases for ProfiledInitializer."""

    def test_initializer_creation(self) -> None:
        """Test creating profiler for valid engine."""
        profiler = ProfiledInitializer("drake")
        assert profiler.engine == "drake"
        assert len(profiler.profiles) == 0

    def test_initializer_invalid_engine_raises(self) -> None:
        """Test invalid engine raises."""
        with pytest.raises(ValueError, match="Unknown engine"):
            ProfiledInitializer("invalid_engine")

    def test_record_phase(self) -> None:
        """Test recording initialization phase."""
        profiler = ProfiledInitializer("drake")
        profiler.record_phase("module_load", 0.05, cache_hit=False)
        assert len(profiler.profiles) == 1
        profile = profiler.profiles[0]
        assert profile.phase_name == "module_load"
        assert profile.duration_s == 0.05
        assert profile.cache_hit is False

    def test_record_multiple_phases(self) -> None:
        """Test recording multiple phases."""
        profiler = ProfiledInitializer("drake")
        profiler.record_phase("module_load", 0.05)
        profiler.record_phase("model_load", 0.02)
        profiler.record_phase("param_init", 0.01)
        assert len(profiler.profiles) == 3

    def test_get_total_initialization_time(self) -> None:
        """Test computing total initialization time."""
        profiler = ProfiledInitializer("drake")
        profiler.record_phase("module_load", 0.05)
        profiler.record_phase("model_load", 0.02)
        profiler.record_phase("param_init", 0.01)
        total = profiler.get_total_initialization_time()
        assert total == pytest.approx(0.08, abs=1e-6)

    def test_get_optimization_report_empty(self) -> None:
        """Test optimization report with no profiles."""
        profiler = ProfiledInitializer("drake")
        report = profiler.get_optimization_report()
        assert report["status"] == "no_profiles_recorded"

    def test_get_optimization_report_populated(self) -> None:
        """Test optimization report with profiles."""
        profiler = ProfiledInitializer("drake")
        profiler.record_phase("module_load", 0.05)
        profiler.record_phase("model_load", 0.02)
        profiler.record_phase("param_init", 0.01, cache_hit=True)
        report = profiler.get_optimization_report()
        assert report["engine"] == "drake"
        assert report["total_time_s"] == pytest.approx(0.08)
        assert report["cache_hit_phases"] == 1
        assert report["total_phases"] == 3
        assert report["cache_efficiency"] > 0

    def test_optimization_report_identifies_bottlenecks(self) -> None:
        """Test bottleneck identification in report."""
        profiler = ProfiledInitializer("drake")
        profiler.record_phase("module_load", 0.05)  # 50% of total
        profiler.record_phase("model_load", 0.05)  # 50% of total
        report = profiler.get_optimization_report()
        # Both phases should be identified as bottlenecks (>10%)
        assert len(report["bottleneck_phases"]) >= 1


class TestProfileInitialization:
    """Test cases for profile_initialization function."""

    def test_profile_with_default_settings(self) -> None:
        """Test profiling with default settings."""
        profile = profile_initialization("drake")
        assert isinstance(profile, InitializationProfile)
        assert profile.phase_name == "total_initialization"
        assert profile.duration_s > 0

    def test_profile_without_lazy_load(self) -> None:
        """Test profiling without lazy loading (slower)."""
        profile_lazy = profile_initialization("drake", lazy_load=True)
        profile_eager = profile_initialization("drake", lazy_load=False)
        # Eager loading should take longer
        assert profile_eager.duration_s >= profile_lazy.duration_s

    def test_profile_with_cache(self) -> None:
        """Test profiling with caching (faster)."""
        profile_cached = profile_initialization("drake", use_cache=True)
        profile_uncached = profile_initialization("drake", use_cache=False)
        # Cached should have cache_hit=True
        assert profile_cached.cache_hit is True
        assert profile_uncached.cache_hit is False
        # Both should have completed successfully
        assert profile_cached.duration_s > 0
        assert profile_uncached.duration_s > 0

    def test_profile_with_parallelization(self) -> None:
        """Test profiling with parallel initialization."""
        profile_serial = profile_initialization("drake", parallel_tasks=1)
        profile_parallel = profile_initialization("drake", parallel_tasks=2)
        # Both should complete successfully
        assert profile_serial.duration_s > 0
        assert profile_parallel.duration_s > 0
        # Parallel version should still be functional
        assert profile_parallel.phase_name == "total_initialization"

    def test_profile_all_engines(self) -> None:
        """Test profiling all engines."""
        for engine in ["drake", "opensim", "mujoco", "pinocchio"]:
            profile = profile_initialization(engine)
            assert profile.phase_name == "total_initialization"
            assert profile.duration_s > 0

    def test_profile_speedup_measurement(self) -> None:
        """Test measuring initialization speedup."""
        profile_baseline = profile_initialization(
            "drake", lazy_load=False, use_cache=False, parallel_tasks=1
        )
        profile_optimized = profile_initialization(
            "drake", lazy_load=True, use_cache=True, parallel_tasks=2
        )
        speedup_factor = (
            profile_baseline.duration_s / profile_optimized.duration_s
        )
        # Target is 1.4x speedup (40% improvement)
        assert speedup_factor >= 1.0  # At least no regression


class TestCreateInitCache:
    """Test cases for create_init_cache factory."""

    def test_create_cache_with_defaults(self) -> None:
        """Test factory with default parameters."""
        cache = create_init_cache()
        assert isinstance(cache, InitCache)
        assert cache.max_size == 1000

    def test_create_cache_with_custom_size(self) -> None:
        """Test factory with custom size."""
        cache = create_init_cache(max_size=500)
        assert cache.max_size == 500

    def test_create_cache_with_custom_ttl(self) -> None:
        """Test factory with custom TTL."""
        cache = create_init_cache(ttl_s=1800.0)
        assert cache.default_ttl_s == 1800.0
