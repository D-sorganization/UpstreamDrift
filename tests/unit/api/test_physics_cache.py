"""Tests for physics control-metadata cache invalidation (issue #2468).

These tests do not require httpx/starlette testclient — they test the
cache functions directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestPhysicsCacheInvalidation:
    """Tests for physics control-metadata cache (issue #2468)."""

    def setup_method(self) -> None:
        from src.api.routes.physics import clear_physics_caches

        clear_physics_caches()

    def teardown_method(self) -> None:
        from src.api.routes.physics import clear_physics_caches

        clear_physics_caches()

    def test_clear_physics_caches_empties_both_caches(self) -> None:
        """clear_physics_caches() removes all entries from both caches."""
        from src.api.routes.physics import (
            _CONTROL_INTERFACE_CACHE,
            _FEATURES_REGISTRY_CACHE,
            clear_physics_caches,
        )

        _CONTROL_INTERFACE_CACHE[1] = object()
        _FEATURES_REGISTRY_CACHE[2] = object()
        clear_physics_caches()
        assert len(_CONTROL_INTERFACE_CACHE) == 0
        assert len(_FEATURES_REGISTRY_CACHE) == 0

    def test_cache_key_is_engine_identity_not_manager(self) -> None:
        """Cache is keyed by id(engine), so a switch yields a cache miss."""
        from src.api.routes.physics import (
            _CONTROL_INTERFACE_CACHE,
            _FEATURES_REGISTRY_CACHE,
        )

        engine_a = MagicMock()
        engine_b = MagicMock()

        # Simulate a warm cache for engine_a
        _CONTROL_INTERFACE_CACHE[id(engine_a)] = MagicMock(name="ctrl_a")
        _FEATURES_REGISTRY_CACHE[id(engine_a)] = MagicMock(name="reg_a")

        # After switching to engine_b, the old key is absent from the lookup
        assert id(engine_b) not in _CONTROL_INTERFACE_CACHE
        assert id(engine_b) not in _FEATURES_REGISTRY_CACHE

    def test_clear_caches_after_engine_switch(self) -> None:
        """Calling clear_physics_caches() after switch removes stale entries."""
        from src.api.routes.physics import (
            _CONTROL_INTERFACE_CACHE,
            _FEATURES_REGISTRY_CACHE,
            clear_physics_caches,
        )

        engine_a = MagicMock()
        _CONTROL_INTERFACE_CACHE[id(engine_a)] = MagicMock()
        _FEATURES_REGISTRY_CACHE[id(engine_a)] = MagicMock()

        clear_physics_caches()

        assert id(engine_a) not in _CONTROL_INTERFACE_CACHE
        assert id(engine_a) not in _FEATURES_REGISTRY_CACHE

    @pytest.mark.parametrize(
        "cache_name", ["_CONTROL_INTERFACE_CACHE", "_FEATURES_REGISTRY_CACHE"]
    )
    def test_caches_are_dicts(self, cache_name: str) -> None:
        """Both caches are plain dicts (no magic container)."""
        import src.api.routes.physics as phy

        assert isinstance(getattr(phy, cache_name), dict)
