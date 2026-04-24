"""Tests for engine_loaders module (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.engine_loaders import LOADER_MAP


class TestEngineLoaders:
    def test_loader_map_not_empty(self) -> None:
        assert len(LOADER_MAP) > 0

    def test_loader_map_values_callable(self) -> None:
        for loader_fn in LOADER_MAP.values():
            assert callable(loader_fn)

    def test_loader_map_has_pendulum(self) -> None:
        names = [e.value for e in LOADER_MAP]
        assert "pendulum" in names

    def test_loader_map_has_golf_swing(self) -> None:
        names = [e.value for e in LOADER_MAP]
        assert "golf_swing_pendulum" in names
