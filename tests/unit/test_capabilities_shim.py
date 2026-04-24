"""Tests for capabilities shim module (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.capabilities import CapabilityLevel, EngineCapabilities


class TestCapabilitiesShim:
    def test_capability_level_importable(self) -> None:
        assert CapabilityLevel is not None

    def test_engine_capabilities_importable(self) -> None:
        assert EngineCapabilities is not None

    def test_capability_level_has_values(self) -> None:
        assert len(list(CapabilityLevel)) > 0

    def test_engine_capabilities_constructable(self) -> None:
        cap = EngineCapabilities()
        assert cap is not None
