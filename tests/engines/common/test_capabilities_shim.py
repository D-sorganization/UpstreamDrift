"""Coverage for src.engines.common.capabilities shim module."""

from __future__ import annotations


def test_capabilities_shim_reexports() -> None:
    """The shim re-exports CapabilityLevel and EngineCapabilities."""
    from src.engines.common import capabilities as shim
    from src.shared.python.engine_core.capabilities import (
        CapabilityLevel,
        EngineCapabilities,
        SPATIAL_JACOBIAN_ORDER,
    )

    assert shim.CapabilityLevel is CapabilityLevel
    assert shim.EngineCapabilities is EngineCapabilities
    assert shim.SPATIAL_JACOBIAN_ORDER is SPATIAL_JACOBIAN_ORDER
    assert "CapabilityLevel" in shim.__all__
    assert "EngineCapabilities" in shim.__all__
