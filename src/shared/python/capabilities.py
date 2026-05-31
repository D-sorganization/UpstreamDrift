"""Backward-compatible shim — canonical location: engine_core.capabilities."""

from src.shared.python.engine_core.capabilities import (  # noqa: F401
    ADAPTER_BOUNDARY_CAPABILITIES,
    ENGINE_CAPABILITY_FIELDS,
    Capability,
    CapabilityLevel,
    CapabilityQuery,
    CapabilityRef,
    EngineCapabilities,
    capability_level_supported,
    normalize_capability,
)

__all__ = [
    "ADAPTER_BOUNDARY_CAPABILITIES",
    "ENGINE_CAPABILITY_FIELDS",
    "Capability",
    "CapabilityLevel",
    "CapabilityQuery",
    "CapabilityRef",
    "EngineCapabilities",
    "capability_level_supported",
    "normalize_capability",
]
