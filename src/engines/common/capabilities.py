"""Engine Capabilities — backward-compatible re-export.

The canonical location for EngineCapabilities and CapabilityLevel is now
``src.shared.python.capabilities``.  This shim preserves the old import
path so existing engine code continues to work without changes.

Migration:
    Old: from src.engines.common.capabilities import EngineCapabilities
    New: from src.shared.python.engine_core.capabilities import EngineCapabilities
"""

from src.shared.python.engine_core.capabilities import (
    ADAPTER_BOUNDARY_CAPABILITIES,
    ENGINE_CAPABILITY_FIELDS,
    SPATIAL_JACOBIAN_ORDER,
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
    "SPATIAL_JACOBIAN_ORDER",
    "Capability",
    "CapabilityLevel",
    "CapabilityQuery",
    "CapabilityRef",
    "EngineCapabilities",
    "capability_level_supported",
    "normalize_capability",
]
