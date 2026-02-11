"""Engine Capabilities — backward-compatible re-export.

The canonical location for EngineCapabilities and CapabilityLevel is now
``src.shared.python.capabilities``.  This shim preserves the old import
path so existing engine code continues to work without changes.

Migration:
    Old: from src.engines.common.capabilities import EngineCapabilities
    New: from src.shared.python.engine_core.capabilities import EngineCapabilities
"""

from src.shared.python.engine_core.capabilities import (
    CapabilityLevel,
    EngineCapabilities,
)

__all__ = ["CapabilityLevel", "EngineCapabilities"]
