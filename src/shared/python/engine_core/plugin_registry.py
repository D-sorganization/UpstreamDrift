"""Plugin registry with thread-safe operations and entry-point discovery.

Extends the base EngineRegistry with:
- Thread-safe registration and unregistration
- Entry-point based plugin discovery for third-party engines
- Plugin metadata support
- Engine lifecycle management (startup/shutdown)

Design by Contract:
    Invariants:
        - _registrations dict is never None
        - _lock is always held during mutations
        - All registered types are valid EngineType values
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any

from src.shared.python.engine_core.engine_registry import (
    EngineFactory,
    EngineRegistration,
    EngineType,
)

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "upstream_drift.engines"


@dataclass
class EnginePluginMetadata:
    """Metadata for a registered engine plugin.

    Attributes:
        name: Human-readable plugin name.
        version: Semantic version string.
        engine_type: The EngineType this plugin provides.
        author: Plugin author (optional).
        description: Short description (optional).
        requires: List of required packages (optional).
    """

    name: str
    version: str
    engine_type: EngineType
    author: str = ""
    description: str = ""
    requires: list[str] = field(default_factory=list)


class PluginRegistry:
    """Thread-safe engine plugin registry with metadata and discovery.

    Extends the engine registry concept with:
    - Thread-safe read/write via threading.Lock
    - Unregistration support for testing and dynamic plugins
    - Plugin metadata storage alongside registrations
    - Entry-point based discovery

    Design by Contract:
        Invariants:
            - _registrations is always a dict[EngineType, EngineRegistration]
            - _metadata is always a dict[EngineType, EnginePluginMetadata]
            - Concurrent access is serialized via _lock
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._registrations: dict[EngineType, EngineRegistration] = {}
        self._metadata: dict[EngineType, EnginePluginMetadata] = {}

    def register(self, registration: EngineRegistration) -> None:
        """Register an engine (thread-safe, overwrites existing).

        Preconditions:
            - registration is not None
            - registration.engine_type is a valid EngineType

        Postconditions:
            - get(registration.engine_type) returns registration
        """
        with self._lock:
            self._registrations[registration.engine_type] = registration

    def unregister(self, engine_type: EngineType) -> bool:
        """Remove an engine registration.

        Preconditions:
            - engine_type is a valid EngineType

        Postconditions:
            - get(engine_type) returns None
            - Returns True if engine was registered, False otherwise
        """
        with self._lock:
            if engine_type in self._registrations:
                del self._registrations[engine_type]
                self._metadata.pop(engine_type, None)
                return True
            return False

    def get(self, engine_type: EngineType) -> EngineRegistration | None:
        """Get registration for an engine type (thread-safe)."""
        with self._lock:
            return self._registrations.get(engine_type)

    def all_types(self) -> list[EngineType]:
        """Get all registered engine types (thread-safe)."""
        with self._lock:
            return list(self._registrations.keys())

    def register_plugin(
        self,
        engine_type: EngineType,
        factory: EngineFactory,
        metadata: EnginePluginMetadata,
    ) -> None:
        """Register a plugin with metadata.

        Preconditions:
            - engine_type, factory, and metadata are not None
            - metadata.engine_type matches engine_type

        Postconditions:
            - get(engine_type) returns an EngineRegistration
            - get_metadata(engine_type) returns metadata
        """
        reg = EngineRegistration(engine_type=engine_type, factory=factory)
        with self._lock:
            self._registrations[engine_type] = reg
            self._metadata[engine_type] = metadata

    def get_metadata(self, engine_type: EngineType) -> EnginePluginMetadata | None:
        """Get metadata for a registered plugin."""
        with self._lock:
            return self._metadata.get(engine_type)

    def load_entry_point_plugins(self) -> int:
        """Discover and register plugins from entry points.

        Returns:
            Number of plugins successfully loaded.
        """
        plugins = discover_entry_point_plugins()
        count = 0
        for plugin_info in plugins:
            et = plugin_info["engine_type"]
            factory = plugin_info["factory"]
            meta = plugin_info.get("metadata")
            if meta:
                self.register_plugin(et, factory, meta)
            else:
                self.register(EngineRegistration(engine_type=et, factory=factory))
            count += 1
        return count


class EngineLifecycle:
    """Tracks active engine instances and manages their lifecycle.

    Ensures engines are properly shut down when replaced or when
    the application exits.

    Design by Contract:
        Invariants:
            - _active is always a dict mapping EngineType to engine instances
    """

    def __init__(self) -> None:
        self._active: dict[EngineType, Any] = {}

    def track(self, engine_type: EngineType, engine: Any) -> None:
        """Start tracking an active engine instance.

        Preconditions:
            - engine_type is a valid EngineType
            - engine is not None
        """
        self._active[engine_type] = engine

    def is_active(self, engine_type: EngineType) -> bool:
        """Check if an engine type is currently active."""
        return engine_type in self._active

    def shutdown(self, engine_type: EngineType) -> None:
        """Shut down a specific engine if it has a shutdown method.

        Postconditions:
            - Engine is no longer tracked
            - shutdown() called if engine supports it
        """
        engine = self._active.pop(engine_type, None)
        if engine is None:
            return
        if hasattr(engine, "shutdown") and callable(engine.shutdown):
            try:
                engine.shutdown()
            except Exception:
                logger.warning(
                    "Engine shutdown failed for %s", engine_type, exc_info=True
                )

    def shutdown_all(self) -> None:
        """Shut down all tracked engines.

        Postconditions:
            - No engines are active
        """
        for engine_type in list(self._active.keys()):
            self.shutdown(engine_type)


def discover_entry_point_plugins() -> list[dict[str, Any]]:
    """Discover engine plugins registered via entry points.

    Looks for entry points in the ``upstream_drift.engines`` group.
    Each entry point should resolve to a dict with:
        - engine_type: EngineType
        - factory: Callable[[], PhysicsEngine]
        - metadata: EnginePluginMetadata (optional)

    Returns:
        List of plugin info dicts. Broken plugins are skipped with a warning.
    """
    results: list[dict[str, Any]] = []
    try:
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:
        # Python 3.9 compatibility: entry_points() doesn't accept group kwarg
        eps = entry_points().get(ENTRY_POINT_GROUP, [])  # type: ignore[union-attr]

    for ep in eps:
        try:
            plugin_info = ep.load()
            if isinstance(plugin_info, dict) and "engine_type" in plugin_info:
                results.append(plugin_info)
            else:
                logger.warning(
                    "Entry point %s did not return a valid plugin dict", ep.name
                )
        except Exception:
            logger.warning(
                "Failed to load engine plugin %s", ep.name, exc_info=True
            )

    return results
