"""Route registry for automatic route discovery and registration.

Replaces the 20+ explicit route imports in server.py with a plugin-style
auto-discovery pattern. Each route module that defines a ``router`` attribute
(an ``APIRouter`` instance) is automatically discovered and registered.

Architecture (#1485):
    - Decouples server.py from individual route modules
    - Adding a new route module requires only creating the file
    - No changes to server.py needed for new routes
    - Supports prefix overrides and route filtering

Design by Contract:
    - Precondition: route modules must define a ``router`` attribute
    - Postcondition: all discovered routers are included in the app
    - Invariant: registration order matches ``_REGISTRATION_ORDER`` for
      modules with overlapping route paths (FastAPI first-match-wins)
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    from fastapi import APIRouter, FastAPI

logger = get_logger(__name__)

# Modules that should NOT be auto-discovered (they are WebSocket-only
# or require special handling)
_EXCLUDED_MODULES: frozenset[str] = frozenset(
    {
        "chat_ws",
        "simulation_ws",
    }
)

# Explicit registration order matching the original server.py.
# This is critical because some modules define overlapping route paths
# (e.g. /simulation/actuators in both physics.py and actuator_controls.py)
# and FastAPI uses first-match-wins semantics.
# Modules not listed here are appended alphabetically after these.
_REGISTRATION_ORDER: tuple[str, ...] = (
    "auth",
    "core",
    "engines",
    "simulation",
    "video",
    "analysis",
    "export",
    "launcher",
    "terrain",
    "dataset",
    "physics",
    "models",
    "analysis_tools",
    "force_overlays",
    "actuator_controls",
    "model_explorer",
    "aip",
    "putting_green",
    "data_explorer",
    "motion_capture",
)


def discover_routes(
    package_path: str = "src.api.routes",
    *,
    exclude: frozenset[str] | None = None,
) -> list[tuple[str, APIRouter]]:
    """Discover all route modules with a ``router`` attribute.

    Scans the given package for Python modules and imports each one.
    If a module exposes a top-level ``router`` attribute that is an
    ``APIRouter`` instance, it is included in the returned list.

    The returned list is ordered according to ``_REGISTRATION_ORDER``
    for modules that appear in that list, with any remaining modules
    appended in alphabetical order.  This preserves FastAPI's
    first-match-wins semantics for overlapping route paths.

    Args:
        package_path: Dotted import path to the routes package.
        exclude: Module names to skip (without package prefix).

    Returns:
        List of (module_name, router) tuples in registration order.

    Raises:
        ImportError: If the routes package itself cannot be imported.
    """
    if exclude is None:
        exclude = _EXCLUDED_MODULES

    package = importlib.import_module(package_path)
    if not hasattr(package, "__path__"):
        raise ImportError(f"{package_path} is not a package (no __path__)")

    # Build a lookup of discovered modules
    discovered_map: dict[str, APIRouter] = {}

    for _finder, module_name, _is_pkg in pkgutil.iter_modules(package.__path__):
        if module_name.startswith("_"):
            continue
        if module_name in exclude:
            logger.debug("Skipping excluded route module: %s", module_name)
            continue

        full_module_path = f"{package_path}.{module_name}"
        try:
            module = importlib.import_module(full_module_path)
        except ImportError:
            logger.warning(
                "Failed to import route module %s — skipping", full_module_path
            )
            continue

        router = getattr(module, "router", None)
        if router is None:
            logger.debug("Module %s has no 'router' attribute — skipping", module_name)
            continue

        discovered_map[module_name] = router
        logger.debug("Discovered route module: %s", module_name)

    # Order: priority modules first (in _REGISTRATION_ORDER), then remainder alphabetically
    ordered: list[tuple[str, APIRouter]] = []
    for name in _REGISTRATION_ORDER:
        if name in discovered_map:
            ordered.append((name, discovered_map.pop(name)))

    # Append any newly added modules not in _REGISTRATION_ORDER (alphabetically)
    for name in sorted(discovered_map):
        ordered.append((name, discovered_map[name]))

    logger.info("Discovered %d route modules", len(ordered))
    return ordered


def register_routes(
    app: FastAPI,
    *,
    prefix: str = "",
    exclude: frozenset[str] | None = None,
) -> int:
    """Discover and register all route modules on the given FastAPI app.

    This is the primary entry point used by ``server.py``.

    Args:
        app: The FastAPI application instance.
        prefix: Optional URL prefix to prepend to all routes (e.g. "/api/v1").
        exclude: Module names to skip.

    Returns:
        Number of routers registered.
    """
    routes = discover_routes(exclude=exclude)
    for module_name, router in routes:
        app.include_router(router, prefix=prefix)
        logger.debug("Registered router from %s with prefix '%s'", module_name, prefix)
    return len(routes)
