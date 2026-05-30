"""API package for Golf Modeling Suite.

Heavy sub-packages (``auth``, ``routes``, ``services``) are imported lazily via
``__getattr__`` so that lightweight consumers — such as ``from src.api import
versioning`` or CLI scripts — can import this package without pulling in
SQLAlchemy and the full web-framework stack.

The public API is unchanged: names that used to be importable as
``from src.api import AnalysisService`` (etc.) are still importable; they just
resolve on first access rather than at module-load time.
"""

from __future__ import annotations

import importlib
import logging
import sys
from types import ModuleType
from typing import Any

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Eagerly-available names: versioning lives here and has no heavy dependencies.
# Nothing else is imported eagerly.
# ---------------------------------------------------------------------------

__all__: list[str] = [
    # auth
    "get_current_user",
    # middleware
    "add_security_headers",
    "handle_api_errors",
    # routes
    "launcher_router",
    "models_router",
    "physics_router",
    "simulation_router",
    # services
    "AnalysisService",
    "ChatService",
    "LauncherService",
    "SimulationService",
    "SimulationStats",
]

# ---------------------------------------------------------------------------
# Mapping of attribute name -> (submodule, attribute_in_submodule)
# for symbols that are re-exported at the package level.
# ---------------------------------------------------------------------------
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # auth
    "get_current_user": ("src.api.auth", "get_current_user"),
    # middleware
    "add_security_headers": ("src.api.middleware", "add_security_headers"),
    "handle_api_errors": ("src.api.middleware", "handle_api_errors"),
    # routes
    "launcher_router": ("src.api.routes", "launcher_router"),
    "models_router": ("src.api.routes", "models_router"),
    "physics_router": ("src.api.routes", "physics_router"),
    "simulation_router": ("src.api.routes", "simulation_router"),
    # services
    "AnalysisService": ("src.api.services", "AnalysisService"),
    "ChatService": ("src.api.services", "ChatService"),
    "LauncherService": ("src.api.services", "LauncherService"),
    "SimulationService": ("src.api.services", "SimulationService"),
    "SimulationStats": ("src.api.services", "SimulationStats"),
}

# Sub-packages that are exposed as sub-module attributes (``src.api.auth`` etc.)
_LAZY_SUBPACKAGES: frozenset[str] = frozenset(
    {"auth", "routes", "services", "middleware"}
)


def __getattr__(name: str) -> Any:
    """Lazily resolve heavy sub-package attributes on first access.

    Preconditions:
        name is a string attribute name on this module.

    Raises:
        ImportError: if the underlying sub-module cannot be imported (e.g.
            SQLAlchemy is absent).
        AttributeError: if *name* is not a known lazy attribute or sub-package.
    """
    if not isinstance(name, str):
        raise TypeError(f"attribute name must be a str, not {type(name)!r}")

    # 1. Direct symbol re-exports (e.g. get_current_user, AnalysisService, …)
    if name in _LAZY_ATTRS:
        module_path, attr = _LAZY_ATTRS[name]
        mod: ModuleType = importlib.import_module(module_path)
        return getattr(mod, attr)

    # 2. Sub-package attributes (e.g. src.api.auth, src.api.routes, …)
    if name in _LAZY_SUBPACKAGES:
        return importlib.import_module(f"src.api.{name}")

    # 3. Versioning is a sibling module; expose it as an attribute too so that
    #    ``import src.api; src.api.versioning`` works without SQLAlchemy.
    if name == "versioning":
        return importlib.import_module("src.api.versioning")

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Make ``sys.modules["src.api"].versioning`` work via the normal attribute
# path without requiring the caller to explicitly import it.
# ---------------------------------------------------------------------------
def _preload_versioning() -> None:
    """Pre-populate the versioning sub-module reference; it has no heavy deps."""
    try:
        versioning_mod = importlib.import_module("src.api.versioning")
        # Attach to this module's namespace so attribute access is O(1) after
        # the first call, and so ``import src.api.versioning`` succeeds without
        # triggering __getattr__ again.
        sys.modules[__name__].__dict__["versioning"] = versioning_mod
    except (ImportError, ModuleNotFoundError) as exc:
        _logger.debug("versioning module unavailable in this environment: %s", exc)


_preload_versioning()
