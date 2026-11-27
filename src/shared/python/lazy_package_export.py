"""Shared lazy ``__getattr__`` resolution for package facades (DRY gate #7315)."""

from __future__ import annotations

import importlib
from typing import Any, Mapping


def resolve_lazy_export(
    name: str,
    *,
    exports: Mapping[str, tuple[str, str]],
    module_globals: dict[str, Any],
    facade_module_name: str,
) -> Any:
    """Import ``exports[name]`` once and cache it on ``module_globals``."""
    spec = exports.get(name)
    if spec is None:
        raise AttributeError(f"module {facade_module_name!r} has no attribute {name!r}")
    module_name, attr_name = spec
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    module_globals[name] = value
    return value
