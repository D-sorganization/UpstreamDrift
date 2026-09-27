"""API route registration."""

from __future__ import annotations

import importlib
from typing import Any

__all__: list[str] = [
    "launcher_router",
    "models_router",
    "physics_router",
    "simulation_router",
]

_ROUTERS: dict[str, tuple[str, str]] = {
    "launcher_router": (".launcher", "router"),
    "models_router": (".models", "router"),
    "physics_router": (".physics", "router"),
    "simulation_router": (".simulation", "router"),
}


def __getattr__(name: str) -> Any:
    if name in _ROUTERS:
        mod_name, attr = _ROUTERS[name]
        mod = importlib.import_module(mod_name, package=__name__)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(__all__)
