"""API business-logic services."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "AnalysisService": ("src.api.services.analysis_service", "AnalysisService"),
    "ChatService": ("src.api.services.chat_service", "ChatService"),
    "LauncherService": ("src.api.services.launcher_service", "LauncherService"),
    "SimulationService": ("src.api.services.simulation_service", "SimulationService"),
    "SimulationStats": ("src.api.services.simulation_service", "SimulationStats"),
}

__all__: list[str] = [
    "AnalysisService",
    "ChatService",
    "LauncherService",
    "SimulationService",
    "SimulationStats",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve concrete services when callers request them."""
    if not isinstance(name, str):
        raise TypeError(f"attribute name must be a str, not {type(name)!r}")

    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr = _LAZY_ATTRS[name]
    module: ModuleType = importlib.import_module(module_path)
    return getattr(module, attr)
