"""API business-logic services.

Service classes are re-exported lazily so importing one service submodule does
not import optional dependencies owned by unrelated services.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__: list[str] = [
    "AnalysisService",
    "ChatService",
    "LauncherService",
    "SimulationService",
    "SimulationStats",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "AnalysisService": ("analysis_service", "AnalysisService"),
    "ChatService": ("chat_service", "ChatService"),
    "LauncherService": ("launcher_service", "LauncherService"),
    "SimulationService": ("simulation_service", "SimulationService"),
    "SimulationStats": ("simulation_service", "SimulationStats"),
}


def __getattr__(name: str) -> Any:
    """Resolve service re-exports only when requested."""
    if not isinstance(name, str):
        raise TypeError(f"attribute name must be a str, not {type(name)!r}")
    try:
        module_name, attr_name = _LAZY_ATTRS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
