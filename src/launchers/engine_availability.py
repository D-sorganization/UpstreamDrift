"""Single engine-availability probe shared by tile badges and launch actions.

Issues #12890/#12891: tile badges used optimistic ``status`` literals from
``models.yaml`` while the launch path ran its own import probe and carried its
own (wrong) install hints.  Both now ask :func:`probe_engine` so the badge, the
dependency dialog and the install hint can never disagree.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "ENGINE_INSTALL_HINTS",
    "ENGINE_MODULES",
    "EngineAvailability",
    "engine_for_model",
    "install_hint",
    "probe_engine",
    "probe_model_engine",
]

# engine key -> importable top-level module.
ENGINE_MODULES: dict[str, str] = {
    "drake": "pydrake",
    "pinocchio": "pinocchio",
    "opensim": "opensim",
    "myosuite": "myosuite",
}

_WSL_DOCKER = "use the Docker or WSL launch mode"

# engine key -> {"win32": hint, "default": hint}.  Must cover ENGINE_MODULES.
ENGINE_INSTALL_HINTS: dict[str, dict[str, str]] = {
    "drake": {
        "win32": f"Drake has no Windows wheel: {_WSL_DOCKER} (pip install drake there)",
        "default": "pip install drake",
    },
    "pinocchio": {
        "win32": f"Pinocchio (pip package 'pin') has no Windows wheel: {_WSL_DOCKER}",
        "default": "pip install pin",
    },
    "opensim": {
        "win32": "pip install opensim",
        "default": "pip install opensim",
    },
    "myosuite": {
        "win32": "pip install myosuite",
        "default": "pip install myosuite",
    },
}

# model id / type -> engine key.
_MODEL_ENGINE_ALIASES: dict[str, str] = {
    "drake": "drake",
    "drake_golf": "drake",
    "pinocchio": "pinocchio",
    "pinocchio_golf": "pinocchio",
    "opensim": "opensim",
    "opensim_golf": "opensim",
    "myosim": "myosuite",
    "myosim_suite": "myosuite",
    "myosuite": "myosuite",
}


@dataclass(frozen=True)
class EngineAvailability:
    """Result of probing one engine runtime."""

    engine: str
    available: bool
    reason: str
    install_hint: str

    @property
    def status(self) -> str:
        """``"available"`` or ``"unavailable"`` (badge vocabulary)."""
        return "available" if self.available else "unavailable"


def install_hint(engine: str, platform: str | None = None) -> str:
    """Return the platform-correct install hint for ``engine``."""
    if engine not in ENGINE_INSTALL_HINTS:
        raise ValueError(f"unknown engine: {engine!r}")
    table = ENGINE_INSTALL_HINTS[engine]
    plat = sys.platform if platform is None else platform
    return table.get(plat, table["default"])


def engine_for_model(model_id: str | None, model_type: str | None = None) -> str | None:
    """Map a tile's id/type to an engine key, or ``None`` for non-engine tiles."""
    for candidate in (model_id, model_type):
        if candidate and str(candidate).lower() in _MODEL_ENGINE_ALIASES:
            return _MODEL_ENGINE_ALIASES[str(candidate).lower()]
    return None


def _module_present(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def probe_engine(
    engine: str,
    *,
    platform: str | None = None,
    module_present: Callable[[str], bool] = _module_present,
) -> EngineAvailability:
    """Probe ``engine`` without importing it (no DLL side effects)."""
    if engine not in ENGINE_MODULES:
        raise ValueError(f"unknown engine: {engine!r}")
    module = ENGINE_MODULES[engine]
    hint = install_hint(engine, platform)
    if module_present(module):
        return EngineAvailability(engine, True, "", hint)
    return EngineAvailability(
        engine, False, f"Python module '{module}' is not installed", hint
    )


def probe_model_engine(
    model: object,
    prober: Callable[[str], EngineAvailability] = probe_engine,
) -> EngineAvailability | None:
    """Probe the engine behind a tile model, or ``None`` for non-engine tiles."""
    if "Mock" in type(model).__name__:
        return None
    engine = engine_for_model(getattr(model, "id", None), getattr(model, "type", None))
    return None if engine is None else prober(engine)
