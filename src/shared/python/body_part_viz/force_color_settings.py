"""Persist the force-colour scale and overlay toggles (Qt-free, engine-agnostic).

Engines store one JSON string per engine key in ``QSettings`` (organisation and
application ``UpstreamDrift``). Reading never raises: corrupt, legacy or
out-of-range values fall back to defaults with a logged warning, so a bad
settings file cannot stop a GUI from starting. Missing keys in a legacy scale
take the :class:`ForceColorScale` defaults.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any, Protocol

from .force_colors import ForceColorScale

logger = logging.getLogger(__name__)

SETTINGS_ORG = "UpstreamDrift"
SETTINGS_APP = "UpstreamDrift"

__all__ = [
    "SETTINGS_APP",
    "SETTINGS_ORG",
    "dump_force_color_state",
    "load_force_color_state",
    "open_force_color_settings",
    "parse_force_color_state",
    "save_force_color_state",
]


class SettingsStore(Protocol):
    """The ``QSettings`` subset used here."""

    def value(self, key: str, default: Any = None) -> Any:
        """Return the stored value or ``default``."""
        ...

    def setValue(self, key: str, value: Any) -> None:  # noqa: N802
        """Store ``value`` under ``key``."""
        ...


def _key(engine: str) -> str:
    if not isinstance(engine, str) or not engine.strip():
        raise ValueError("engine key must be a non-empty string")
    return f"force_color/{engine.strip()}"


def dump_force_color_state(scale: ForceColorScale, toggles: Mapping[str, bool]) -> str:
    """Serialise ``scale`` and boolean ``toggles`` to a JSON string.

    Raises:
        TypeError: ``scale`` is not a ForceColorScale or a toggle is not a bool.
    """
    if not isinstance(scale, ForceColorScale):
        raise TypeError("scale must be ForceColorScale")
    if not isinstance(toggles, Mapping) or any(
        not isinstance(k, str) or not isinstance(v, bool) for k, v in toggles.items()
    ):
        raise TypeError("toggles must map str to bool")
    return json.dumps({"scale": scale.to_dict(), "toggles": dict(toggles)})


def parse_force_color_state(text: Any) -> tuple[ForceColorScale, dict[str, bool]]:
    """Parse :func:`dump_force_color_state` output; never raises.

    Postcondition: returns a valid scale (defaults on any failure; ``None`` is
    a silent first run) and only the boolean toggles.
    """
    if text is None:
        return ForceColorScale(), {}
    try:
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise TypeError("stored force-colour state must be an object")
        scale = ForceColorScale.from_dict(payload.get("scale", {}))
        raw = payload.get("toggles", {})
        if not isinstance(raw, dict):
            raise TypeError("stored toggles must be an object")
    except (ValueError, TypeError, UnicodeDecodeError) as error:
        logger.warning("Ignoring invalid stored force-colour settings: %s", error)
        return ForceColorScale(), {}
    return scale, {k: v for k, v in raw.items() if isinstance(v, bool)}


def save_force_color_state(
    settings: SettingsStore,
    engine: str,
    scale: ForceColorScale,
    toggles: Mapping[str, bool],
) -> None:
    """Write the state for ``engine`` into ``settings``."""
    settings.setValue(_key(engine), dump_force_color_state(scale, toggles))


def load_force_color_state(
    settings: SettingsStore, engine: str
) -> tuple[ForceColorScale, dict[str, bool]]:
    """Read the state for ``engine``; defaults when absent or invalid."""
    return parse_force_color_state(settings.value(_key(engine), None))


def open_force_color_settings() -> SettingsStore:
    """Return the shared ``QSettings`` (lazy import keeps this module Qt-free)."""
    from PyQt6.QtCore import QSettings

    return QSettings(SETTINGS_ORG, SETTINGS_APP)  # type: ignore[return-value]
