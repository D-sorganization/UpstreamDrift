"""Persist stored kinetic-energy shading and debug-proxy toggles (#15396)."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any, Protocol

from src.shared.python.segment_kinetic_energy.shading import (
    KE_COMPONENTS,
    default_stored_ke_color_scale,
)

from .scalar_shading import ScalarColorScale

logger = logging.getLogger(__name__)

SETTINGS_ORG = "UpstreamDrift"
SETTINGS_APP = "UpstreamDrift"

__all__ = [
    "load_segment_ke_state",
    "parse_segment_ke_state",
    "save_segment_ke_state",
]


class SettingsStore(Protocol):
    def value(self, key: str, default: Any = None) -> Any: ...

    def setValue(self, key: str, value: Any) -> None:  # noqa: N802
        ...


def _key(engine: str) -> str:
    if not isinstance(engine, str) or not engine.strip():
        raise ValueError("engine key must be a non-empty string")
    return f"segment_ke_shading/{engine.strip()}"


def dump_segment_ke_state(
    scale: ScalarColorScale,
    *,
    mode: str,
    toggles: Mapping[str, bool],
) -> str:
    if not isinstance(scale, ScalarColorScale):
        raise TypeError("scale must be ScalarColorScale")
    if mode not in KE_COMPONENTS:
        raise ValueError(f"mode must be one of {KE_COMPONENTS}")
    if not isinstance(toggles, Mapping) or any(
        not isinstance(k, str) or not isinstance(v, bool) for k, v in toggles.items()
    ):
        raise TypeError("toggles must map str to bool")
    return json.dumps(
        {"scale": scale.to_dict(), "mode": mode, "toggles": dict(toggles)}
    )


def parse_segment_ke_state(text: Any) -> tuple[ScalarColorScale, str, dict[str, bool]]:
    if text is None:
        return default_stored_ke_color_scale(), "total", {}
    try:
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise TypeError("stored segment KE state must be an object")
        scale = ScalarColorScale.from_dict(payload.get("scale", {}))
        mode = payload.get("mode", "total")
        if mode not in KE_COMPONENTS:
            raise ValueError(f"unknown KE component mode: {mode!r}")
        raw = payload.get("toggles", {})
        if not isinstance(raw, dict):
            raise TypeError("stored toggles must be an object")
    except (ValueError, TypeError, UnicodeDecodeError) as error:
        logger.warning("Ignoring invalid stored segment KE settings: %s", error)
        return default_stored_ke_color_scale(), "total", {}
    return scale, mode, {k: v for k, v in raw.items() if isinstance(v, bool)}


def save_segment_ke_state(
    settings: SettingsStore,
    engine: str,
    scale: ScalarColorScale,
    *,
    mode: str,
    toggles: Mapping[str, bool],
) -> None:
    settings.setValue(
        _key(engine), dump_segment_ke_state(scale, mode=mode, toggles=toggles)
    )


def load_segment_ke_state(
    settings: SettingsStore, engine: str
) -> tuple[ScalarColorScale, str, dict[str, bool]]:
    return parse_segment_ke_state(settings.value(_key(engine), None))
