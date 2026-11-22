"""Persist segment net-power overlay settings (Qt-free)."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any, Protocol

from .scalar_shading import ScalarColorScale
from .segment_net_power import default_net_power_color_scale

logger = logging.getLogger(__name__)

SETTINGS_ORG = "UpstreamDrift"
SETTINGS_APP = "UpstreamDrift"

__all__ = [
    "load_net_power_state",
    "open_net_power_settings",
    "save_net_power_state",
]


class SettingsStore(Protocol):
    def value(self, key: str, default: Any = None) -> Any: ...

    def setValue(self, key: str, value: Any) -> None: ...  # noqa: N802


def _key(engine: str) -> str:
    if not isinstance(engine, str) or not engine.strip():
        raise ValueError("engine key must be a non-empty string")
    return f"net_power/{engine.strip()}"


def _parse_scale(payload: Mapping[str, Any]) -> ScalarColorScale:
    base = default_net_power_color_scale()
    merged = {**base.to_dict(), **dict(payload)}
    return ScalarColorScale.from_dict(merged)


def load_net_power_state(
    store: SettingsStore, engine: str
) -> tuple[ScalarColorScale, dict[str, bool]]:
    text = store.value(_key(engine))
    if text is None:
        return default_net_power_color_scale(), {}
    try:
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        payload = json.loads(text)
        scale = _parse_scale(payload.get("scale", {}))
        toggles = {
            str(k): bool(v)
            for k, v in (payload.get("toggles") or {}).items()
            if isinstance(k, str)
        }
        return scale, toggles
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Invalid net-power settings for %s: %s", engine, exc)
        return default_net_power_color_scale(), {}


def save_net_power_state(
    store: SettingsStore,
    engine: str,
    scale: ScalarColorScale,
    toggles: Mapping[str, bool],
) -> None:
    if not isinstance(scale, ScalarColorScale):
        raise TypeError("scale must be ScalarColorScale")
    payload = {"scale": scale.to_dict(), "toggles": dict(toggles)}
    store.setValue(_key(engine), json.dumps(payload))


def open_net_power_settings() -> SettingsStore:
    from PyQt6.QtCore import QSettings

    return QSettings(SETTINGS_ORG, SETTINGS_APP)
