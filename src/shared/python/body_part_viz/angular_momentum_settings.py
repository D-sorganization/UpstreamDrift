"""Persist angular-momentum overlay toggles (Qt-free)."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any, Protocol

from .angular_momentum_glyphs import AngularMomentumArrowStyle

logger = logging.getLogger(__name__)

SETTINGS_ORG = "UpstreamDrift"
SETTINGS_APP = "UpstreamDrift"
SCHEMA_KEY = "angular_momentum_overlay"

__all__ = [
    "dump_angular_momentum_state",
    "load_angular_momentum_state",
    "open_angular_momentum_settings",
    "parse_angular_momentum_state",
    "save_angular_momentum_state",
]


class SettingsStore(Protocol):
    def value(self, key: str, default: Any = None) -> Any: ...

    def setValue(self, key: str, value: Any) -> None:  # noqa: N802
        ...


def _key(engine: str) -> str:
    if not isinstance(engine, str) or not engine.strip():
        raise ValueError("engine key must be a non-empty string")
    return f"{SCHEMA_KEY}/{engine.strip()}"


def dump_angular_momentum_state(
    enabled: bool, style: AngularMomentumArrowStyle
) -> str:
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be bool")
    if not isinstance(style, AngularMomentumArrowStyle):
        raise TypeError("style must be AngularMomentumArrowStyle")
    return json.dumps({"enabled": enabled, "style": style.to_dict()})


def parse_angular_momentum_state(
    text: Any,
) -> tuple[bool, AngularMomentumArrowStyle]:
    if text is None:
        return False, AngularMomentumArrowStyle()
    try:
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict")
        enabled = bool(payload.get("enabled", False))
        style_raw = payload.get("style", {})
        style = (
            AngularMomentumArrowStyle.from_dict(style_raw)
            if isinstance(style_raw, Mapping)
            else AngularMomentumArrowStyle()
        )
        return enabled, style
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Invalid angular-momentum settings; using defaults: %s", exc)
        return False, AngularMomentumArrowStyle()


def save_angular_momentum_state(
    store: SettingsStore,
    engine: str,
    enabled: bool,
    style: AngularMomentumArrowStyle,
) -> None:
    store.setValue(_key(engine), dump_angular_momentum_state(enabled, style))


def load_angular_momentum_state(
    store: SettingsStore, engine: str
) -> tuple[bool, AngularMomentumArrowStyle]:
    return parse_angular_momentum_state(store.value(_key(engine)))


def open_angular_momentum_settings() -> SettingsStore:
    from PyQt6.QtCore import QSettings

    return QSettings(SETTINGS_ORG, SETTINGS_APP)