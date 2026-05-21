"""Theme policy and persistence contract for Sidekick."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from .design_tokens import (
    _HEX_COLOR_RE,
    SIDEKICK_DESIGN_TOKENS,
    SidekickDesignTokens,
    _normalize_token_values,
)

MIN_FONT_SIZE_PX = 9
MAX_FONT_SIZE_PX = 24
MAX_FONT_FAMILY_LENGTH = 64


class SidekickThemeMode(str, Enum):
    """Constrained Sidekick theme resolution modes."""

    INHERIT_PARENT = "inherit_parent"
    CUSTOM = "custom"


@dataclass(frozen=True)
class SidekickFontSettings:
    """Validated custom font settings for Sidekick tokens."""

    family: str | None = None
    size_px: int | None = None

    def __post_init__(self) -> None:
        family = _normalize_font_family(self.family)
        size_px = _normalize_font_size(self.size_px)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "size_px", size_px)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.family is not None:
            payload["family"] = self.family
        if self.size_px is not None:
            payload["size_px"] = self.size_px
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> SidekickFontSettings:
        if not payload:
            return cls()
        return cls(
            family=payload.get("family"),
            size_px=payload.get("size_px"),
        )


@dataclass(frozen=True)
class SidekickThemeSettings:
    """Serializable Sidekick theme settings stored with sidebar state."""

    mode: SidekickThemeMode = SidekickThemeMode.INHERIT_PARENT
    colors: Mapping[str, str] = field(default_factory=dict)
    font: SidekickFontSettings = field(default_factory=SidekickFontSettings)

    def __post_init__(self) -> None:
        mode = _coerce_theme_mode(self.mode)
        colors = _normalize_custom_colors(self.colors)
        font = (
            self.font
            if isinstance(self.font, SidekickFontSettings)
            else SidekickFontSettings.from_dict(self.font)
        )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "colors", MappingProxyType(colors))
        object.__setattr__(self, "font", font)

    @classmethod
    def custom(
        cls,
        *,
        colors: Mapping[str, str] | None = None,
        font_family: str | None = None,
        font_size_px: int | None = None,
    ) -> SidekickThemeSettings:
        """Create custom theme settings with validated color and font payloads."""
        return cls(
            mode=SidekickThemeMode.CUSTOM,
            colors=colors or {},
            font=SidekickFontSettings(family=font_family, size_px=font_size_px),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> SidekickThemeSettings:
        if not payload:
            return cls()
        custom = payload.get("custom")
        custom_payload = custom if isinstance(custom, Mapping) else {}
        return cls(
            mode=payload.get("mode", SidekickThemeMode.INHERIT_PARENT),
            colors=custom_payload.get("colors", {}),
            font=SidekickFontSettings.from_dict(custom_payload.get("font")),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"mode": self.mode.value}
        custom: dict[str, Any] = {}
        if self.colors:
            custom["colors"] = dict(self.colors)
        font_payload = self.font.to_dict()
        if font_payload:
            custom["font"] = font_payload
        if custom:
            payload["custom"] = custom
        return payload


def resolve_sidekick_theme(
    *,
    parent_tokens: SidekickDesignTokens | None = None,
    settings: SidekickThemeSettings | Mapping[str, Any] | None = None,
) -> SidekickDesignTokens:
    """Resolve parent-inherited or custom Sidekick design tokens."""
    theme_settings = (
        settings
        if isinstance(settings, SidekickThemeSettings)
        else SidekickThemeSettings.from_dict(settings)
    )
    base = parent_tokens or SIDEKICK_DESIGN_TOKENS
    if theme_settings.mode is SidekickThemeMode.INHERIT_PARENT:
        return base
    overrides = dict(theme_settings.colors)
    if theme_settings.font.family is not None:
        overrides["font.family"] = theme_settings.font.family
    if theme_settings.font.size_px is not None:
        overrides["font.size"] = f"{theme_settings.font.size_px}px"
    return base.with_overrides(**overrides)


def _coerce_theme_mode(value: Any) -> SidekickThemeMode:
    try:
        if isinstance(value, SidekickThemeMode):
            return value

        # When moving from StrEnum (Python 3.11+) to str, Enum (Python 3.10)
        # str(enum_member) changes from "inherit_parent" to a full class name
        val_str = str(value)
        if val_str.startswith("SidekickThemeMode."):
            val_str = val_str.split(".")[1].lower()

        return SidekickThemeMode(val_str)
    except ValueError as exc:
        message = "Sidekick theme mode must be inherit_parent or custom"
        raise ValueError(message) from exc


def _normalize_custom_colors(values: Mapping[str, str]) -> dict[str, str]:
    normalized = _normalize_token_values(values)
    result: dict[str, str] = {}
    for name, value in normalized.items():
        if not name.startswith("color."):
            continue
        color = str(value).strip()
        if not _HEX_COLOR_RE.match(color):
            raise ValueError(f"Invalid custom Sidekick color for {name}: {color}")
        result[name] = color
    return result


def _normalize_font_family(value: Any) -> str | None:
    if value is None:
        return None
    family = str(value).strip()
    if not family or len(family) > MAX_FONT_FAMILY_LENGTH:
        raise ValueError("Sidekick font family must be 1-64 characters")
    if any(char in family for char in "\n\r\t;{}"):
        raise ValueError("Sidekick font family contains unsupported characters")
    return family


def _normalize_font_size(value: Any) -> int | None:
    if value is None:
        return None
    size = int(value)
    if not MIN_FONT_SIZE_PX <= size <= MAX_FONT_SIZE_PX:
        raise ValueError("Sidekick font size must be between 9 and 24 px")
    return size
