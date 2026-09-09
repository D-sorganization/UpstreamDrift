"""Renderer-independent signed axial-force color policy (newtons, tension positive).

Only display colors change: the policy never changes geometry or physical state.
Opaque sRGB hex colors provide an identical contract for Python and web clients.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from numbers import Real
from typing import Any


@dataclass(frozen=True)
class ForceColorScale:
    """Validated fixed scale; values beyond the limits saturate without rescaling.

    Limits must exceed the nonnegative neutral deadband. Missing/nonfinite samples
    retain the supplied base color, distinct from a measured zero (neutral color).
    Alpha is deliberately excluded so renderers preserve their original opacity.
    """

    enabled: bool = False
    tension_limit_n: float = 1000.0
    compression_limit_n: float = 1000.0
    deadband_n: float = 0.0
    tension_color: str = "#0000ff"
    compression_color: str = "#ff0000"
    neutral_color: str = "#ffffff"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be bool")
        for name in ("tension_limit_n", "compression_limit_n", "deadband_n"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))
        if self.deadband_n < 0:
            raise ValueError("deadband_n must be nonnegative")
        if min(self.tension_limit_n, self.compression_limit_n) <= self.deadband_n:
            raise ValueError("both limits must exceed deadband_n")
        for name in ("tension_color", "compression_color", "neutral_color"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if re.fullmatch(r"#[0-9a-fA-F]{6}", value) is None:
                raise ValueError(f"{name} must be an opaque #RRGGBB color")
            object.__setattr__(self, name, value.lower())

    def color(self, force_n: Real | None, base_color: str) -> str:
        """Return base styling when off/unavailable, otherwise clipped sRGB color."""
        if not isinstance(base_color, str):
            raise TypeError("base_color must be a string")
        if force_n is not None and (
            isinstance(force_n, bool) or not isinstance(force_n, Real)
        ):
            raise TypeError("force_n must be a real number or None")
        if not self.enabled or force_n is None or not math.isfinite(force_n):
            return base_color
        magnitude = abs(force_n)
        if magnitude <= self.deadband_n:
            return self.neutral_color
        endpoint = self.tension_color if force_n > 0 else self.compression_color
        limit = self.tension_limit_n if force_n > 0 else self.compression_limit_n
        fraction = min(1.0, (magnitude - self.deadband_n) / (limit - self.deadband_n))
        channels = []
        for offset in (1, 3, 5):
            start = int(self.neutral_color[offset : offset + 2], 16)
            end = int(endpoint[offset : offset + 2], 16)
            # Half-up rounding matches JavaScript Math.round for RGB channels.
            channels.append(math.floor(start + fraction * (end - start) + 0.5))
        return "#" + "".join(f"{channel:02x}" for channel in channels)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe settings without renderer or model state."""
        return asdict(self)

    @classmethod
    def from_dict(cls, settings: Mapping[str, Any]) -> ForceColorScale:
        """Validate persisted settings; unknown keys fail rather than disappear."""
        if not isinstance(settings, Mapping):
            raise TypeError("settings must be a mapping")
        unknown = set(settings) - {entry.name for entry in fields(cls)}
        if unknown:
            raise ValueError(
                f"unknown force-color settings: {sorted(unknown, key=str)}"
            )
        return cls(**settings)
