"""Renderer-independent signed axial-force color policy (newtons, tension positive).

Only display colors change: the policy never changes geometry or physical state.
Opaque sRGB hex colors provide an identical contract for Python and web clients.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from numbers import Real
from typing import Any, Final

DEFAULT_TENSION_COLOR: Final[str] = "#0000ff"
DEFAULT_COMPRESSION_COLOR: Final[str] = "#ff0000"
DEFAULT_NEUTRAL_COLOR: Final[str] = "#ffffff"
#: ``replace`` swaps the segment colour for the load colour (blended with the base
#: by ``opacity``); ``tint`` shifts the original colour toward the tension/
#: compression colour by ``opacity * fraction`` so unloaded segments keep their own.
BLEND_MODES: Final[tuple[str, ...]] = ("replace", "tint")

__all__ = [
    "BLEND_MODES",
    "DEFAULT_COMPRESSION_COLOR",
    "DEFAULT_NEUTRAL_COLOR",
    "DEFAULT_TENSION_COLOR",
    "ForceColorScale",
]


@dataclass(frozen=True)
class ForceColorScale:
    """Validated fixed scale; values beyond the limits saturate without rescaling.

    Limits must exceed the nonnegative neutral deadband. Missing/nonfinite samples
    retain the supplied base color, distinct from a measured zero (neutral color).
    Alpha is deliberately excluded so renderers preserve their original opacity.
    ``opacity`` (0..1) is the overlay strength used by :meth:`shade_rgb` and
    :meth:`shade_hex`; ``blend_mode`` selects ``replace`` or ``tint`` (see
    ``BLEND_MODES``). :meth:`color` is always the opaque load color.
    """

    enabled: bool = False
    tension_limit_n: float = 1000.0
    compression_limit_n: float = 1000.0
    deadband_n: float = 0.0
    tension_color: str = DEFAULT_TENSION_COLOR
    compression_color: str = DEFAULT_COMPRESSION_COLOR
    neutral_color: str = DEFAULT_NEUTRAL_COLOR
    opacity: float = 1.0
    blend_mode: str = "replace"

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
        if isinstance(self.opacity, bool) or not isinstance(self.opacity, Real):
            raise TypeError("opacity must be a real number")
        if not math.isfinite(self.opacity) or not 0.0 <= self.opacity <= 1.0:
            raise ValueError("opacity must be finite and within [0, 1]")
        object.__setattr__(self, "opacity", float(self.opacity))
        if not isinstance(self.blend_mode, str):
            raise TypeError("blend_mode must be a string")
        if self.blend_mode not in BLEND_MODES:
            raise ValueError(f"blend_mode must be one of {BLEND_MODES}")
        for name in ("tension_color", "compression_color", "neutral_color"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if re.fullmatch(r"#[0-9a-fA-F]{6}", value) is None:
                raise ValueError(f"{name} must be an opaque #RRGGBB color")
            object.__setattr__(self, name, value.lower())

    def _scalar_scale(self):  # noqa: ANN202 — lazy import avoids cycle at load time
        from .scalar_shading import ScalarColorScale

        return ScalarColorScale.from_force_color_scale(self)

    @staticmethod
    def _check_force(force_n: float | Real | None) -> None:
        if force_n is not None and (
            isinstance(force_n, bool) or not isinstance(force_n, Real)
        ):
            raise TypeError("force_n must be a real number or None")

    def color(self, force_n: float | Real | None, base_color: str) -> str:
        """Return base styling when off/unavailable, otherwise clipped sRGB color."""
        if not isinstance(base_color, str):
            raise TypeError("base_color must be a string")
        self._check_force(force_n)
        return self._scalar_scale().color(force_n, base_color)

    def shade_rgb(
        self, force_n: float | Real | None, base_rgb: Sequence[float]
    ) -> tuple[float, float, float] | None:
        """Blend the load color over ``base_rgb`` per ``opacity`` and ``blend_mode``."""
        self._check_force(force_n)
        return self._scalar_scale().shade_rgb(force_n, base_rgb)

    def shade_hex(self, force_n: float | Real | None, base_color: str) -> str | None:
        """Hex form of :meth:`shade_rgb`; None means the caller keeps ``base_color``."""
        if not isinstance(base_color, str):
            raise TypeError("base_color must be a string")
        self._check_force(force_n)
        return self._scalar_scale().shade_hex(force_n, base_color)

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
