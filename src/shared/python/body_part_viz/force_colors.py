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

    @staticmethod
    def _check_force(force_n: float | Real | None) -> None:
        if force_n is not None and (
            isinstance(force_n, bool) or not isinstance(force_n, Real)
        ):
            raise TypeError("force_n must be a real number or None")

    def _active(self, force_n: float | Real | None) -> bool:
        return self.enabled and force_n is not None and math.isfinite(force_n)

    def _endpoint_fraction(self, force_n: float | Real) -> tuple[str, float]:
        """Return (endpoint color, saturation fraction); the deadband yields 0."""
        force = float(force_n)
        magnitude = abs(force)
        if magnitude <= self.deadband_n:
            return self.neutral_color, 0.0
        endpoint = self.tension_color if force > 0 else self.compression_color
        limit = self.tension_limit_n if force > 0 else self.compression_limit_n
        fraction = min(1.0, (magnitude - self.deadband_n) / (limit - self.deadband_n))
        return endpoint, fraction

    @staticmethod
    def _channels(color: str) -> tuple[int, int, int]:
        return (
            int(color[1:3], 16),
            int(color[3:5], 16),
            int(color[5:7], 16),
        )

    def _load_channels(self, endpoint: str, fraction: float) -> tuple[int, int, int]:
        """Neutral-to-endpoint interpolation, half-up rounded like JS Math.round."""
        start, end = self._channels(self.neutral_color), self._channels(endpoint)
        return (
            math.floor(start[0] + fraction * (end[0] - start[0]) + 0.5),
            math.floor(start[1] + fraction * (end[1] - start[1]) + 0.5),
            math.floor(start[2] + fraction * (end[2] - start[2]) + 0.5),
        )

    def color(self, force_n: float | Real | None, base_color: str) -> str:
        """Return base styling when off/unavailable, otherwise clipped sRGB color."""
        if not isinstance(base_color, str):
            raise TypeError("base_color must be a string")
        self._check_force(force_n)
        if not self._active(force_n):
            return base_color
        assert force_n is not None
        endpoint, fraction = self._endpoint_fraction(force_n)
        return "#" + "".join(
            f"{c:02x}" for c in self._load_channels(endpoint, fraction)
        )

    def shade_rgb(
        self, force_n: float | Real | None, base_rgb: Sequence[float]
    ) -> tuple[float, float, float] | None:
        """Blend the load color over ``base_rgb`` per ``opacity`` and ``blend_mode``.

        Preconditions: ``base_rgb`` is three finite floats in [0, 1].
        Postcondition: None when disabled or the force is missing/non-finite (the
        caller keeps its base); otherwise three floats in [0, 1]. ``replace``
        returns ``base + opacity * (load_color - base)``; ``tint`` returns
        ``base + opacity * fraction * (endpoint - base)`` so zero load is unchanged.
        """
        base = self._check_rgb(base_rgb)
        self._check_force(force_n)
        if not self._active(force_n):
            return None
        assert force_n is not None
        endpoint, fraction = self._endpoint_fraction(force_n)
        if self.blend_mode == "tint":
            target, weight = self._channels(endpoint), self.opacity * fraction
        else:
            target, weight = self._load_channels(endpoint, fraction), self.opacity
        return (
            base[0] + weight * (target[0] / 255.0 - base[0]),
            base[1] + weight * (target[1] / 255.0 - base[1]),
            base[2] + weight * (target[2] / 255.0 - base[2]),
        )

    def shade_hex(self, force_n: float | Real | None, base_color: str) -> str | None:
        """Hex form of :meth:`shade_rgb`; None means the caller keeps ``base_color``.

        An opaque ``replace`` scale ignores the base, so legacy callers passing a
        blank base keep working; any blending needs a valid ``#RGB``/``#RRGGBB``.
        """
        if not isinstance(base_color, str):
            raise TypeError("base_color must be a string")
        self._check_force(force_n)
        if not self._active(force_n):
            return None
        if self.opacity == 1.0 and self.blend_mode == "replace":
            return self.color(force_n, base_color)
        # Deferred: plot_style.colormaps imports this module's default colors.
        from ..plot_style.color_utils import hex_to_rgba, rgba_to_hex

        rgb = self.shade_rgb(force_n, hex_to_rgba(base_color)[:3])
        return None if rgb is None else rgba_to_hex(rgb)

    @staticmethod
    def _check_rgb(base_rgb: Sequence[float]) -> tuple[float, float, float]:
        if isinstance(base_rgb, (str, bytes)) or not isinstance(base_rgb, Sequence):
            raise TypeError("base_rgb must be a sequence of three floats")
        if len(base_rgb) != 3:
            raise ValueError("base_rgb must have exactly three channels")
        for value in base_rgb:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("base_rgb channels must be real numbers")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("base_rgb channels must be finite and within [0, 1]")
        return float(base_rgb[0]), float(base_rgb[1]), float(base_rgb[2])

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
