"""Quantity-agnostic per-segment scalar series and colour scales (#15395)."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from .force_colors import ForceColorScale

from .series_validation import (
    coerce_finite_real_fields,
    normalize_segment_samples,
    validate_strictly_increasing_times,
)

DEFAULT_NEUTRAL_COLOR: Final[str] = "#ffffff"
DEFAULT_SEQ_COLORMAP_ID: Final[str] = "viridis"
SCALE_KINDS: Final[tuple[str, ...]] = ("sequential", "diverging")
DISPLAY_TRANSFORMS: Final[tuple[str, ...]] = ("linear", "sqrt")
SIGN_CONVENTIONS: Final[tuple[str, ...]] = (
    "unsigned",
    "signed-zero-center",
    "tension-positive",
)
BLEND_MODES: Final[tuple[str, ...]] = ("replace", "tint")
_SATURATION_NOTE: Final[str] = "Values beyond limits use endpoint colour (saturated)."
_COLORMAP_CACHE: dict[str, Any] = {}

__all__ = [
    "BLEND_MODES",
    "DISPLAY_TRANSFORMS",
    "SCALE_KINDS",
    "SIGN_CONVENTIONS",
    "ScalarColorLegend",
    "ScalarColorScale",
    "SegmentScalarSeries",
    "ShadeOutcome",
    "scalar_for_force_color_scale",
    "shade_outcome_hex",
]


def _hex_channels(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def _lerp_hex(neutral: str, endpoint: str, fraction: float) -> str:
    start, end = _hex_channels(neutral), _hex_channels(endpoint)
    channels = (
        math.floor(start[0] + fraction * (end[0] - start[0]) + 0.5),
        math.floor(start[1] + fraction * (end[1] - start[1]) + 0.5),
        math.floor(start[2] + fraction * (end[2] - start[2]) + 0.5),
    )
    return "#" + "".join(f"{c:02x}" for c in channels)


def _validate_opaque_hex(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if re.fullmatch(r"#[0-9a-fA-F]{6}", value) is None:
        raise ValueError(f"{name} must be an opaque #RRGGBB color")
    return value.lower()


@dataclass(frozen=True)
class ShadeOutcome:
    """Resolved display colour plus whether the sample hit a scale limit."""

    color: str
    saturated: bool


@dataclass(frozen=True)
class ScalarColorLegend:
    """Colour-bar hints for UI layers (units stay in physical space)."""

    units: str
    scale_kind: str
    sign_convention: str
    display_transform: str
    low_limit: float
    high_limit: float
    tick_values: tuple[float, ...]
    saturation_note: str
    colormap_id: str


@dataclass(frozen=True)
class SegmentScalarSeries:
    """Time-aligned per-segment scalars; None or non-finite means unavailable."""

    time_s: Sequence[float]
    values: Mapping[str, Sequence[float | None]]
    source: str
    units: str
    sign_convention: str

    def __post_init__(self) -> None:
        if not isinstance(self.units, str) or not self.units.strip():
            raise ValueError("units must be a nonempty string")
        if self.sign_convention not in SIGN_CONVENTIONS:
            raise ValueError(f"sign_convention must be one of {SIGN_CONVENTIONS}")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must describe the scalar producer")
        times = validate_strictly_increasing_times(self.time_s)
        copied = normalize_segment_samples(
            times, self.values, segment_value_label="sample count"
        )
        object.__setattr__(self, "time_s", times)
        object.__setattr__(self, "values", copied)

    def value_at(self, segment: str, frame_idx: int) -> float | None:
        if isinstance(frame_idx, bool) or not isinstance(frame_idx, Integral):
            raise TypeError("frame_idx must be an integer")
        if not 0 <= frame_idx < len(self.time_s):
            raise IndexError("frame_idx outside scalar recording")
        samples = self.values.get(segment)
        return None if samples is None else samples[frame_idx]


@dataclass(frozen=True)
class ScalarColorScale:
    """Fixed-limit scalar shading with optional display-only transform."""

    enabled: bool = False
    scale_kind: str = "sequential"
    units: str = "1"
    sign_convention: str = "unsigned"
    colormap_id: str = DEFAULT_SEQ_COLORMAP_ID
    low_limit: float = 0.0
    high_limit: float = 1.0
    negative_limit: float | None = None
    positive_limit: float | None = None
    deadband: float = 0.0
    low_color: str = "#440154"
    high_color: str = "#fde725"
    neutral_color: str = DEFAULT_NEUTRAL_COLOR
    display_transform: str = "linear"
    opacity: float = 1.0
    blend_mode: str = "replace"

    def __post_init__(self) -> None:
        self._validate_flags()
        self._validate_limits_and_colors()

    def _validate_flags(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be bool")
        if self.scale_kind not in SCALE_KINDS:
            raise ValueError(f"scale_kind must be one of {SCALE_KINDS}")
        if not isinstance(self.units, str) or not self.units.strip():
            raise ValueError("units must be a nonempty string")
        if self.sign_convention not in SIGN_CONVENTIONS:
            raise ValueError(f"sign_convention must be one of {SIGN_CONVENTIONS}")
        if self.display_transform not in DISPLAY_TRANSFORMS:
            raise ValueError(f"display_transform must be one of {DISPLAY_TRANSFORMS}")
        if not isinstance(self.blend_mode, str) or self.blend_mode not in BLEND_MODES:
            raise ValueError(f"blend_mode must be one of {BLEND_MODES}")
        if isinstance(self.opacity, bool) or not isinstance(self.opacity, Real):
            raise TypeError("opacity must be a real number")
        if not math.isfinite(self.opacity) or not 0.0 <= self.opacity <= 1.0:
            raise ValueError("opacity must be finite and within [0, 1]")
        object.__setattr__(self, "opacity", float(self.opacity))

    def _validate_limits_and_colors(self) -> None:
        coerce_finite_real_fields(
            self,
            ("low_limit", "high_limit", "deadband"),
            nonnegative=("deadband",),
        )
        for name in ("low_color", "high_color", "neutral_color"):
            object.__setattr__(
                self, name, _validate_opaque_hex(name, getattr(self, name))
            )
        if self.scale_kind == "sequential":
            if self.sign_convention != "unsigned":
                raise ValueError("sequential scales require unsigned sign_convention")
            if self.low_limit < 0:
                raise ValueError("sequential low_limit must be nonnegative")
            if self.high_limit <= self.low_limit:
                raise ValueError("high_limit must exceed low_limit")
            return
        if self.sign_convention == "tension-positive":
            for name, key in (
                ("negative_limit", "negative_limit"),
                ("positive_limit", "positive_limit"),
            ):
                raw = getattr(self, name)
                if raw is None:
                    raise ValueError(
                        f"{key} is required for tension-positive diverging"
                    )
                if isinstance(raw, bool) or not isinstance(raw, Real):
                    raise TypeError(f"{key} must be a real number")
                if not math.isfinite(raw) or raw <= self.deadband:
                    raise ValueError(f"{key} must be finite and exceed deadband")
                object.__setattr__(self, name, float(raw))
            return
        if self.sign_convention != "signed-zero-center":
            raise ValueError(
                "diverging scales need signed-zero-center or tension-positive"
            )
        if self.low_limit >= 0 or self.high_limit <= 0:
            raise ValueError("signed diverging limits must bracket zero")

    @classmethod
    def from_force_color_scale(cls, scale: Any) -> ScalarColorScale:
        return scalar_for_force_color_scale(scale)

    def _transform_magnitude(self, magnitude: float) -> float:
        if self.display_transform == "linear":
            return magnitude
        return math.sqrt(max(0.0, magnitude))

    def _magnitude_fraction(self, magnitude: float, limit: float) -> tuple[float, bool]:
        if magnitude <= self.deadband:
            return 0.0, False
        saturated = magnitude > limit
        span = self._transform_magnitude(limit) - self._transform_magnitude(
            self.deadband
        )
        if span <= 0:
            raise ValueError("limit must exceed deadband after display transform")
        capped = min(
            self._transform_magnitude(magnitude), self._transform_magnitude(limit)
        )
        fraction = min(
            1.0, max(0.0, (capped - self._transform_magnitude(self.deadband)) / span)
        )
        return fraction, saturated

    def _display_limits(self) -> tuple[float, float]:
        if (
            self.scale_kind == "diverging"
            and self.sign_convention == "tension-positive"
        ):
            return -float(self.negative_limit or 0), float(self.positive_limit or 0)
        return self.low_limit, self.high_limit

    def legend_descriptor(self, tick_count: int = 5) -> ScalarColorLegend:
        lo, hi = self._display_limits()
        if tick_count < 2:
            raise ValueError("tick_count must be at least 2")
        step = (hi - lo) / (tick_count - 1)
        ticks = tuple(lo + step * index for index in range(tick_count))
        return ScalarColorLegend(
            units=self.units,
            scale_kind=self.scale_kind,
            sign_convention=self.sign_convention,
            display_transform=self.display_transform,
            low_limit=lo,
            high_limit=hi,
            tick_values=ticks,
            saturation_note=_SATURATION_NOTE,
            colormap_id=self.colormap_id,
        )

    @staticmethod
    def _check_sample(sample: float | Real | None) -> None:
        if sample is not None and (
            isinstance(sample, bool) or not isinstance(sample, Real)
        ):
            raise TypeError("sample value must be a real number or None")

    def _active(self, sample: float | Real | None) -> bool:
        return self.enabled and sample is not None and math.isfinite(sample)

    def _diverging_endpoint(self, value: float) -> tuple[str, float, bool]:
        if self.sign_convention == "tension-positive":
            return self._tension_positive_endpoint(value)
        return self._signed_center_endpoint(value)

    def _tension_positive_endpoint(self, value: float) -> tuple[str, float, bool]:
        magnitude = abs(value)
        if magnitude <= self.deadband:
            return self.neutral_color, 0.0, False
        if value > 0:
            assert self.positive_limit is not None
            limit = self.positive_limit
            endpoint = self.high_color
        else:
            assert self.negative_limit is not None
            limit = self.negative_limit
            endpoint = self.low_color
        fraction, saturated = self._magnitude_fraction(magnitude, limit)
        return endpoint, fraction, saturated

    def _signed_center_endpoint(self, value: float) -> tuple[str, float, bool]:
        if abs(value) <= self.deadband:
            return self.neutral_color, 0.0, False
        if value > 0:
            limit = self.high_limit
            endpoint = self.high_color
            magnitude = value
        else:
            limit = abs(self.low_limit)
            endpoint = self.low_color
            magnitude = abs(value)
        fraction, saturated = self._magnitude_fraction(magnitude, limit)
        return endpoint, fraction, saturated

    def _resolve_load_rgb(self, endpoint: str, fraction: float) -> tuple[int, int, int]:
        return _hex_channels(_lerp_hex(self.neutral_color, endpoint, fraction))

    def _sequential_hex(self, value: float) -> tuple[str, bool]:
        transformed = self._transform_magnitude(value)
        lo = self._transform_magnitude(self.low_limit)
        hi = self._transform_magnitude(self.high_limit)
        saturated = value < self.low_limit or value > self.high_limit
        if hi <= lo:
            raise ValueError("invalid sequential limits after transform")
        t = min(1.0, max(0.0, (transformed - lo) / (hi - lo)))
        return self._sample_colormap(t), saturated

    def _sample_colormap(self, position: float) -> str:
        cmap = _COLORMAP_CACHE.get(self.colormap_id)
        if cmap is None:
            from matplotlib import colormaps

            from ..plot_style.colormaps import ColormapId, resolve_colormap_alias

            cmap_key = resolve_colormap_alias(ColormapId(self.colormap_id))
            cmap = colormaps[str(cmap_key.value)]
            _COLORMAP_CACHE[self.colormap_id] = cmap
        rgba = cmap(position)
        return "#" + "".join(f"{int(round(channel * 255)):02x}" for channel in rgba[:3])

    def evaluate(self, sample: float | Real | None, base_color: str) -> ShadeOutcome:
        if not isinstance(base_color, str):
            raise TypeError("base_color must be a string")
        self._check_sample(sample)
        if sample is None or not self._active(sample):
            return ShadeOutcome(base_color, False)
        value = float(sample)
        if self.scale_kind == "sequential":
            color, saturated = self._sequential_hex(value)
            return ShadeOutcome(color, saturated)
        endpoint, fraction, saturated = self._diverging_endpoint(value)
        color = _lerp_hex(self.neutral_color, endpoint, fraction)
        return ShadeOutcome(color, saturated)

    def color(self, sample: float | Real | None, base_color: str) -> str:
        return self.evaluate(sample, base_color).color

    def shade_rgb(
        self, sample: float | Real | None, base_rgb: Sequence[float]
    ) -> tuple[float, float, float] | None:
        base = self._check_rgb(base_rgb)
        self._check_sample(sample)
        if sample is None or not self._active(sample):
            return None
        outcome = self.evaluate(sample, "#000000")
        value = float(sample)
        if self.scale_kind == "diverging":
            endpoint, fraction, _ = self._diverging_endpoint(value)
            if self.blend_mode == "tint":
                target, weight = _hex_channels(endpoint), self.opacity * fraction
            else:
                target = self._resolve_load_rgb(endpoint, fraction)
                weight = self.opacity
        else:
            target = _hex_channels(outcome.color)
            weight = self.opacity
        return self._blend_channels(base, target, weight)

    @staticmethod
    def _blend_channels(
        base: tuple[float, float, float],
        target: tuple[int, int, int],
        weight: float,
    ) -> tuple[float, float, float]:
        return (
            base[0] + weight * (target[0] / 255.0 - base[0]),
            base[1] + weight * (target[1] / 255.0 - base[1]),
            base[2] + weight * (target[2] / 255.0 - base[2]),
        )

    def shade_hex(self, sample: float | Real | None, base_color: str) -> str | None:
        if not isinstance(base_color, str):
            raise TypeError("base_color must be a string")
        self._check_sample(sample)
        if sample is None or not self._active(sample):
            return None
        if self.opacity == 1.0 and self.blend_mode == "replace":
            return self.color(sample, base_color)
        from ..plot_style.color_utils import hex_to_rgba, rgba_to_hex

        rgb = self.shade_rgb(sample, hex_to_rgba(base_color)[:3])
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
        return asdict(self)

    @classmethod
    def from_dict(cls, settings: Mapping[str, Any]) -> ScalarColorScale:
        if not isinstance(settings, Mapping):
            raise TypeError("settings must be a mapping")
        allowed = {entry.name for entry in fields(cls)}
        unknown = set(settings) - allowed
        if unknown:
            raise ValueError(
                f"unknown scalar-color settings: {sorted(unknown, key=str)}"
            )
        return cls(**settings)


@lru_cache(maxsize=128)
def _cached_scalar_for_force(scale: ForceColorScale) -> ScalarColorScale:
    from .force_colors import ForceColorScale

    if not isinstance(scale, ForceColorScale):
        raise TypeError("scale must be ForceColorScale")
    return ScalarColorScale(
        enabled=scale.enabled,
        scale_kind="diverging",
        units="N",
        sign_convention="tension-positive",
        positive_limit=scale.tension_limit_n,
        negative_limit=scale.compression_limit_n,
        deadband=scale.deadband_n,
        high_color=scale.tension_color,
        low_color=scale.compression_color,
        neutral_color=scale.neutral_color,
        opacity=scale.opacity,
        blend_mode=scale.blend_mode,
        display_transform="linear",
    )


def scalar_for_force_color_scale(scale: Any) -> ScalarColorScale:
    from .force_colors import ForceColorScale

    if not isinstance(scale, ForceColorScale):
        raise TypeError("scale must be ForceColorScale")
    return _cached_scalar_for_force(scale)


def shade_outcome_hex(
    scale: ScalarColorScale, sample: float | Real | None, base_color: str
) -> ShadeOutcome:
    """Return resolved hex colour and saturation flag (blending uses shade_hex rules)."""
    shaded = scale.shade_hex(sample, base_color)
    if shaded is None:
        return ShadeOutcome(base_color, False)
    outcome = scale.evaluate(sample, base_color)
    if scale.opacity == 1.0 and scale.blend_mode == "replace":
        return outcome
    return ShadeOutcome(shaded, outcome.saturated)
