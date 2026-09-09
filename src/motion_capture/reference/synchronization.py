"""Invertible, bounded event synchronization for reference playback (#9881)."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

MIN_RATE = 0.25
MAX_RATE = 4.0


class EventAnchors(BaseModel):
    """Paired events with immutable, strictly increasing reference/scene times."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    reference: Mapping[str, float] = Field(min_length=1, max_length=32)
    scene: Mapping[str, float] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_anchors(self) -> Self:
        if set(self.reference) != set(self.scene):
            raise ValueError("Each event needs both reference and scene timestamps")
        if any(not key.strip() or len(key) > 100 for key in self.reference):
            raise ValueError("Event names must be visible and at most 100 characters")
        keys = sorted(self.reference, key=lambda key: self.reference[key])
        r = np.array([self.reference[key] for key in keys])
        s = np.array([self.scene[key] for key in keys])
        if not np.isfinite(r).all() or not np.isfinite(s).all():
            raise ValueError("Event timestamps must be finite")
        dr, ds = np.diff(r), np.diff(s)
        if (dr <= 0).any() or (ds <= 0).any():
            raise ValueError(
                "Events must have unique, increasing times in both clocks (strictly monotonic)"
            )
        if ((ds / dr < MIN_RATE) | (ds / dr > MAX_RATE)).any():
            raise ValueError(
                "Event interval rates must be positive and bounded within [0.25, 4.0]"
            )
        object.__setattr__(self, "reference", MappingProxyType(dict(self.reference)))
        object.__setattr__(self, "scene", MappingProxyType(dict(self.scene)))
        return self

    @field_serializer("reference", "scene")
    def serialize_times(self, value: Mapping[str, float]) -> dict[str, float]:
        return dict(value)


def _warp(
    values: npt.NDArray[np.float64],
    source: Mapping[str, float],
    target: Mapping[str, float],
    rate: float,
) -> npt.NDArray[np.float64]:
    keys = sorted(source, key=lambda key: source[key])
    x = np.array([source[key] for key in keys])
    y = np.array([target[key] for key in keys])
    if len(keys) == 1:
        eff_offset = y[0] - x[0] * rate
        return values * rate + eff_offset
    result = np.asarray(np.interp(values, x, y))
    for edge, neighbor, mask in ((0, 1, values < x[0]), (-1, -2, values > x[-1])):
        slope = (y[neighbor] - y[edge]) / (x[neighbor] - x[edge])
        result = np.where(mask, y[edge] + (values - x[edge]) * slope, result)
    return result


class TimeMapping(BaseModel):
    """Affine alignment or paired events, with linear endpoint extrapolation.

    Event pairs replace the affine offset. One pair retains rate_scale about
    that event; two or more pairs define their own bounded interval rates.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    offset_s: float = 0.0
    rate_scale: float = Field(default=1.0, ge=MIN_RATE, le=MAX_RATE)
    event_anchors: EventAnchors | None = None

    def _map(self, value: float | npt.NDArray[np.float64], inverse: bool) -> Any:
        arr = np.asarray(value, dtype=float)
        if not np.isfinite(arr).all():
            raise ValueError("Playback timestamps must be finite")
        anchors = self.event_anchors
        if anchors is not None:
            source, target = anchors.reference, anchors.scene
            if len(source) == 1:
                k = next(iter(source))
                eff_offset = target[k] - source[k] * self.rate_scale
                if inverse:
                    result = (arr - eff_offset) / self.rate_scale
                else:
                    result = arr * self.rate_scale + eff_offset
            else:
                if inverse:
                    source, target = target, source
                result = _warp(
                    arr,
                    source,
                    target,
                    1 / self.rate_scale if inverse else self.rate_scale,
                )
        elif inverse:
            result = (arr - self.offset_s) / self.rate_scale
        else:
            result = arr * self.rate_scale + self.offset_s
        return float(result) if arr.ndim == 0 else result

    def reference_to_scene(self, t_ref: float | npt.NDArray[np.float64]) -> Any:
        return self._map(t_ref, inverse=False)

    def scene_to_reference(self, t_scene: float | npt.NDArray[np.float64]) -> Any:
        return self._map(t_scene, inverse=True)
