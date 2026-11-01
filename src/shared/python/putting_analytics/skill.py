"""Putting skill profile: stroke execution and green-reading dispersion.

All spreads are one standard deviation. No preset is calibrated yet; presets
wait for validation against published make rates (#14176).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace

from src.shared.python.contracts import require

_STROKE_FIELDS = {
    "face": "face_sd_deg",
    "path": "path_sd_deg",
    "speed": "speed_cv",
    "strike": "strike_toe_sd_mm",
}
_READ_FIELDS = {
    "read_grade": "grade_rel_sd",
    "read_aspect": "aspect_sd_deg",
    "read_stimp": "stimp_rel_sd",
}
CHANNELS: tuple[str, ...] = (*_STROKE_FIELDS, *_READ_FIELDS)
DEFAULT_FACE_WEIGHT = 0.83  # share of start line set by face; literature cites ~0.8-0.9


def _require_spreads(obj: object, names: dict[str, str]) -> None:
    for attr in names.values():
        value = getattr(obj, attr)
        require(value >= 0.0, f"{attr} must be non-negative", value)


@dataclass(frozen=True)
class StrokeDispersion:
    """Execution spread: face and path angle (deg), speed CV, toe strike (mm).

    ``face_bias_deg`` is a systematic face error; ``speed_bias`` is a
    fractional mean speed error (0.05 means 5 % too firm on average).
    """

    face_sd_deg: float
    path_sd_deg: float
    speed_cv: float
    strike_toe_sd_mm: float = 0.0
    face_bias_deg: float = 0.0
    speed_bias: float = 0.0

    def __post_init__(self) -> None:
        _require_spreads(self, _STROKE_FIELDS)
        require(self.speed_bias > -1.0, "speed_bias must exceed -1", self.speed_bias)


@dataclass(frozen=True)
class ReadDispersion:
    """Green-reading spread: relative grade, slope aspect (deg), relative stimp."""

    grade_rel_sd: float = 0.0
    aspect_sd_deg: float = 0.0
    stimp_rel_sd: float = 0.0
    grade_rel_bias: float = 0.0

    def __post_init__(self) -> None:
        _require_spreads(self, _READ_FIELDS)
        require(self.grade_rel_bias > -1.0, "grade_rel_bias must exceed -1")


@dataclass(frozen=True)
class PuttingSkillProfile:
    """One player's stroke and read dispersion, plus face/path start-line blend."""

    name: str
    stroke: StrokeDispersion
    read: ReadDispersion = field(default_factory=ReadDispersion)
    face_weight: float = DEFAULT_FACE_WEIGHT

    def __post_init__(self) -> None:
        require(0.0 <= self.face_weight <= 1.0, "face_weight must be in [0, 1]")

    def scaled(self, factor: float) -> PuttingSkillProfile:
        """Return a profile with every spread multiplied by ``factor``; biases kept."""
        require(factor >= 0.0, "factor must be non-negative", factor)
        return self._map_spreads(CHANNELS, lambda value: value * factor)

    def without(self, *channels: str) -> PuttingSkillProfile:
        """Return a profile with the named channels' spreads set to zero."""
        for channel in channels:
            require(channel in CHANNELS, f"unknown channel; expected one of {CHANNELS}")
        return self._map_spreads(channels, lambda _value: 0.0)

    def _map_spreads(
        self, channels: Iterable[str], transform: Callable[[float], float]
    ) -> PuttingSkillProfile:
        channels = set(channels)
        stroke = {
            attr: transform(getattr(self.stroke, attr))
            for name, attr in _STROKE_FIELDS.items()
            if name in channels
        }
        read = {
            attr: transform(getattr(self.read, attr))
            for name, attr in _READ_FIELDS.items()
            if name in channels
        }
        return replace(
            self, stroke=replace(self.stroke, **stroke), read=replace(self.read, **read)
        )
