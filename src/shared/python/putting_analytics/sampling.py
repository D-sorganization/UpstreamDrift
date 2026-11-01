"""Vectorised Monte Carlo draws of stroke execution and read error.

Each channel draws from its own ``SeedSequence`` substream, so ablating one
channel (``profile.without(...)``) leaves every other channel's draws
identical: differences between runs are then due to the ablation, not noise.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.putting_analytics.skill import (
    CHANNELS,
    READ_CHANNELS,
    STROKE_CHANNELS,
    ReadDispersion,
    StrokeDispersion,
)


@dataclass(frozen=True)
class ExecutionSample:
    """Per-putt executed face/path error (deg), speed ratio and toe strike (mm)."""

    face_deg: np.ndarray
    path_deg: np.ndarray
    speed_ratio: np.ndarray
    strike_toe_mm: np.ndarray

    def start_line_error_deg(self, face_weight: float) -> np.ndarray:
        """Start direction error as a face/path blend."""
        return face_weight * self.face_deg + (1.0 - face_weight) * self.path_deg


@dataclass(frozen=True)
class ReadSample:
    """Per-putt perceived/true grade ratio, aspect error (deg), stimp ratio."""

    grade_ratio: np.ndarray
    aspect_error_deg: np.ndarray
    stimp_ratio: np.ndarray


def _standard_normal(seed: int, channel: str, n: int) -> np.ndarray:
    stream = np.random.SeedSequence(seed, spawn_key=(CHANNELS.index(channel),))
    return np.random.default_rng(stream).standard_normal(n)


def _deviates(seed: int, channels: tuple[str, ...], n: int) -> dict[str, np.ndarray]:
    return {name: _standard_normal(seed, name, n) for name in channels}


def _ratio(z: np.ndarray, cv: float, bias: float) -> np.ndarray:
    """Log-normal multiplier: always positive, exact mean ``1 + bias``, exact CV."""
    sigma = np.sqrt(np.log1p(cv**2))
    return (1.0 + bias) * np.exp(sigma * z - 0.5 * sigma**2)


def _check(n: int, seed: int) -> None:
    require(n >= 1, "n must be at least 1", n)
    require(seed >= 0, "seed must be non-negative", seed)


def execution_from_deviates(
    stroke: StrokeDispersion, z: Mapping[str, np.ndarray]
) -> ExecutionSample:
    """Map standard-normal deviates keyed by stroke channel to executed strokes."""
    return ExecutionSample(
        face_deg=stroke.face_bias_deg + stroke.face_sd_deg * z["face"],
        path_deg=stroke.path_sd_deg * z["path"],
        speed_ratio=_ratio(z["speed"], stroke.speed_cv, stroke.speed_bias),
        strike_toe_mm=stroke.strike_toe_sd_mm * z["strike"],
    )


def read_from_deviates(read: ReadDispersion, z: Mapping[str, np.ndarray]) -> ReadSample:
    """Map standard-normal deviates keyed by read channel to green reads."""
    return ReadSample(
        grade_ratio=_ratio(z["read_grade"], read.grade_rel_sd, read.grade_rel_bias),
        aspect_error_deg=read.aspect_sd_deg * z["read_aspect"],
        stimp_ratio=_ratio(z["read_stimp"], read.stimp_rel_sd, 0.0),
    )


def sample_execution(stroke: StrokeDispersion, n: int, seed: int) -> ExecutionSample:
    """Draw ``n`` executed strokes for ``stroke``; reproducible for a given seed."""
    _check(n, seed)
    return execution_from_deviates(stroke, _deviates(seed, STROKE_CHANNELS, n))


def sample_read(read: ReadDispersion, n: int, seed: int) -> ReadSample:
    """Draw ``n`` green reads for ``read``; reproducible for a given seed."""
    _check(n, seed)
    return read_from_deviates(read, _deviates(seed, READ_CHANNELS, n))
