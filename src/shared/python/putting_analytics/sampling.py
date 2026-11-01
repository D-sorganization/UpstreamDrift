"""Vectorised Monte Carlo draws of stroke execution and read error.

Each channel draws from its own ``SeedSequence`` substream, so ablating one
channel (``profile.without(...)``) leaves every other channel's draws
identical: differences between runs are then due to the ablation, not noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.putting_analytics.skill import (
    CHANNELS,
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


def _normal(seed: int, channel: str, n: int, mean: float, sd: float) -> np.ndarray:
    return mean + sd * _standard_normal(seed, channel, n)


def _ratio(seed: int, channel: str, n: int, cv: float, bias: float) -> np.ndarray:
    """Log-normal multiplier: always positive, exact mean ``1 + bias``, exact CV."""
    sigma = np.sqrt(np.log1p(cv**2))
    z = _standard_normal(seed, channel, n)
    return (1.0 + bias) * np.exp(sigma * z - 0.5 * sigma**2)


def _check(n: int, seed: int) -> None:
    require(n >= 1, "n must be at least 1", n)
    require(seed >= 0, "seed must be non-negative", seed)


def sample_execution(stroke: StrokeDispersion, n: int, seed: int) -> ExecutionSample:
    """Draw ``n`` executed strokes for ``stroke``; reproducible for a given seed."""
    _check(n, seed)
    return ExecutionSample(
        face_deg=_normal(seed, "face", n, stroke.face_bias_deg, stroke.face_sd_deg),
        path_deg=_normal(seed, "path", n, 0.0, stroke.path_sd_deg),
        speed_ratio=_ratio(seed, "speed", n, stroke.speed_cv, stroke.speed_bias),
        strike_toe_mm=_normal(seed, "strike", n, 0.0, stroke.strike_toe_sd_mm),
    )


def sample_read(read: ReadDispersion, n: int, seed: int) -> ReadSample:
    """Draw ``n`` green reads for ``read``; reproducible for a given seed."""
    _check(n, seed)
    return ReadSample(
        grade_ratio=_ratio(
            seed, "read_grade", n, read.grade_rel_sd, read.grade_rel_bias
        ),
        aspect_error_deg=_normal(seed, "read_aspect", n, 0.0, read.aspect_sd_deg),
        stimp_ratio=_ratio(seed, "read_stimp", n, read.stimp_rel_sd, 0.0),
    )
